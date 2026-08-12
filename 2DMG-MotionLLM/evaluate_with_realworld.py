"""
Evaluate adapter + 2D VQ-VAE on real-world VitPose-estimated keypoints (M2T task).

Pipeline (with adapter):
    VitPose JSON (COCO-17) → COCO-13 → 81-dim feature
        → Adapter → [zero-pad to 263] → HumanVQVAE 2D encoder (frozen)
        → token string → T5 → predicted text

Pipeline (no adapter):
    VitPose JSON (COCO-17) → COCO-13 → 68-dim feature (no confidence)
        → [zero-pad to 263] → HumanVQVAE 2D encoder (frozen)
        → token string → T5 → predicted text

Expected data layout:
    <estimated_motion_dir>/json/<motion_id>.json
Each JSON: list of frame dicts [{..., "instances": [{"keypoints": ..., "keypoint_scores": ...}]}, ...]

Usage:
    cd 2DMG-MotionLLM
    conda run -n mg-motionllm python evaluate_with_realworld.py \\
        --estimated_motion_dir ./datasets/real_world_dataset_ver2/pred \\
        --adapter_ckpt ./checkpoints/adapter/MG-MotionLLM/adapter-bleu4-bs64-lr1e-3/best_adapter_epoch330_bleu40.1163.pt \\
        [--mode both]
"""

import argparse
import codecs as cs
import glob
import json
import os
import random
import sys
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import models.vqvae as vqvae_module
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.evaluate import calculate_bleu_rouge_cider


# ---------------------------------------------------------------------------
# Adapter models
# ---------------------------------------------------------------------------

class AdapterLinear(nn.Module):
    def __init__(self, dim=81):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.linear(self.norm(x))


class AdapterResidual(nn.Module):
    def __init__(self, dim=81, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class AdapterConv1d(nn.Module):
    def __init__(self, dim=81, hidden=512, kernel_size=3, dropout=0.1):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):              # x: (B, T, dim)
        r = self.norm(x)
        r = r.permute(0, 2, 1)
        r = self.conv1(r)
        r = self.act(r)
        r = self.drop(r)
        r = self.conv2(r)
        r = r.permute(0, 2, 1)
        return x + r


def build_adapter(adapter_type, dim=81, hidden=512, kernel_size=3):
    if adapter_type == "linear":
        return AdapterLinear(dim=dim)
    elif adapter_type == "residual":
        return AdapterResidual(dim=dim, hidden=hidden)
    elif adapter_type == "conv1d":
        return AdapterConv1d(dim=dim, hidden=hidden, kernel_size=kernel_size)
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")


# ---------------------------------------------------------------------------
# VitPose feature helpers  (COCO-17 → COCO-13)
# ---------------------------------------------------------------------------

_COCO13_LHIP    = 7
_COCO13_RHIP    = 8
_COCO17_TO_13   = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def _normalize_2d_coco13_midhip(joints_2d):
    joints_2d = np.asarray(joints_2d)
    root_pos   = 0.5 * (joints_2d[:, _COCO13_LHIP, :] + joints_2d[:, _COCO13_RHIP, :])
    joints_rel = joints_2d - root_pos[:, None, :]
    abs_xy     = np.abs(joints_rel).reshape(-1, 2)
    s          = max(np.percentile(abs_xy[:, 0], 99), np.percentile(abs_xy[:, 1], 99), 1e-8)
    return root_pos, joints_rel, s


def _preprocess_estimated(motion_2d, conf):
    """COCO-13 (T,13,2) + conf (T,13) → 81-dim feature (with confidence)."""
    root_pos, joints_rel, s = _normalize_2d_coco13_midhip(motion_2d)
    T = root_pos.shape[0]

    root_y_2d     = (root_pos[:, 1:2] / s).astype(np.float32)
    root_y_2d    -= root_y_2d[0:1]
    joints_pos_2d = (joints_rel / s).reshape(T, -1).astype(np.float32)
    root_norm     = (root_pos / s).astype(np.float32)
    root_vel_2d   = np.zeros_like(root_norm)
    root_vel_2d[1:] = root_norm[1:] - root_norm[:-1]

    jrn  = (joints_rel / s).astype(np.float32)
    rot  = np.arctan2(jrn[:, :, 1], jrn[:, :, 0])
    vel  = np.zeros_like(jrn); vel[1:] = jrn[1:] - jrn[:-1]
    vel  = vel.reshape(T, -1)

    c = conf.reshape(T, -1).astype(np.float32)
    return np.concatenate([root_vel_2d, root_y_2d, joints_pos_2d, rot, vel, c], axis=-1)


def _preprocess_no_conf(motion_2d):
    """COCO-13 (T,13,2) → 68-dim feature (without confidence)."""
    root_pos, joints_rel, s = _normalize_2d_coco13_midhip(motion_2d)
    T = root_pos.shape[0]

    root_y_2d     = (root_pos[:, 1:2] / s).astype(np.float32)
    root_y_2d    -= root_y_2d[0:1]
    joints_pos_2d = (joints_rel / s).reshape(T, -1).astype(np.float32)
    root_norm     = (root_pos / s).astype(np.float32)
    root_vel_2d   = np.zeros_like(root_norm)
    root_vel_2d[1:] = root_norm[1:] - root_norm[:-1]

    jrn  = (joints_rel / s).astype(np.float32)
    rot  = np.arctan2(jrn[:, :, 1], jrn[:, :, 0])
    vel  = np.zeros_like(jrn); vel[1:] = jrn[1:] - jrn[:-1]
    vel  = vel.reshape(T, -1)

    return np.concatenate([root_vel_2d, root_y_2d, joints_pos_2d, rot, vel], axis=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealWorldEvalDataset(Dataset):
    """
    Loads real-world VitPose JSONs from <estimated_motion_dir>/json/<motion_id>.json.
    GT captions matched from HumanML3D texts/ by motion_id.
    Returns (motion_id, feat_81dim, feat_68dim, m_length, all_captions).
    """

    def __init__(self, estimated_motion_dir, data_root, split,
                 mean_est, std_est, mean_2d, std_2d,
                 unit_length=4, min_len=40, max_len=196):
        self.unit_length = unit_length
        self.min_len     = min_len
        self.max_len     = max_len
        self.mean_est    = mean_est
        self.std_est     = std_est
        self.mean_2d     = mean_2d
        self.std_2d      = std_2d

        split_ids = set()
        split_path = pjoin(data_root, f'{split}.txt')
        if os.path.exists(split_path):
            with cs.open(split_path) as f:
                for line in f:
                    split_ids.add(line.strip())

        json_paths = sorted(glob.glob(os.path.join(estimated_motion_dir, "json", "*.json")))
        print(f"[RealWorldEvalDataset] Found {len(json_paths)} JSON files")

        self.samples = []
        skipped = 0
        for jp in json_paths:
            mid = os.path.basename(jp).replace(".json", "")
            if split_ids and mid not in split_ids:
                skipped += 1
                continue
            text_path = pjoin(data_root, "texts", mid + ".txt")
            if not os.path.exists(text_path):
                skipped += 1
                continue

            data = json.load(open(jp))
            motions, confs = [], []
            for frame in data:
                motions.append(frame["instances"][0]["keypoints"])
                confs.append(frame["instances"][0]["keypoint_scores"])
            motions = np.array(motions)[:, _COCO17_TO_13, :]
            confs   = np.array(confs)[:,   _COCO17_TO_13]

            if motions.shape[0] < min_len:
                skipped += 1
                continue

            captions = []
            with cs.open(text_path) as f:
                for line in f.readlines():
                    try:
                        parts  = line.strip().split('#')
                        t_toks = parts[1].split(' ')
                        cap    = ' '.join(tok.split('/')[0] for tok in t_toks)
                        f_tag  = float(parts[2]) if len(parts) > 2 else 0.0
                        to_tag = float(parts[3]) if len(parts) > 3 else 0.0
                        if np.isnan(f_tag):  f_tag  = 0.0
                        if np.isnan(to_tag): to_tag = 0.0
                        if f_tag == 0.0 and to_tag == 0.0:
                            captions.append(cap)
                    except Exception:
                        pass
            if not captions:
                skipped += 1
                continue

            if len(captions) > 3:   captions = captions[:3]
            elif len(captions) == 2: captions = captions + captions[:1]
            elif len(captions) == 1: captions = captions * 3

            self.samples.append((mid, motions, confs, captions))

        print(f"[RealWorldEvalDataset] {len(self.samples)} samples loaded, {skipped} skipped (split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, motion_2d, conf, captions = self.samples[idx]
        T = motion_2d.shape[0]

        m_len = min(T, self.max_len)
        m_len = (m_len // self.unit_length) * self.unit_length
        m_len = max(m_len, self.unit_length)

        start = random.randint(0, max(0, T - m_len))
        motion_2d = motion_2d[start:start + m_len]
        conf      = conf[start:start + m_len]

        feat_81 = _preprocess_estimated(motion_2d, conf)
        feat_68 = _preprocess_no_conf(motion_2d)

        feat_81 = (feat_81 - self.mean_est) / (self.std_est + 1e-8)
        feat_68 = (feat_68 - self.mean_2d)  / (self.std_2d  + 1e-8)

        return mid, feat_81.astype(np.float32), feat_68.astype(np.float32), m_len, captions


def collate_fn(batch):
    mids, feat81s, feat68s, lengths, caps = zip(*batch)
    max_t = max(f.shape[0] for f in feat81s)

    pad81 = torch.zeros(len(feat81s), max_t, feat81s[0].shape[1])
    pad68 = torch.zeros(len(feat68s), max_t, feat68s[0].shape[1])
    for i, (f81, f68) in enumerate(zip(feat81s, feat68s)):
        pad81[i, :f81.shape[0]] = torch.from_numpy(f81)
        pad68[i, :f68.shape[0]] = torch.from_numpy(f68)

    return {
        "mid":          list(mids),
        "feat_81":      pad81,
        "feat_68":      pad68,
        "length":       torch.LongTensor(list(lengths)),
        "all_captions": list(caps),
    }


# ---------------------------------------------------------------------------
# T5 generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_texts_t5(vae_2d, tokenizer, t5_model, feat_padded, device,
                      prompt="Generate text: ", max_new_tokens=40):
    """feat_padded: (1, T, 263) tensor on CPU."""
    feat_padded = feat_padded.to(device)
    tokenized   = vae_2d.encode(feat_padded)          # (1, L) tensor
    token_list  = tokenized.cpu().numpy()[0].tolist()

    motion_string = '<Motion Tokens>'
    for tok in token_list:
        motion_string += f'<{tok}>'
    motion_string += '</Motion Tokens>'

    input_ids = tokenizer(prompt + motion_string, return_tensors="pt").input_ids.to(device)
    outputs   = t5_model.generate(input_ids, max_length=max_new_tokens, num_beams=1, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip('"')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Dir with json/<motion_id>.json real-world VitPose keypoints")
    p.add_argument("--adapter_ckpt", type=str, default=None,
                   help="Adapter checkpoint (.pt, key 'model_state_dict')")
    p.add_argument("--vqvae_2d_ckpt", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "2d_vq_train" / "t2m" /
                               "best_2dvq_epoch1881_ratio0.5557.pt"))
    p.add_argument("--vqvae_3d_pth", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "pretrained_vqvae" / "t2m.pth"))
    p.add_argument("--model_name", type=str,
                   default=str(ROOT_DIR / "m2t-ft-from-GSPretrained-base"))
    p.add_argument("--meta_dir", type=str,
                   default="../2DMotionGPT/deps/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta")
    p.add_argument("--data_root", type=str,
                   default=str(ROOT_DIR / "dataset" / "HumanML3D"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--mode", type=str, default="both",
                   choices=["adapter", "no_adapter", "both"])
    p.add_argument("--adapter_type", type=str, default="residual",
                   choices=["linear", "residual", "conv1d"],)
    p.add_argument("--adapter_hidden", type=int, default=512,)
    p.add_argument("--adapter_kernel_size", type=int, default=3,)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--prompt", type=str, default="Generate text: ")
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--unit_length", type=int, default=4)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Stats ---
    mean_est = np.load(pjoin(args.meta_dir, "mean_2d_coco_estimated_concatenate.npy"))
    std_est  = np.load(pjoin(args.meta_dir, "std_2d_coco_estimated_concatenate.npy"))
    mean_2d  = np.load(pjoin(args.meta_dir, "mean_2d_coco_normalized.npy"))
    std_2d   = np.load(pjoin(args.meta_dir, "std_2d_coco_normalized.npy"))

    # --- Dataset ---
    dataset = RealWorldEvalDataset(
        estimated_motion_dir=args.estimated_motion_dir,
        data_root=args.data_root,
        split=args.split,
        mean_est=mean_est, std_est=std_est,
        mean_2d=mean_2d,   std_2d=std_2d,
        unit_length=args.unit_length,
    )
    if len(dataset) == 0:
        print("No samples found. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    # --- HumanVQVAE (3D base + 2D encoder override) ---
    import argparse as _ap
    vae_args = _ap.Namespace(dataname='t2m', quantizer='ema_reset', mu=0.99)

    def _build_vae():
        return vqvae_module.HumanVQVAE(
            vae_args, nb_code=512, code_dim=512, output_emb_width=512,
            down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
        ).to(device)

    ckpt_3d = torch.load(args.vqvae_3d_pth, map_location="cpu", weights_only=False)
    vae_2d  = _build_vae()
    vae_2d.load_state_dict(ckpt_3d['net'], strict=True)
    ckpt_2d = torch.load(args.vqvae_2d_ckpt, map_location="cpu", weights_only=False)
    vae_2d.load_state_dict(
        {k: v for k, v in ckpt_2d['net'].items() if k.startswith("vqvae.encoder")},
        strict=False,
    )
    vae_2d.eval()
    print(f"vae_2d loaded (2D encoder from {args.vqvae_2d_ckpt})")

    # --- T5 ---
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    t5_model  = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    t5_model.eval()
    print(f"T5 loaded from {args.model_name}")

    # --- Adapter ---
    adapter = None
    if args.adapter_ckpt:
        adapter = build_adapter(
            args.adapter_type, dim=81,
            hidden=args.adapter_hidden,
            kernel_size=args.adapter_kernel_size,
        ).to(device)
        ckpt_a  = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ckpt_a.get("model_state_dict", ckpt_a))
        adapter.eval()
        print(f"Adapter ({args.adapter_type}) loaded from {args.adapter_ckpt}")

    # --- Inference helper ---
    def run_inference(label, feat_key, use_adapter):
        all_preds, all_refs, video_ids = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"[{label}]"):
                l       = batch["length"][0].item()
                feat    = batch[feat_key][0:1, :l].to(device)  # (1, T, dim)

                if use_adapter and adapter is not None:
                    feat = adapter(feat)                        # (1, T, 81) → adapter → (1, T, 81)

                # zero-pad to 263 after adapter
                if feat.shape[2] < 263:
                    pad  = torch.zeros(1, feat.shape[1], 263 - feat.shape[2], device=device)
                    feat = torch.cat([feat, pad], dim=-1)       # (1, T, 263)

                pred = generate_texts_t5(
                    vae_2d, tokenizer, t5_model, feat,
                    device, args.prompt, args.max_new_tokens,
                )
                all_preds.append(pred)
                all_refs.append(batch["all_captions"][0])
                video_ids.append(batch["mid"][0])
        return all_preds, all_refs, video_ids

    def print_results(label, all_preds, all_refs, video_ids):
        print(f"\n=== [{label}] Results ({len(all_preds)} samples) ===")
        bleu1, bleu4, rouge, cider = calculate_bleu_rouge_cider(all_refs, all_preds)
        print(f"  BLEU-1   : {bleu1:.4f}")
        print(f"  BLEU-4   : {bleu4:.4f}")
        print(f"  ROUGE-L  : {rouge:.4f}")
        print(f"  CIDEr    : {cider:.4f}")
        try:
            from bert_score import score as score_bert
            _, _, F1 = score_bert(all_preds, all_refs, lang='en',
                                  rescale_with_baseline=True, idf=True,
                                  device=str(device), verbose=False)
            bert_f1 = F1.mean().item()
        except Exception as e:
            print(f"  [BERTScore error: {e}]")
            bert_f1 = 0.0
        print(f"  BERTScore: {bert_f1:.4f}")
        print()
        for i, (vid, pred, refs) in enumerate(zip(video_ids, all_preds, all_refs)):
            print(f"[{i+1:3d}] {vid}")
            print(f"       GT  : {refs[0]}")
            print(f"       Pred: {pred}")

    # --- Run ---
    if args.mode in ("adapter", "both"):
        if adapter is None:
            print("[Warning] --adapter_ckpt not specified, skipping 'adapter' mode.")
        else:
            preds, refs, vids = run_inference("with adapter", "feat_81", use_adapter=True)
            print_results("with adapter", preds, refs, vids)

    if args.mode in ("no_adapter", "both"):
        preds, refs, vids = run_inference("no adapter", "feat_68", use_adapter=False)
        print_results("no adapter", preds, refs, vids)


if __name__ == "__main__":
    main()
