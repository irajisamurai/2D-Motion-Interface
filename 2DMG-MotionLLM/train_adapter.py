"""
Adapter training for real-world VitPose 2D keypoints → HumanVQVAE latent space.

The adapter is a lightweight residual MLP (dim=263) placed before the frozen
2D VQ-VAE encoder.  It bridges the domain gap between:
  - synthetic 2D projections used to train the 2D encoder
  - real VitPose-estimated 2D keypoints (noisy / real-world)

Architecture:
  VitPose 2D (81-dim) → Adapter → [zero-pad to 263] → 2D VQ-VAE encoder (frozen)
                                                              ↓
                                         L1 loss vs. HumanVQVAE 3D encoder output

Key differences vs. 2DMotionGPT/train_adapter.py:
  - VQ-VAE: HumanVQVAE (wraps VQVAE_251 as .vqvae);
            encoder access: vae.vqvae.encoder / vae.vqvae.preprocess
  - 3D ref checkpoint: checkpoints/pretrained_vqvae/t2m.pth (key 'net')
  - 2D encoder checkpoint: checkpoints/2d_vq_train/t2m/best_2dvq_epoch*.pt (key 'net',
                            filter startswith('vqvae.encoder'))
  - Validation LM: T5ForConditionalGeneration (not MLM)
  - No OmegaConf/pytorch-lightning dependency

Execution:
  cd 2DMG-MotionLLM
  conda run -n mg-motionllm python train_adapter.py \\
      --vqvae_2d_ckpt ./checkpoints/2d_vq_train/t2m/best_2dvq_epoch1881_ratio0.5557.pt \\
      --estimated_motion_dir /path/to/humanml3d_for_render
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from os.path import join as pjoin
from pathlib import Path

import codecs as cs
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

try:
    from bert_score import score as score_bert
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("[Warning] bert-score not installed — BERTScore checkpoint criterion unavailable.")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


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
    """Lightweight residual adapter: identity at init, learns correction."""

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
    """1D Conv over time + residual."""

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
# VitPose feature helpers  (COCO-17 → COCO-13, 81-dim)
# ---------------------------------------------------------------------------

_COCO13_LHIP = 7   # 'left_hip_extra' in COCO-13
_COCO13_RHIP = 8   # 'right_hip_extra' in COCO-13
_COCO17_TO_13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def _normalize_2d_coco13_midhip(joints_2d):
    """Return (root_pos, joints_rel, scale_s) for COCO-13 keypoints (T,13,2)."""
    joints_2d = np.asarray(joints_2d)
    root_pos = 0.5 * (joints_2d[:, _COCO13_LHIP, :] + joints_2d[:, _COCO13_RHIP, :])
    joints_rel = joints_2d - root_pos[:, None, :]
    abs_xy = np.abs(joints_rel).reshape(-1, 2)
    s = max(np.percentile(abs_xy[:, 0], 99), np.percentile(abs_xy[:, 1], 99), 1e-8)
    return root_pos, joints_rel, s


def _preprocess_estimated(motion_2d, conf):
    """Convert COCO-13 keypoints (T,13,2) + confidence (T,13) → 81-dim feature.

    Feature layout (same as 2DMotionGPT):
      root_vel_2d  (T, 2)
      root_y_2d    (T, 1)
      joints_pos   (T, 26)   -- 13 joints × 2
      joints_rot   (T, 13)   -- atan2 of relative position
      joints_vel   (T, 26)   -- 13 joints × 2
      conf         (T, 13)
      total = 81
    """
    root_pos, joints_rel, s = _normalize_2d_coco13_midhip(motion_2d)
    T = root_pos.shape[0]

    root_y_2d = (root_pos[:, 1:2] / s).astype(np.float32)
    root_y_2d = root_y_2d - root_y_2d[0:1]
    joints_pos_2d = (joints_rel / s).reshape(T, -1).astype(np.float32)
    root_norm = (root_pos / s).astype(np.float32)
    root_vel_2d = np.zeros_like(root_norm)
    root_vel_2d[1:] = root_norm[1:] - root_norm[:-1]

    joints_rel_norm = (joints_rel / s).astype(np.float32)
    rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0])
    vel = np.zeros_like(joints_rel_norm)
    vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
    vel = vel.reshape(T, -1)

    c = conf.reshape(T, -1).astype(np.float32)
    return np.concatenate([root_vel_2d, root_y_2d, joints_pos_2d, rot, vel, c], axis=-1)


def _load_vitpose_jsons(estimated_motion_dir):
    """Scan <dir>/json/<motion_id>/<view_idx>.json.
    Returns {motion_id: {view_idx: {"motions": (T,13,2), "confs": (T,13)}}}.
    """
    json_list = glob.glob(os.path.join(estimated_motion_dir, "json", "**", "*.json"), recursive=True)
    result = {}
    for jp in json_list:
        vid_name = jp.split("/")[-2]
        view_idx = jp.split("/")[-1].replace(".json", "")
        try:
            data = json.load(open(jp))
        except Exception:
            continue
        motions, confs = [], []
        for frame_data in data:
            motions.append(frame_data["instances"][0]["keypoints"])
            confs.append(frame_data["instances"][0]["keypoint_scores"])
        motions = np.array(motions)
        confs = np.array(confs)
        if motions.shape[1] != 17:
            continue
        # COCO-17 → COCO-13
        motions = motions[:, _COCO17_TO_13, :]
        confs = confs[:, _COCO17_TO_13]
        if vid_name not in result:
            result[vid_name] = {}
        result[vid_name][view_idx] = {"motions": motions, "confs": confs}
    return result


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class VitPoseTrainDataset(Dataset):
    """HumanML3D train split + VitPose JSON pairs.
    Returns (motion_3d, feat_2d_81dim, m_length).
    """

    def __init__(self, data_root, split, mean_3d, std_3d, mean_estimate, std_estimate,
                 estimated_motion_dir, min_len=40, max_len=196, unit_length=4):
        self.mean_3d = mean_3d
        self.std_3d = std_3d
        self.mean_estimate = mean_estimate
        self.std_estimate = std_estimate
        self.unit_length = unit_length
        self.max_len = max_len

        # Load split IDs
        split_ids = []
        split_path = pjoin(data_root, f'{split}.txt')
        with cs.open(split_path) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    split_ids.append(mid)

        # Load VitPose JSONs
        print(f"[{split}] Loading VitPose JSONs from {estimated_motion_dir}...")
        vitpose_dict = _load_vitpose_jsons(estimated_motion_dir)
        print(f"  Found {len(vitpose_dict)} motions with JSON")

        # Build sample list
        self.samples = []
        skipped = 0
        for mid in split_ids:
            if mid not in vitpose_dict:
                skipped += 1
                continue
            npy_path = pjoin(data_root, "new_joint_vecs", mid + ".npy")
            if not os.path.exists(npy_path):
                skipped += 1
                continue
            motion = np.load(npy_path).astype(np.float32)
            if motion.shape[0] < min_len:
                skipped += 1
                continue
            self.samples.append((mid, motion, vitpose_dict[mid]))
        print(f"[VitPoseTrainDataset] {len(self.samples)} samples loaded, {skipped} skipped (split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, motion, vitpose_views = self.samples[idx]
        T = motion.shape[0]

        # Sample effective length (aligned to unit_length)
        if self.unit_length < 10:
            coin = np.random.choice(["single", "single", "double"])
        else:
            coin = "single"
        m_len = min(T, self.max_len)
        if coin == "double":
            m_len = (m_len // self.unit_length - 1) * self.unit_length
        else:
            m_len = (m_len // self.unit_length) * self.unit_length
        m_len = max(m_len, self.unit_length)

        # Select random view
        view_idx = random.choice(list(vitpose_views.keys()))
        motion_2d = vitpose_views[view_idx]["motions"]  # (T_json, 13, 2)
        conf_2d = vitpose_views[view_idx]["confs"]      # (T_json, 13)

        # Align lengths (JSON and motion may differ by 1-2 frames)
        T_use = min(T, motion_2d.shape[0])
        m_len = min(m_len, T_use)
        m_len = (m_len // self.unit_length) * self.unit_length
        m_len = max(m_len, self.unit_length)

        start = random.randint(0, max(0, T_use - m_len))
        motion   = motion[:T_use][start:start + m_len]
        motion_2d = motion_2d[start:start + m_len]
        conf_2d   = conf_2d[start:start + m_len]

        feat_2d = _preprocess_estimated(motion_2d, conf_2d)

        motion  = (motion  - self.mean_3d)      / (self.std_3d      + 1e-8)
        feat_2d = (feat_2d - self.mean_estimate) / (self.std_estimate + 1e-8)

        return motion, feat_2d, m_len


# ---------------------------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------------------------

class VitPoseValDataset(Dataset):
    """HumanML3D val split + VitPose JSON pairs + GT captions (multi-reference).
    Returns (motion_id, feat_2d_81dim, m_length, all_captions).
    """

    def __init__(self, data_root, split, mean_3d, std_3d, mean_estimate, std_estimate,
                 estimated_motion_dir, min_len=40, max_len=196, unit_length=4):
        self.mean_3d = mean_3d
        self.std_3d = std_3d
        self.mean_estimate = mean_estimate
        self.std_estimate = std_estimate
        self.unit_length = unit_length
        self.max_len = max_len

        split_ids = []
        split_path = pjoin(data_root, f'{split}.txt')
        with cs.open(split_path) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    split_ids.append(mid)

        print(f"[{split}] Loading VitPose JSONs from {estimated_motion_dir}...")
        vitpose_dict = _load_vitpose_jsons(estimated_motion_dir)

        self.samples = []
        skipped = 0
        for mid in split_ids:
            if mid not in vitpose_dict:
                skipped += 1
                continue
            npy_path = pjoin(data_root, "new_joint_vecs", mid + ".npy")
            text_path = pjoin(data_root, "texts", mid + ".txt")
            if not os.path.exists(npy_path) or not os.path.exists(text_path):
                skipped += 1
                continue
            motion = np.load(npy_path).astype(np.float32)
            if motion.shape[0] < min_len:
                skipped += 1
                continue

            # GT captions — lemmatized (TM2T/2DMotionGPT protocol)
            captions = []
            with cs.open(text_path) as f:
                for line in f.readlines():
                    try:
                        parts = line.strip().split('#')
                        t_tokens = parts[1].split(' ')
                        cap = ' '.join(tok.split('/')[0] for tok in t_tokens)
                        f_tag = float(parts[2]) if len(parts) > 2 else 0.0
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

            # Pad to exactly 3 references
            if len(captions) > 3:
                captions = captions[:3]
            elif len(captions) == 2:
                captions = captions + captions[:1]
            elif len(captions) == 1:
                captions = captions * 3

            self.samples.append((mid, motion, vitpose_dict[mid], captions))
        print(f"[VitPoseValDataset] {len(self.samples)} samples loaded, {skipped} skipped (split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, motion, vitpose_views, captions = self.samples[idx]
        T = motion.shape[0]

        m_len = min(T, self.max_len)
        m_len = (m_len // self.unit_length) * self.unit_length
        m_len = max(m_len, self.unit_length)

        view_idx = random.choice(list(vitpose_views.keys()))
        motion_2d = vitpose_views[view_idx]["motions"]
        conf_2d   = vitpose_views[view_idx]["confs"]

        T_use = min(T, motion_2d.shape[0])
        m_len = min(m_len, T_use)
        m_len = (m_len // self.unit_length) * self.unit_length
        m_len = max(m_len, self.unit_length)

        start = random.randint(0, max(0, T_use - m_len))
        motion_3d = motion[:T_use][start:start + m_len]
        motion_2d = motion_2d[start:start + m_len]
        conf_2d   = conf_2d[start:start + m_len]

        feat_2d = _preprocess_estimated(motion_2d, conf_2d)
        motion_3d = (motion_3d - self.mean_3d)       / (self.std_3d       + 1e-8)
        feat_2d   = (feat_2d   - self.mean_estimate) / (self.std_estimate  + 1e-8)

        return mid, feat_2d, motion_3d.astype(np.float32), m_len, captions


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_train(batch):
    motions, feats, lengths = zip(*batch)
    max_t = max(m.shape[0] for m in motions)
    dim_3d = motions[0].shape[1]
    dim_2d = feats[0].shape[1]

    pad_motions = torch.zeros(len(motions), max_t, dim_3d)
    pad_feats   = torch.zeros(len(feats),   max_t, dim_2d)
    for i, (m, f, l) in enumerate(zip(motions, feats, lengths)):
        pad_motions[i, :m.shape[0]] = torch.from_numpy(m)
        pad_feats[i,   :f.shape[0]] = torch.from_numpy(f)

    return {
        "motion_3d": pad_motions,
        "feat_2d":   pad_feats,
        "length":    torch.LongTensor(list(lengths)),
    }


def collate_val(batch):
    mids, feats, motions_3d, lengths, all_caps = zip(*batch)
    max_t  = max(f.shape[0] for f in feats)
    dim_2d = feats[0].shape[1]
    dim_3d = motions_3d[0].shape[1]

    pad_feats    = torch.zeros(len(feats),      max_t, dim_2d)
    pad_motion3d = torch.zeros(len(motions_3d), max_t, dim_3d)
    for i, (f, m) in enumerate(zip(feats, motions_3d)):
        pad_feats[i,    :f.shape[0]] = torch.from_numpy(f)
        pad_motion3d[i, :m.shape[0]] = torch.from_numpy(m)

    return {
        "mid":          list(mids),
        "feat_2d":      pad_feats,
        "motion_3d":    pad_motion3d,
        "length":       torch.LongTensor(list(lengths)),
        "all_captions": list(all_caps),
    }


# ---------------------------------------------------------------------------
# T5 text generation (adapter + 2D VQ-VAE → T5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text_t5(vae_2d, adapter, tokenizer, t5_model,
                     feat_2d_np, device, prompt="Generate text: ", max_new_tokens=40):
    """feat_2d_np: (T, 81) numpy array (already normalized)."""
    feat = torch.from_numpy(feat_2d_np).float().unsqueeze(0).to(device)  # (1, T, 81)
    feat = adapter(feat)                                                  # (1, T, 81)
    pad  = torch.zeros(1, feat.shape[1], 263 - feat.shape[2], device=device)
    feat_adapted = torch.cat([feat, pad], dim=-1)                        # (1, T, 263)

    tokenized = vae_2d.encode(feat_adapted)   # HumanVQVAE.encode → (1, L) tensor
    token_list = tokenized.cpu().numpy()[0].reshape(-1).tolist()

    motion_string = '<Motion Tokens>'
    for tok in token_list:
        motion_string += f'<{tok}>'
    motion_string += '</Motion Tokens>'

    input_ids = tokenizer(prompt + motion_string, return_tensors="pt").input_ids.to(device)
    outputs = t5_model.generate(input_ids, max_length=max_new_tokens, num_beams=1, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip('"')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # --- model checkpoints ---
    p.add_argument("--vqvae_2d_ckpt", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "2d_vq_train" / "t2m" /
                               "best_2dvq_epoch1881_ratio0.5557.pt"),
                   help="2D encoder checkpoint (.pt, key 'net', filter 'vqvae.encoder')")
    p.add_argument("--vqvae_3d_pth", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "pretrained_vqvae" / "t2m.pth"),
                   help="HumanVQVAE 3D reference checkpoint (.pth, key 'net')")
    p.add_argument("--model_name", type=str,
                   default=str(ROOT_DIR / "m2t-ft-from-GSPretrained-base"),
                   help="T5 model dir for validation generation")

    # --- data ---
    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Root dir with json/<motion_id>/<view>.json (VitPose rendered HumanML3D)")
    p.add_argument("--data_root", type=str,
                   default=str(ROOT_DIR / "dataset" / "HumanML3D"))
    p.add_argument("--meta_dir", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "t2m" / "VQVAEV3_CB1024_CMT_H1024_NRES3" / "meta"),
                   help="Dir with mean.npy, std.npy, mean_2d_coco_estimated_concatenate.npy, etc.")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split",   type=str, default="val")
    p.add_argument("--unit_length", type=int, default=4)

    # --- training ---
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--val_batch_size", type=int,   default=32)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--max_epochs",     type=int,   default=3000)
    p.add_argument("--val_every",      type=int,   default=10)
    p.add_argument("--gpu_id",         type=int,   default=0)

    # --- generation ---
    p.add_argument("--prompt",         type=str, default="Generate text: ")
    p.add_argument("--max_new_tokens", type=int, default=40)

    # --- output ---
    p.add_argument("--ckpt_save_dir", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "adapter" / "MG-MotionLLM"))
    p.add_argument("--wandb_project", type=str, default="2DMG-MotionLLM")
    p.add_argument("--no_wandb",      action="store_true")

    # --- adapter architecture ---
    p.add_argument("--adapter_type", type=str, default="residual",
                   choices=["linear", "residual", "conv1d"],
    p.add_argument("--adapter_hidden", type=int, default=512,
    p.add_argument("--adapter_kernel_size", type=int, default=3,

    # --- resume ---
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to adapter checkpoint (.pt) to resume training from")

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

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=f"adapter-{args.adapter_type}-bs{args.batch_size}-lr{args.lr}")

    os.makedirs(args.ckpt_save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------
    mean_3d  = np.load(pjoin(args.meta_dir, "mean.npy"))
    std_3d   = np.load(pjoin(args.meta_dir, "std.npy"))
    mean_est = np.load(pjoin(args.meta_dir, "mean_2d_coco_estimated_concatenate.npy"))
    std_est  = np.load(pjoin(args.meta_dir, "std_2d_coco_estimated_concatenate.npy"))

    # ----------------------------------------------------------------
    # Datasets & dataloaders
    # ----------------------------------------------------------------
    train_dataset = VitPoseTrainDataset(
        data_root=args.data_root,
        split=args.train_split,
        mean_3d=mean_3d, std_3d=std_3d,
        mean_estimate=mean_est, std_estimate=std_est,
        estimated_motion_dir=args.estimated_motion_dir,
        unit_length=args.unit_length,
    )
    val_dataset = VitPoseValDataset(
        data_root=args.data_root,
        split=args.val_split,
        mean_3d=mean_3d, std_3d=std_3d,
        mean_estimate=mean_est, std_estimate=std_est,
        estimated_motion_dir=args.estimated_motion_dir,
        unit_length=args.unit_length,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_train,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_val,
        persistent_workers=(args.num_workers > 0),
    )

    # ----------------------------------------------------------------
    # Models
    # ----------------------------------------------------------------
    import argparse as _ap
    vae_args = _ap.Namespace(dataname='t2m', quantizer='ema_reset', mu=0.99)

    def _build_vae():
        return vqvae_module.HumanVQVAE(
            vae_args, nb_code=512, code_dim=512, output_emb_width=512,
            down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
        ).to(device)

    # ref_vae: 3D VQ-VAE (frozen, provides reference encoder latents)
    ref_vae = _build_vae()
    ckpt_3d = torch.load(args.vqvae_3d_pth, map_location="cpu", weights_only=False)
    ref_vae.load_state_dict(ckpt_3d['net'], strict=True)

    # vae_2d: same 3D base + 2D encoder weights overwritten (frozen)
    vae_2d = _build_vae()
    vae_2d.load_state_dict(ckpt_3d['net'], strict=True)
    ckpt_2d = torch.load(args.vqvae_2d_ckpt, map_location="cpu", weights_only=False)
    vae_2d.load_state_dict(
        {k: v for k, v in ckpt_2d['net'].items() if k.startswith("vqvae.encoder")},
        strict=False,
    )

    # T5 (frozen, used only for validation text generation)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    t5_model  = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    for model in [ref_vae, vae_2d, t5_model]:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

    print(f"ref_vae loaded from: {args.vqvae_3d_pth}")
    print(f"vae_2d  loaded from: {args.vqvae_2d_ckpt} (encoder overridden)")
    print(f"T5      loaded from: {args.model_name}")

    # Adapter (trainable)
    adapter = build_adapter(
        args.adapter_type, dim=81,
        hidden=args.adapter_hidden,
        kernel_size=args.adapter_kernel_size,
    ).to(device)
    if use_wandb:
        wandb.watch(adapter)

    # ----------------------------------------------------------------
    # Optimizer & loss
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()

    # ----------------------------------------------------------------
    # CSV log
    # ----------------------------------------------------------------
    log_path = pjoin(args.ckpt_save_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_encoder_loss", "val_encoder_loss", "BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "BERTScore_F1"]
        )

    best_val_loss      = float("inf")
    best_val_loss_ckpt = None

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    for epoch in range(args.max_epochs):
        adapter.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()

            feats_3d = batch["motion_3d"].to(device)          # (B, T, 263)
            feats_2d = adapter(batch["feat_2d"].to(device))   # (B, T, 81)
            pad      = torch.zeros(feats_2d.shape[0], feats_2d.shape[1],
                                   263 - feats_2d.shape[2], device=device)
            feats_2d = torch.cat([feats_2d, pad], dim=-1)     # (B, T, 263)

            with torch.no_grad():
                ref_encoded = ref_vae.vqvae.encoder(ref_vae.vqvae.preprocess(feats_3d))

            enc_2d = vae_2d.vqvae.encoder(vae_2d.vqvae.preprocess(feats_2d))
            loss = loss_fn(enc_2d, ref_encoded)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}  Encoder Loss: {avg_loss:.4f}")

        metrics_log  = {"train/encoder_loss": avg_loss}
        bertscore_f1 = ""

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        if epoch % args.val_every == 0:
            adapter.eval()
            all_preds, all_refs = [], []
            val_total_enc_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Val", leave=False):
                    # val encoder loss
                    feats_3d = batch["motion_3d"].to(device)
                    feats_2d = adapter(batch["feat_2d"].to(device))
                    pad      = torch.zeros(feats_2d.shape[0], feats_2d.shape[1],
                                           263 - feats_2d.shape[2], device=device)
                    feats_2d_padded = torch.cat([feats_2d, pad], dim=-1)
                    ref_encoded = ref_vae.vqvae.encoder(ref_vae.vqvae.preprocess(feats_3d))
                    enc_2d      = vae_2d.vqvae.encoder(vae_2d.vqvae.preprocess(feats_2d_padded))
                    val_total_enc_loss += loss_fn(enc_2d, ref_encoded).item()

                    # text generation
                    feats_2d_raw = batch["feat_2d"]
                    for i in range(len(batch["mid"])):
                        l = batch["length"][i].item()
                        feat_np = feats_2d_raw[i, :l].numpy()
                        pred = generate_text_t5(
                            vae_2d, adapter, tokenizer, t5_model,
                            feat_np, device, args.prompt, args.max_new_tokens,
                        )
                        all_preds.append(pred)
                        all_refs.append(batch["all_captions"][i])

            avg_val_enc_loss = val_total_enc_loss / len(val_loader)
            print(f"  val/encoder_loss: {avg_val_enc_loss:.4f}")
            metrics_log["val/encoder_loss"] = avg_val_enc_loss

            if all_preds:
                bleu1, bleu4, rouge, cider = calculate_bleu_rouge_cider(all_refs, all_preds)
                print(f"  BLEU-1: {bleu1:.4f}  BLEU-4: {bleu4:.4f}  ROUGE-L: {rouge:.4f}  CIDEr: {cider:.4f}")
                metrics_log.update({
                    "val/BLEU-1": bleu1, "val/BLEU-4": bleu4,
                    "val/ROUGE-L": rouge, "val/CIDEr": cider,
                })

                if HAS_BERTSCORE:
                    _, _, F1 = score_bert(
                        all_preds, all_refs, lang='en',
                        rescale_with_baseline=True, idf=True,
                        device=device, verbose=False,
                    )
                    bertscore_f1 = F1.mean().item()
                    print(f"  BERTScore_F1: {bertscore_f1:.4f}")
                    metrics_log["val/BERTScore_F1"] = bertscore_f1

            # Best checkpoint criterion: val encoder loss (lower is better)
            if avg_val_enc_loss < best_val_loss:
                best_val_loss = avg_val_enc_loss
                new_ckpt = pjoin(
                    args.ckpt_save_dir,
                    f"best_adapter_epoch{epoch}_valloss{best_val_loss:.4f}.pt",
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": adapter.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, new_ckpt)
                if best_val_loss_ckpt and os.path.exists(best_val_loss_ckpt):
                    os.remove(best_val_loss_ckpt)
                best_val_loss_ckpt = new_ckpt
                print(f"  -> Best val_loss checkpoint saved: {new_ckpt}")

        if use_wandb:
            wandb.log(metrics_log, step=epoch)
        val_enc_loss_log = avg_val_enc_loss if epoch % args.val_every == 0 else ""
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss, val_enc_loss_log,
                                    bleu1, bleu4, rouge, cider, bertscore_f1])


if __name__ == "__main__":
    main()
