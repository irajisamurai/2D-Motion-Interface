"""
Evaluate TM2T M2T pipeline on real-world VitPose-estimated keypoints.

Uses the same adapter + MotionGPT 2D VQ-VAE encoder as 2DMotionGPT,
but replaces the T5 LM with TM2T's TransformerV2 + beam search decoding.

Pipeline (adapter mode):
  VitPose 2D keypoints (17 joints)
    → COCO-17 to COCO-13 selection
    → feature extraction (81-dim, with confidence)
    → normalize
    → zero-pad to 263
    → AdapterResidual
    → MotionGPT VQ-VAE encoder → token indices
    → TransformerV2 (M2T_MotionGPT) beam search
    → predicted text

Pipeline (no-adapter mode):
  Same up to feature extraction (68-dim, no confidence)
    → normalize
    → zero-pad to 263
    → MotionGPT VQ-VAE encoder → token indices
    → TransformerV2 beam search
    → predicted text

Usage:
    python evaluate_with_realworld_tm2t.py \\
        --estimated_motion_dir ./datasets/real_world_dataset_ver2/pred \\
        --motiongpt_ckpt /path/to/MotionGPT-base/motiongpt_s3_h3d.tar \\
        --vqvae_2d_ckpt /path/to/2d_vqvae.tar \\
        --adapter_ckpt /path/to/best_adapter.tar \\
        --m2t_name M2T_MotionGPT_EL4_DL4_NH8_PS \\
        [--mode both] [--split test]
"""

import argparse
import codecs as cs
import glob
import json
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.mgpt_vq import VQVae
from networks.transformer import TransformerV2
from networks.translator import Translator
from utils.word_vectorizer import WordVectorizerV2

try:
    from nlgeval import NLGEval
    HAS_NLGEVAL = True
except ImportError:
    HAS_NLGEVAL = False

try:
    from nlgmetricverse import NLGMetricverse, load_metric
    HAS_NLGMETRICVERSE = True
except ImportError:
    HAS_NLGMETRICVERSE = False

if not HAS_NLGEVAL and not HAS_NLGMETRICVERSE:
    print("[Warning] nlgeval / nlgmetricverse not installed — BLEU/ROUGE/CIDEr skipped.")

try:
    from bert_score import score as score_bert
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("[Warning] bert-score not installed — BERTScore skipped.")


# ---------------------------------------------------------------------------
# VitPose feature extraction helpers (self-contained copy of _VitPoseMixin)
# ---------------------------------------------------------------------------

def normalize_2d_coco13_midhip(joints_2d, eps=1e-8, q=99):
    joints_2d = np.asarray(joints_2d)
    lhip, rhip = 7, 8  # COCO-13 indices for left/right hip
    root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])
    joints_rel = joints_2d - root_pos[:, None, :]
    abs_xy = np.abs(joints_rel).reshape(-1, 2)
    s = max(np.percentile(abs_xy[:, 0], q), np.percentile(abs_xy[:, 1], q), eps)
    return root_pos, joints_rel, s


def decompose_2d_motion_coco13_midhip_root(joints_2d):
    root_pos, joints_rel, s = normalize_2d_coco13_midhip(joints_2d)
    root_y_2d = (root_pos[:, 1:2] / s).astype(np.float32)
    root_y_2d = root_y_2d - root_y_2d[0:1]
    joints_pos_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)
    root_norm = (root_pos / s).astype(np.float32)
    root_vel_2d = np.zeros_like(root_norm)
    root_vel_2d[1:] = root_norm[1:] - root_norm[:-1]
    return root_y_2d, joints_pos_2d, root_vel_2d


def compute_joint_features_2d_coco13(joints_2d):
    _, joints_rel, s = normalize_2d_coco13_midhip(joints_2d)
    joints_rel_norm = (joints_rel / s).astype(np.float32)
    rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)
    vel = np.zeros_like(joints_rel_norm)
    vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
    vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)
    return rot, vel


def preprocess_vitpose_68dim(joints_2d_coco13):
    """68-dim features without confidence (matches tokenize_script_motiongpt_2d.py)."""
    root_y, pos, vel = decompose_2d_motion_coco13_midhip_root(joints_2d_coco13)
    rot, jvel = compute_joint_features_2d_coco13(joints_2d_coco13)
    return np.concatenate([vel, root_y, pos, rot, jvel], axis=-1)  # (T, 2+1+26+13+26) = 68


def preprocess_vitpose_81dim(joints_2d_coco13, conf_coco13):
    """81-dim features with confidence (adapter input)."""
    root_y, pos, vel = decompose_2d_motion_coco13_midhip_root(joints_2d_coco13)
    rot, jvel = compute_joint_features_2d_coco13(joints_2d_coco13)
    T = root_y.shape[0]
    c = conf_coco13.reshape(T, -1).astype(np.float32)
    return np.concatenate([vel, root_y, pos, rot, jvel, c], axis=-1)  # (T, 68+13) = 81


# ---------------------------------------------------------------------------
# Adapter
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
# Dataset
# ---------------------------------------------------------------------------

class RealWorldDataset(Dataset):
    """
    Each JSON file: one video, VitPose-estimated 17-joint keypoints per frame.
    The JSON filename (without .json) must match a HumanML3D motion ID.
    """

    COCO17_TO_COCO13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def __init__(self, json_dir, data_root, split, mean_2d, std_2d,
                 mean_est=None, std_est=None, unit_length=4, min_len=40, max_len=196):
        self.data_root = data_root
        self.mean_2d = mean_2d
        self.std_2d = std_2d
        self.mean_est = mean_est
        self.std_est = std_est
        self.unit_length = unit_length
        self.min_len = min_len
        self.max_len = max_len

        # Filter to IDs present in split file
        split_ids = set()
        split_path = pjoin(data_root, f'{split}.txt')
        if os.path.exists(split_path):
            with cs.open(split_path) as f:
                for line in f:
                    split_ids.add(line.strip())

        json_paths = sorted(glob.glob(os.path.join(json_dir, "json", "*.json")))
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
            kps, confs = [], []
            for fd in data:
                kps.append(fd["instances"][0]["keypoints"])
                confs.append(fd["instances"][0]["keypoint_scores"])
            kps = np.array(kps, dtype=np.float32)     # (T, 17, 2)
            confs = np.array(confs, dtype=np.float32) # (T, 17)
            self.samples.append((mid, jp, kps, confs, text_path))
        print(f"[RealWorldDataset] {len(self.samples)} samples loaded ({skipped} skipped).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, jp, kps17, conf17, text_path = self.samples[idx]

        # COCO-17 → COCO-13
        kps = kps17[:, self.COCO17_TO_COCO13, :]  # (T, 13, 2)
        conf = conf17[:, self.COCO17_TO_COCO13]   # (T, 13)

        T = kps.shape[0]
        m_len = (T // self.unit_length) * self.unit_length
        m_len = max(self.min_len, min(m_len, self.max_len))
        if T < m_len:
            m_len = (T // self.unit_length) * self.unit_length
        start = random.randint(0, max(0, T - m_len))
        kps = kps[start:start + m_len]
        conf = conf[start:start + m_len]

        # 68-dim (no conf)
        feat68 = preprocess_vitpose_68dim(kps)
        feat68_norm = (feat68 - self.mean_2d) / (self.std_2d + 1e-8)

        # 81-dim (with conf) — only if stats provided
        feat81_norm = None
        if self.mean_est is not None:
            feat81 = preprocess_vitpose_81dim(kps, conf)
            feat81_norm = (feat81 - self.mean_est) / (self.std_est + 1e-8)

        # GT captions — lemmatized (TM2T official protocol: token.split('/')[0])
        captions = []
        with cs.open(text_path) as f:
            for line in f.readlines():
                try:
                    parts = line.strip().split('#')
                    t_tokens = parts[1].split(' ')
                    cap = ' '.join(tok.split('/')[0] for tok in t_tokens)
                    f_tag = float(parts[2]) if len(parts) > 2 else 0.0
                    to_tag = float(parts[3]) if len(parts) > 3 else 0.0
                    if np.isnan(f_tag):
                        f_tag = 0.0
                    if np.isnan(to_tag):
                        to_tag = 0.0
                    if f_tag == 0.0 and to_tag == 0.0:
                        captions.append(cap)
                except Exception:
                    pass

        return mid, feat68_norm, feat81_norm, m_len, captions


# ---------------------------------------------------------------------------
# Token formatting
# ---------------------------------------------------------------------------

def tokens_to_input(token_list, mot_start_idx, mot_end_idx, mot_pad_idx, max_motion_token=55):
    """Format VQ-VAE token indices for TransformerV2 input."""
    toks = [mot_start_idx] + list(token_list) + [mot_end_idx]
    pad_len = max_motion_token - len(toks)
    if pad_len > 0:
        toks = toks + [mot_pad_idx] * pad_len
    else:
        toks = toks[:max_motion_token]
    return torch.LongTensor(toks).unsqueeze(0)  # (1, max_motion_token)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Dir with json/<id>.json VitPose outputs")
    p.add_argument("--motiongpt_ckpt", type=str, required=True,
                   help="MotionGPT base checkpoint (motiongpt_s3_h3d.tar)")
    p.add_argument("--vqvae_2d_ckpt", type=str, required=True,
                   help="2D VQ-VAE encoder checkpoint (encoder weights only)")
    p.add_argument("--adapter_ckpt", type=str, default=None,
                   help="Adapter checkpoint. Required for adapter/both mode.")
    p.add_argument("--m2t_name", type=str, default="M2T_MotionGPT_EL4_DL4_NH8_PS",
                   help="M2T checkpoint name under checkpoints/t2m/")
    p.add_argument("--m2t_epoch", type=str, default="finest",
                   help="Checkpoint epoch tag (default: finest)")
    p.add_argument("--data_root", type=str,
                   default=str(ROOT_DIR / "dataset" / "HumanML3D"))
    p.add_argument("--meta_dir", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "t2m" / "VQVAEV3_CB1024_CMT_H1024_NRES3" / "meta"),
                   help="Dir with mean/std .npy files for 2D features")
    p.add_argument("--estimated_stats_dir", type=str,
                   default="../2DMotionGPT/deps/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta",
                   help="Dir with mean_2d_coco_estimated_concatenate.npy (for adapter mode)")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--mode", type=str, default="both",
                   choices=["adapter", "no_adapter", "both"])
    p.add_argument("--adapter_type", type=str, default="residual",
                   choices=["linear", "residual", "conv1d"],
    p.add_argument("--adapter_hidden", type=int, default=512,
    p.add_argument("--adapter_kernel_size", type=int, default=3,
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--beam_size", type=int, default=2)
    p.add_argument("--max_motion_token", type=int, default=55)
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
    mean_2d = np.load(pjoin(args.meta_dir, "mean_2d_coco_normalized.npy"))
    std_2d  = np.load(pjoin(args.meta_dir, "std_2d_coco_normalized.npy"))

    mean_est = std_est = None
    est_stats_path = args.estimated_stats_dir
    mean_est_path = pjoin(est_stats_path, "mean_2d_coco_estimated_concatenate.npy")
    std_est_path  = pjoin(est_stats_path, "std_2d_coco_estimated_concatenate.npy")
    if os.path.exists(mean_est_path) and os.path.exists(std_est_path):
        mean_est = np.load(mean_est_path)
        std_est  = np.load(std_est_path)
    elif args.mode in ("adapter", "both"):
        raise FileNotFoundError(
            f"Estimated stats not found at {est_stats_path}. "
            "Pass --estimated_stats_dir or use --mode no_adapter."
        )

    # --- Dataset ---
    dataset = RealWorldDataset(
        json_dir=args.estimated_motion_dir,
        data_root=args.data_root,
        split=args.split,
        mean_2d=mean_2d,
        std_2d=std_2d,
        mean_est=mean_est,
        std_est=std_est,
        unit_length=args.unit_length,
    )
    if len(dataset) == 0:
        print("No samples found. Exiting.")
        return

    # --- VQ-VAE (MotionGPT base + 2D encoder override) ---
    vqvae = VQVae(
        nfeats=263,
        quantizer="ema_reset",
        code_num=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        norm="none",
        activation="relu",
    ).to(device)

    ckpt_base = torch.load(args.motiongpt_ckpt, map_location="cpu", weights_only=False)
    vqvae.load_state_dict(
        {k.replace("vae.", ""): v for k, v in ckpt_base["state_dict"].items() if "vae" in k}
    )
    ckpt_2d = torch.load(args.vqvae_2d_ckpt, map_location="cpu", weights_only=False)
    raw_2d = ckpt_2d.get("model_state_dict", ckpt_2d)
    vqvae.load_state_dict({k: v for k, v in raw_2d.items() if "encoder" in k}, strict=False)
    vqvae.eval()
    print("VQ-VAE loaded.")

    # --- Adapter ---
    adapter = None
    if args.mode in ("adapter", "both"):
        if args.adapter_ckpt is None:
            raise ValueError("--adapter_ckpt required for adapter/both mode")
        adapter = build_adapter(
            args.adapter_type, dim=81,
            hidden=args.adapter_hidden,
            kernel_size=args.adapter_kernel_size,
        ).to(device)
        ckpt_adapter = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ckpt_adapter.get("model_state_dict", ckpt_adapter))
        adapter.eval()
        print("Adapter loaded.")

    # --- WordVectorizer ---
    glove_dir = str(ROOT_DIR / "glove")
    w_vectorizer = WordVectorizerV2(glove_dir, "our_vab")
    n_txt_vocab = len(w_vectorizer) + 1
    _, _, txt_start_idx = w_vectorizer["sos/OTHER"]
    _, _, txt_end_idx   = w_vectorizer["eos/OTHER"]
    txt_pad_idx = len(w_vectorizer)

    # --- M2T Transformer ---
    codebook_size = 512
    mot_start_idx = codebook_size
    mot_end_idx   = codebook_size + 1
    mot_pad_idx   = codebook_size + 2
    n_mot_vocab   = codebook_size + 3

    m2t_transformer = TransformerV2(
        n_mot_vocab, mot_pad_idx, n_txt_vocab, txt_pad_idx,
        d_src_word_vec=512, d_trg_word_vec=512,
        d_model=512, d_inner=2048,
        n_enc_layers=4, n_dec_layers=4,
        n_head=8, d_k=64, d_v=64,
        dropout=0.1,
        n_src_position=100, n_trg_position=50,
        trg_emb_prj_weight_sharing=True,
    )
    m2t_ckpt_path = str(ROOT_DIR / "checkpoints" / "t2m" / args.m2t_name / "model" / f"{args.m2t_epoch}.tar")
    ckpt_m2t = torch.load(m2t_ckpt_path, map_location="cpu", weights_only=False)
    m2t_transformer.load_state_dict(ckpt_m2t["m2t_transformer"])
    print(f"M2T transformer loaded from {m2t_ckpt_path} (ep={ckpt_m2t.get('ep', '?')})")

    translator = Translator(
        m2t_transformer, beam_size=args.beam_size, max_seq_len=30,
        src_pad_idx=mot_pad_idx, trg_pad_idx=txt_pad_idx,
        trg_sos_idx=txt_start_idx, trg_eos_idx=txt_end_idx,
    ).to(device)

    # --- Inference helper ---
    def encode_and_translate(feat_np, _adapter):
        """
        feat_np: (T, D) numpy array, already normalized.
        _adapter: AdapterResidual or None.
        Returns predicted text string.
        """
        feat = torch.from_numpy(feat_np).float().unsqueeze(0).to(device)  # (1, T, D)

        with torch.no_grad():
            if _adapter is not None:
                feat = _adapter(feat)              # adapter on 81-dim
            # Zero-pad to 263 after adapter
            if feat.shape[2] < 263:
                pad = torch.zeros(feat.shape[0], feat.shape[1], 263 - feat.shape[2], device=device)
                feat = torch.cat([feat, pad], dim=-1)

            token_indices, _ = vqvae.encode(feat)  # (1, L)
            toks = token_indices[0].cpu().numpy().tolist()

            m_input = tokens_to_input(
                toks, mot_start_idx, mot_end_idx, mot_pad_idx,
                max_motion_token=args.max_motion_token,
            ).to(device)  # (1, max_motion_token)

            pred_ids = translator.translate_sentence(m_input)
            pred_ids = pred_ids[1:-1]  # strip SOS/EOS
            text = " ".join(w_vectorizer.itos(i) for i in pred_ids)
        return text

    # --- Evaluation loop ---
    def evaluate(label, use_adapter):
        _adapter = adapter if use_adapter else None

        all_preds, all_refs, video_ids = [], [], []
        for i in range(len(dataset)):
            mid, feat68, feat81, m_len, captions = dataset[i]
            feat = feat81 if (use_adapter and feat81 is not None) else feat68
            pred = encode_and_translate(feat, _adapter)
            all_preds.append(pred)
            # Use first caption as primary reference for display
            all_refs.append(captions)
            video_ids.append(mid)

        print(f"\n=== [{label}] Results ({len(all_preds)} samples) ===")

        # Corpus-level BLEU / ROUGE / CIDEr (nlgeval preferred — consistent with TM2T paper)
        if HAS_NLGEVAL:
            nlg_eval = NLGEval(metrics_to_omit=[
                "METEOR", "EmbeddingAverageCosineSimilarity",
                "SkipThoughtCS", "VectorExtremaCosineSimilarity", "GreedyMatchingScore"
            ])
            max_refs = max(len(r) for r in all_refs)
            ref_list = []
            for ri in range(max_refs):
                ref_list.append([refs[ri] if ri < len(refs) else refs[-1] for refs in all_refs])
            scores = nlg_eval.compute_metrics(ref_list, all_preds)
            for key in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr"]:
                print(f"  {key:>10}: {scores.get(key, 0.0):.4f}")
        elif HAS_NLGMETRICVERSE:
            # Sentence-level (nlgmetricverse) — note: different from TM2T corpus-level BLEU
            _metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            scorer = NLGMetricverse(_metrics)
            scores = scorer(predictions=all_preds, references=all_refs)
            b1 = scores["bleu_1"]["score"]
            b4 = scores["bleu_4"]["score"]
            rouge = scores["rouge"]["rougeL"]
            cider = scores["cider"]["score"]
            print(f"  {'BLEU-1':>10}: {b1:.4f}  (sentence-level)")
            print(f"  {'BLEU-4':>10}: {b4:.4f}  (sentence-level)")
            print(f"  {'ROUGE-L':>10}: {rouge:.4f}")
            print(f"  {'CIDEr':>10}: {cider:.4f}")
        else:
            scores = {}

        # BERTScore
        bert_f1 = None
        if HAS_BERTSCORE:
            _, _, F1 = score_bert(
                all_preds, all_refs, lang="en",
                rescale_with_baseline=True, idf=True,
                device=device, verbose=False,
            )
            bert_f1 = F1
            print(f"  {'BERTScore':>10}: {F1.mean().item():.4f}")

        # Per-sample output
        print()
        for i, (vid, pred, refs) in enumerate(zip(video_ids, all_preds, all_refs)):
            bert_str = f"{bert_f1[i].item():.4f}" if bert_f1 is not None else "N/A"
            print(f"[{i+1:3d}] {vid}")
            print(f"       GT  : {refs[0]}")
            print(f"       Pred: {pred}")
            print(f"       BERT: {bert_str}")

        return all_preds, all_refs, scores

    if args.mode in ("adapter", "both"):
        evaluate("with adapter", use_adapter=True)

    if args.mode in ("no_adapter", "both"):
        evaluate("no adapter", use_adapter=False)


if __name__ == "__main__":
    main()
