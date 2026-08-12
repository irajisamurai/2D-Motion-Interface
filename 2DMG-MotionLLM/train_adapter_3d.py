# -*- coding: utf-8 -*-
"""
Adapter training for WHAM-estimated 3D motion -> HumanVQVAE latent space.

The adapter is a lightweight residual MLP (dim=263) placed before the frozen
3D VQ-VAE encoder.  It bridges the domain gap between:
  - clean HumanML3D 3D motion (used to train HumanVQVAE)
  - WHAM-estimated 3D motion (noisy, from real-world video)

Architecture:
  WHAM 3D (263-dim, WHAM-norm) -> Adapter -> HumanVQVAE encoder (frozen)
                                                       |
                                    L1 loss vs. GT 3D encoder output

Key differences vs. train_adapter.py (2D adapter):
  - dim=263 (no zero-padding needed)
  - Single VQ-VAE (no separate 2D encoder)
  - WHAM-specific normalization stats (mean_wham_3d / std_wham_3d)
  - Training data: adapter_training_WAHA/<motion_id>/<view>.npy (multi-view WHAM)
  - Val data:      new_joint_vecs/<motion_id>.npy (single-view WHAM)

Execution:
  cd 2DMG-MotionLLM
  CUDA_VISIBLE_DEVICES=1 conda run -n mg-motionllm python train_adapter_3d.py \\
      --estimated_motion_dir ../2DMotionGPT/datasets/humanml3d_for_render_wham
"""

import argparse
import codecs as cs
import csv
import glob
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
# Adapter models  (dim=263, no zero-padding)
# ---------------------------------------------------------------------------

class AdapterLinear(nn.Module):
    def __init__(self, dim=263):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.linear(self.norm(x))


class AdapterResidual(nn.Module):
    """Lightweight residual adapter: identity at init, learns correction."""

    def __init__(self, dim=263, hidden=512, dropout=0.1):
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

    def __init__(self, dim=263, hidden=512, kernel_size=3, dropout=0.1):
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


def build_adapter(adapter_type, dim=263, hidden=512, kernel_size=3):
    if adapter_type == "linear":
        return AdapterLinear(dim=dim)
    elif adapter_type == "residual":
        return AdapterResidual(dim=dim, hidden=hidden)
    elif adapter_type == "conv1d":
        return AdapterConv1d(dim=dim, hidden=hidden, kernel_size=kernel_size)
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")


# ---------------------------------------------------------------------------
# WAHA loading helper
# ---------------------------------------------------------------------------

def _build_waha_dict(estimated_motion_dir, min_frames=40):
    """Scan <estimated_motion_dir>/adapter_training_WAHA/<motion_id>/<view>.npy.

    Returns {motion_id: [list of valid .npy paths]}.
    NaN files and files shorter than min_frames are skipped.
    """
    waha_root = os.path.join(estimated_motion_dir, "adapter_training_WAHA")
    waha_dict = {}
    for mid in sorted(os.listdir(waha_root)):
        mid_dir = os.path.join(waha_root, mid)
        if not os.path.isdir(mid_dir):
            continue
        valid_paths = []
        for npy_path in sorted(glob.glob(os.path.join(mid_dir, "*.npy"))):
            arr = np.load(npy_path, mmap_mode='r')
            if arr.shape[0] < min_frames:
                continue
            if np.any(np.isnan(arr)):
                continue
            valid_paths.append(npy_path)
        if valid_paths:
            waha_dict[mid] = valid_paths
    return waha_dict


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class WhamTrainDataset(Dataset):
    """HumanML3D train split paired with WAHA (multi-view WHAM) files.

    Returns (motion_3d_norm, wham_3d_norm, m_length).
    """

    def __init__(self, data_root, split, mean, std, mean_wham, std_wham,
                 estimated_motion_dir, min_len=40, max_len=196, unit_length=4):
        self.mean      = mean
        self.std       = std
        self.mean_wham = mean_wham
        self.std_wham  = std_wham
        self.unit_length = unit_length
        self.max_len   = max_len

        # Load split IDs from HumanML3D
        split_ids = []
        with cs.open(pjoin(data_root, f'{split}.txt')) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    split_ids.append(mid)

        # Load WAHA dict
        print(f"[{split}] Loading WAHA files from {estimated_motion_dir}...")
        waha_dict = _build_waha_dict(estimated_motion_dir, min_frames=min_len)
        print(f"  Found {len(waha_dict)} motions with WAHA data")

        # Build sample list
        self.samples = []
        skipped = 0
        for mid in split_ids:
            if mid not in waha_dict:
                skipped += 1
                continue
            gt_path = pjoin(data_root, "new_joint_vecs", mid + ".npy")
            if not os.path.exists(gt_path):
                skipped += 1
                continue
            motion_gt = np.load(gt_path).astype(np.float32)
            if motion_gt.shape[0] < min_len:
                skipped += 1
                continue
            self.samples.append((mid, motion_gt, waha_dict[mid]))
        print(f"[WhamTrainDataset] {len(self.samples)} samples loaded, "
              f"{skipped} skipped (split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, motion_gt, waha_paths = self.samples[idx]
        T_gt = motion_gt.shape[0]

        # Pick a random WAHA view
        waha_path = random.choice(waha_paths)
        wham_raw  = np.load(waha_path).astype(np.float32)
        T_wham    = wham_raw.shape[0]

        # Effective length (aligned to unit_length)
        if self.unit_length < 10:
            coin = np.random.choice(["single", "single", "double"])
        else:
            coin = "single"

        T_use  = min(T_gt, T_wham)
        m_len  = min(T_use, self.max_len)
        if coin == "double":
            m_len = (m_len // self.unit_length - 1) * self.unit_length
        else:
            m_len = (m_len // self.unit_length) * self.unit_length
        m_len  = max(m_len, self.unit_length)

        start    = random.randint(0, max(0, T_use - m_len))
        motion   = motion_gt[start:start + m_len]
        wham_clip = wham_raw[start:start + m_len]

        motion    = (motion    - self.mean)      / (self.std      + 1e-8)
        wham_clip = (wham_clip - self.mean_wham) / (self.std_wham + 1e-8)

        return motion, wham_clip, m_len


# ---------------------------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------------------------

class WhamValDataset(Dataset):
    """HumanML3D val split paired with WAHA (multi-view synthetic WHAM) + GT captions.

    Mirrors 2DMotionGPT Text2MotionDatasetEval_wham_3d: uses WAHA for WHAM input
    so train/val share the same data distribution.

    Returns (motion_id, wham_3d_norm, motion_3d_norm, m_length, all_captions).
    """

    def __init__(self, data_root, split, mean, std, mean_wham, std_wham,
                 estimated_motion_dir, min_len=40, max_len=196, unit_length=4):
        self.mean      = mean
        self.std       = std
        self.mean_wham = mean_wham
        self.std_wham  = std_wham
        self.unit_length = unit_length
        self.max_len   = max_len

        split_ids = []
        with cs.open(pjoin(data_root, f'{split}.txt')) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    split_ids.append(mid)

        # Use WAHA (same distribution as training) — mirrors 2DMotionGPT val setup
        waha_dict = _build_waha_dict(estimated_motion_dir, min_frames=min_len)
        print(f"[WhamValDataset] Found {len(waha_dict)} motions with WAHA data")

        self.samples = []
        skipped = 0
        for mid in split_ids:
            if mid not in waha_dict:
                skipped += 1
                continue
            gt_path   = pjoin(data_root, "new_joint_vecs", mid + ".npy")
            text_path = pjoin(data_root, "texts", mid + ".txt")
            if not os.path.exists(gt_path) or not os.path.exists(text_path):
                skipped += 1
                continue
            motion_gt = np.load(gt_path).astype(np.float32)
            if motion_gt.shape[0] < min_len:
                skipped += 1
                continue

            # GT captions -- lemmatized (TM2T official protocol)
            captions = []
            with cs.open(text_path) as f:
                for line in f.readlines():
                    try:
                        parts  = line.strip().split('#')
                        t_tokens = parts[1].split(' ')
                        cap    = ' '.join(tok.split('/')[0] for tok in t_tokens)
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

            if len(captions) > 3:
                captions = captions[:3]
            elif len(captions) == 2:
                captions = captions + captions[:1]
            elif len(captions) == 1:
                captions = captions * 3

            self.samples.append((mid, waha_dict[mid], motion_gt, captions))
        print(f"[WhamValDataset] {len(self.samples)} samples loaded, "
              f"{skipped} skipped (split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, waha_paths, motion_gt, captions = self.samples[idx]

        # Pick a random WAHA view (same as training)
        wham_raw = np.load(random.choice(waha_paths)).astype(np.float32)

        T_gt   = motion_gt.shape[0]
        T_wham = wham_raw.shape[0]

        T_use = min(T_gt, T_wham)
        m_len = min(T_use, self.max_len)
        m_len = (m_len // self.unit_length) * self.unit_length
        m_len = max(m_len, self.unit_length)

        start    = random.randint(0, max(0, T_use - m_len))
        wham_clip = wham_raw[start:start + m_len]
        motion    = motion_gt[start:start + m_len]

        wham_clip = (wham_clip - self.mean_wham) / (self.std_wham + 1e-8)
        motion    = (motion    - self.mean)      / (self.std      + 1e-8)

        return mid, wham_clip.astype(np.float32), motion.astype(np.float32), m_len, captions


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_train(batch):
    motions, whams, lengths = zip(*batch)
    max_t = max(m.shape[0] for m in motions)
    dim   = motions[0].shape[1]   # 263

    pad_motions = torch.zeros(len(motions), max_t, dim)
    pad_whams   = torch.zeros(len(whams),   max_t, dim)
    for i, (m, w) in enumerate(zip(motions, whams)):
        pad_motions[i, :m.shape[0]] = torch.from_numpy(m)
        pad_whams[i,   :w.shape[0]] = torch.from_numpy(w)

    return {
        "motion_3d": pad_motions,
        "wham_3d":   pad_whams,
        "length":    torch.LongTensor(list(lengths)),
    }


def collate_val(batch):
    mids, whams, motions, lengths, all_caps = zip(*batch)
    max_t  = max(w.shape[0] for w in whams)
    dim    = whams[0].shape[1]   # 263

    pad_whams   = torch.zeros(len(whams),   max_t, dim)
    pad_motions = torch.zeros(len(motions), max_t, dim)
    for i, (w, m) in enumerate(zip(whams, motions)):
        pad_whams[i,   :w.shape[0]] = torch.from_numpy(w)
        pad_motions[i, :m.shape[0]] = torch.from_numpy(m)

    return {
        "mid":          list(mids),
        "wham_3d":      pad_whams,
        "motion_3d":    pad_motions,
        "length":       torch.LongTensor(list(lengths)),
        "all_captions": list(all_caps),
    }


# ---------------------------------------------------------------------------
# T5 text generation (adapter + 3D VQ-VAE -> T5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text_t5_3d(vae_3d, adapter, tokenizer, t5_model,
                         wham_np, device, prompt="Generate text: ", max_new_tokens=40):
    """wham_np: (T, 263) numpy array (WHAM-normalized)."""
    feat = torch.from_numpy(wham_np).float().unsqueeze(0).to(device)  # (1, T, 263)
    feat = adapter(feat)                                               # (1, T, 263)

    tokenized   = vae_3d.encode(feat)                                 # (1, L)
    token_list  = tokenized.cpu().numpy()[0].reshape(-1).tolist()

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

    # --- model checkpoints ---
    p.add_argument("--vqvae_3d_pth", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "pretrained_vqvae" / "t2m.pth"),
                   help="HumanVQVAE 3D checkpoint (.pth, key 'net')")
    p.add_argument("--model_name", type=str,
                   default=str(ROOT_DIR / "m2t-ft-from-GSPretrained-base"),
                   help="T5 model dir for validation text generation")

    # --- data ---
    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Root dir with adapter_training_WAHA/ and new_joint_vecs/ (WHAM data)")
    p.add_argument("--data_root", type=str,
                   default=str(ROOT_DIR / "dataset" / "HumanML3D"))
    p.add_argument("--meta_dir", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "t2m" / "VQVAEV3_CB1024_CMT_H1024_NRES3" / "meta"),
                   help="Dir with mean.npy / std.npy (GT HumanML3D stats)")
    p.add_argument("--wham_meta_dir", type=str,
                   default="../2DMotionGPT/deps/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta",
                   help="Dir with mean_wham_3d.npy / std_wham_3d.npy (WHAM-specific stats)")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split",   type=str, default="val")
    p.add_argument("--unit_length", type=int, default=4)

    # --- training ---
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--val_batch_size", type=int,   default=32)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--max_epochs",     type=int,   default=1000)
    p.add_argument("--val_every",      type=int,   default=10,
                   help="Encoder loss validation interval (fast)")
    p.add_argument("--val_text_every", type=int,   default=100,
                   help="T5 text generation + BLEU/BERTScore interval (slow)")
    p.add_argument("--gpu_id",         type=int,   default=0)

    # --- generation ---
    p.add_argument("--prompt",         type=str, default="Generate text: ")
    p.add_argument("--max_new_tokens", type=int, default=40)

    # --- output ---
    p.add_argument("--ckpt_save_dir", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "adapter3d" / "MG-MotionLLM"))
    p.add_argument("--wandb_project", type=str, default="2DMG-MotionLLM")
    p.add_argument("--no_wandb",      action="store_true")

    # --- adapter architecture ---
    p.add_argument("--adapter_type", type=str, default="residual",
                   choices=["linear", "residual", "conv1d"])
    p.add_argument("--adapter_hidden", type=int, default=512)
    p.add_argument("--adapter_kernel_size", type=int, default=3)

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
    run_name  = f"adapter3d-{args.adapter_type}-bs{args.batch_size}-lr{args.lr}"
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args), name=run_name)

    os.makedirs(args.ckpt_save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Normalization stats
    # GT motion: HumanML3D mean/std; WHAM: WHAM-specific stats
    # Both normalized to std≈1.0 before the adapter.
    # ----------------------------------------------------------------
    mean      = np.load(pjoin(args.meta_dir,      "mean.npy"))
    std       = np.load(pjoin(args.meta_dir,      "std.npy"))
    mean_wham = np.load(pjoin(args.wham_meta_dir, "mean_wham_3d.npy"))
    std_wham  = np.load(pjoin(args.wham_meta_dir, "std_wham_3d.npy"))

    # ----------------------------------------------------------------
    # Datasets & dataloaders
    # ----------------------------------------------------------------
    train_dataset = WhamTrainDataset(
        data_root=args.data_root,
        split=args.train_split,
        mean=mean, std=std,
        mean_wham=mean_wham, std_wham=std_wham,
        estimated_motion_dir=args.estimated_motion_dir,
        unit_length=args.unit_length,
    )
    val_dataset = WhamValDataset(
        data_root=args.data_root,
        split=args.val_split,
        mean=mean, std=std,
        mean_wham=mean_wham, std_wham=std_wham,
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

    # 3D VQ-VAE: used for both GT reference encoding and WHAM tokenization
    vae_3d = _build_vae()
    ckpt_3d = torch.load(args.vqvae_3d_pth, map_location="cpu", weights_only=False)
    vae_3d.load_state_dict(ckpt_3d['net'], strict=True)

    # T5 (frozen, used only for validation text generation)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    t5_model  = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    for model in [vae_3d, t5_model]:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

    print(f"HumanVQVAE (3D) loaded from: {args.vqvae_3d_pth}")
    print(f"T5 loaded from: {args.model_name}")

    # Adapter (trainable)
    adapter = build_adapter(
        args.adapter_type, dim=263,
        hidden=args.adapter_hidden,
        kernel_size=args.adapter_kernel_size,
    ).to(device)

    start_epoch = 0
    if args.resume_from:
        ckpt_resume = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ckpt_resume["model_state_dict"])
        start_epoch = ckpt_resume.get("epoch", 0) + 1
        print(f"Resumed from {args.resume_from} (epoch {start_epoch})")

    if use_wandb:
        wandb.watch(adapter)

    # ----------------------------------------------------------------
    # Optimizer & loss
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn   = nn.L1Loss()

    # ----------------------------------------------------------------
    # CSV log
    # ----------------------------------------------------------------
    log_path = pjoin(args.ckpt_save_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_encoder_loss", "val_encoder_loss",
             "BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "BERTScore_F1"]
        )

    best_val_loss      = float("inf")
    best_val_loss_ckpt = None

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    for epoch in range(start_epoch, args.max_epochs):
        adapter.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()

            feats_gt   = batch["motion_3d"].to(device)  # GT 3D: (B, T, 263)
            feats_wham = adapter(batch["wham_3d"].to(device))  # adapted WHAM: (B, T, 263)

            with torch.no_grad():
                ref_encoded = vae_3d.vqvae.encoder(vae_3d.vqvae.preprocess(feats_gt))

            enc_wham = vae_3d.vqvae.encoder(vae_3d.vqvae.preprocess(feats_wham))
            loss = loss_fn(enc_wham, ref_encoded)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}  Encoder Loss: {avg_loss:.4f}")

        metrics_log  = {"train/encoder_loss": avg_loss}
        bleu1 = bleu4 = rouge = cider = bertscore_f1 = ""
        val_enc_loss_log = ""

        # ----------------------------------------------------------------
        # Validation: encoder loss (fast, every val_every epochs)
        # ----------------------------------------------------------------
        if epoch % args.val_every == 0:
            adapter.eval()
            val_total_enc_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Val (enc)", leave=False):
                    feats_gt   = batch["motion_3d"].to(device)
                    feats_wham = adapter(batch["wham_3d"].to(device))
                    ref_encoded = vae_3d.vqvae.encoder(vae_3d.vqvae.preprocess(feats_gt))
                    enc_wham    = vae_3d.vqvae.encoder(vae_3d.vqvae.preprocess(feats_wham))
                    val_total_enc_loss += loss_fn(enc_wham, ref_encoded).item()

            avg_val_enc_loss = val_total_enc_loss / len(val_loader)
            val_enc_loss_log = avg_val_enc_loss
            print(f"  val/encoder_loss: {avg_val_enc_loss:.4f}")
            metrics_log["val/encoder_loss"] = avg_val_enc_loss

            if avg_val_enc_loss < best_val_loss:
                best_val_loss = avg_val_enc_loss
                new_ckpt = pjoin(
                    args.ckpt_save_dir,
                    f"best_adapter3d_epoch{epoch}_valloss{best_val_loss:.4f}.pt",
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

        # ----------------------------------------------------------------
        # Validation: T5 text generation + BLEU/BERTScore (slow, every val_text_every epochs)
        # ----------------------------------------------------------------
        if epoch % args.val_text_every == 0:
            adapter.eval()
            all_preds, all_refs = [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Val (text)", leave=False):
                    wham_raw = batch["wham_3d"]
                    for i in range(len(batch["mid"])):
                        l       = batch["length"][i].item()
                        wham_np = wham_raw[i, :l].numpy()
                        pred = generate_text_t5_3d(
                            vae_3d, adapter, tokenizer, t5_model,
                            wham_np, device, args.prompt, args.max_new_tokens,
                        )
                        all_preds.append(pred)
                        all_refs.append(batch["all_captions"][i])

            if all_preds:
                bleu1, bleu4, rouge, cider = calculate_bleu_rouge_cider(all_refs, all_preds)
                print(f"  BLEU-1: {bleu1:.4f}  BLEU-4: {bleu4:.4f}  "
                      f"ROUGE-L: {rouge:.4f}  CIDEr: {cider:.4f}")
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

        if use_wandb:
            wandb.log(metrics_log, step=epoch)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, avg_loss, val_enc_loss_log,
                 bleu1, bleu4, rouge, cider, bertscore_f1]
            )


if __name__ == "__main__":
    main()
