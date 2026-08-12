# -*- coding: utf-8 -*-
"""
Evaluate 3D adapter + MG-MotionLLM on real-world WHAM-estimated 3D poses (M2T task).

Pipeline (adapter mode):
    WHAM 3D (263D, WHAM-norm) -> Adapter -> HumanVQVAE (frozen) -> tokens -> T5 -> text

Pipeline (no_adapter mode):
    WHAM 3D (263D, GT-norm)   ->           HumanVQVAE (frozen) -> tokens -> T5 -> text

Evaluation method is identical to evaluate_with_realworld_3d.py.

Usage:
    cd 2DMG-MotionLLM
    CUDA_VISIBLE_DEVICES=X python evaluate_with_realworld_3d_adapter.py \\
        --estimated_motion_dir /path/to/wham_output \\
        --adapter_ckpt ./checkpoints/adapter3d/MG-MotionLLM/best_adapter3d_*.pt \\
        [--mode both]
"""

import argparse
import codecs as cs
import glob
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from os.path import join as pjoin
from tqdm import tqdm
from torch.utils.data import Dataset

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import models.vqvae as vqvae_module
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.evaluate import calculate_bleu_rouge_cider, evaluate_bert_score


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealWorld3DAdapterDataset(Dataset):
    """WHAM-estimated 3D poses for adapter evaluation.

    Returns (mid, wham_wham_norm, wham_gt_norm, captions):
      - wham_wham_norm : WHAM-specific normalization  -> adapter input
      - wham_gt_norm   : GT HumanML3D normalization   -> no-adapter baseline
    """

    def __init__(self, estimated_motion_dir, data_root, split, mean, std,
                 mean_wham, std_wham, unit_length=4, min_len=40, max_len=196):
        self.mean      = mean
        self.std       = std
        self.mean_wham = mean_wham
        self.std_wham  = std_wham
        self.unit_length = unit_length
        self.min_len   = min_len
        self.max_len   = max_len

        split_ids = set()
        split_path = pjoin(data_root, f'{split}.txt')
        if os.path.exists(split_path):
            with cs.open(split_path) as f:
                for line in f:
                    split_ids.add(line.strip())

        npy_paths = sorted(glob.glob(
            os.path.join(estimated_motion_dir, "new_joint_vecs", "*.npy")
        ))
        if not npy_paths:
            raise FileNotFoundError(
                f"No .npy files under {estimated_motion_dir}/new_joint_vecs/"
            )

        self.samples = []
        skipped = 0
        for np_path in npy_paths:
            mid = os.path.basename(np_path).replace(".npy", "")
            if split_ids and mid not in split_ids:
                skipped += 1
                continue
            text_path = pjoin(data_root, "texts", mid + ".txt")
            if not os.path.exists(text_path):
                skipped += 1
                continue
            arr = np.load(np_path, mmap_mode='r')
            if np.any(np.isnan(arr)) or arr.shape[0] < min_len:
                skipped += 1
                continue
            self.samples.append((mid, np_path, text_path))

        print(f"[RealWorld3DAdapterDataset] {len(self.samples)} samples loaded "
              f"({skipped} skipped, split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, np_path, text_path = self.samples[idx]

        wham_raw = np.load(np_path).astype(np.float32)  # (T, 263)
        T = wham_raw.shape[0]
        m_len = (T // self.unit_length) * self.unit_length
        m_len = max(self.min_len, min(m_len, self.max_len))
        start = random.randint(0, max(0, T - m_len))
        wham_clip = wham_raw[start:start + m_len]

        wham_wham_norm = (wham_clip - self.mean_wham) / (self.std_wham + 1e-8)
        wham_gt_norm   = (wham_clip - self.mean)      / (self.std      + 1e-8)

        captions = []
        with cs.open(text_path) as f:
            for line in f.readlines():
                try:
                    parts    = line.strip().split('#')
                    t_tokens = parts[1].split(' ')
                    cap      = ' '.join(tok.split('/')[0] for tok in t_tokens)
                    f_tag    = float(parts[2]) if len(parts) > 2 else 0.0
                    to_tag   = float(parts[3]) if len(parts) > 3 else 0.0
                    if np.isnan(f_tag):  f_tag  = 0.0
                    if np.isnan(to_tag): to_tag = 0.0
                    if f_tag == 0.0 and to_tag == 0.0:
                        captions.append(cap)
                except Exception:
                    pass

        if len(captions) > 3:  captions = captions[:3]
        elif len(captions) == 2: captions = captions + captions[:1]
        elif len(captions) == 1: captions = captions * 3

        return mid, wham_wham_norm.astype(np.float32), wham_gt_norm.astype(np.float32), captions


# ---------------------------------------------------------------------------
# Adapter models  (identical to train_adapter_3d.py)
# ---------------------------------------------------------------------------

class AdapterResidual(nn.Module):
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


class AdapterLinear(nn.Module):
    def __init__(self, dim=263):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.linear(self.norm(x))


class AdapterConv1d(nn.Module):
    def __init__(self, dim=263, hidden=512, kernel_size=3, dropout=0.1):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        r = self.norm(x).permute(0, 2, 1)
        r = self.drop(self.act(self.conv1(r)))
        r = self.conv2(r).permute(0, 2, 1)
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Dir with new_joint_vecs/<id>.npy (WHAM output, 263-dim)")
    p.add_argument("--adapter_ckpt", type=str, required=True,
                   help="Trained 3D adapter checkpoint (.pt)")
    p.add_argument("--vqvae_pth", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "pretrained_vqvae" / "t2m.pth"))
    p.add_argument("--model_name", type=str,
                   default=str(ROOT_DIR / "m2t-ft-from-GSPretrained-base"))
    p.add_argument("--meta_dir", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "t2m" / "VQVAEV3_CB1024_CMT_H1024_NRES3" / "meta"))
    p.add_argument("--data_root", type=str,
                   default=str(ROOT_DIR / "dataset" / "HumanML3D"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--mode", type=str, default="both",
                   choices=["adapter", "no_adapter", "both"],
                   help="'adapter': WHAM->adapter->VQ-VAE, "
                        "'no_adapter': WHAM->VQ-VAE (baseline), "
                        "'both': run both (default)")
    p.add_argument("--adapter_type", type=str, default="residual",
                   choices=["linear", "residual", "conv1d"])
    p.add_argument("--adapter_hidden", type=int, default=512)
    p.add_argument("--adapter_kernel_size", type=int, default=3)
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
    mean      = np.load(pjoin(args.meta_dir, "mean.npy"))
    std       = np.load(pjoin(args.meta_dir, "std.npy"))
    mean_wham = np.load(pjoin(args.meta_dir, "mean_wham_3d.npy"))
    std_wham  = np.load(pjoin(args.meta_dir, "std_wham_3d.npy"))

    # --- Dataset ---
    dataset = RealWorld3DAdapterDataset(
        estimated_motion_dir=args.estimated_motion_dir,
        data_root=args.data_root,
        split=args.split,
        mean=mean,
        std=std,
        mean_wham=mean_wham,
        std_wham=std_wham,
        unit_length=args.unit_length,
    )
    if len(dataset) == 0:
        print("No samples found. Exiting.")
        return

    # --- VQ-VAE ---
    import argparse as _ap
    vae_args = _ap.Namespace(dataname='t2m', quantizer='ema_reset', mu=0.99)
    vae = vqvae_module.HumanVQVAE(
        vae_args,
        nb_code=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
    ).to(device)
    ckpt = torch.load(args.vqvae_pth, map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt['net'], strict=True)
    vae.eval()
    print(f"VQ-VAE loaded from {args.vqvae_pth}")

    # --- T5 ---
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    t5_model  = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    t5_model.eval()
    print(f"T5 loaded from {args.model_name}")

    # --- Adapter ---
    adapter = build_adapter(
        args.adapter_type, dim=263,
        hidden=args.adapter_hidden,
        kernel_size=args.adapter_kernel_size,
    ).to(device)
    ckpt_a = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ckpt_a.get("model_state_dict", ckpt_a))
    adapter.eval()
    print(f"Adapter loaded from {args.adapter_ckpt}")

    # --- Inference helper (identical pipeline to evaluate_with_realworld_3d.py) ---
    def run_inference(label, use_adapter):
        all_preds, all_refs, video_ids = [], [], []
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc=f"Inference [{label}]"):
                mid, wham_wham_norm, wham_gt_norm, captions = dataset[i]

                feat_np = wham_wham_norm if use_adapter else wham_gt_norm
                feat = torch.from_numpy(feat_np).float().unsqueeze(0).to(device)  # (1, T, 263)

                if use_adapter:
                    feat = adapter(feat)

                tokenized  = vae.encode(feat)         # (1, L)
                token_list = tokenized.cpu().numpy()[0].reshape(-1).tolist()

                motion_string = '<Motion Tokens>'
                for tok in token_list:
                    motion_string += f'<{tok}>'
                motion_string += '</Motion Tokens>'

                prompt    = args.prompt + motion_string
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                outputs   = t5_model.generate(
                    input_ids,
                    max_length=args.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
                pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip('"')

                all_preds.append(pred_text)
                all_refs.append(captions)
                video_ids.append(mid)

        return all_preds, all_refs, video_ids

    def print_results(label, all_preds, all_refs, video_ids):
        print(f"\n=== [{label}] Results ({len(all_preds)} samples) ===")
        bleu1, bleu4, rouge, cider = calculate_bleu_rouge_cider(all_refs, all_preds)
        print(f"  {'BLEU-1':>10}: {bleu1:.4f}  (sentence-level)")
        print(f"  {'BLEU-4':>10}: {bleu4:.4f}  (sentence-level)")
        print(f"  {'ROUGE-L':>10}: {rouge:.4f}")
        print(f"  {'CIDEr':>10}: {cider:.4f}")
        bert_f1_mean = evaluate_bert_score(all_preds, all_refs)
        print(f"  {'BERTScore':>10}: {bert_f1_mean:.4f}")
        print()
        for i, (vid, pred, refs) in enumerate(zip(video_ids, all_preds, all_refs)):
            print(f"[{i+1:3d}] {vid}")
            print(f"       GT  : {refs[0]}")
            print(f"       Pred: {pred}")

    mode = args.mode

    if mode in ("adapter", "both"):
        preds, refs, vids = run_inference("with adapter", use_adapter=True)
        print_results("with adapter", preds, refs, vids)

    if mode in ("no_adapter", "both"):
        preds, refs, vids = run_inference("no adapter", use_adapter=False)
        print_results("no adapter", preds, refs, vids)


if __name__ == "__main__":
    main()
