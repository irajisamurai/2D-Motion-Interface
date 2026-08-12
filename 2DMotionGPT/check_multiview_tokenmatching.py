"""
Multi-view token matching analysis for 2DMotionGPT.

Measures:
  1. 2D multi-view consistency  : same motion, N random views → token match rate
  2. 3D vs 2D cross-modal       : 3D tokens vs 2D (random view) tokens
  3. Different-motion baseline  : shuffled inter-sample match rate (upper bound of chance)

Usage:
    CUDA_VISIBLE_DEVICES=7 python check_multiview_tokenmatching.py \
        --vqvae_ckpt ./checkpoints/2d_vqvae_ver3/.../best_vqvae.tar \
        --n_passes 5 --gpu_id 0
"""

import argparse
import copy
import os
import random
import sys
from datetime import datetime
from itertools import combinations
from os.path import join as pjoin

MOTION_DIM = 263

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from evaluate_motionGPT import (
    DEFAULT_MOTIONGPT_CKPT,
    DEFAULT_VQVAE_2D_CKPT,
    build_dataloader,
    build_models,
    build_word_vectorizer,
    encode_motion_tokens,
    get_device,
    load_2d_encoder_weights,
    load_motiongpt_checkpoint,
    load_statistics,
    prepare_batch_2d,
    prepare_batch_3d,
    save_results,
    seed_everything,
)
from src.config import parse_args


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vqvae_ckpt", type=str, default=DEFAULT_VQVAE_2D_CKPT,
    parser.add_argument("--motiongpt_ckpt", type=str, default=DEFAULT_MOTIONGPT_CKPT,
    parser.add_argument("--n_passes", type=int, default=5,
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/multiview_tokenmatching")
    script_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return script_args


# ──────────────────────────────────────────────────────────────────────────────
# Token collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_tokens_single(vqvae, dataloader, batch_preparer, device):
    tokens = []
    vqvae.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            feats, _ = batch_preparer(batch, device)
            motion_tokens, _ = encode_motion_tokens(vqvae, feats)
            tokens.append(motion_tokens[0].reshape(-1).cpu().numpy())
    return tokens


def collect_tokens_n_views_fixed_window(vqvae_2d, dataloader_2d, device, n_views, stats):
    """
    mean_3d = stats["mean"]          # (263,)
    std_3d  = stats["std"]           # (263,)
    mean_2d = stats["mean_2d"]       # (68,)
    std_2d  = stats["std_2d"]        # (68,)
    create_fn = dataloader_2d.dataset.create_2d_joints_from_features

    per_sample = []
    vqvae_2d.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader_2d, desc="multi-view (fixed window)", leave=False):
            # normalized 3D → raw 3D (denormalize)
            motion_norm = batch["motion"][0].numpy()          # (T, 263)
            motion_raw  = motion_norm * std_3d + mean_3d      # (T, 263) raw

            views_tokens = []
            for _ in range(n_views):
                feat_2d      = create_fn(motion_raw)                          # (T, 68) raw 2D
                feat_2d_norm = (feat_2d - mean_2d) / (std_2d + 1e-8)         # normalized
                pad          = np.zeros((motion_raw.shape[0], MOTION_DIM - 68), dtype=np.float32)
                feat         = np.concatenate([feat_2d_norm, pad], axis=-1)   # (T, 263)
                feat_t       = torch.from_numpy(feat).unsqueeze(0).to(device)
                tok, _       = vqvae_2d.encode(feat_t)
                views_tokens.append(tok[0].reshape(-1).cpu().numpy())
            per_sample.append(views_tokens)
    return per_sample


# ──────────────────────────────────────────────────────────────────────────────
# Match rate computation
# ──────────────────────────────────────────────────────────────────────────────

def _match_rate(a, b):
    n = min(len(a), len(b))
    return float(np.sum(a[:n] == b[:n]) / n)


def pairwise_match_rates(per_sample):
    rates = []
    for sample_runs in per_sample:
        pair_rates = [_match_rate(a, b) for a, b in combinations(sample_runs, 2)]
        rates.append(np.mean(pair_rates) if pair_rates else 0.0)
    return rates


def all_agree_rates(per_sample):
    rates = []
    for sample_runs in per_sample:
        tokens = np.stack(sample_runs, axis=0)          # (n_views, T)
        all_agree = np.all(tokens == tokens[0:1], axis=0)  # (T,) bool
        rates.append(float(all_agree.mean()))
    return rates


def cross_match_rates(tokens_a, tokens_b):
    return [_match_rate(a, b) for a, b in zip(tokens_a, tokens_b)]


def collect_cross_tokens(vqvae_3d, vqvae_2d, dataloader_2d, device):
    """
    tokens_3d, tokens_2d = [], []
    vqvae_3d.eval()
    vqvae_2d.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader_2d, desc="3D vs 2D (single pass)", leave=False):
            # 3D tokens
            feats_3d = batch["motion"].to(device)
            tok_3d, _ = encode_motion_tokens(vqvae_3d, feats_3d)
            tokens_3d.append(tok_3d[0].reshape(-1).cpu().numpy())

            m2d = batch["motion_2d"].to(device)
            pad = torch.zeros(m2d.shape[0], m2d.shape[1], MOTION_DIM - m2d.shape[2], device=device)
            feats_2d = torch.cat([m2d, pad], dim=-1)
            tok_2d, _ = encode_motion_tokens(vqvae_2d, feats_2d)
            tokens_2d.append(tok_2d[0].reshape(-1).cpu().numpy())
    return tokens_3d, tokens_2d


def shuffled_baseline_rates(tokens_a, tokens_b):
    shuffled = tokens_b.copy()
    random.shuffle(shuffled)
    return [_match_rate(tokens_a[i], shuffled[i]) for i in range(len(tokens_a))]


def compute_stats(rates):
    arr = np.array(rates) * 100
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_histogram(data_dict, out_path):
    """data_dict: {label: [0..1 rates]}"""
    plt.figure(figsize=(9, 5))
    for label, rates in data_dict.items():
        plt.hist(np.array(rates) * 100, alpha=0.6, bins=40, label=label)
    plt.xlim(0, 100)
    plt.xlabel("Token match rate (%)")
    plt.ylabel("Frequency")
    plt.title("Token match rate distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    script_args = parse_script_args()

    extra = []
    if "--cfg" not in sys.argv:
        extra += ["--cfg", "configs/config_h3d_stage1.yaml"]
    if "--nodebug" not in sys.argv:
        extra += ["--nodebug"]
    extra += ["--device", str(script_args.gpu_id), "--num_nodes", "1"]
    sys.argv.extend(extra)

    cfg = parse_args(phase="train")
    seed_everything(cfg.SEED_VALUE)
    device = get_device(script_args.gpu_id)

    print(f"motiongpt_ckpt : {script_args.motiongpt_ckpt}")
    print(f"vqvae_ckpt     : {script_args.vqvae_ckpt}")
    print(f"n_passes       : {script_args.n_passes}")

    stats = load_statistics(cfg)
    w_vectorizer = build_word_vectorizer(cfg)
    dataloader_2d = build_dataloader(cfg, w_vectorizer, stats, use_2d=True)

    vqvae, lm = build_models(cfg, device)

    load_motiongpt_checkpoint(vqvae, lm, script_args.motiongpt_ckpt)
    vqvae_3d = copy.deepcopy(vqvae)
    load_2d_encoder_weights(vqvae, script_args.vqvae_ckpt)
    vqvae_2d = vqvae

    # ──────────────────────────────────────────────
    # [1] 2D multi-view consistency
    # ──────────────────────────────────────────────
    print(f"\n[1] 2D multi-view consistency ({script_args.n_passes} views, fixed window)")
    per_sample_2d = collect_tokens_n_views_fixed_window(
        vqvae_2d, dataloader_2d, device, script_args.n_passes, stats
    )
    mv_rates      = all_agree_rates(per_sample_2d)
    mv_pair_rates = pairwise_match_rates(per_sample_2d)

    # Different-motion baseline (pass 0 vs shuffled pass 1)
    tokens_2d_pass0 = [s[0] for s in per_sample_2d]
    tokens_2d_pass1 = [s[1] for s in per_sample_2d]
    diff_rates = shuffled_baseline_rates(tokens_2d_pass0, tokens_2d_pass1)

    # ──────────────────────────────────────────────
    # [2] 3D vs 2D cross-modal alignment
    # ──────────────────────────────────────────────
    print("\n[2] 3D vs 2D cross-modal alignment")
    tokens_3d, tokens_2d_single = collect_cross_tokens(vqvae_3d, vqvae_2d, dataloader_2d, device)
    cross_rates = cross_match_rates(tokens_3d, tokens_2d_single)

    # ──────────────────────────────────────────────
    # Output
    # ──────────────────────────────────────────────
    os.makedirs(script_args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": timestamp,
        "motiongpt_ckpt": script_args.motiongpt_ckpt,
        "vqvae_ckpt": script_args.vqvae_ckpt,
        "n_passes": script_args.n_passes,
        "2d_multiview_all_agree": compute_stats(mv_rates),
        "2d_multiview_pairwise": compute_stats(mv_pair_rates),
        "different_motion_baseline": compute_stats(diff_rates),
        "3d_vs_2d_alignment": compute_stats(cross_rates),
    }

    json_path = pjoin(script_args.out_dir, f"tokenmatching_{timestamp}.json")
    save_results(results, json_path)

    for key in ("2d_multiview_all_agree", "2d_multiview_pairwise", "different_motion_baseline", "3d_vs_2d_alignment"):
        s = results[key]
        print(f"  {key}: mean={s['mean']:.1f}%, std={s['std']:.1f}%, median={s['median']:.1f}%")

    png_path = pjoin(script_args.out_dir, f"tokenmatching_{timestamp}.png")
    plot_histogram(
        {
            f"2D multi-view all-agree ({script_args.n_passes} views)": mv_rates,
            "Different motion (baseline)": diff_rates,
            "3D vs 2D (cross-modal)": cross_rates,
        },
        png_path,
    )


if __name__ == "__main__":
    main()
