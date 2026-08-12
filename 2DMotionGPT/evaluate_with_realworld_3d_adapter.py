# -*- coding: utf-8 -*-
"""
Evaluate 3D adapter + MotionGPT on real-world WHAM-estimated 3D poses (M2T task).

Input data layout:
    <estimated_motion_dir>/new_joint_vecs/<motion_id>.npy
Each .npy: shape (T, 263) -- HumanML3D 263-dim feature format.
motion_id must match a HumanML3D motion ID (for GT text lookup).

Pipeline (adapter mode):
    WHAM 3D (263D, WHAM-norm) -> Adapter -> MotionGPT 3D VQ-VAE (frozen) -> tokens -> LM -> text

Pipeline (no_adapter mode):
    WHAM 3D (263D, GT-norm)   ->           MotionGPT 3D VQ-VAE (frozen) -> tokens -> LM -> text

Usage:
    python evaluate_with_realworld_3d_adapter.py \\
        --cfg configs/config_h3d_stage1.yaml --nodebug \\
        --estimated_motion_dir ./real_world_dataset/wham_output \\
        --adapter_ckpt ./checkpoints/adapter3d/.../best_adapter3d.tar \\
        [--mode both]           # adapter / no_adapter / both (default: both)
        [--adapter_type residual]
        [--split test]
"""

import argparse
import codecs as cs
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from os.path import join as pjoin
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import parse_args
from src.data.humanml.dataset_t2m import Text2MotionDataset
from src.data.utils import humanml3d_collate_2d
from src.models.mgpt_vq import VQVae
from src.data.humanml.utils.word_vectorizer import WordVectorizer
from src.archs.mgpt_lm import MLM

try:
    from nlgmetricverse import NLGMetricverse, load_metric
    HAS_NLG = True
except ImportError:
    HAS_NLG = False
    print("[Warning] nlgmetricverse not installed -- BLEU/ROUGE/CIDEr skipped.")

try:
    from bert_score import score as score_bert
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("[Warning] bert-score not installed -- BERTScore skipped.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealWorld3DAdapterEvalDataset(Text2MotionDataset):
    """Evaluation dataset for WHAM-estimated 3D poses with/without 3D adapter.

    Returns a 10-tuple:
        (caption, motion_3d_gt_norm, wham_wham_norm, m_length,
         None, None, None, None, all_captions, wham_gt_norm)

    - wham_wham_norm : WHAM features normalized with WHAM-specific stats
                       (input to the adapter)
    - wham_gt_norm   : WHAM features normalized with GT HumanML3D stats
                       (used for no-adapter baseline, same distribution as the
                        original evaluate_with_realworld_3d.py)
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_wham,
        std_wham,
        w_vectorizer,
        estimated_motion_dir,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer
        self.data_root    = data_root
        self.mean_wham    = mean_wham
        self.std_wham     = std_wham

        npy_dir   = os.path.join(estimated_motion_dir, "new_joint_vecs")
        npy_paths = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        if not npy_paths:
            raise FileNotFoundError(
                f"No .npy files found under {npy_dir}\n"
                "WHAM output must be preprocessed into HumanML3D 263-dim format first."
            )

        # filter out NaN files
        valid_paths = []
        for p in npy_paths:
            arr = np.load(p, mmap_mode='r')
            if not np.any(np.isnan(arr)):
                valid_paths.append(p)
        skipped = len(npy_paths) - len(valid_paths)
        if skipped:
            print(f"[RealWorld3DAdapterEvalDataset] Skipped {skipped} NaN files")

        print(f"[RealWorld3DAdapterEvalDataset] Found {len(valid_paths)} WHAM .npy files")
        self.npy_paths   = valid_paths
        self.motion_list = [np.load(p).astype(np.float32) for p in valid_paths]

    def __len__(self):
        return len(self.motion_list)

    def __getitem__(self, item):
        wham_raw   = self.motion_list[item]   # (T, 263) raw WHAM features
        m_length   = wham_raw.shape[0]

        motion_id  = os.path.basename(self.npy_paths[item]).replace(".npy", "")
        text_path  = pjoin(self.data_root, "texts", motion_id + ".txt")
        motion_3d  = np.load(pjoin(self.data_root, "new_joint_vecs", motion_id + ".npy"))
        m_length_3d = motion_3d.shape[0]

        # GT text
        text_data = []
        with cs.open(text_path) as f:
            for line in f.readlines():
                try:
                    parts   = line.strip().split('#')
                    caption = parts[0]
                    t_tokens = parts[1].split(' ')
                    f_tag   = float(parts[2])
                    to_tag  = float(parts[3])
                    f_tag   = 0.0 if np.isnan(f_tag)  else f_tag
                    to_tag  = 0.0 if np.isnan(to_tag) else to_tag
                    if f_tag == 0.0 and to_tag == 0.0:
                        text_data.append({'caption': caption, 'tokens': t_tokens})
                except Exception:
                    pass

        all_captions = [' '.join(t.split('/')[0] for t in td['tokens']) for td in text_data]
        if   len(all_captions) > 3: all_captions = all_captions[:3]
        elif len(all_captions) == 2: all_captions = all_captions + all_captions[:1]
        elif len(all_captions) == 1: all_captions = all_captions * 3

        caption = random.choice(text_data)['caption']

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length    = (m_length    // self.unit_length - 1) * self.unit_length
            m_length_3d = (m_length_3d // self.unit_length - 1) * self.unit_length
        else:
            m_length    = (m_length    // self.unit_length) * self.unit_length
            m_length_3d = (m_length_3d // self.unit_length) * self.unit_length

        start      = random.randint(0, wham_raw.shape[0] - m_length)
        wham_clip  = wham_raw[start:start + m_length]

        start_3d   = random.randint(0, motion_3d.shape[0] - m_length_3d)
        motion_3d  = motion_3d[start_3d:start_3d + m_length_3d]

        # two normalizations of the same WHAM clip
        wham_wham_norm = (wham_clip - self.mean_wham) / self.std_wham   # adapter input
        wham_gt_norm   = (wham_clip - self.mean)      / self.std        # no-adapter baseline

        motion_3d_norm = (motion_3d - self.mean) / self.std

        # position 2 -> batch["motion_2d"] (adapter input, WHAM-normalized)
        # position 9  -> extra field for no-adapter path (GT-normalized)
        return (caption, motion_3d_norm, wham_wham_norm, m_length,
                None, None, None, None, all_captions, wham_gt_norm)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def rw3d_collate(batch):
    """Like rw_collate in evaluate_with_realworld.py: handles the extra 10th field."""
    assert len(batch) == 1, "rw3d_collate only supports batch_size=1"
    item = batch[0]
    base = humanml3d_collate_2d([item[:9]])
    wham_gt = torch.from_numpy(item[9]).float()            # (T, 263)
    base["wham_gt_norm"] = wham_gt.unsqueeze(0)            # (1, T, 263)
    return base


# ---------------------------------------------------------------------------
# Adapter models  (same as train_adapter_3d.py, dim=263)
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
# CLI
# ---------------------------------------------------------------------------

def parse_script_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Root dir with new_joint_vecs/<id>.npy (WHAM output, 263-dim)")
    p.add_argument("--adapter_ckpt", type=str, required=True,
                   help="Trained 3D adapter checkpoint (.tar)")
    p.add_argument("--motiongpt_ckpt", type=str,
                   default="./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar",
                   help="MotionGPT-base checkpoint (.tar)")
    p.add_argument("--split", type=str, default="test",
                   help="HumanML3D split for GT text lookup (default: test)")
    p.add_argument("--mode", type=str, default="both",
                   choices=["adapter", "no_adapter", "both"],
                   help="'adapter': WHAM->adapter->VQ-VAE, "
                        "'no_adapter': WHAM->VQ-VAE (baseline), "
                        "'both': run both (default)")
    p.add_argument("--adapter_type", type=str, default="residual",
                   choices=["linear", "residual", "conv1d"])
    p.add_argument("--adapter_hidden", type=int, default=512)
    p.add_argument("--adapter_kernel_size", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of samples (for quick testing)")
    script_args, _ = p.parse_known_args()
    return script_args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_args = parse_script_args()
    cfg = parse_args(phase="train")
    pl.seed_everything(1000)
    random.seed(1000)

    device = torch.device(f"cuda:{cfg.DEVICE[0]}" if torch.cuda.is_available() else "cpu")

    # Normalization stats
    data_root     = cfg.DATASET.HUMANML3D.ROOT
    dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m',
                          "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
    mean      = np.load(pjoin(dis_data_root, "mean.npy"))           # GT stats
    std       = np.load(pjoin(dis_data_root, "std.npy"))
    mean_wham = np.load(pjoin(dis_data_root, "mean_wham_3d.npy"))   # WHAM-specific stats
    std_wham  = np.load(pjoin(dis_data_root, "std_wham_3d.npy"))
    w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

    # Dataset & dataloader
    dataset = RealWorld3DAdapterEvalDataset(
        data_root=data_root,
        split=script_args.split,
        mean=mean,
        std=std,
        mean_wham=mean_wham,
        std_wham=std_wham,
        w_vectorizer=w_vectorizer,
        estimated_motion_dir=script_args.estimated_motion_dir,
        max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
        min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
        unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
    )
    if script_args.max_samples:
        dataset.npy_paths   = dataset.npy_paths[:script_args.max_samples]
        dataset.motion_list = dataset.motion_list[:script_args.max_samples]
        print(f"[max_samples] Truncated to {len(dataset.motion_list)} samples")

    num_workers = cfg.TRAIN.NUM_WORKERS
    dataloader  = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rw3d_collate,
        persistent_workers=(num_workers > 0),
    )

    # Models: MotionGPT 3D VQ-VAE + LM (both frozen)
    vqvae = VQVae(
        nfeats=cfg.vq.default.params.nfeats,
        quantizer=cfg.vq.default.params.quantizer,
        code_num=cfg.vq.default.params.code_num,
        code_dim=cfg.vq.default.params.code_dim,
        output_emb_width=cfg.vq.default.params.output_emb_width,
        down_t=cfg.vq.default.params.down_t,
        stride_t=cfg.vq.default.params.stride_t,
        width=cfg.vq.default.params.width,
        depth=cfg.vq.default.params.depth,
        dilation_growth_rate=cfg.vq.default.params.dilation_growth_rate,
        norm=cfg.vq.default.params.norm,
        activation=cfg.vq.default.params.activation,
    ).to(device)

    lm = MLM(
        model_path=cfg.lm.default.params.model_path,
        model_type=cfg.lm.default.params.model_type,
        stage=cfg.lm.default.params.stage,
        motion_codebook_size=cfg.lm.default.params.motion_codebook_size,
    ).to(device)

    ckpt_base = torch.load(script_args.motiongpt_ckpt, map_location="cpu", weights_only=True)
    vqvae.load_state_dict(
        {k.replace("vae.", ""): v for k, v in ckpt_base["state_dict"].items() if "vae" in k}
    )
    lm.load_state_dict(
        {k.replace("lm.", ""): v for k, v in ckpt_base["state_dict"].items() if "lm" in k}
    )

    # 3D Adapter
    adapter = build_adapter(
        script_args.adapter_type,
        dim=263,
        hidden=script_args.adapter_hidden,
        kernel_size=script_args.adapter_kernel_size,
    ).to(device)
    ckpt_adapter = torch.load(script_args.adapter_ckpt, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ckpt_adapter.get("model_state_dict", ckpt_adapter))

    for model in [vqvae, lm, adapter]:
        model.eval()

    # Inference helper
    def run_inference(label, feats_fn):
        gt_list, pred_list = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Inference [{label}]"):
                feats = feats_fn(batch)                    # (1, T, 263)
                motion_token, _ = vqvae.encode(feats)
                outputs = lm.generate_conditional(
                    motion_tokens=[motion_token[0]],
                    lengths=[motion_token.shape[1]],
                    task="m2t",
                    stage='test',
                )
                gt_list.extend(batch['all_captions'])
                pred_list.extend(outputs)
        return gt_list, pred_list

    def print_results(label, gt_list, pred_list):
        print(f"\n=== [{label}] Results ({len(pred_list)} samples) ===")
        if HAS_NLG:
            metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            scores = NLGMetricverse(metrics)(predictions=pred_list, references=gt_list)
            for key in ["bleu_1", "bleu_4", "rouge", "cider"]:
                v = scores.get(key, {})
                v = v.get("score", list(v.values())[0]) if isinstance(v, dict) else v
                print(f"  {key:>8}: {v:.4f}")
        if HAS_BERTSCORE:
            _, _, F1 = score_bert(pred_list, gt_list, lang='en',
                                  rescale_with_baseline=True, idf=True,
                                  device=device, verbose=False)
            print(f"  BERTScore F1: {F1.mean().item():.4f}")
        else:
            F1 = None
        print()
        for i, (p, g) in enumerate(zip(pred_list, gt_list)):
            motion_id = os.path.basename(dataset.npy_paths[i]).replace(".npy", "")
            bert_str  = f"{F1[i].item():.4f}" if F1 is not None else "N/A"
            gt_str    = g[0] if isinstance(g, list) and len(g) == 1 else g
            print(f"[{i+1:3d}] {motion_id}")
            print(f"       GT  : {gt_str}")
            print(f"       Pred: {p}")
            print(f"       BERT: {bert_str}")

    mode = script_args.mode

    if mode in ("adapter", "both"):
        # WHAM (WHAM-normalized) -> adapter -> VQ-VAE
        def feats_with_adapter(batch):
            return adapter(batch["motion_2d"].to(device))   # (1, T, 263)

        gt_list, pred_list = run_inference("with adapter", feats_with_adapter)
        print_results("with adapter", gt_list, pred_list)

    if mode in ("no_adapter", "both"):
        # WHAM (GT-normalized) -> VQ-VAE directly (baseline)
        def feats_no_adapter(batch):
            return batch["wham_gt_norm"].to(device)          # (1, T, 263)

        gt_list, pred_list = run_inference("no adapter", feats_no_adapter)
        print_results("no adapter", gt_list, pred_list)


if __name__ == "__main__":
    main()
