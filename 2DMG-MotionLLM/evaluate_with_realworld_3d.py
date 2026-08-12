"""
Evaluate 2DMG-MotionLLM on real-world TRACE-estimated 3D poses (M2T task).

Pipeline:
    TRACE 3D (263-dim HumanML3D format)
        → normalize
        → HumanVQVAE 3D encoder (pretrained_vqvae/t2m.pth)
        → token string  '<Motion Tokens><0><3>...</Motion Tokens>'
        → T5 (m2t-ft-from-GSPretrained-base)
        → predicted text

Usage:
    cd 2DMG-MotionLLM
    python evaluate_with_realworld_3d.py \\
        --estimated_motion_dir /path/to/custom_real_world_dataset \\
        [--vqvae_pth ./checkpoints/pretrained_vqvae/t2m.pth] \\
        [--model_name ./m2t-ft-from-GSPretrained-base/] \\
        [--split test] [--gpu_id 0]

expected directory layout:
    <estimated_motion_dir>/new_joint_vecs/<motion_id>.npy   # (T, 263) float32
"""

import argparse
import codecs as cs
import glob
import os
import sys
import random

import numpy as np
import torch
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

class RealWorld3DDataset(Dataset):
    """
    Loads TRACE-estimated 3D poses (HumanML3D 263-dim format).
    GT captions: lemmatized (TM2T/2DMotionGPT-compatible, token.split('/')[0]).
    """

    def __init__(self, estimated_motion_dir, data_root, split, mean, std,
                 unit_length=4, min_len=40, max_len=196):
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.min_len = min_len
        self.max_len = max_len

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
            self.samples.append((mid, np_path, text_path))
        print(f"[RealWorld3DDataset] {len(self.samples)} samples loaded ({skipped} skipped).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, np_path, text_path = self.samples[idx]

        motion = np.load(np_path).astype(np.float32)
        T = motion.shape[0]
        m_len = (T // self.unit_length) * self.unit_length
        m_len = max(self.min_len, min(m_len, self.max_len))
        if T < m_len:
            m_len = (T // self.unit_length) * self.unit_length
        start = random.randint(0, max(0, T - m_len))
        motion = motion[start:start + m_len]
        motion = (motion - self.mean) / (self.std + 1e-8)

        # GT captions — lemmatized (consistent with 2DMotionGPT / TM2T)
        captions = []
        with cs.open(text_path) as f:
            for line in f.readlines():
                try:
                    parts = line.strip().split('#')
                    t_tokens = parts[1].split(' ')
                    cap = ' '.join(tok.split('/')[0] for tok in t_tokens)
                    f_tag = float(parts[2]) if len(parts) > 2 else 0.0
                    to_tag = float(parts[3]) if len(parts) > 3 else 0.0
                    if np.isnan(f_tag): f_tag = 0.0
                    if np.isnan(to_tag): to_tag = 0.0
                    if f_tag == 0.0 and to_tag == 0.0:
                        captions.append(cap)
                except Exception:
                    pass

        # Pad to 3 captions (consistent with evaluation_m2t)
        if len(captions) > 3:
            captions = captions[:3]
        elif len(captions) == 2:
            captions = captions + captions[:1]
        elif len(captions) == 1:
            captions = captions * 3

        return mid, motion, captions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Dir with new_joint_vecs/<id>.npy (TRACE, 263-dim HumanML3D format)")
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
    mean = np.load(pjoin(args.meta_dir, "mean.npy"))
    std  = np.load(pjoin(args.meta_dir, "std.npy"))

    # --- Dataset ---
    dataset = RealWorld3DDataset(
        estimated_motion_dir=args.estimated_motion_dir,
        data_root=args.data_root,
        split=args.split,
        mean=mean,
        std=std,
        unit_length=args.unit_length,
    )
    if len(dataset) == 0:
        print("No samples found. Exiting.")
        return

    # --- VQ-VAE (HumanVQVAE, 3D weights, no encoder override) ---
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
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    model.eval()
    print(f"T5 loaded from {args.model_name}")

    # --- Inference ---
    all_preds, all_refs, video_ids = [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Inference [TRACE 3D → T5]"):
            mid, motion_np, captions = dataset[i]

            motion = torch.from_numpy(motion_np).float().unsqueeze(0).to(device)  # (1, T, 263)
            tokenized = vae.encode(motion)             # (1, L) — no tuple
            token_list = tokenized.cpu().numpy()[0].reshape(-1).tolist()

            motion_string = '<Motion Tokens>'
            for tok in token_list:
                motion_string += f'<{tok}>'
            motion_string += '</Motion Tokens>'

            prompt = args.prompt + motion_string
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(
                input_ids,
                max_length=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip('"')

            all_preds.append(pred_text)
            all_refs.append(captions)
            video_ids.append(mid)

    # --- Metrics ---
    print(f"\n=== Results ({len(all_preds)} samples) ===")
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


if __name__ == "__main__":
    main()
