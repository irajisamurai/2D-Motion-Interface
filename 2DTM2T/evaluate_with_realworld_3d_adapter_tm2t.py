# -*- coding: utf-8 -*-
"""
Evaluate TM2T M2T pipeline on real-world WHAM-estimated 3D poses, with/without 3D adapter.

Pipeline (adapter mode):
    WHAM 3D (263-dim, WHAM-norm) -> Adapter -> MotionGPT 3D VQ-VAE -> tokens
    -> TransformerV2 (M2T_MotionGPT) beam search -> text

Pipeline (no_adapter mode):
    WHAM 3D (263-dim, GT-norm)   ->           MotionGPT 3D VQ-VAE -> tokens
    -> TransformerV2 beam search -> text

Usage:
    python evaluate_with_realworld_3d_adapter_tm2t.py \\
        --estimated_motion_dir /path/to/wham_output \\
        --adapter_ckpt ./checkpoints/adapter3d/.../best_adapter3d.tar \\
        [--mode both]             # adapter / no_adapter / both (default: both)
        [--adapter_type residual]
        [--split test]

Expected directory layout:
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
import torch.nn as nn
from os.path import join as pjoin
from tqdm import tqdm
from torch.utils.data import Dataset

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
    print("[Warning] nlgeval / nlgmetricverse not installed -- BLEU/ROUGE/CIDEr skipped.")

try:
    from bert_score import score as score_bert
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("[Warning] bert-score not installed -- BERTScore skipped.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealWorld3DAdapterDataset(Dataset):
    """
    WHAM-estimated 3D poses (HumanML3D 263-dim format) for TM2T M2T evaluation.
    GT captions are lemmatized (TM2T official protocol).

    Returns:
        mid          : motion ID string
        wham_wham_norm: WHAM features normalized with WHAM-specific stats (adapter input)
        wham_gt_norm : WHAM features normalized with GT HumanML3D stats (no-adapter baseline)
        captions     : list of lemmatized GT caption strings
    """

    def __init__(self, estimated_motion_dir, data_root, split, mean, std,
                 mean_wham, std_wham, unit_length=4, min_len=40, max_len=196):
        self.data_root = data_root
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
                f"No .npy files under {estimated_motion_dir}/new_joint_vecs/\n"
                "WHAM output must be preprocessed into HumanML3D 263-dim format."
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
            if np.any(np.isnan(arr)):
                skipped += 1
                continue
            self.samples.append((mid, np_path, text_path))

        print(f"[RealWorld3DAdapterDataset] {len(self.samples)} samples loaded "
              f"({skipped} skipped).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mid, np_path, text_path = self.samples[idx]

        motion = np.load(np_path).astype(np.float32)  # (T, 263)
        T = motion.shape[0]
        m_len = (T // self.unit_length) * self.unit_length
        m_len = max(self.min_len, min(m_len, self.max_len))
        if T < m_len:
            m_len = (T // self.unit_length) * self.unit_length
        start = random.randint(0, max(0, T - m_len))
        motion = motion[start:start + m_len]          # (m_len, 263)

        # two normalizations of the same WHAM clip
        wham_wham_norm = (motion - self.mean_wham) / (self.std_wham + 1e-8)  # adapter input
        wham_gt_norm   = (motion - self.mean)       / (self.std      + 1e-8)  # no-adapter baseline

        # GT captions -- lemmatized (TM2T official protocol)
        captions = []
        with cs.open(text_path) as f:
            for line in f.readlines():
                try:
                    parts = line.strip().split('#')
                    t_tokens = parts[1].split(' ')
                    cap = ' '.join(tok.split('/')[0] for tok in t_tokens)
                    f_tag  = float(parts[2]) if len(parts) > 2 else 0.0
                    to_tag = float(parts[3]) if len(parts) > 3 else 0.0
                    if np.isnan(f_tag):  f_tag  = 0.0
                    if np.isnan(to_tag): to_tag = 0.0
                    if f_tag == 0.0 and to_tag == 0.0:
                        captions.append(cap)
                except Exception:
                    pass

        return mid, wham_wham_norm, wham_gt_norm, captions


# ---------------------------------------------------------------------------
# Token formatting
# ---------------------------------------------------------------------------

def tokens_to_input(token_list, mot_start_idx, mot_end_idx, mot_pad_idx, max_motion_token=55):
    toks = [mot_start_idx] + list(token_list) + [mot_end_idx]
    pad_len = max_motion_token - len(toks)
    if pad_len > 0:
        toks = toks + [mot_pad_idx] * pad_len
    else:
        toks = toks[:max_motion_token]
    return torch.LongTensor(toks).unsqueeze(0)


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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Dir with new_joint_vecs/<id>.npy (WHAM output, 263-dim HumanML3D format)")
    p.add_argument("--adapter_ckpt", type=str, required=True,
                   help="Trained 3D adapter checkpoint (.tar)")
    p.add_argument("--motiongpt_ckpt", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "MotionGPT-base" / "motiongpt_s3_h3d.tar"))
    p.add_argument("--m2t_name", type=str, default="M2T_MotionGPT_EL4_DL4_NH8_PS")
    p.add_argument("--m2t_epoch", type=str, default="finest")
    p.add_argument("--data_root", type=str,
                   default=str(ROOT_DIR / "dataset" / "HumanML3D"))
    p.add_argument("--meta_dir", type=str,
                   default=str(ROOT_DIR / "checkpoints" / "t2m" / "VQVAEV3_CB1024_CMT_H1024_NRES3" / "meta"),
                   help="Dir with mean.npy / std.npy (GT HumanML3D stats)")
    p.add_argument("--wham_meta_dir", type=str,
                   default="../2DMotionGPT/deps/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta",
                   help="Dir with mean_wham_3d.npy / std_wham_3d.npy (WHAM-specific stats)")
    p.add_argument("--split", type=str, default="test")
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

    # --- Normalization stats ---
    mean      = np.load(pjoin(args.meta_dir,      "mean.npy"))
    std       = np.load(pjoin(args.meta_dir,      "std.npy"))
    mean_wham = np.load(pjoin(args.wham_meta_dir, "mean_wham_3d.npy"))
    std_wham  = np.load(pjoin(args.wham_meta_dir, "std_wham_3d.npy"))

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
    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"[max_samples] Truncated to {len(dataset.samples)} samples")
    if len(dataset) == 0:
        print("No samples found. Exiting.")
        return

    # --- VQ-VAE (MotionGPT 3D, frozen) ---
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
    vqvae.eval()
    print("VQ-VAE (3D) loaded.")

    # --- WordVectorizer ---
    w_vectorizer = WordVectorizerV2(str(ROOT_DIR / "glove"), "our_vab")
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
    print(f"M2T loaded: {m2t_ckpt_path} (ep={ckpt_m2t.get('ep', '?')})")

    translator = Translator(
        m2t_transformer, beam_size=args.beam_size, max_seq_len=30,
        src_pad_idx=mot_pad_idx, trg_pad_idx=txt_pad_idx,
        trg_sos_idx=txt_start_idx, trg_eos_idx=txt_end_idx,
    ).to(device)

    # --- 3D Adapter ---
    adapter = build_adapter(
        args.adapter_type,
        dim=263,
        hidden=args.adapter_hidden,
        kernel_size=args.adapter_kernel_size,
    ).to(device)
    ckpt_adapter = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ckpt_adapter.get("model_state_dict", ckpt_adapter))
    adapter.eval()
    print(f"Adapter ({args.adapter_type}) loaded: {args.adapter_ckpt}")

    # --- Inference helper ---
    def run_inference(label, use_adapter):
        all_preds, all_refs, video_ids = [], [], []
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc=f"Inference [{label}]"):
                mid, wham_wham_norm, wham_gt_norm, captions = dataset[i]

                if use_adapter:
                    feat_np = wham_wham_norm  # WHAM-normalized -> adapter
                else:
                    feat_np = wham_gt_norm    # GT-normalized -> VQ-VAE directly

                feat = torch.from_numpy(feat_np).float().unsqueeze(0).to(device)  # (1, T, 263)

                if use_adapter:
                    feat = adapter(feat)

                token_indices, _ = vqvae.encode(feat)
                toks = token_indices[0].cpu().numpy().tolist()

                m_input = tokens_to_input(
                    toks, mot_start_idx, mot_end_idx, mot_pad_idx,
                    max_motion_token=args.max_motion_token,
                ).to(device)

                pred_ids = translator.translate_sentence(m_input)
                pred_ids = pred_ids[1:-1]
                text = " ".join(w_vectorizer.itos(i) for i in pred_ids)

                all_preds.append(text)
                all_refs.append(captions)
                video_ids.append(mid)

        return all_preds, all_refs, video_ids

    def print_results(label, all_preds, all_refs, video_ids):
        print(f"\n=== [{label}] Results ({len(all_preds)} samples) ===")

        if HAS_NLGEVAL:
            nlg_eval = NLGEval(metrics_to_omit=[
                "METEOR", "EmbeddingAverageCosineSimilarity",
                "SkipThoughtCS", "VectorExtremaCosineSimilarity", "GreedyMatchingScore"
            ])
            max_refs = max(len(r) for r in all_refs)
            ref_list = [[refs[ri] if ri < len(refs) else refs[-1] for refs in all_refs]
                        for ri in range(max_refs)]
            scores = nlg_eval.compute_metrics(ref_list, all_preds)
            for key in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr"]:
                print(f"  {key:>10}: {scores.get(key, 0.0):.4f}")
        elif HAS_NLGMETRICVERSE:
            _metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            scorer = NLGMetricverse(_metrics)
            scores = scorer(predictions=all_preds, references=all_refs)
            b1    = scores["bleu_1"]["score"]
            b4    = scores["bleu_4"]["score"]
            rouge = scores["rouge"]["rougeL"]
            cider = scores["cider"]["score"]
            print(f"  {'BLEU-1':>10}: {b1:.4f}  (sentence-level)")
            print(f"  {'BLEU-4':>10}: {b4:.4f}  (sentence-level)")
            print(f"  {'ROUGE-L':>10}: {rouge:.4f}")
            print(f"  {'CIDEr':>10}: {cider:.4f}")

        bert_f1 = None
        if HAS_BERTSCORE:
            _, _, F1 = score_bert(
                all_preds, all_refs, lang="en",
                rescale_with_baseline=True, idf=True,
                device=device, verbose=False,
            )
            bert_f1 = F1
            print(f"  {'BERTScore':>10}: {F1.mean().item():.4f}")

        print()
        for i, (vid, pred, refs) in enumerate(zip(video_ids, all_preds, all_refs)):
            bert_str = f"{bert_f1[i].item():.4f}" if bert_f1 is not None else "N/A"
            print(f"[{i+1:3d}] {vid}")
            print(f"       GT  : {refs[0] if refs else ''}")
            print(f"       Pred: {pred}")
            print(f"       BERT: {bert_str}")

    mode = args.mode

    if mode in ("adapter", "both"):
        preds, refs, vids = run_inference("WHAM + 3D Adapter", use_adapter=True)
        print_results("WHAM + 3D Adapter", preds, refs, vids)

    if mode in ("no_adapter", "both"):
        preds, refs, vids = run_inference("WHAM no adapter", use_adapter=False)
        print_results("WHAM no adapter", preds, refs, vids)


if __name__ == "__main__":
    main()
