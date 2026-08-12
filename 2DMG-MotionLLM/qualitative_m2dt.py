"""
Qualitative evaluation of m2dt on HumanML3D (FineMotion) test set.

Runs 3D and (optionally) 2D conditions, computes per-sample BERTScore / ROUGE-L,
sorts results by a chosen metric, and saves CSV + formatted text.

Usage:
    CUDA_VISIBLE_DEVICES=7 python qualitative_m2dt.py \\
        --model_name ./m2dt-ft-from-GSPretrained-base \\
        --vqvae_2d_ckpt ./checkpoints/2d_vq_train/t2m/best_2dvq_epoch1881_ratio0.5557.pt \\
        --n_samples 50 --gpu_id 0

Sort options (--sort_by):
    bert_3d   : BERTScore(3D) descending  [default]
    bert_2d   : BERTScore(2D) descending
    bert_diff : BERTScore(3D) - BERTScore(2D) descending  (where 3D is better)
    rouge_3d  : ROUGE-L(3D) descending
    rouge_2d  : ROUGE-L(2D) descending
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as rouge_scorer_module
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import models.vqvae as vqvae_module
from dataloader.eval_loader import Motion2MotionScriptDataset, _create_2d_joints_from_features


# ── text helpers ──────────────────────────────────────────────────────────────

def _snippets(raw: str):
    if "### Motion Script ###" in raw:
        raw = raw.split("### Motion Script ###")[1]
    return [s.strip() for s in raw.split("<SEP>") if s.strip()]


def _format_snippets(snippets, label, bert=None, rouge=None, indent=2):
    pad = " " * indent
    metrics = ""
    if bert is not None:
        metrics += f"  BERTScore={bert:.3f}"
    if rouge is not None:
        metrics += f"  ROUGE-L={rouge:.3f}"
    lines = [f"{pad}{label} ({len(snippets)} snippets){metrics}:"]
    for i, s in enumerate(snippets):
        lines.append(f"{pad}  [{i*0.5:.1f}s] {s}")
    return "\n".join(lines)


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer(motion_tensor, vae, tokenizer, t5_model, device, prompt, max_new_tokens):
    motion_tensor = motion_tensor.to(device)
    tokens = vae.encode(motion_tensor).cpu().numpy()[0].reshape(-1).tolist()
    motion_str = "<Motion Tokens>" + "".join(f"<{t}>" for t in tokens) + "</Motion Tokens>"
    input_ids = tokenizer(prompt + motion_str, return_tensors="pt").input_ids.to(device)
    out = t5_model.generate(input_ids, max_length=max_new_tokens, num_beams=1, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Motion Script ###" in text:
        text = text.split("### Motion Script ###")[1].strip()
    return text


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_bertscore(preds, refs, device):
    """Return per-sample BERTScore F1 list (float)."""
    _, _, F1 = bert_score_fn(
        preds, refs,
        lang="en",
        rescale_with_baseline=True,
        idf=True,
        device=str(device),
        verbose=False,
    )
    return F1.tolist()


def _compute_rougeL(preds, refs):
    """Return per-sample ROUGE-L F1 list (float)."""
    scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


# ── dataset iteration ─────────────────────────────────────────────────────────

def iter_dataset(ds, n_samples, seed):
    total = min(len(ds), n_samples) if n_samples else len(ds)
    for i in range(total):
        random.seed(seed + i)
        np.random.seed(seed + i)

        idx = ds.pointer + i
        name = ds.name_list[idx]
        data = ds.data_dict[name]
        motion, m_length = data["motion"], data["length"]
        text_entry = data["text"][0]  # deterministic: first entry

        summary = text_entry.get("summary", "")
        bpmsd = text_entry["detail"][:]

        m_length = (m_length // 20) * 20
        if m_length == 0:
            continue
        motion_crop = motion[:m_length]
        bpmsd = bpmsd[: m_length // 10]
        bpmsd = ["<Motionless>" if b == "" else b for b in bpmsd]
        gt_script = " <SEP> ".join(bpmsd)

        motion_3d = (motion_crop - ds.mean) / ds.std
        m3d = torch.from_numpy(motion_3d.astype(np.float32)).unsqueeze(0)

        motion_2d = _create_2d_joints_from_features(motion_crop)
        motion_2d_norm = (motion_2d - ds.mean_2d) / ds.std_2d
        d2 = motion_2d_norm.shape[1]
        motion_2d_pad = np.concatenate(
            [motion_2d_norm, np.zeros((motion_2d_norm.shape[0], 263 - d2), dtype=np.float32)], axis=-1
        )
        m2d = torch.from_numpy(motion_2d_pad.astype(np.float32)).unsqueeze(0)

        yield name, summary, gt_script, m3d, m2d, motion_crop, motion_2d_norm


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name",     default="./m2dt-ft-from-GSPretrained-base")
    ap.add_argument("--vqvae_2d_ckpt",  default=None)
    ap.add_argument("--vqvae_3d_pth",   default="./checkpoints/pretrained_vqvae/t2m.pth")
    ap.add_argument("--dataname",       default="t2m")
    ap.add_argument("--split",          default="test")
    ap.add_argument("--n_samples",      type=int, default=None)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--gpu_id",         type=int, default=0)
    ap.add_argument("--out_dir",        default="results")
    ap.add_argument("--prompt",         default="Generate the motion script: ")
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--sort_by",        default="bert_3d",
                    choices=["bert_3d", "bert_2d", "bert_diff", "rouge_3d", "rouge_2d"],
                    help="Sort key for output (descending)")
    ap.add_argument("--vis_top_n",      type=int, default=None,
                    help="Generate MP4 videos for the top-N ranked samples (requires ffmpeg)")
    ap.add_argument("--vis_utils_dir",  default=None,
                    help="Path to directory containing vis.py (required with --vis_top_n)")
    args = ap.parse_args()

    # Validate: 2D sort metrics require a 2D checkpoint
    if args.sort_by in ("bert_2d", "rouge_2d", "bert_diff") and not args.vqvae_2d_ckpt:
        ap.error(f"--sort_by {args.sort_by} requires --vqvae_2d_ckpt")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── dataset ──
    print("Loading dataset...")
    ds = Motion2MotionScriptDataset(args.dataname, args.split)
    n_run = min(len(ds), args.n_samples) if args.n_samples else len(ds)
    print(f"Dataset: {len(ds)} samples  →  processing {n_run}")

    # ── models ──
    print("Loading 3D VQ-VAE...")
    vae_args = argparse.Namespace(dataname=args.dataname, quantizer="ema_reset", mu=0.99)
    ckpt_3d = torch.load(args.vqvae_3d_pth, map_location="cpu", weights_only=False)

    def _build_vae():
        v = vqvae_module.HumanVQVAE(
            vae_args, nb_code=512, code_dim=512, output_emb_width=512,
            down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
        ).to(device).eval()
        v.load_state_dict(ckpt_3d["net"], strict=True)
        return v

    vae_3d = _build_vae()

    vae_2d = None
    if args.vqvae_2d_ckpt:
        print(f"Loading 2D encoder: {args.vqvae_2d_ckpt}")
        vae_2d = _build_vae()
        ckpt_2d = torch.load(args.vqvae_2d_ckpt, map_location="cpu", weights_only=False)
        vae_2d.load_state_dict(
            {k: v for k, v in ckpt_2d["net"].items() if k.startswith("vqvae.encoder")},
            strict=False,
        )

    print(f"Loading T5: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    t5_model  = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device).eval()

    # ── inference ──
    results = []
    for name, summary, gt_script, m3d, m2d, motion_crop_np, motion_2d_68_np in tqdm(
        iter_dataset(ds, args.n_samples, args.seed), total=n_run, desc="Inference"
    ):
        pred_3d = _infer(m3d, vae_3d, tokenizer, t5_model, device,
                         args.prompt, args.max_new_tokens)
        pred_2d = None
        if vae_2d:
            pred_2d = _infer(m2d, vae_2d, tokenizer, t5_model, device,
                             args.prompt, args.max_new_tokens)
        results.append({
            "id": name, "gt_summary": summary,
            "gt_script": gt_script, "pred_3d": pred_3d, "pred_2d": pred_2d,
            "motion_crop": motion_crop_np,
            "motion_2d_68": motion_2d_68_np,
        })

    # ── metrics ──
    print("\nComputing metrics...")
    gt_list   = [r["gt_script"]  for r in results]
    pred3_list = [r["pred_3d"]   for r in results]

    bert_3d = _compute_bertscore(pred3_list, gt_list, device)
    rouge_3d = _compute_rougeL(pred3_list, gt_list)

    bert_2d = rouge_2d = [None] * len(results)
    if vae_2d:
        pred2_list = [r["pred_2d"] for r in results]
        bert_2d  = _compute_bertscore(pred2_list, gt_list, device)
        rouge_2d = _compute_rougeL(pred2_list, gt_list)

    for i, r in enumerate(results):
        r["bert_3d"]  = bert_3d[i]
        r["rouge_3d"] = rouge_3d[i]
        r["bert_2d"]  = bert_2d[i]
        r["rouge_2d"] = rouge_2d[i]
        r["bert_diff"] = (bert_3d[i] - bert_2d[i]) if vae_2d else None

    # ── sort ──
    def _sort_key(r):
        v = r[args.sort_by]
        return v if v is not None else -999.0

    results.sort(key=_sort_key, reverse=True)
    print(f"Sorted by: {args.sort_by} (descending)")

    # ── aggregate stats ──
    def _mean(lst): return np.mean([x for x in lst if x is not None])
    b2_mean  = f"{_mean(bert_2d):8.4f}"  if vae_2d else "     N/A"
    ro2_mean = f"{_mean(rouge_2d):8.4f}" if vae_2d else "     N/A"
    print(f"\n{'Metric':<16} {'3D':>8} {'2D':>8}")
    print("-" * 35)
    print(f"{'BERTScore':<16} {_mean(bert_3d):8.4f} {b2_mean}")
    print(f"{'ROUGE-L':<16} {_mean(rouge_3d):8.4f} {ro2_mean}")

    # ── write output ──
    suffix = f"_{args.split}"
    if args.n_samples:
        suffix += f"_n{args.n_samples}"
    suffix += f"_sort{args.sort_by}"
    csv_path = Path(args.out_dir) / f"qualitative_m2dt{suffix}.csv"
    txt_path = Path(args.out_dir) / f"qualitative_m2dt{suffix}.txt"

    fieldnames = ["rank", "id", "gt_summary",
                  "bert_3d", "rouge_3d", "pred_3d", "gt_n_snippets", "pred_3d_n_snippets"]
    if vae_2d:
        fieldnames += ["bert_2d", "rouge_2d", "bert_diff", "pred_2d", "pred_2d_n_snippets"]
    fieldnames += ["gt_script"]

    SEP = "=" * 80

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_f, \
         open(txt_path, "w", encoding="utf-8") as txt_f:

        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, r in enumerate(results, 1):
            gt_snips      = _snippets(r["gt_script"])
            pred_3d_snips = _snippets(r["pred_3d"])
            pred_2d_snips = _snippets(r["pred_2d"]) if r["pred_2d"] else []

            row = {
                "rank":              rank,
                "id":                r["id"],
                "gt_summary":        r["gt_summary"],
                "bert_3d":           f"{r['bert_3d']:.4f}",
                "rouge_3d":          f"{r['rouge_3d']:.4f}",
                "pred_3d":           r["pred_3d"],
                "gt_n_snippets":     len(gt_snips),
                "pred_3d_n_snippets": len(pred_3d_snips),
                "gt_script":         r["gt_script"],
            }
            if vae_2d:
                row["bert_2d"]          = f"{r['bert_2d']:.4f}" if r["bert_2d"] is not None else ""
                row["rouge_2d"]         = f"{r['rouge_2d']:.4f}" if r["rouge_2d"] is not None else ""
                row["bert_diff"]        = f"{r['bert_diff']:+.4f}" if r["bert_diff"] is not None else ""
                row["pred_2d"]          = r["pred_2d"] or ""
                row["pred_2d_n_snippets"] = len(pred_2d_snips)
            writer.writerow(row)

            # text
            b3 = r["bert_3d"]; ro3 = r["rouge_3d"]
            b2 = r["bert_2d"]; ro2 = r["rouge_2d"]
            diff_str = f"  diff={r['bert_diff']:+.3f}" if r["bert_diff"] is not None else ""
            b2_str  = f"{b2:.3f}"  if b2  is not None else "N/A"
            ro2_str = f"{ro2:.3f}" if ro2 is not None else "N/A"
            bert_line  = f"  BERTScore: 3D={b3:.3f}  2D={b2_str}{diff_str}" if vae_2d else f"  BERTScore: 3D={b3:.3f}"
            rouge_line = f"  ROUGE-L  : 3D={ro3:.3f}  2D={ro2_str}"         if vae_2d else f"  ROUGE-L  : 3D={ro3:.3f}"
            lines = [
                SEP,
                f"Rank #{rank:4d}  [{r['id']}]  GT: {r['gt_summary']}",
                bert_line,
                rouge_line,
                "-" * 80,
            ]
            lines.append(_format_snippets(gt_snips, "GT Script"))
            lines.append("")
            lines.append(_format_snippets(pred_3d_snips, "Pred (3D)", bert=b3, rouge=ro3))
            if r["pred_2d"]:
                lines.append("")
                lines.append(_format_snippets(pred_2d_snips, "Pred (2D)", bert=b2, rouge=ro2))
            lines.append(SEP + "\n")
            txt_f.write("\n".join(lines) + "\n")

    print(f"\nDone.  CSV : {csv_path}")
    print(f"       Text: {txt_path}")

    # ── visualization ──
    if args.vis_top_n:
        if args.vis_utils_dir is None:
            ap.error("--vis_utils_dir is required when --vis_top_n is specified")
        sys.path.insert(0, args.vis_utils_dir)
        from vis import save_triplet_mp4_with_gt_and_pred_caption

        vis_dir = Path(args.out_dir) / f"vis_m2dt{suffix}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        n_vis = min(args.vis_top_n, len(results))
        print(f"\nGenerating {n_vis} visualization videos → {vis_dir}")

        for rank_i, r in enumerate(results[:n_vis], 1):
            gt_snips = _snippets(r["gt_script"])
            T_frames = r["motion_crop"].shape[0]
            gt_pf = [gt_snips[min(f // 10, len(gt_snips) - 1)] for f in range(T_frames)]

            # Align visualization prediction with sort criterion
            if args.sort_by in ("bert_2d", "rouge_2d") and vae_2d and r.get("pred_2d"):
                vis_conds = [("pred_2d", r["pred_2d"])]
            elif args.sort_by == "bert_diff" and vae_2d and r.get("pred_2d"):
                vis_conds = [("pred_3d", r["pred_3d"]), ("pred_2d", r["pred_2d"])]
            else:
                vis_conds = [("pred_3d", r["pred_3d"])]

            for pred_cond, pred_text in vis_conds:
                pred_snips = _snippets(pred_text)
                pred_pf = [pred_snips[min(f // 10, len(pred_snips) - 1)] for f in range(T_frames)]
                vis_path = vis_dir / f"{rank_i:03d}_{r['id']}_{pred_cond}.mp4"
                print(f"  [{rank_i:3d}/{n_vis}] {r['id']} ({pred_cond}) → {vis_path.name}")
                save_triplet_mp4_with_gt_and_pred_caption(
                    gt_motion_3d=r["motion_crop"],
                    in_motion_2d=r["motion_2d_68"],
                    gt_caption_text=gt_pf,
                    pred_caption_text=pred_pf,
                    save_path=str(vis_path),
                    mean_2d=ds.mean_2d,
                    std_2d=ds.std_2d,
                    in_2d_is_normalized=True,
                    fps=20,
                )
        print(f"Videos saved to: {vis_dir}")


if __name__ == "__main__":
    main()
