"""
Evaluate original MotionGPT on real-world TRACE-estimated 3D poses (M2T task).

TRACE output must be preprocessed into HumanML3D format (263-dim feature vectors)
and stored as:
    <estimated_motion_dir>/new_joint_vecs/<motion_id>.npy

Each .npy has shape (T, 263) — same format as HumanML3D new_joint_vecs.
The motion_id must match a HumanML3D motion ID (for GT text lookup).

Pipeline:
    TRACE 3D pose → HumanML3D 263-dim features
        → MotionGPT 3D VQ-VAE (frozen) → tokens → MotionGPT LM → text

Usage:
    python evaluate_with_realworld_3d.py \\
        --cfg configs/config_h3d_stage1.yaml --nodebug \\
        --estimated_motion_dir ./real_world_dataset/trace_output \\
        [--motiongpt_ckpt ./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar] \\
        [--split test]
"""

import argparse
import codecs as cs
import glob
import os
import random

import numpy as np
import torch
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
    print("[Warning] nlgmetricverse not installed — BLEU/ROUGE/CIDEr skipped.")
    print("          Install with: pip install nlgmetricverse")

try:
    from bert_score import score as score_bert
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("[Warning] bert-score not installed — BERTScore skipped.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealWorld3DEvalDataset(Text2MotionDataset):
    """Evaluation dataset for TRACE-estimated 3D poses.

    Expects:
        <estimated_motion_dir>/new_joint_vecs/<motion_id>.npy
    Each .npy is shape (T, 263) — HumanML3D feature format.
    motion_id must match a HumanML3D ID (for GT text lookup).
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
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
        self.data_root = data_root

        npy_paths = sorted(glob.glob(
            os.path.join(estimated_motion_dir, "new_joint_vecs", "*.npy")
        ))
        if len(npy_paths) == 0:
            raise FileNotFoundError(
                f"No .npy files found under {estimated_motion_dir}/new_joint_vecs/\n"
                "TRACE output must be preprocessed into HumanML3D 263-dim format first."
            )
        print(f"[RealWorld3DEvalDataset] Found {len(npy_paths)} TRACE .npy files")
        self.npy_paths = npy_paths
        self.motion_list = [np.load(p, allow_pickle=True) for p in npy_paths]

    def __len__(self):
        return len(self.motion_list)

    def __getitem__(self, item):
        estimated_motion_3d = np.array(self.motion_list[item])  # (T, 263)
        m_length = estimated_motion_3d.shape[0]

        # GT text and 3D motion matched by filename (e.g. 000021.npy → 000021)
        motion_id = os.path.basename(self.npy_paths[item]).replace(".npy", "")
        text_path = pjoin(self.data_root, "texts", motion_id + ".txt")
        motion_3d = np.load(pjoin(self.data_root, "new_joint_vecs", motion_id + ".npy"))
        m_length_3d = motion_3d.shape[0]

        text_data = []
        with cs.open(text_path) as f:
            for line in f.readlines():
                try:
                    parts = line.strip().split('#')
                    caption = parts[0]
                    t_tokens = parts[1].split(' ')
                    f_tag = float(parts[2])
                    to_tag = float(parts[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag
                    if f_tag == 0.0 and to_tag == 0.0:
                        text_data.append({'caption': caption, 'tokens': t_tokens})
                except Exception:
                    pass

        all_captions = [' '.join(t.split('/')[0] for t in td['tokens']) for td in text_data]
        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        caption = random.choice(text_data)['caption']

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
            m_length_3d = (m_length_3d // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            m_length_3d = (m_length_3d // self.unit_length) * self.unit_length

        start = random.randint(0, estimated_motion_3d.shape[0] - m_length)
        estimated_motion_3d = estimated_motion_3d[start:start + m_length]

        start_3d = random.randint(0, motion_3d.shape[0] - m_length_3d)
        motion_3d = motion_3d[start_3d:start_3d + m_length_3d]

        # Normalize with 3D stats (same as HumanML3D training)
        estimated_motion_3d = (estimated_motion_3d - self.mean) / self.std
        motion_3d = (motion_3d - self.mean) / self.std

        # estimated_motion_3d goes into position 2 → batch["motion_2d"] via collate
        return caption, motion_3d, estimated_motion_3d, m_length, None, None, None, None, all_captions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_script_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--estimated_motion_dir", type=str, required=True,
                   help="Root dir with new_joint_vecs/<id>.npy (TRACE output, 263-dim)")
    p.add_argument("--motiongpt_ckpt", type=str,
                   default="./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar",
                   help="MotionGPT-base checkpoint (.tar)")
    p.add_argument("--split", type=str, default="test",
                   help="HumanML3D split for GT text lookup (default: test)")
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

    # Stats (3D only)
    data_root = cfg.DATASET.HUMANML3D.ROOT
    dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m',
                          "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
    mean = np.load(pjoin(dis_data_root, "mean.npy"))
    std  = np.load(pjoin(dis_data_root, "std.npy"))
    w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

    # Dataset & dataloader
    dataset = RealWorld3DEvalDataset(
        data_root=data_root,
        split=script_args.split,
        mean=mean,
        std=std,
        w_vectorizer=w_vectorizer,
        estimated_motion_dir=script_args.estimated_motion_dir,
        max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
        min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
        unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
    )
    num_workers = cfg.TRAIN.NUM_WORKERS
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=humanml3d_collate_2d,
        persistent_workers=(num_workers > 0),
    )

    # Models: original MotionGPT 3D VQ-VAE + LM (both frozen)
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

    ckpt = torch.load(script_args.motiongpt_ckpt, map_location="cpu", weights_only=True)
    vqvae.load_state_dict(
        {k.replace("vae.", ""): v for k, v in ckpt["state_dict"].items() if "vae" in k}
    )
    lm.load_state_dict(
        {k.replace("lm.", ""): v for k, v in ckpt["state_dict"].items() if "lm" in k}
    )

    vqvae.eval()
    lm.eval()

    # Inference: TRACE 3D features → 3D VQ-VAE → LM → text
    gt_list, pred_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference [TRACE 3D → MotionGPT]"):
            # TRACE 3D features are in batch["motion_2d"] (position 2 of the tuple)
            feats_3d = batch["motion_2d"].to(device)  # (1, T, 263)
            motion_token, _ = vqvae.encode(feats_3d)
            outputs = lm.generate_conditional(
                motion_tokens=[motion_token[0]],
                lengths=[motion_token.shape[1]],
                task="m2t",
                stage='test',
            )
            gt_list.extend(batch['all_captions'])
            pred_list.extend(outputs)

    print(f"\n=== Results ({len(pred_list)} samples) ===")

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
        bert_str = f"{F1[i].item():.4f}" if F1 is not None else "N/A"
        print(f"[{i+1:3d}] {motion_id}")
        print(f"       GT  : {g}")
        print(f"       Pred: {p}")
        print(f"       BERT: {bert_str}")


if __name__ == "__main__":
    main()
