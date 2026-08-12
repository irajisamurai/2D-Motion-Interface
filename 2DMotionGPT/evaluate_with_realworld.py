"""
Evaluate adapter + 2D VQ-VAE on real-world VitPose-estimated keypoints (M2T task).

Expected data layout:
    <estimated_motion_dir>/json/<motion_id>.json
Each JSON is a list of frame dicts: [{"frame_id": ..., "instances": [{"keypoints": ..., "keypoint_scores": ...}]}, ...]
The motion_id must match a HumanML3D motion ID (for GT text / 3D motion).

Usage:
    python evaluate_with_realworld.py \\
        --cfg configs/config_h3d_stage1.yaml --nodebug \\
        --estimated_motion_dir ./real_world_dataset/pred \\
        --vqvae_ckpt ./checkpoints/2d_vqvae_ver3/.../best_vqvae.tar \\
        --adapter_ckpt ./checkpoints/adapter/.../best_adapter.tar \\
        [--motiongpt_ckpt ./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar] \\
        [--split test]

idea400 evaluation (Motion-X++ subset):
    python evaluate_with_realworld.py \\
        --cfg configs/config_h3d_stage1.yaml --nodebug \\
        --idea400_dir /path/to/Motion-X++ \\
        --vqvae_ckpt ./checkpoints/2d_vqvae_ver3/.../best_vqvae.tar \\
        --adapter_ckpt ./checkpoints/adapter/.../best_adapter.tar \\
        [--mode both]

    Keypoints: <idea400_dir>/motion/keypoints/idea400/<name>.json
    GT:        <idea400_dir>/text/semantic_label/idea400/<name>.txt

Requires:
    pip install nlgmetricverse   # for BLEU/ROUGE/CIDEr
    bert-score is already installed in mgpt_2d
"""

import argparse
import codecs as cs
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from os.path import join as pjoin
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import parse_args
from src.data.humanml.dataset_t2m import Text2MotionDataset
from src.data.utils import humanml3d_collate_2d
from src.models.mgpt_vq import VQVae
from src.data.humanml.utils.word_vectorizer import WordVectorizer
from src.archs.mgpt_lm import MLM
from src.data.adapter_datasets import _VitPoseMixin

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

class RealWorldEvalDataset(_VitPoseMixin, Text2MotionDataset):
    """Evaluation dataset for real-world VitPose keypoints.

    Each JSON file covers all frames of one video.
    The filename (without .json) is matched to HumanML3D for GT text/3D motion.
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d,
        std_2d,
        mean_estimate,
        std_estimate,
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
        self.mean_2d = mean_2d
        self.std_2d = std_2d
        self.mean_estimate = mean_estimate
        self.std_estimate = std_estimate
        self.data_root = data_root

        json_paths = sorted(glob.glob(os.path.join(estimated_motion_dir, "json", "*.json")))
        print(f"[RealWorldEvalDataset] Found {len(json_paths)} JSON files")
        self.json_list_path = json_paths
        self.motion_list, self.conf_list = self._load_keypoints(json_paths)

    def _load_keypoints(self, json_paths):
        motion_list, conf_list = [], []
        for json_path in json_paths:
            data = json.load(open(json_path))
            motions, confs = [], []
            for frame_data in data:
                motions.append(frame_data["instances"][0]["keypoints"])
                confs.append(frame_data["instances"][0]["keypoint_scores"])
            motion_list.append(motions)
            conf_list.append(confs)
        return motion_list, conf_list

    def __len__(self):
        return len(self.motion_list)

    def __getitem__(self, item):
        estimated_motion_2d = np.array(self.motion_list[item])
        estimated_conf = np.array(self.conf_list[item])

        # COCO-17 → COCO-13 (drop eyes/ears: indices 1,2,3,4)
        estimated_motion_2d = estimated_motion_2d[:, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]
        estimated_conf = estimated_conf[:, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        m_length = estimated_motion_2d.shape[0]

        # HumanML3D GT text and 3D motion matched by JSON filename
        motion_id = os.path.basename(self.json_list_path[item]).replace(".json", "")
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

        start_2d = random.randint(0, estimated_motion_2d.shape[0] - m_length)
        estimated_motion_2d = estimated_motion_2d[start_2d:start_2d + m_length]
        estimated_conf = estimated_conf[start_2d:start_2d + m_length]

        start_3d = random.randint(0, motion_3d.shape[0] - m_length_3d)
        motion_3d = motion_3d[start_3d:start_3d + m_length_3d]

        # Features WITH confidence scores (adapter input)
        estimated_feat = self._preprocess_estimated(estimated_motion_2d, estimated_conf)
        estimated_feat = (estimated_feat - self.mean_estimate) / self.std_estimate

        # Features WITHOUT confidence scores (adapter-less input, same as synthetic 2D)
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(
            estimated_motion_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(estimated_motion_2d)
        motion_2d_no_conf = np.concatenate(
            [root_vel_2d, root_y_2d, joints_pos_2d, joints_rot_2d, joints_vel_2d], axis=-1)
        motion_2d_no_conf = (motion_2d_no_conf - self.mean_2d) / self.std_2d

        motion_3d = (motion_3d - self.mean) / self.std

        # estimated_feat (81-dim, with conf) at position 2 for humanml3d_collate_2d
        # motion_2d_no_conf (68-dim, no conf) as extra 10th element
        return caption, motion_3d, estimated_feat, m_length, None, None, None, None, all_captions, motion_2d_no_conf


# ---------------------------------------------------------------------------
# Idea400 dataset (Motion-X++ idea400 subset)
# ---------------------------------------------------------------------------

class Idea400EvalDataset(_VitPoseMixin):
    """Evaluation dataset for Motion-X++ idea400.

    Keypoints : <idea400_dir>/motion/keypoints/idea400/<name>.json
                body_kpts = [[x, y, score], ...]×17 per frame (COCO-17)
    GT caption: <idea400_dir>/text/semantic_label/idea400/<name>.txt (one line)
    """

    def __init__(self, idea400_dir, mean_2d, std_2d, mean_estimate, std_estimate,
                 max_motion_length=196, min_motion_length=40, unit_length=4):
        self.mean_2d = mean_2d
        self.std_2d = std_2d
        self.mean_estimate = mean_estimate
        self.std_estimate = std_estimate
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length

        kp_dir  = os.path.join(idea400_dir, "motion", "keypoints", "idea400")
        txt_dir = os.path.join(idea400_dir, "text", "semantic_label", "idea400")

        self.samples = []
        skipped = 0
        for jp in sorted(glob.glob(os.path.join(kp_dir, "*.json"))):
            name = os.path.basename(jp).replace(".json", "")
            txt_path = os.path.join(txt_dir, name + ".txt")
            if not os.path.exists(txt_path):
                skipped += 1
                continue
            caption = open(txt_path).read().strip()
            if not caption:
                skipped += 1
                continue
            self.samples.append((jp, caption, name))

        self.json_list_path = [s[0] for s in self.samples]
        print(f"[Idea400EvalDataset] {len(self.samples)} samples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jp, caption, _ = self.samples[idx]
        data = json.load(open(jp))

        motions, confs = [], []
        for ann in data.get("annotations", []):
            kpts = ann["body_kpts"]          # list of 17 [x, y, score]
            motions.append([[k[0], k[1]] for k in kpts])
            confs.append([k[2] for k in kpts])

        motion_2d = np.array(motions, dtype=np.float32)   # (T, 17, 2)
        conf      = np.array(confs,   dtype=np.float32)   # (T, 17)

        # COCO-17 → COCO-13 (drop eyes/ears: indices 1,2,3,4)
        coco13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        motion_2d = motion_2d[:, coco13, :]
        conf      = conf[:,      coco13]

        T = motion_2d.shape[0]
        m_length = min(T, self.max_motion_length)
        m_length = (m_length // self.unit_length) * self.unit_length
        m_length = max(m_length, self.unit_length)

        if T > m_length:
            start = random.randint(0, T - m_length)
            motion_2d = motion_2d[start:start + m_length]
            conf      = conf[start:start + m_length]

        # 81-dim feature with confidence (adapter input)
        feat_est = self._preprocess_estimated(motion_2d, conf)
        feat_est = (feat_est - self.mean_estimate) / (self.std_estimate + 1e-8)

        # 68-dim feature without confidence (no-adapter input)
        root_y, joints_pos, root_vel = self.decompose_2d_motion_coco13_midhip_root(motion_2d)
        joints_rot, joints_vel = self.compute_joint_features_2d_coco13(motion_2d)
        feat_nc = np.concatenate([root_vel, root_y, joints_pos, joints_rot, joints_vel], axis=-1)
        feat_nc = (feat_nc - self.mean_2d) / (self.std_2d + 1e-8)

        return feat_est.astype(np.float32), feat_nc.astype(np.float32), m_length, caption


def idea400_collate(batch):
    assert len(batch) == 1, "idea400_collate only supports batch_size=1"
    feat_est, feat_nc, m_length, caption = batch[0]

    # motion_2d: (1, T, 81) — adapter input
    motion_2d = torch.from_numpy(feat_est).unsqueeze(0)

    # motion_2d_no_conf: (1, T, 263) zero-padded — no-adapter input
    feat_nc_t = torch.from_numpy(feat_nc)
    pad = torch.zeros(feat_nc_t.shape[0], 263 - feat_nc_t.shape[1])
    motion_2d_no_conf = torch.cat([feat_nc_t, pad], dim=-1).unsqueeze(0)

    return {
        "motion_2d":         motion_2d,
        "motion_2d_no_conf": motion_2d_no_conf,
        "lengths":           torch.LongTensor([m_length]),
        "all_captions":      [[caption]],
    }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def rw_collate(batch):
    """Wraps humanml3d_collate_2d and handles the extra motion_2d_no_conf field."""
    assert len(batch) == 1, "rw_collate only supports batch_size=1"
    item = batch[0]
    base = humanml3d_collate_2d([item[:9]])
    m2d = torch.from_numpy(item[9]).float()  # (T, 68)
    pad = torch.zeros(m2d.shape[0], 263 - m2d.shape[1])
    base["motion_2d_no_conf"] = torch.cat([m2d, pad], dim=-1).unsqueeze(0)  # (1, T, 263)
    return base


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
        r = r.permute(0, 2, 1)        # (B, dim, T)
        r = self.conv1(r)
        r = self.act(r)
        r = self.drop(r)
        r = self.conv2(r)
        r = r.permute(0, 2, 1)        # (B, T, dim)
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
# CLI
# ---------------------------------------------------------------------------

def parse_script_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--estimated_motion_dir", type=str, default=None,
                   help="Root dir with json/<id>.json real-world keypoints "
                        "(required unless --idea400_dir is set)")
    p.add_argument("--vqvae_ckpt", type=str, required=True,
                   help="Trained 2D encoder checkpoint (.tar)")
    p.add_argument("--adapter_ckpt", type=str, required=True,
                   help="Trained adapter checkpoint (.tar)")
    p.add_argument("--motiongpt_ckpt", type=str,
                   default="./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar")
    p.add_argument("--split", type=str, default="test",
                   help="HumanML3D split used for the GT text lookup (default: test)")
    p.add_argument("--mode", type=str, default="both",
                   choices=["adapter", "no_adapter", "both"],
                   help="Evaluation mode: 'adapter' (with adapter), "
                        "'no_adapter' (direct 2D features), or 'both' (default)")
    p.add_argument("--idea400_dir", type=str, default=None,
                   help="Motion-X++ root dir for idea400 evaluation. "
                        "If set, overrides --estimated_motion_dir and uses idea400 dataset.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit the number of evaluation samples (for quick testing).")
    p.add_argument("--adapter_type", type=str, default="residual",
                   choices=["linear", "residual", "conv1d"],
    p.add_argument("--adapter_hidden", type=int, default=512,
    p.add_argument("--adapter_kernel_size", type=int, default=3,
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

    # Stats
    data_root = cfg.DATASET.HUMANML3D.ROOT
    dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m',
                          "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
    mean     = np.load(pjoin(dis_data_root, "mean.npy"))
    std      = np.load(pjoin(dis_data_root, "std.npy"))
    mean_2d  = np.load(pjoin(dis_data_root, "mean_2d_coco_normalized.npy"))
    std_2d   = np.load(pjoin(dis_data_root, "std_2d_coco_normalized.npy"))
    mean_est = np.load(pjoin(dis_data_root, "mean_2d_coco_estimated_concatenate.npy"))
    std_est  = np.load(pjoin(dis_data_root, "std_2d_coco_estimated_concatenate.npy"))
    w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

    # Dataset & dataloader
    num_workers = cfg.TRAIN.NUM_WORKERS
    if script_args.idea400_dir:
        dataset = Idea400EvalDataset(
            idea400_dir=script_args.idea400_dir,
            mean_2d=mean_2d,
            std_2d=std_2d,
            mean_estimate=mean_est,
            std_estimate=std_est,
            max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
            min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
            unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
        )
        if script_args.max_samples:
            dataset.samples = dataset.samples[:script_args.max_samples]
            dataset.json_list_path = dataset.json_list_path[:script_args.max_samples]
            print(f"[max_samples] Truncated to {len(dataset.samples)} samples")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=idea400_collate,
            persistent_workers=(num_workers > 0),
        )
    else:
        dataset = RealWorldEvalDataset(
            data_root=data_root,
            split=script_args.split,
            mean=mean,
            std=std,
            mean_2d=mean_2d,
            std_2d=std_2d,
            mean_estimate=mean_est,
            std_estimate=std_est,
            w_vectorizer=w_vectorizer,
            estimated_motion_dir=script_args.estimated_motion_dir,
            max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
            min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
            unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=rw_collate,
            persistent_workers=(num_workers > 0),
        )

    # Models
    def _build_vqvae():
        return VQVae(
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

    ckpt_base = torch.load(script_args.motiongpt_ckpt, map_location="cpu", weights_only=True)

    vqvae = _build_vqvae()
    vqvae.load_state_dict(
        {k.replace("vae.", ""): v for k, v in ckpt_base["state_dict"].items() if "vae" in k}
    )
    ckpt_2d = torch.load(script_args.vqvae_ckpt, map_location="cpu", weights_only=False)
    raw_2d = ckpt_2d.get("model_state_dict", ckpt_2d)
    vqvae.load_state_dict({k: v for k, v in raw_2d.items() if "encoder" in k}, strict=False)

    lm = MLM(
        model_path=cfg.lm.default.params.model_path,
        model_type=cfg.lm.default.params.model_type,
        stage=cfg.lm.default.params.stage,
        motion_codebook_size=cfg.lm.default.params.motion_codebook_size,
    ).to(device)
    lm.load_state_dict(
        {k.replace("lm.", ""): v for k, v in ckpt_base["state_dict"].items() if "lm" in k}
    )

    adapter = build_adapter(
        script_args.adapter_type,
        dim=81,
        hidden=script_args.adapter_hidden,
        kernel_size=script_args.adapter_kernel_size,
    ).to(device)
    ckpt_adapter = torch.load(script_args.adapter_ckpt, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ckpt_adapter.get("model_state_dict", ckpt_adapter))

    for model in [vqvae, lm, adapter]:
        model.eval()

    def run_inference(label, feats_fn):
        """feats_fn(batch) → (T, 263) tensor already on device."""
        gt_list, pred_list = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Inference [{label}]"):
                feats = feats_fn(batch)
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
            video_name = os.path.basename(dataset.json_list_path[i]).replace(".json", "")
            bert_str = f"{F1[i].item():.4f}" if F1 is not None else "N/A"
            gt_str = g[0] if isinstance(g, list) and len(g) == 1 else g
            print(f"[{i+1:3d}] {video_name}")
            print(f"       GT  : {gt_str}")
            print(f"       Pred: {p}")
            print(f"       BERT: {bert_str}")

    mode = script_args.mode

    if mode in ("adapter", "both"):
        def feats_with_adapter(batch):
            feats = adapter(batch["motion_2d"].to(device))   # (B, T, 81)
            feats = torch.cat([
                feats,
                torch.zeros(feats.shape[0], feats.shape[1],
                            263 - feats.shape[2], device=device),
            ], dim=-1)                                        # (B, T, 263)
            return feats

        gt_list, pred_list = run_inference("with adapter", feats_with_adapter)
        print_results("with adapter", gt_list, pred_list)

    if mode in ("no_adapter", "both"):
        def feats_no_adapter(batch):
            return batch["motion_2d_no_conf"].to(device)

        gt_list, pred_list = run_inference("no adapter", feats_no_adapter)
        print_results("no adapter", gt_list, pred_list)


if __name__ == "__main__":
    main()
