"""
Adapter training for real-world VitPose 2D keypoints → MotionGPT latent space.

The adapter is a lightweight residual MLP (dim=263) placed before the frozen
2D VQ-VAE encoder. It bridges the domain gap between:
  - synthetic 2D projections (used to train the 2D encoder)
  - real VitPose-estimated 2D keypoints (noisy)

Architecture:
  VitPose 2D keypoints → [zero-pad to 263] → Adapter → 2D VQ-VAE encoder (frozen)
                                                               ↓
                                              L1 loss vs. MotionGPT 3D encoder output

Execution:
  python -m train_adapter \\
      --cfg configs/config_h3d_stage1.yaml --nodebug \\
      --vqvae_ckpt ./checkpoints/2d_vqvae_ver3/.../best_vqvae_epoch50_valacc0.3495.tar \\
      --estimated_motion_dir /path/to/vitpose_json_root
"""

import argparse
import os
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
from os.path import join as pjoin
from tqdm import tqdm

try:
    from bert_score import score as score_bert
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("[Warning] bert-score not installed — BERTScore checkpoint criterion unavailable.")

from src.config import parse_args
from src.data.utils import humanml3d_collate_2d
from src.data.humanml.utils.word_vectorizer import WordVectorizer
from src.models.mgpt_vq import VQVae
from src.archs.mgpt_lm import MLM
from src.metrics.m2t import M2TMetrics
from src.losses.utils import compose_loss
from src.data.adapter_datasets import (
    MotionDataset_vitpose_2DCOCO_normalized,
    Text2MotionDatasetEval_vitpose_2DCOCO_normalized,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Adapter model
# ---------------------------------------------------------------------------

class AdapterResidual(nn.Module):
    """Lightweight residual adapter: identity at init, learns correction."""

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


class AdapterLinear(nn.Module):

    def __init__(self, dim=81):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.linear(self.norm(x))


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
# Script-specific argument parser
# ---------------------------------------------------------------------------

def parse_script_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--vqvae_ckpt",
        type=str,
        default="./checkpoints/2d_vqvae_ver3/MotionGPT_2DEncoder/encoder_only-l1-bs64-lr0.0001/best_vqvae_epoch2960_valacc0.4471.tar",
        help="Path to trained 2D encoder checkpoint (.tar) from train_2d_vqvae_ver4_coco_normalized.py or train_vqvae_2d.py",
    )
    p.add_argument(
        "--estimated_motion_dir",
        type=str,
        default="./datasets/humanml3d_for_render",
        help="Root directory containing VitPose JSON keypoints (expects <dir>/json/<vid>/<frame>.json)",
    )
    p.add_argument(
        "--motiongpt_ckpt",
        type=str,
        default="./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar",
        help="Path to MotionGPT base checkpoint (.tar)",
    )
    p.add_argument(
        "--adapter_type",
        type=str,
        default="residual",
        choices=["linear", "residual", "conv1d"],
    )
    p.add_argument(
        "--adapter_hidden",
        type=int,
        default=512,
    )
    p.add_argument(
        "--adapter_kernel_size",
        type=int,
        default=3,
    )
    script_args, _ = p.parse_known_args()
    return script_args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_args = parse_script_args()
    cfg = parse_args(phase="train")
    pl.seed_everything(cfg.SEED_VALUE)

    device = torch.device(f"cuda:{cfg.DEVICE[0]}" if torch.cuda.is_available() else "cpu")

    run_name = f"adapter-{script_args.adapter_type}-bs{cfg.TRAIN.BATCH_SIZE}-lr1e-3"

    with wandb.init(
        project=cfg.LOGGER.WANDB.params.project,
        config=OmegaConf.to_container(cfg, resolve=False),
        name=run_name,
        group=cfg.LOGGER.WANDB.params.group,
        save_code=True,
    ):
        artifact = wandb.Artifact(name="train_adapter", type="code")
        artifact.add_file(local_path=str(Path(__file__)), name="train scripts")
        wandb.log_artifact(artifact)

        ckpt_save_path = f"./checkpoints/adapter/{wandb.run.group}/{wandb.run.name}"
        os.makedirs(ckpt_save_path, exist_ok=True)

        # ----------------------------------------------------------------
        # Stats / data paths
        # ----------------------------------------------------------------
        data_root = cfg.DATASET.HUMANML3D.ROOT
        dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        mean     = np.load(pjoin(dis_data_root, "mean.npy"))
        std      = np.load(pjoin(dis_data_root, "std.npy"))
        mean_2d  = np.load(pjoin(dis_data_root, "mean_2d_coco_normalized.npy"))
        std_2d   = np.load(pjoin(dis_data_root, "std_2d_coco_normalized.npy"))
        mean_est = np.load(pjoin(dis_data_root, "mean_2d_coco_estimated_concatenate.npy"))
        std_est  = np.load(pjoin(dis_data_root, "std_2d_coco_estimated_concatenate.npy"))

        # ----------------------------------------------------------------
        # Datasets & dataloaders
        # ----------------------------------------------------------------
        w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        _train_dataset = MotionDataset_vitpose_2DCOCO_normalized(
            data_root=data_root,
            split='train',
            mean=mean,
            std=std,
            mean_2d_coco=mean_2d,
            std_2d_coco=std_2d,
            mean_estimate=mean_est,
            std_estimate=std_est,
            max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
            min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
            win_size=64,
            estimated_motion_dir=script_args.estimated_motion_dir,
            unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
        )
        _val_dataset = Text2MotionDatasetEval_vitpose_2DCOCO_normalized(
            data_root=data_root,
            split='val',
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

        _train_dataloader = DataLoader(
            dataset=_train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            collate_fn=humanml3d_collate_2d,
            persistent_workers=True,
        )
        _val_dataloader = DataLoader(
            dataset=_val_dataset,
            batch_size=cfg.EVAL.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            collate_fn=humanml3d_collate_2d,
            persistent_workers=True,
        )

        # ----------------------------------------------------------------
        # Models
        # ----------------------------------------------------------------
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

        # ref_vqvae: MotionGPT 3D VQ-VAE (frozen, provides reference latents)
        ref_vqvae = _build_vqvae()
        ckpt_base = torch.load(script_args.motiongpt_ckpt, map_location="cpu", weights_only=True)
        ref_vqvae.load_state_dict({
            k.replace("vae.", ""): v
            for k, v in ckpt_base["state_dict"].items() if "vae" in k
        })

        # vqvae: MotionGPT base + 2D encoder overwrite (frozen after loading)
        vqvae = _build_vqvae()
        vqvae.load_state_dict({
            k.replace("vae.", ""): v
            for k, v in ckpt_base["state_dict"].items() if "vae" in k
        })
        ckpt_2d = torch.load(script_args.vqvae_ckpt, map_location="cpu", weights_only=False)
        raw_2d = ckpt_2d.get("model_state_dict", ckpt_2d)
        vqvae.load_state_dict(
            {k: v for k, v in raw_2d.items() if "encoder" in k},
            strict=False,
        )

        # lm: MotionGPT language model (frozen, used only for val M2T generation)
        lm = MLM(
            model_path=cfg.lm.default.params.model_path,
            model_type=cfg.lm.default.params.model_type,
            stage=cfg.lm.default.params.stage,
            motion_codebook_size=cfg.lm.default.params.motion_codebook_size,
        ).to(device)
        lm.load_state_dict({
            k.replace("lm.", ""): v
            for k, v in ckpt_base["state_dict"].items() if "lm" in k
        })

        # freeze everything except adapter
        for model in [ref_vqvae, vqvae, lm]:
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()

        # adapter (trainable)
        adapter = build_adapter(
            script_args.adapter_type,
            dim=81,
            hidden=script_args.adapter_hidden,
            kernel_size=script_args.adapter_kernel_size,
        ).to(device)
        wandb.watch(adapter)

        # ----------------------------------------------------------------
        # Optimizer & loss
        # ----------------------------------------------------------------
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.L1Loss()

        # ----------------------------------------------------------------
        # M2T metrics for validation
        # ----------------------------------------------------------------
        cfg.model.params.task = 'm2t'
        m2t_metrics = M2TMetrics(
            cfg=cfg,
            w_vectorizer=w_vectorizer,
            diversity_times=30,
            dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
        ).to(device)

        # ----------------------------------------------------------------
        # CSV logging
        # ----------------------------------------------------------------
        log_path = pjoin(ckpt_save_path, "training_log.csv")
        csv_header = ["epoch", "train_encoder_loss", "val_encoder_loss",
                      "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr",
                      "Matching_Score", "R_precision_top1", "R_precision_top2", "R_precision_top3",
                      "BERTScore_F1"]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

        best_val_loss = float("inf")
        best_val_loss_ckpt_path = None

        # ----------------------------------------------------------------
        # Training loop
        # ----------------------------------------------------------------
        for epoch in range(1000):
            adapter.train()
            train_total_encoder_loss = 0.0

            for batch in tqdm(_train_dataloader, desc=f"Epoch {epoch}", leave=False):
                optimizer.zero_grad()
                feats_3d = batch["motion"].to(device)
                feats_2d = adapter(batch["motion_2d"].to(device))   # (B, T, 68)
                feats_2d = torch.cat([
                    feats_2d,
                    torch.zeros(feats_2d.shape[0], feats_2d.shape[1],
                                263 - feats_2d.shape[2], device=device),
                ], dim=-1)                                           # (B, T, 263)

                with torch.no_grad():
                    ref_encoded = ref_vqvae.encoder(ref_vqvae.preprocess(feats_3d))

                enc_2d = vqvae.encoder(vqvae.preprocess(feats_2d))
                loss_encoder = loss_fn(enc_2d, ref_encoded)

                loss_terms = {"encoder": loss_encoder, "recon": 0}
                loss, _ = compose_loss(loss_terms, cfg.LOSS.terms)
                loss.backward()
                optimizer.step()
                train_total_encoder_loss += loss_encoder.item()

            avg_train_enc_loss = train_total_encoder_loss / len(_train_dataloader)
            print(f"Epoch {epoch}  Train Encoder Loss: {avg_train_enc_loss:.4f}")

            metrics_log = {"train/encoder_loss": avg_train_enc_loss}
            csv_row = [epoch, avg_train_enc_loss, ""] + [""] * 11

            if epoch % cfg.LOGGER.VAL_EVERY_STEPS == 0:
                adapter.eval()
                print("Starting Validation...")
                all_pred_texts, all_gt_texts = [], []
                val_total_encoder_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(_val_dataloader, desc="Val", leave=False):
                        feats_3d = batch["motion"].to(device)
                        feats_ref = adapter(batch["motion_2d"].to(device))  # (B, T, 68)
                        feats_ref = torch.cat([
                            feats_ref,
                            torch.zeros(feats_ref.shape[0], feats_ref.shape[1],
                                        263 - feats_ref.shape[2], device=device),
                        ], dim=-1)                                           # (B, T, 263)
                        lengths = batch["length"]

                        ref_encoded = ref_vqvae.encoder(ref_vqvae.preprocess(feats_3d))
                        enc_2d = vqvae.encoder(vqvae.preprocess(feats_ref))
                        val_total_encoder_loss += loss_fn(enc_2d, ref_encoded).item()

                        motion_tokens, lengths_tokens = [], []
                        for i in range(len(feats_ref)):
                            motion_token, _ = vqvae.encode(feats_ref[i:i + 1])
                            motion_tokens.append(motion_token[0])
                            lengths_tokens.append(motion_token.shape[1])

                        outputs = lm.generate_conditional(
                            motion_tokens=motion_tokens,
                            lengths=lengths_tokens,
                            task="m2t",
                            stage='test',
                        )
                        m2t_metrics.update(
                            feats_ref=feats_ref,
                            pred_texts=outputs,
                            gt_texts=batch["all_captions"],
                            lengths=lengths,
                            word_embs=batch["word_embs"].to(device),
                            pos_ohot=batch["pos_ohot"].to(device),
                            text_lengths=batch["text_len"].to(device),
                        )
                        all_pred_texts.extend(outputs)
                        all_gt_texts.extend(batch["all_captions"])

                avg_val_enc_loss = val_total_encoder_loss / len(_val_dataloader)
                print(f"  val/encoder_loss: {avg_val_enc_loss:.4f}")
                metrics_log["val/encoder_loss"] = avg_val_enc_loss
                csv_row[2] = avg_val_enc_loss

                val_result = m2t_metrics.compute(sanity_flag=False)
                m2t_metrics.reset()

                result_dict = {k: v.item() for k, v in val_result.items()}
                for k, v in result_dict.items():
                    print(f"  {k}: {v:.4f}")
                    metrics_log[f"val/{k}"] = v

                # BERTScore
                bertscore_f1 = 0.0
                if HAS_BERTSCORE and all_pred_texts:
                    _, _, F1 = score_bert(
                        all_pred_texts, all_gt_texts, lang='en',
                        rescale_with_baseline=True, idf=True,
                        device=device, verbose=False,
                    )
                    bertscore_f1 = F1.mean().item()
                    print(f"  BERTScore_F1: {bertscore_f1:.4f}")
                    metrics_log["val/BERTScore_F1"] = bertscore_f1

                csv_row[3:] = [
                    result_dict.get("Bleu_1", ""), result_dict.get("Bleu_2", ""),
                    result_dict.get("Bleu_3", ""), result_dict.get("Bleu_4", ""),
                    result_dict.get("ROUGE_L", ""), result_dict.get("CIDEr", ""),
                    result_dict.get("Matching_Score", ""),
                    result_dict.get("R_precision_top1", ""),
                    result_dict.get("R_precision_top2", ""),
                    result_dict.get("R_precision_top3", ""),
                    bertscore_f1 if HAS_BERTSCORE else "",
                ]

                # best checkpoint by val encoder loss (lower is better)
                if avg_val_enc_loss < best_val_loss:
                    best_val_loss = avg_val_enc_loss
                    new_ckpt = pjoin(ckpt_save_path, f"best_adapter_epoch{epoch}_valloss{best_val_loss:.4f}.tar")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": adapter.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, new_ckpt)
                    if best_val_loss_ckpt_path is not None and os.path.exists(best_val_loss_ckpt_path):
                        os.remove(best_val_loss_ckpt_path)
                    best_val_loss_ckpt_path = new_ckpt
                    print(f"  -> Best val_loss checkpoint saved: {new_ckpt}")


            wandb.log(metrics_log, step=epoch)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(csv_row)


if __name__ == "__main__":
    main()
