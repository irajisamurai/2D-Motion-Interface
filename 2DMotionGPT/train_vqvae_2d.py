from src.config import parse_args
from os.path import join as pjoin
import os
import csv
from pathlib import Path
import torch
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm

from src.data.HumanML3D import MotionDatasetV_2DCOCO_normalized_non_windowsize
from src.data.humanml.dataset_t2m_eval import Text2MotionDatasetEval_2D_COCO_normalized
from src.data.utils import humanml3d_collate_2d
from src.data.humanml.utils.word_vectorizer import WordVectorizer
from torch.utils.data import DataLoader
from src.models.mgpt_vq import VQVae
from src.losses.mgpt import GPTLosses_2D
from src.data.humanml.scripts.motion_process import recover_from_ric


def main():
    cfg = parse_args(phase="train")
    pl.seed_everything(cfg.SEED_VALUE)

    device = torch.device(f"cuda:{cfg.DEVICE[0]}" if torch.cuda.is_available() else "cpu")

    run_name = (
        f"full_recon_2d"
        f"-{cfg.LOSS.ABLATION.RECONS_LOSS}"
        f"-bs{cfg.TRAIN.BATCH_SIZE}"
        f"-lr{cfg.TRAIN.OPTIM.params.lr}"
    )

    with wandb.init(
        project=cfg.LOGGER.WANDB.params.project,
        config=OmegaConf.to_container(cfg, resolve=False),
        name=run_name,
        group=cfg.LOGGER.WANDB.params.group,
        save_code=True,
    ):
        artifact = wandb.Artifact(name="train_vqvae_2d", type="code")
        artifact.add_file(local_path=str(Path(__file__)), name="train scripts")
        wandb.log_artifact(artifact)

        ckpt_save_path = f"./checkpoints/2d_vqvae_full/{wandb.run.group}/{wandb.run.name}"
        os.makedirs(ckpt_save_path, exist_ok=True)

        data_root = cfg.DATASET.HUMANML3D.ROOT
        dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        mean    = np.load(pjoin(dis_data_root, "mean.npy"))
        std     = np.load(pjoin(dis_data_root, "std.npy"))
        mean_2d = np.load(pjoin(dis_data_root, "mean_2d_coco_normalized.npy"))
        std_2d  = np.load(pjoin(dis_data_root, "std_2d_coco_normalized.npy"))
        dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "Comp_v6_KLD01", "meta")
        mean_eval = np.load(pjoin(dis_data_root_eval, "mean.npy"))
        std_eval  = np.load(pjoin(dis_data_root_eval, "std.npy"))

        _train_dataset = MotionDatasetV_2DCOCO_normalized_non_windowsize(
            data_root=data_root,
            split='train',
            mean=mean,
            std=std,
            mean_2d_coco=mean_2d,
            std_2d_coco=std_2d,
            win_size=64,
            max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
            min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
            unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
        )
        w_vectorizer = WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        _val_dataset = Text2MotionDatasetEval_2D_COCO_normalized(
            data_root=data_root,
            split='test',
            mean=mean,
            std=std,
            mean_2d=mean_2d,
            std_2d=std_2d,
            max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
            min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
            win_size=64,
            unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
            w_vectorizer=w_vectorizer,
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

        optimizer = torch.optim.AdamW(
            vqvae.parameters(),
            lr=cfg.TRAIN.OPTIM.params.lr,
            weight_decay=cfg.TRAIN.OPTIM.params.weight_decay,
            betas=cfg.TRAIN.OPTIM.params.betas,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.TRAIN.LR_SCHEDULER.params.T_max,
            eta_min=cfg.TRAIN.LR_SCHEDULER.params.eta_min,
        )

        # loss（GPTLosses_2D: recons_feature + recons_velocity + vq_commit）
        _losses = torch.nn.ModuleDict({
            split: GPTLosses_2D(cfg, "vae", 13)
            for split in ["losses_train", "losses_test"]
        }).to(device)

        log_path = pjoin(ckpt_save_path, "training_log.csv")
        csv_header = ["epoch", "train_loss", "val_loss"]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

        best_val_loss = float("inf")
        best_ckpt_path = None

        for epoch in range(cfg.TRAIN.END_EPOCH):
            vqvae.train()
            train_total_loss = 0.0
            for batch in tqdm(_train_dataloader, desc=f"Epoch {epoch}", leave=False):
                optimizer.zero_grad()
                feats_ref = torch.cat([
                    batch["motion_2d"],
                    torch.zeros(batch["motion_2d"].shape[0], batch["motion_2d"].shape[1],
                                263 - batch["motion_2d"].shape[2]),
                ], dim=-1).to(device)
                feats_rst, loss_commit, perplexity = vqvae(feats_ref)
                rs_set = {
                    "m_ref": feats_ref,
                    "m_rst": feats_rst,
                    "loss_commit": loss_commit,
                    "perplexity": perplexity,
                }
                loss = _losses["losses_train"].update(rs_set)
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()
            scheduler.step()

            avg_train_loss = train_total_loss / len(_train_dataloader)
            print(f"Epoch {epoch}  Train Loss: {avg_train_loss:.4f}")

            metrics = {"train/loss": avg_train_loss}
            csv_row = [epoch, avg_train_loss, ""]

            if epoch % cfg.LOGGER.VAL_EVERY_STEPS == 0:
                vqvae.eval()
                val_total_loss = 0.0
                with torch.no_grad():
                    for batch in _val_dataloader:
                        feats_ref = torch.cat([
                            batch["motion_2d"],
                            torch.zeros(batch["motion_2d"].shape[0], batch["motion_2d"].shape[1],
                                        263 - batch["motion_2d"].shape[2]),
                        ], dim=-1).to(device)
                        feats_rst, loss_commit, perplexity = vqvae(feats_ref)
                        rs_set = {
                            "m_ref": feats_ref,
                            "m_rst": feats_rst,
                            "loss_commit": loss_commit,
                            "perplexity": perplexity,
                        }
                        loss = _losses["losses_test"].update(rs_set)
                        val_total_loss += loss.item()

                avg_val_loss = val_total_loss / len(_val_dataloader)
                print(f"Epoch {epoch}  Val Loss: {avg_val_loss:.4f}")
                metrics["val/loss"] = avg_val_loss
                csv_row[2] = avg_val_loss

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    new_ckpt_path = pjoin(
                        ckpt_save_path,
                        f"best_vqvae_epoch{epoch}_valloss{best_val_loss:.4f}.tar",
                    )
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": vqvae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, new_ckpt_path)
                    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                        os.remove(best_ckpt_path)
                    best_ckpt_path = new_ckpt_path
                    print(f"  -> Best checkpoint saved: {new_ckpt_path}")

            wandb.log(metrics, step=epoch)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(csv_row)


if __name__ == "__main__":
    main()
