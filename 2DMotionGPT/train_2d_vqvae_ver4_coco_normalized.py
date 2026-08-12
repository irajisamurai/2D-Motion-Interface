from src.config import parse_args
from os.path import join as pjoin
import os
import csv
import argparse
import torch
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
import random
import numpy as np

from src.data.build import build_humanml3d_dataloader_2d
from src.models.build import build_vqvae
from src.losses.build import build_loss_functions, build_optim
from src.train.trainer import train_one_epoch, eval_one_epoch
from src.data.HumanML3D import MotionDatasetV_2DCOCO_normalized_non_windowsize
from src.data.humanml.dataset_t2m_eval import Text2MotionDatasetEval_2D_COCO_normalized
from src.data.utils import humanml3d_collate_2d
from src.data.humanml.utils.word_vectorizer import WordVectorizer
from torch.utils.data import DataLoader
from src.metrics.t2m import TM2TMetrics
from src.metrics.mr import MRMetrics

cfg = parse_args(phase="train")

_seed_parser = argparse.ArgumentParser(add_help=False)
_seed_parser.add_argument("--seed", type=int, default=0)
_seed_args, _ = _seed_parser.parse_known_args()
seed_value = _seed_args.seed
cfg.SEED_VALUE = seed_value

random.seed(seed_value)
pl.seed_everything(seed_value)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    run_name = (
        f"encoder_only"
        f"-{cfg.LOSS.ABLATION.ENCODER_LOSS}"
        f"-bs{cfg.TRAIN.BATCH_SIZE}"
        f"-lr{cfg.TRAIN.OPTIM.params.lr}"
        f"-seed{seed_value}"
    )
    with wandb.init(
        project=cfg.LOGGER.WANDB.params.project,
        config=OmegaConf.to_container(cfg, resolve=False),
        name=run_name,
        group=cfg.LOGGER.WANDB.params.group,
    ):
        ckpt_save_path = f"./checkpoints/2d_vqvae_ver3/{wandb.run.group}/{wandb.run.name}"

        data_root = cfg.DATASET.HUMANML3D.ROOT
        dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        mean = np.load(pjoin(dis_data_root, "mean.npy"))
        std = np.load(pjoin(dis_data_root, "std.npy"))
        mean_2d = np.load(pjoin(dis_data_root, "mean_2d_coco_normalized.npy"))
        std_2d = np.load(pjoin(dis_data_root, "std_2d_coco_normalized.npy"))
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
                        unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,)
        w_vectorizer = WordVectorizer(
                cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
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
                w_vectorizer=w_vectorizer,)

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

        ref_vqvae = build_vqvae(cfg).to(device)
        checkpoint = torch.load("./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar", weights_only=True)
        new_state_dict = {}
        for key, values in checkpoint["state_dict"].items():
            if "vae" in key:
                new_state_dict[key.replace("vae.", "")] = values
        ref_vqvae.load_state_dict(state_dict=new_state_dict)

        vqvae = build_vqvae(cfg).to(device)
        new_state_dict = {}
        for key, values in checkpoint["state_dict"].items():
            if "vae" in key and not "encoder" in key:
                new_state_dict[key.replace("vae.", "")] = values
        vqvae.load_state_dict(state_dict=new_state_dict, strict=False)
        for name, param in vqvae.named_parameters():
            if not "encoder" in name:
                param.requires_grad_(False)

        recon_losses, loss_fn = build_loss_functions(cfg, device)
        optimizer = build_optim(cfg, vqvae)

        tm2t_metric = TM2TMetrics(cfg, dataname="humanml3d").to(device)
        mrmetrics = MRMetrics(njoints=22).to(device)

        os.makedirs(ckpt_save_path, exist_ok=True)
        log_path = pjoin(ckpt_save_path, "training_log.csv")
        csv_header = [
            "epoch",
            "train_loss", "train_encoder_loss", "train_rst_loss",
            "val_loss", "val_encoder_loss", "val_rst_loss", "val_accuracy",
            "FID", "Div", "MPJPE", "PAMPJPE", "ACCEL",
        ]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)
        best_val_accuracy = 0.0
        best_ckpt_path = None

        for epoch in range(cfg.TRAIN.END_EPOCH):
            log = train_one_epoch(cfg, vqvae, ref_vqvae, _train_dataloader, recon_losses, loss_fn, optimizer, device)
            print(f'Epoch {epoch} Train Loss: {log["train_loss"]} Encoder Loss: {log["train_encoder_loss"]} Rst Loss: {log["train_rst_loss"]}')

            metrics = {
                "train/loss": log["train_loss"],
                "train/encoder_loss": log["train_encoder_loss"],
                "train/rst_loss": log["train_rst_loss"],
            }
            csv_row = [
                epoch,
                log["train_loss"], log["train_encoder_loss"], log["train_rst_loss"],
                "", "", "", "", "", "", "", "", "",
            ]

            if epoch % 10 == 0:
                val_log = eval_one_epoch(cfg, vqvae, ref_vqvae, _val_dataloader, recon_losses, loss_fn, device, mr_metrics=mrmetrics, tm2t_metric=tm2t_metric)
                print(f'Epoch {epoch} Encoder Loss: {val_log["val_encoder_loss"]} Rst Loss: {val_log["val_rst_loss"]} Acc: {val_log["val_accuracy"]}')
                metrics.update({
                    (f"val/{k[4:]}" if k.startswith("val_") else f"val/{k}"): v
                    for k, v in val_log.items()
                })
                csv_row[4:] = [
                    val_log.get("val_loss", ""),
                    val_log.get("val_encoder_loss", ""),
                    val_log.get("val_rst_loss", ""),
                    val_log.get("val_accuracy", ""),
                    val_log.get("FID", ""), val_log.get("Div", ""),
                    val_log.get("MPJPE", ""), val_log.get("PAMPJPE", ""), val_log.get("ACCEL", ""),
                ]
                if val_log["val_accuracy"] > best_val_accuracy:
                    best_val_accuracy = val_log["val_accuracy"]
                    new_ckpt_path = pjoin(ckpt_save_path, f'best_vqvae_epoch{epoch}_valacc{best_val_accuracy:.4f}.tar')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': vqvae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, new_ckpt_path)
                    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                        os.remove(best_ckpt_path)
                    best_ckpt_path = new_ckpt_path

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(csv_row)
            wandb.log(metrics, step=epoch)

train()