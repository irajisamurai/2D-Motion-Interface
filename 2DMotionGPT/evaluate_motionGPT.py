import argparse
import json
import os
import random
import sys
from datetime import datetime
from os.path import join as pjoin

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.archs.mgpt_lm import MLM
from src.config import parse_args
from src.data.humanml.dataset_t2m_eval import (
    Text2MotionDatasetEval,
    Text2MotionDatasetEval_2D,
    Text2MotionDatasetEval_2D_COCO_normalized
)
from src.data.humanml.utils.word_vectorizer import WordVectorizer
from src.data.utils import humanml3d_collate, humanml3d_collate_2d
from src.metrics.m2t import M2TMetrics
from src.models.mgpt_vq import VQVae

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MOTION_DIM = 263
DEFAULT_MOTIONGPT_CKPT = "./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar"
DEFAULT_VQVAE_2D_CKPT = "./checkpoints/2d_vqvae_ver3/MotionGPT_2DEncoder/encoder_only-l1-bs64-lr0.0001/best_vqvae_epoch30_valacc0.3095.tar"


def parse_eval_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--vqvae_ckpt", type=str, default=DEFAULT_VQVAE_2D_CKPT,
    )
    parser.add_argument(
        "--motiongpt_ckpt", type=str, default=DEFAULT_MOTIONGPT_CKPT,
    )
    parser.add_argument(
        "--eval_3d", action="store_true",
    )
    parser.add_argument(
        "--output", type=str, default=None,
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
    )
    eval_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return eval_args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


def get_device(gpu_id: int = 0) -> torch.device:
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def load_statistics(cfg):
    data_root = cfg.DATASET.HUMANML3D.ROOT
    dis_data_root = pjoin(
        cfg.DATASET.HUMANML3D.MEAN_STD_PATH,
        "t2m",
        "VQVAEV3_CB1024_CMT_H1024_NRES3",
        "meta",
    )
    return {
        "data_root": data_root,
        "mean": np.load(pjoin(dis_data_root, "mean.npy")),
        "std": np.load(pjoin(dis_data_root, "std.npy")),
        "mean_2d": np.load(pjoin(dis_data_root, "mean_2d_coco_normalized.npy")),
        "std_2d": np.load(pjoin(dis_data_root, "std_2d_coco_normalized.npy")),
    }


def build_word_vectorizer(cfg) -> WordVectorizer:
    return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")


def build_dataloader(cfg, w_vectorizer, stats, *, use_2d: bool = False):
    dataset_kwargs = dict(
        data_root=stats["data_root"],
        split="test",
        mean=stats["mean"],
        std=stats["std"],
        max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
        min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
        win_size=64,
        unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
        w_vectorizer=w_vectorizer,
    )
    if use_2d:
        dataset_cls = Text2MotionDatasetEval_2D_COCO_normalized
        dataset_kwargs.update(mean_2d=stats["mean_2d"], std_2d=stats["std_2d"])
        collate_fn = humanml3d_collate_2d
    else:
        dataset_cls = Text2MotionDatasetEval
        collate_fn = humanml3d_collate

    dataset = dataset_cls(**dataset_kwargs)
    return DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=True,
    )


def build_models(cfg, device):
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
    return vqvae, lm


def extract_prefixed_state_dict(state_dict, prefix: str):
    prefix_with_dot = f"{prefix}."
    return {
        key.replace(prefix_with_dot, "", 1): value
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    }


def load_motiongpt_checkpoint(vqvae, lm, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    vqvae.load_state_dict(extract_prefixed_state_dict(state_dict, "vae"))
    lm.load_state_dict(extract_prefixed_state_dict(state_dict, "lm"))


def load_2d_encoder_weights(vqvae, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_only = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if "encoder" in key
    }
    vqvae.load_state_dict(encoder_only, strict=False)


def create_metrics(cfg, w_vectorizer, device):
    return M2TMetrics(
        cfg=cfg,
        w_vectorizer=w_vectorizer,
        diversity_times=30,
        dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
    ).to(device)


def encode_motion_tokens(vqvae, feats):
    motion_tokens = []
    lengths_tokens = []
    for idx in range(len(feats)):
        motion_token, _ = vqvae.encode(feats[idx : idx + 1])
        motion_tokens.append(motion_token[0])
        lengths_tokens.append(motion_token.shape[1])
    return motion_tokens, lengths_tokens


def prepare_batch_3d(batch, device):
    feats = batch["motion"].to(device)
    return feats, feats


def prepare_batch_2d(batch, device):
    motion_2d = batch["motion_2d"].to(device)
    pad_dim = MOTION_DIM - motion_2d.shape[2]
    if pad_dim < 0:
        raise ValueError(f"2D motion feature dim {motion_2d.shape[2]} exceeds {MOTION_DIM}")
    padding = torch.zeros(motion_2d.shape[0], motion_2d.shape[1], pad_dim, device=device)
    feats_for_encoder = torch.cat([motion_2d, padding], dim=-1)
    feats_for_metrics = batch["motion"].to(device)
    return feats_for_encoder, feats_for_metrics


def evaluate_model(description, dataloader, batch_preparer, vqvae, lm, metrics, device):
    metrics.reset()
    vqvae.eval()
    lm.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            feats_for_encoder, reference_feats = batch_preparer(batch, device)
            motion_tokens, lengths_tokens = encode_motion_tokens(vqvae, feats_for_encoder)
            outputs = lm.generate_conditional(
                motion_tokens=motion_tokens,
                lengths=lengths_tokens,
                task="m2t",
                stage="test",
            )
            metrics.update(
                feats_ref=reference_feats,
                pred_texts=outputs,
                gt_texts=batch["all_captions"],
                lengths=batch["length"],
                word_embs=batch["word_embs"].to(device),
                pos_ohot=batch["pos_ohot"].to(device),
                text_lengths=batch["text_len"].to(device),
            )

    results = metrics.compute(sanity_flag=False)
    metrics.reset()

    for key, value in results.items():
        print(f"  {key}: {value.item():.4f}")
    return {k: v.item() for k, v in results.items()}


def save_results(results_all: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)


def main():
    eval_args = parse_eval_args()
    extra = []
    if "--cfg" not in sys.argv:
        extra += ["--cfg", "configs/config_h3d_stage1.yaml"]
    if "--nodebug" not in sys.argv:
        extra += ["--nodebug"]
    extra += ["--device", str(eval_args.gpu_id), "--num_nodes", "1"]
    sys.argv.extend(extra)
    cfg = parse_args(phase="train")
    seed_everything(cfg.SEED_VALUE)
    device = get_device(eval_args.gpu_id)

    print(f"motiongpt_ckpt : {eval_args.motiongpt_ckpt}")
    print(f"vqvae_ckpt     : {eval_args.vqvae_ckpt}")

    stats = load_statistics(cfg)
    w_vectorizer = build_word_vectorizer(cfg)
    dataloader_2d = build_dataloader(cfg, w_vectorizer, stats, use_2d=True)
    if eval_args.eval_3d:
        dataloader_3d = build_dataloader(cfg, w_vectorizer, stats, use_2d=False)

    vqvae, lm = build_models(cfg, device)
    metrics = create_metrics(cfg, w_vectorizer, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_all = {
        "timestamp": timestamp,
        "motiongpt_ckpt": eval_args.motiongpt_ckpt,
        "vqvae_ckpt": eval_args.vqvae_ckpt,
    }

    if eval_args.eval_3d:
        load_motiongpt_checkpoint(vqvae, lm, eval_args.motiongpt_ckpt)
        results_3d = evaluate_model(
            "3D baseline", dataloader_3d, prepare_batch_3d, vqvae, lm, metrics, device
        )
        results_all["3d"] = results_3d

    load_motiongpt_checkpoint(vqvae, lm, eval_args.motiongpt_ckpt)
    load_2d_encoder_weights(vqvae, eval_args.vqvae_ckpt)
    results_2d = evaluate_model(
        "2D encoder", dataloader_2d, prepare_batch_2d, vqvae, lm, metrics, device
    )
    results_all["2d"] = results_2d

    output_path = eval_args.output or pjoin(
        os.path.dirname(eval_args.vqvae_ckpt), f"eval_{timestamp}.json"
    )
    save_results(results_all, output_path)


if __name__ == "__main__":
    main()
