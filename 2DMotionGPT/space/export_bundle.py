"""Export a slim, inference-only weight bundle for the Hugging Face Space.

Builds the models exactly the way evaluate_with_realworld.py does, then
serialises only what m2t inference needs:

  * lm.safetensors      flan-t5-base + motion tokens, tied embedding copies dropped
  * vqvae.safetensors   2D encoder + quantizer (decoder dropped, `encode` only)
  * adapter.safetensors  A_real
  * stats.npz           mean/std for the 68-dim and 81-dim features
  * flan-t5-base/       config + tokenizer only (no 990MB model.safetensors)

Everything stays fp32 — see the size report at the end for why fp16 is not needed.

Run from the 2DMotionGPT directory:
    python space/export_bundle.py \
        --vqvae_ckpt ./checkpoints/2d_vqvae_ver3/.../best_vqvae_epoch2960_valacc0.4471.tar \
        --adapter_ckpt ./checkpoints/adapter/.../best_adapter_epoch690_valloss1.2251.tar
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from omegaconf import OmegaConf
from os.path import join as pjoin
from safetensors.torch import save_file

from src.config import parse_args

from space.adapters import build_adapter
from space.mgpt.mgpt_lm import MLM
from space.mgpt.mgpt_vq import VQVae

# config + tokenizer are all the Space needs; the weights come from lm.safetensors
TOKENIZER_FILES = ["config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "special_tokens_map.json"]


def mb(x):
    return x / 1e6


def state_dict_bytes(sd):
    return sum(v.numel() * v.element_size() for v in sd.values())


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vqvae_ckpt", required=True)
    p.add_argument("--adapter_ckpt", required=True)
    p.add_argument("--motiongpt_ckpt", default="./checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar")
    p.add_argument("--adapter_type", default="residual")
    p.add_argument("--adapter_hidden", type=int, default=512)
    p.add_argument("--out", default="./space/bundle")
    a, _ = p.parse_known_args()

    cfg = parse_args(phase="train")
    os.makedirs(a.out, exist_ok=True)
    report = {}

    ckpt_base = torch.load(a.motiongpt_ckpt, map_location="cpu", weights_only=True)
    report["source_checkpoint_total"] = mb(state_dict_bytes(ckpt_base["state_dict"]))

    # ---- VQ-VAE: base weights, then the trained 2D encoder on top ---------
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
    )
    vqvae.load_state_dict({k.replace("vae.", ""): v
                           for k, v in ckpt_base["state_dict"].items() if "vae" in k})
    ckpt_2d = torch.load(a.vqvae_ckpt, map_location="cpu", weights_only=False)
    raw_2d = ckpt_2d.get("model_state_dict", ckpt_2d)
    missing, unexpected = vqvae.load_state_dict(
        {k: v for k, v in raw_2d.items() if "encoder" in k}, strict=False)
    assert not unexpected, f"unexpected 2D encoder keys: {unexpected[:5]}"
    assert all("encoder" not in k for k in missing), "2D checkpoint did not cover the encoder"

    vq_full = vqvae.state_dict()
    vq_slim = {k: v.contiguous() for k, v in vq_full.items() if "decoder" not in k}
    report["vqvae_full"] = mb(state_dict_bytes(vq_full))
    report["vqvae_slim"] = mb(state_dict_bytes(vq_slim))
    save_file(vq_slim, pjoin(a.out, "vqvae.safetensors"))

    # ---- language model --------------------------------------------------
    lm = MLM(
        model_path=cfg.lm.default.params.model_path,
        model_type=cfg.lm.default.params.model_type,
        stage=cfg.lm.default.params.stage,
        motion_codebook_size=cfg.lm.default.params.motion_codebook_size,
    )
    lm.load_state_dict({k.replace("lm.", ""): v
                        for k, v in ckpt_base["state_dict"].items() if "lm" in k})

    lm_full = lm.state_dict()
    tied = ["language_model.encoder.embed_tokens.weight",
            "language_model.decoder.embed_tokens.weight"]
    shared = lm_full["language_model.shared.weight"]
    for k in tied:
        assert torch.equal(lm_full[k], shared), f"{k} is not tied to shared.weight"
    lm_slim = {k: v.contiguous() for k, v in lm_full.items() if k not in tied}
    report["lm_full"] = mb(state_dict_bytes(lm_full))
    report["lm_slim"] = mb(state_dict_bytes(lm_slim))
    save_file(lm_slim, pjoin(a.out, "lm.safetensors"))

    # ---- adapter ---------------------------------------------------------
    adapter = build_adapter(a.adapter_type, dim=81, hidden=a.adapter_hidden)
    ck_ad = torch.load(a.adapter_ckpt, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ck_ad.get("model_state_dict", ck_ad))
    ad_sd = {k: v.contiguous() for k, v in adapter.state_dict().items()}
    report["adapter"] = mb(state_dict_bytes(ad_sd))
    save_file(ad_sd, pjoin(a.out, "adapter.safetensors"))

    # ---- normalisation stats --------------------------------------------
    meta = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, "t2m",
                 "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
    np.savez(pjoin(a.out, "stats.npz"),
             mean_2d=np.load(pjoin(meta, "mean_2d_coco_normalized.npy")),
             std_2d=np.load(pjoin(meta, "std_2d_coco_normalized.npy")),
             mean_est=np.load(pjoin(meta, "mean_2d_coco_estimated_concatenate.npy")),
             std_est=np.load(pjoin(meta, "std_2d_coco_estimated_concatenate.npy")))

    # ---- tokenizer / config (no pretrained weights) ----------------------
    tok_out = pjoin(a.out, "flan-t5-base")
    os.makedirs(tok_out, exist_ok=True)
    src = cfg.lm.default.params.model_path
    for fn in TOKENIZER_FILES:
        sp = pjoin(src, fn)
        if os.path.exists(sp):
            shutil.copy(sp, pjoin(tok_out, fn))
        else:
            print(f"[warn] missing {sp}")

    # ---- manifest --------------------------------------------------------
    OmegaConf.save(OmegaConf.create({
        "adapter_type": a.adapter_type,
        "adapter_hidden": a.adapter_hidden,
        "adapter_dim": 81,
        "nfeats": cfg.vq.default.params.nfeats,
        "unit_length": cfg.DATASET.HUMANML3D.UNIT_LEN,
        "max_motion_length": cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
        "motion_codebook_size": cfg.lm.default.params.motion_codebook_size,
        "vq": dict(cfg.vq.default.params),
    }), pjoin(a.out, "model_config.yaml"))

    # ---- report ----------------------------------------------------------
    on_disk = {f: os.path.getsize(pjoin(a.out, f)) for f in os.listdir(a.out)
               if os.path.isfile(pjoin(a.out, f))}
    print("\n=== bundle ===")
    for f, b in sorted(on_disk.items(), key=lambda kv: -kv[1]):
        print(f"  {f:24s} {mb(b):8.1f} MB")
    tok_bytes = sum(os.path.getsize(pjoin(tok_out, f)) for f in os.listdir(tok_out))
    print(f"  flan-t5-base/{'':11s} {mb(tok_bytes):8.1f} MB")
    print(f"  {'TOTAL':24s} {mb(sum(on_disk.values()) + tok_bytes):8.1f} MB")

    print("\n=== what was dropped ===")
    print(f"  source checkpoint state_dict   {report['source_checkpoint_total']:8.1f} MB")
    print(f"  lm   {report['lm_full']:7.1f} -> {report['lm_slim']:7.1f} MB "
          f"(tied embedding copies)")
    print(f"  vae  {report['vqvae_full']:7.1f} -> {report['vqvae_slim']:7.1f} MB (decoder)")
    print(f"  metrics.* evaluator            dropped entirely")
    json.dump(report, open(pjoin(a.out, "export_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
