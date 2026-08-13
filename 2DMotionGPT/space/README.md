---
title: 2D Motion Interface
emoji: 🕺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

> **Hardware note.** This Space is set to `zero-a10g`, not `cpu-basic`: a free personal
> account cannot create a Gradio Space on `cpu-basic` (HTTP 402). Inference still runs on
> CPU (~1 s per caption) — the GPU is not used, and ZeroGPU's supported torch versions are
> the reason `requirements.txt` pins `torch==2.9.1`.

# A Plug-and-Play 2D Motion Interface

Motion captioning of real monocular video through **2D keypoints only** — no 3D pose
estimation anywhere in the pipeline.

Pick one of four clips from our real-world video dataset. The ViTPose 2D keypoints
released with the clip are fed through `A_real` (0.3M params) → the 2D encoder →
a frozen MotionGPT, and captioned live on CPU. The adapter-less baseline is captioned
too, so the adapter's contribution is visible side by side.

## What runs here

```
COCO-17 keypoints + confidence   (precomputed, shipped with the Space)
  → COCO-13
  → 81-dim features (with confidence)  --A_real-->  zero-pad to 263  ─┐
  → 68-dim features (no confidence)  ───────────── zero-pad to 263  ─┴→ VQ-VAE.encode
                                                                      → MotionGPT (m2t)
                                                                      → caption
```

Pose estimation does **not** run in the Space — the keypoints come from the released
dataset. That keeps the demo CPU-only (~1–2 s per caption) and free of any mmpose
install. Decoding is greedy and the full clip is used from frame 0, so each caption is
reproducible; the expected outputs are recorded in `assets/clips.yaml` and checked by
`verify_parity.py`.

## Layout

| Path | |
|---|---|
| `app.py` | Gradio UI |
| `captioner.py` | the whole inference pipeline |
| `features2d.py` | 2D keypoints → features (extracted from `src/data/adapter_datasets.py`) |
| `adapters.py` | `A_real` definitions |
| `mgpt/` | VQ-VAE + language model, vendored from `2DMotionGPT/src` |
| `assets/` | the four clips: overlay video, keypoints, ground-truth captions |
| `bundle/` | weights (~1.03 GB, fp32) — not committed |

## Weights

`bundle/` is produced from the training checkpoints by `export_bundle.py`, which keeps
only what m2t inference needs:

| | |
|---|---|
| `lm.safetensors` | 993.3 MB — flan-t5-base + motion tokens, tied embedding copies dropped |
| `vqvae.safetensors` | 39.4 MB — 2D encoder + quantizer (decoder dropped) |
| `adapter.safetensors` | 0.3 MB — `A_real` |
| `flan-t5-base/` | 2.4 MB — config + tokenizer only, no pretrained weights |
| `stats.npz` | feature mean/std |

Everything stays fp32: dropping the evaluator weights, the duplicated embeddings and the
VQ-VAE decoder takes the 1.53 GB training checkpoint down to 1.03 GB with no numerical
change at all, so fp16 is not needed.

The Space finds the bundle via `BUNDLE_DIR`, or downloads it from the model repo named in
`BUNDLE_REPO` — currently
[`KanameYOkoYAMA/2d-motion-interface`](https://huggingface.co/KanameYOkoYAMA/2d-motion-interface).

That repo is **private**, so the Space needs an `HF_TOKEN` secret with read access to it.
Set both under Space *Settings*:

| | | |
|---|---|---|
| `BUNDLE_REPO` | variable | `KanameYOkoYAMA/2d-motion-interface` |
| `HF_TOKEN` | secret | a read token for the account owning the model repo |

## Reproducing locally

From the `2DMotionGPT` directory:

```bash
# 1. export the weight bundle
PYTHONPATH=. python space/export_bundle.py \
    --cfg configs/config_h3d_stage1.yaml --nodebug \
    --vqvae_ckpt ./checkpoints/2d_vqvae_ver3/MotionGPT_2DEncoder/encoder_only-l1-bs64-lr0.0001/best_vqvae_epoch2960_valacc0.4471.tar \
    --adapter_ckpt ./checkpoints/adapter/MotionGPT_2DEncoder/adapter-residual-bs64-lr1e-3/best_adapter_epoch690_valloss1.2251.tar

# 2. collect the demo clips (needs the real-world dataset)
PYTHONPATH=. python space/build_assets.py --reference <all_clips.json>

# 3. confirm the bundle reproduces the evaluation captions exactly
PYTHONPATH=.:space python space/verify_parity.py --device cpu

# 4. run the UI
PYTHONPATH=.:space python space/app.py
```

## Licence

MIT. The demo videos come from our own real-world dataset. The weights derive from
[MotionGPT](https://github.com/OpenMotionLab/MotionGPT) (MIT) and
[HumanML3D](https://github.com/EricGuo5513/HumanML3D) (MIT) statistics, on top of
[flan-t5-base](https://huggingface.co/google/flan-t5-base) (Apache-2.0). No AMASS or
HumanML3D motion data is redistributed — see the
[model repo](https://huggingface.co/KanameYOkoYAMA/2d-motion-interface) for details.
