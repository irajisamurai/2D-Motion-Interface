---
library_name: pytorch
tags:
  - arxiv:2608.15984
  - motion-captioning
  - motion-language
  - human-motion
  - 2d-pose
pipeline_tag: other
license: mit
---

# 2D Motion Interface — inference bundle (MotionGPT)

[Paper (arXiv:2608.15984)](https://arxiv.org/abs/2608.15984) · [Code](https://github.com/irajisamurai/2D-Motion-Interface) · [HCMIW @ ECCV 2026](https://hcmiw.github.io/hcmiw-eccv2026/) (Oral Presentation)

Inference-only weights for **A Plug-and-Play 2D Motion Interface for Real-World Motion
Language Models**, in the MotionGPT configuration. This repo backs the demo Space; it is
not a general-purpose checkpoint.

The interface captions human motion from **2D keypoints only** — no 3D pose estimation
anywhere in the pipeline. The pretrained motion-language model is frozen; only the 2D
encoder (9.6M params) and the real-video adapter `A_real` (0.3M params) are trained.

## Contents

| File | Size | |
|---|---|---|
| `lm.safetensors` | 993.3 MB | flan-t5-base + 515 motion tokens. The three tied embedding copies present in the training checkpoint are dropped; `lm_head` is untied and kept. |
| `vqvae.safetensors` | 39.4 MB | 2D encoder + quantizer. The decoder is dropped — only `encode` is ever called. |
| `adapter.safetensors` | 0.3 MB | `A_real`, the real-video adapter (residual, hidden 512, 81-dim). |
| `flan-t5-base/` | 2.4 MB | Config + tokenizer only. No pretrained weights: `lm.safetensors` supplies all of them. |
| `stats.npz` | 3 KB | Feature mean/std for the 68-dim and 81-dim layouts. |
| `model_config.yaml` | 1 KB | VQ-VAE hyperparameters, adapter type, `unit_length`, `max_motion_length`. |

Everything is **fp32**. Dropping the evaluator weights, the duplicated embeddings and the
VQ-VAE decoder takes the 1.53 GB training checkpoint down to 1.03 GB with no numerical
change, so fp16 is unnecessary — and flan-T5 is known to be unstable in fp16 anyway.

## Pipeline

```
COCO-17 keypoints + confidence
  -> COCO-13 (drop eyes/ears)
  -> 81-dim features (with confidence)  --A_real-->  zero-pad to 263
  -> VQ-VAE.encode  ->  MotionGPT (m2t)  ->  caption
```

Features are mid-hip centred and scale-normalised, so raw pixel coordinates work at any
resolution. Input is expected at 20 fps.

## Usage

```python
from huggingface_hub import snapshot_download
from captioner import MotionCaptioner          # from the demo Space

bundle = snapshot_download("KanameYOkoYAMA/2d-motion-interface")
cap = MotionCaptioner(bundle, device="cpu")
print(cap.caption_json("clip.json")["caption"])
```

Decoding is greedy and the full clip is used from frame 0, so captions are reproducible.
A caption takes ~1 s on CPU.

## Reproducibility

Captions were verified byte-for-byte against the evaluation script on the four demo clips
(8/8 exact, adapter and adapter-less paths). Across all 132 real-world clips the match rate
is 256/264; the 8 differences are GPU-vs-CPU floating-point noise flipping a greedy argmax,
not a difference in weights.

Verified with `torch==2.9.0`, `transformers==4.57.1`. Bumping transformers can change greedy
decoding — re-run the parity check after any upgrade.

## Licence and provenance

MIT, following the licences of the upstream work this builds on:

- [**MotionGPT**](https://github.com/OpenMotionLab/MotionGPT) (MIT) — `lm.safetensors` and the
  VQ-VAE quantizer derive from its pretrained checkpoint.
- [**HumanML3D**](https://github.com/EricGuo5513/HumanML3D) (MIT) — `stats.npz` holds mean/std
  vectors computed over its features.
- [**flan-t5-base**](https://huggingface.co/google/flan-t5-base) (Apache-2.0) — the language
  model architecture and tokenizer.

HumanML3D is itself derived from **AMASS**, whose distribution policy does not permit
redistributing the motion data. **No AMASS or HumanML3D motion data is included here** — this
repo contains trained weights and a few hundred aggregate mean/std floats, nothing from which
motion sequences could be recovered.

MotionGPT's README notes that its dependencies (SMPL, SMPL-X, PyTorch3D) and the datasets it
uses each carry their own licences, which apply to downstream use of these weights as well.
