# A Plug-and-Play 2D Motion Interface for Real-World Motion Language Models

## Overview

This repository contains the implementation for three MoLM baselines:

| Directory | Base Model |
|-----------|-----------|
| `2DMotionGPT/` | MotionGPT |
| `2DTM2T/` | TM2T |
| `2DMG-MotionLLM/` | MG-MotionLLM |

## Environment Setup

Each sub-directory uses a separate conda environment.

**2DMotionGPT**
```bash
cd 2DMotionGPT
conda env create -f configs/environment.yaml
conda activate mgpt_2d
```

**2DTM2T**
```bash
cd 2DTM2T
conda env create -f environment.yaml
conda activate tm2t
```

**2DMG-MotionLLM**
```bash
cd 2DMG-MotionLLM
conda env create -f environment.yml
conda activate mg-motionllm
```

## Dataset Preparation

All three sub-projects use the **HumanML3D** dataset. Please follow the HumanML3D repository to download and prepare the dataset. Expected structure:

```
HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy
├── Std.npy
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```

**2DMG-MotionLLM** additionally uses the **FineMotion** dataset for the Motion-to-Detailed Text task. Download `BPMSD_auto.zip` and `BPMSD_human.zip` from the FineMotion repository and place them under `dataset/HumanML3D/finemotion_texts/`:

```
dataset/HumanML3D/
├── finemotion_texts/
│   ├── BPMSD_auto.zip
│   ├── BPMSD_auto.json
│   ├── BPMSD_human.zip
│   └── BPMSD_human.json
└── ...
```

The **real-world video dataset** constructed in this work (paired 2D/3D estimated motions and captions) cannot be shared at this time due to the anonymity requirements of the review process. The dataset will be made publicly available upon acceptance.

## Pre-trained Checkpoints

For base model checkpoints and other dependencies (GloVe embeddings, evaluators, etc.), please follow the setup instructions in each original repository.

The checkpoints trained in this work cannot be shared at this time due to the anonymity requirements of the review process. They will be made publicly available upon acceptance.

## Reproducing Key Results

### 1. M2T Evaluation on HumanML3D (2D encoder vs 3D baseline)

**2DMotionGPT** — evaluate 2D encoder and 3D baseline side by side:
```bash
cd 2DMotionGPT
# 3D baseline
python evaluate_motionGPT.py --cfg configs/config_h3d_stage1.yaml --nodebug --eval_3d
# 2D encoder
python evaluate_motionGPT.py --cfg configs/config_h3d_stage1.yaml --nodebug \
    --vqvae_ckpt <path/to/2d_encoder.tar>
```

<details><summary>Paper Results (HumanML3D test set)</summary>

| Method | R-Prec@1↑ | R-Prec@2↑ | R-Prec@3↑ | MM-Dist↓ | BLEU-1↑ | BLEU-4↑ | ROUGE-L↑ | CIDEr↑ | BERTScore↑ |
|---|---|---|---|---|---|---|---|---|---|
| 3D baseline | 0.5162 | 0.7069 | 0.8026 | 2.9941 | 0.4293 | 0.0616 | **0.3453** | **0.0789** | 0.3155 |
| 2D encoder (Ours) | **0.5256** | **0.7166** | **0.8034** | **2.9703** | **0.4312** | **0.0628** | 0.3448 | 0.0771 | **0.3197** |
| 2D scratch | — | — | — | — | — | — | — | — | — |

</details>

---

**2DTM2T** — first tokenize, then evaluate:
```bash
cd 2DTM2T
# 3D baseline
python tokenize_script_motiongpt.py --gpu_id 0 --name MotionGPT-base --dataset_name t2m --motiongpt_ckpt <path/to/motiongpt_s3_h3d.tar>
python final_evaluations_m2t.py --tokenizer_name MotionGPT-base \
    --m2t_name M2T_MotionGPT_EL4_DL4_NH8_PS --gpu_id 0 --codebook_size 512
# 2D scratch (full VQ-VAE trained from scratch on 2D data)
python tokenize_script_motiongpt_2d_rec.py --gpu_id 0 --name MotionGPT_2DReconstruction \
    --dataset_name t2m --recon_ckpt <path/to/2d_scratch_encoder.tar>
python final_evaluations_m2t.py --tokenizer_name MotionGPT_2DReconstruction \
    --m2t_name M2T_MotionGPT_2DRecon_EL4_DL4_NH8_PS --gpu_id 0
# 2D encoder
python tokenize_script_motiongpt_2d.py --gpu_id 0 --name MotionGPT-base-2D \
    --dataset_name t2m --encoder_2d_ckpt <path/to/2d_encoder.tar>
python final_evaluations_m2t.py --tokenizer_name MotionGPT-base-2D \
    --m2t_name M2T_MotionGPT_EL4_DL4_NH8_PS --gpu_id 0
```

<details><summary>Paper Results (HumanML3D test set)</summary>

| Method | R-Prec@1↑ | R-Prec@2↑ | R-Prec@3↑ | MM-Dist↓ | BLEU-1↑ | BLEU-4↑ | ROUGE-L↑ | CIDEr↑ | BERTScore↑ |
|---|---|---|---|---|---|---|---|---|---|
| 3D baseline | **0.4881** | 0.6890 | 0.7862 | 3.1366 | 0.6148 | 0.2326 | 0.4922 | 0.6910 | 0.3680 |
| 2D scratch | 0.2789 | 0.4470 | 0.5616 | 4.9210 | 0.5502 | 0.1748 | 0.4374 | 0.4988 | 0.3115 |
| 2D encoder (Ours) | 0.4847 | **0.6981** | **0.7978** | **3.1318** | 0.6137 | 0.2320 | **0.4923** | **0.6937** | **0.3687** |

</details>

---

**2DMG-MotionLLM** — evaluate 3D baseline and 2D encoder sequentially:
```bash
cd 2DMG-MotionLLM
# 3D baseline + 2D encoder (2D encoder is auto-detected from checkpoints/2d_vq_train/t2m/)
# M2T
python eval_m2t.py --model_name ./m2t-ft-from-GSPretrained-base
# M2DT
python eval_m2dt.py --model_name ./m2dt-ft-from-GSPretrained-base

# 2D scratch (full VQ-VAE trained from scratch on 2D data)
# M2T
python eval_m2t.py --model_name ./m2t-2drecon-100k/checkpoint-100000 \
    --vqvae_2drecon_ckpt <path/to/2d_scratch_encoder.tar>
# M2DT
python eval_m2dt.py --model_name ./m2dt-2drecon/checkpoint-300000 \
    --vqvae_2drecon_ckpt <path/to/2d_scratch_encoder.tar>
```

<details><summary>Paper Results (HumanML3D test set)</summary>

**M2T:**

| Method | R-Prec@1↑ | MM-Dist↓ | BLEU-1↑ | BLEU-4↑ | ROUGE-L↑ | CIDEr↑ | BERTScore↑ |
|---|---|---|---|---|---|---|---|
| 3D baseline | **0.583** | **2.571** | **0.507** | **0.091** | **0.402** | **0.095** | **0.386** |
| 2D scratch | 0.487 | 3.289 | 0.494 | 0.078 | 0.385 | 0.074 | 0.364 |
| 2D encoder (Ours) | **0.583** | 2.672 | 0.499 | 0.086 | 0.395 | 0.090 | 0.379 |

**M2DT (sequence-level):**

| Method | BLEU-1↑ | BLEU-4↑ | ROUGE-L↑ | CIDEr↑ | BERTScore↑ |
|---|---|---|---|---|---|
| 3D baseline | 0.828 | **0.667** | **0.651** | **0.099** | **0.523** |
| 2D scratch | **0.814** | 0.637 | 0.629 | 0.229 | 0.483 |
| 2D encoder (Ours) | **0.829** | 0.665 | 0.648 | 0.024 | 0.517 |

</details>

---

### 2. Real-world Evaluation

Requires the real-world video dataset (see Dataset Preparation).

**2DMotionGPT:**
```bash
cd 2DMotionGPT
# ViTPose 2D + 2D Adapter (main result)
python evaluate_with_realworld.py \
    --cfg configs/config_h3d_stage1.yaml --nodebug \
    --estimated_motion_dir <path/to/real_world_dataset> \
    --vqvae_ckpt <path/to/2d_encoder.tar> \
    --adapter_ckpt <path/to/2d_adapter.tar> \
    --adapter_type residual \
    --mode both
# WHAM 3D + 3D Adapter
python evaluate_with_realworld_3d_adapter.py \
    --cfg configs/config_h3d_stage1.yaml --nodebug \
    --estimated_motion_dir <path/to/real_world_dataset> \
    --adapter_ckpt <path/to/3d_adapter.tar> \
    --split test --mode both
```

<details><summary>Paper Results (132 real-world samples)</summary>

| Method | BLEU-1↑ | BLEU-4↑ | ROUGE-L↑ | CIDEr↑ | BERTScore↑ |
|---|---|---|---|---|---|
| 2D encoder (no adapter) | 0.3614 | 0.0468 | 0.3209 | 0.0745 | 0.2362 |
| 2D + Adapter (Ours) | **0.4166** | **0.0568** | **0.3611** | **0.1015** | **0.2884** |
| WHAM 3D (no adapter) | 0.3726 | 0.0294 | 0.3299 | 0.0501 | 0.2652 |
| WHAM + 3D Adapter | 0.4084 | 0.0458 | 0.3653 | 0.1024 | 0.2853 |
| GT 3D (upper bound) | 0.4611 | 0.0691 | 0.4050 | 0.1490 | 0.3521 |

</details>

---

**2DTM2T:**
```bash
cd 2DTM2T
# ViTPose 2D + 2D Adapter (main result)
python evaluate_with_realworld_tm2t.py \
    --estimated_motion_dir <path/to/real_world_dataset> \
    --motiongpt_ckpt <path/to/motiongpt_s3_h3d.tar> \
    --vqvae_2d_ckpt <path/to/2d_encoder.tar> \
    --adapter_ckpt <path/to/2d_adapter.tar> \
    --adapter_type residual \
    --m2t_name M2T_MotionGPT_EL4_DL4_NH8_PS \
    --mode both --gpu_id 0 --split test
# WHAM 3D + 3D Adapter
python evaluate_with_realworld_3d_adapter_tm2t.py \
    --estimated_motion_dir <path/to/real_world_dataset> \
    --adapter_ckpt <path/to/3d_adapter.tar> \
    --gpu_id 0 --mode both
```

<details><summary>Paper Results (132 real-world samples)</summary>

| Method | BLEU-1↑ | BLEU-4↑ | ROUGE-L↑ | CIDEr↑ | BERTScore↑ |
|---|---|---|---|---|---|
| 2D encoder (no adapter) | 0.5541 | 0.1471 | 0.4034 | 0.4028 | 0.2797 |
| 2D + Adapter (Ours) | **0.6186** | **0.2183** | **0.4614** | **0.6453** | **0.3444** |
| WHAM 3D (no adapter) | 0.4772 | 0.1046 | 0.3600 | 0.2617 | 0.2221 |
| WHAM + 3D Adapter | 0.5875 | 0.1661 | 0.4358 | 0.4856 | 0.3037 |
| GT 3D (upper bound) | 0.5987 | 0.1897 | 0.4604 | 0.6089 | 0.3541 |

</details>

---

**2DMG-MotionLLM:**
```bash
cd 2DMG-MotionLLM
# ViTPose 2D + 2D Adapter (main result)
python evaluate_with_realworld.py \
    --estimated_motion_dir <path/to/real_world_dataset> \
    --adapter_ckpt <path/to/2d_adapter.pt> \
    --adapter_type residual \
    --meta_dir <path/to/meta_dir> \
    --data_root <path/to/HumanML3D> \
    --split test --mode both --gpu_id 0
# WHAM 3D + 3D Adapter
python evaluate_with_realworld_3d_adapter.py \
    --estimated_motion_dir <path/to/real_world_dataset> \
    --adapter_ckpt <path/to/3d_adapter.pt> \
    --split test --gpu_id 0 --mode both
```

<details><summary>Paper Results (129 real-world samples)</summary>

| Method | BLEU-1↑ | BLEU-4↑ | ROUGE-L↑ | CIDEr↑ | BERTScore↑ |
|---|---|---|---|---|---|
| 2D encoder (no adapter) | 0.3919 | 0.0441 | 0.3213 | 0.0619 | 0.2641 |
| 2D + Adapter (Ours) | **0.5065** | **0.0926** | **0.4113** | **0.1505** | **0.3738** |
| WHAM 3D (no adapter) | 0.4327 | 0.0640 | 0.3499 | 0.0948 | 0.3081 |
| WHAM + 3D Adapter | 0.4688 | 0.0355 | 0.3784 | 0.1060 | 0.3517 |
| GT 3D (upper bound) | 0.5556 | 0.1170 | 0.4496 | 0.2331 | 0.4359 |

</details>

---

## Training

The training pipeline consists of three steps: (1) train the 2D encoder, (2) train the 2D adapter for ViTPose inputs, and (3) train the 3D adapter for WHAM inputs. Note that **2DTM2T reuses the 2D encoder and adapters trained in 2DMotionGPT** and does not require its own training.

### Step 1: Train 2D Encoder

**2DMotionGPT** (encoder-only; also used by 2DTM2T):
```bash
cd 2DMotionGPT
python train_2d_vqvae_ver4_coco_normalized.py \
    --cfg configs/config_h3d_stage1.yaml --nodebug
```

**2DMG-MotionLLM** (trains its own 2D encoder independently):
```bash
cd 2DMG-MotionLLM
python 2d_vq_train.py --dataname t2m
```

### Step 2: Train 2D Adapter (ViTPose → latent space)

**2DMotionGPT** (adapter also used by 2DTM2T):
```bash
cd 2DMotionGPT
python -m train_adapter \
    --cfg configs/config_h3d_stage1.yaml --nodebug \
    --vqvae_ckpt <path/to/2d_encoder.tar> \
    --estimated_motion_dir <path/to/humanml3d_vitpose_json_root>
```

**2DMG-MotionLLM**:
```bash
cd 2DMG-MotionLLM
python train_adapter.py \
    --vqvae_2d_ckpt <path/to/2d_encoder.pt> \
    --estimated_motion_dir <path/to/humanml3d_vitpose_json_root> \
    --meta_dir <path/to/meta_dir> \
    --data_root <path/to/HumanML3D> \
    --adapter_type residual \
    --max_epochs 3000 --batch_size 64
```

### Step 3: Train 3D Adapter (WHAM → latent space)

**2DMotionGPT** (adapter also used by 2DTM2T):
```bash
cd 2DMotionGPT
python train_adapter_3d.py \
    --cfg configs/config_h3d_stage1.yaml --nodebug \
    --estimated_motion_dir <path/to/humanml3d_wham_root>
```

**2DMG-MotionLLM**:
```bash
cd 2DMG-MotionLLM
python train_adapter_3d.py \
    --estimated_motion_dir <path/to/humanml3d_wham_root> \
    --val_every 10
```

## Acknowledgements

This work builds on HumanML3D, TM2T, MotionGPT, MG-MotionLLM, and FineMotion. We thank the authors for their contributions.
