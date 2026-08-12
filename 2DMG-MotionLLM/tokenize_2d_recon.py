"""
Tokenize HumanML3D motions with 2D COCO projection + MotionGPT 2D Reconstruction VQ-VAE.

Output: dataset/HumanML3D/{token_dir}/*.npy  shape=(num_replics, T)  dtype=int64

Usage:
    # for M2T (random crop per replication)
    python tokenize_2d_recon.py --gpu_id 0 --dataset_name t2m \
        --token_dir VQVAE_2DRecon \
        --recon_ckpt /path/to/best_vqvae_epoch2990_valloss0.0354.tar

    # for M2DT (start from frame 0, saves to VQVAE_2DRecon_start0)
    python tokenize_2d_recon.py --gpu_id 0 --dataset_name t2m \
        --token_dir VQVAE_2DRecon_start0 \
        --recon_ckpt /path/to/best_vqvae_epoch2990_valloss0.0354.tar \
        --start_from_zero
"""

import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import codecs as cs
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
MOTIONGPT_DIR = ROOT_DIR.parent / '2DMotionGPT'
sys.path.insert(0, str(MOTIONGPT_DIR))

from src.models.mgpt_vq import VQVae


# ---------------------------------------------------------------------------
# 2D projection pipeline  (identical to 2d_vq_train.py)
# ---------------------------------------------------------------------------

def _build_global_joints(root_y_position, joints_positions, root_linear_velocity,
                         vel_scale=1.0):
    T = joints_positions.shape[0]
    joints_local = joints_positions.reshape(T, 21, 3).copy()
    root_delta = root_linear_velocity * vel_scale
    root_pos_xz = np.cumsum(root_delta, axis=0) - np.cumsum(root_delta, axis=0)[:1]
    hips = np.zeros((T, 3), dtype=joints_positions.dtype)
    hips[:, 0] = root_pos_xz[:, 0]
    hips[:, 1] = root_y_position[:, 0]
    hips[:, 2] = root_pos_xz[:, 1]
    joints_no_hips = joints_local.copy()
    joints_no_hips[:, :, 0] += root_pos_xz[:, 0:1]
    joints_no_hips[:, :, 2] += root_pos_xz[:, 1:2]
    out = np.zeros((T, 22, 3), dtype=joints_positions.dtype)
    out[:, 0, :] = hips
    out[:, 1:, :] = joints_no_hips
    return out


def _convert_smpl22_to_coco(smpl):
    SMPL22 = ['pelvis', 'left_hip_extra', 'right_hip_extra', 'spine_1',
              'left_knee', 'right_knee', 'spine_2', 'left_ankle', 'right_ankle',
              'spine_3', 'left_foot', 'right_foot', 'neck', 'left_collar',
              'right_collar', 'nose', 'left_shoulder', 'right_shoulder',
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
    COCO = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip_extra', 'right_hip_extra',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    out = np.zeros((smpl.shape[0], len(COCO), 3))
    for t in range(smpl.shape[0]):
        for idx, joint in enumerate(smpl[t]):
            ci = COCO.index(SMPL22[idx]) if SMPL22[idx] in COCO else -1
            if ci != -1:
                out[t, ci] = joint
    return out


def _build_2D_joints(joints_global, yaw_deg=0.0, pitch_deg=0.0, invert_y=True):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    R = Rx @ Ry
    center = joints_global.reshape(-1, 3).mean(axis=0)
    joints_cam = (joints_global - center) @ R.T
    x = joints_cam[..., 0]
    y = -joints_cam[..., 1] if invert_y else joints_cam[..., 1]
    return np.stack([x, y], axis=-1), R


def _normalize_2d_coco13_midhip(joints_2d, eps=1e-8, q=99):
    COCO13 = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip_extra', 'right_hip_extra',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    joints_2d = np.asarray(joints_2d)
    n2i = {n: i for i, n in enumerate(COCO13)}
    root_pos = 0.5 * (joints_2d[:, n2i['left_hip_extra'], :]
                      + joints_2d[:, n2i['right_hip_extra'], :])
    joints_rel = joints_2d - root_pos[:, None, :]
    abs_xy = np.abs(joints_rel).reshape(-1, 2)
    s = max(np.percentile(abs_xy[:, 0], q), np.percentile(abs_xy[:, 1], q), eps)
    return root_pos, joints_rel, s


def _decompose_2d(joints_2d):
    root_pos, joints_rel, s = _normalize_2d_coco13_midhip(joints_2d)
    root_y = (root_pos[:, 1:2] / s).astype(np.float32)
    root_y -= root_y[:1]
    joints_pos = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)
    root_norm = (root_pos / s).astype(np.float32)
    root_vel = np.zeros_like(root_norm)
    root_vel[1:] = root_norm[1:] - root_norm[:-1]
    return root_y, joints_pos, root_vel


def _joint_features_2d(joints_2d):
    _, joints_rel, s = _normalize_2d_coco13_midhip(joints_2d)
    jn = (joints_rel / s).astype(np.float32)
    rot = np.arctan2(jn[:, :, 1], jn[:, :, 0]).astype(np.float32)
    vel = np.zeros_like(jn)
    vel[1:] = jn[1:] - jn[:-1]
    return rot, vel.reshape(jn.shape[0], -1)


def create_2d_features(motion, yaw_deg, pitch_deg):
    """3D HumanML3D motion (T, 263) → 2D COCO13 features (T, 68)."""
    root_y  = motion[:, 3:4]
    joints  = motion[:, 4:4 + 21 * 3]
    root_v  = motion[:, 1:3]

    jg = _build_global_joints(root_y, joints, root_v)
    jg = _convert_smpl22_to_coco(jg)
    # drop eye/ear rows (indices 1-4), keep nose (0) + shoulders onward (5-)
    jg = np.concatenate([jg[:, :1, :], jg[:, 5:, :]], axis=1)  # (T, 13, 3)

    j2d, _ = _build_2D_joints(jg, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    root_y2d, joints_pos, root_vel = _decompose_2d(j2d)
    rot, vel = _joint_features_2d(j2d)
    return np.concatenate([root_vel, root_y2d, joints_pos, rot, vel], axis=-1)  # (T, 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_id',        type=int,  default=0)
    p.add_argument('--dataset_name',  type=str,  default='t2m',
                   choices=['t2m', 'kit'])
    p.add_argument('--token_dir',     type=str,  default='VQVAE_2DRecon',
                   help='Sub-directory under dataset root where .npy files are saved')
    p.add_argument('--recon_ckpt',    type=str,  required=True,
                   help='Path to full 2D Reconstruction VQ-VAE checkpoint (.tar)')
    p.add_argument('--num_replics',   type=int,  default=5,
                   help='Number of random projections per motion')
    p.add_argument('--start_from_zero', action='store_true',
                   help='Always crop from frame 0 (use for M2DT / VQVAE_start0)')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device('cpu' if args.gpu_id == -1 else f'cuda:{args.gpu_id}')
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    if args.dataset_name == 't2m':
        data_root  = ROOT_DIR / 'dataset' / 'HumanML3D'
        motion_dir = data_root / 'new_joint_vecs'
        min_len, max_len = 40, 200
        dim_feat = 263
    elif args.dataset_name == 'kit':
        data_root  = ROOT_DIR / 'dataset' / 'KIT-ML'
        motion_dir = data_root / 'new_joint_vecs'
        min_len, max_len = 24, 200
        dim_feat = 251

    mean_2d = np.load(ROOT_DIR / f'{args.dataset_name}_motion_2d_mean.npy')  # (68,)
    std_2d  = np.load(ROOT_DIR / f'{args.dataset_name}_motion_2d_std.npy')   # (68,)

    # ------------------------------------------------------------------
    # Load VQ-VAE (full 2D Reconstruction weights)
    # ------------------------------------------------------------------
    vqvae = VQVae(
        nfeats=dim_feat,
        quantizer='ema_reset',
        code_num=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        norm='none',
        activation='relu',
    ).to(device)

    print(f'Loading 2D Reconstruction checkpoint: {args.recon_ckpt}')
    ckpt = torch.load(args.recon_ckpt, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    vqvae.load_state_dict(state)
    vqvae.eval()

    # ------------------------------------------------------------------
    # Motion list (all splits)
    # ------------------------------------------------------------------
    all_split = data_root / 'all.txt'
    with cs.open(str(all_split), 'r') as f:
        id_list = [l.strip() for l in f.readlines()]

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    token_dir = data_root / args.token_dir
    token_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {token_dir}')

    unit_length = 4  # each token = 4 frames (down_t=2, stride_t=2)
    skipped = 0

    with torch.no_grad():
        for name in tqdm(id_list):
            motion_path = motion_dir / f'{name}.npy'
            if not motion_path.exists():
                skipped += 1
                continue
            motion = np.load(str(motion_path))
            m_len = len(motion)
            if m_len < min_len or m_len >= max_len:
                skipped += 1
                continue

            m_len_aligned = (m_len // unit_length) * unit_length

            replics = []
            for _ in range(args.num_replics):
                yaw   = random.choice(range(-180, 180))
                pitch = random.choice(range(0, 60))

                if args.start_from_zero:
                    idx = 0
                else:
                    idx = random.randint(0, m_len - m_len_aligned)

                crop = motion[idx: idx + m_len_aligned]

                feat_2d = create_2d_features(crop, yaw_deg=yaw, pitch_deg=pitch)
                feat_2d = (feat_2d - mean_2d) / std_2d  # (T, 68)

                # Zero-pad to dim_feat so VQ-VAE input width matches
                pad = np.zeros((feat_2d.shape[0], dim_feat - feat_2d.shape[1]),
                               dtype=np.float32)
                feat_2d_padded = np.concatenate([feat_2d, pad], axis=-1)  # (T, 263)

                inp = torch.from_numpy(feat_2d_padded).float().unsqueeze(0).to(device)
                indices, _ = vqvae.encode(inp)           # (1, T//4)
                replics.append(indices[0].cpu().numpy())  # (T//4,)

            tokens = np.stack(replics, axis=0).astype(np.int64)  # (num_replics, T//4)
            np.save(str(token_dir / f'{name}.npy'), tokens)

    print(f'Done. Skipped {skipped} motions (length out of range or file missing).')
    print(f'Tokens saved to: {token_dir}')


if __name__ == '__main__':
    main()
