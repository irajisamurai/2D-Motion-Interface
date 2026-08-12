import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from os.path import join as pjoin
import random

import numpy as np
import torch
import codecs as cs
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import DataLoader

import utils.paramUtil as paramUtil
from options.train_options import TrainVQTokenizerOptions
from src.models.mgpt_vq import VQVae


# ---------------------------------------------------------------------------
# Dataset: 3D HumanML3D motion → 2D COCO keypoints → normalized features
# Same pipeline as tokenize_script_motiongpt_2d.py.
# ---------------------------------------------------------------------------

class Motion2DTokenizeDataset(data.Dataset):
    def __init__(self, opt, mean, std, mean_2d, std_2d, split_file):
        self.opt = opt
        self.mean_2d = mean_2d
        self.std_2d = std_2d
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
        max_motion_len = 200

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= max_motion_len):
                    continue
                data_dict[name] = {'motion': motion, 'length': len(motion), 'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                pass

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        motion_2d = self._create_2d_joints_from_features(motion)
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d
        return motion_2d, name

    # ------------------------------------------------------------------
    # 2D projection pipeline
    # ------------------------------------------------------------------

    def _create_2d_joints_from_features(self, motion):
        root_y_position = motion[:, 3].reshape(-1, 1)
        joints_positions = motion[:, 4:4 + 21 * 3]
        root_linear_velocity = motion[:, 1:3]

        joints_global = self._build_global_joints(
            root_y_position, joints_positions, root_linear_velocity)
        joints_global = self._convert_smpl22_to_coco(joints_global)
        joints_global = np.concatenate([
            joints_global[:, 0, :].reshape(joints_global.shape[0], 1, joints_global.shape[2]),
            joints_global[:, 5:, :]
        ], axis=1)

        random_yaw   = random.choice(range(-180, 180))
        random_pitch = random.choice(range(0, 60))
        joints_2d, _ = self._build_2D_joints(
            joints_global, yaw_deg=random_yaw, pitch_deg=random_pitch)

        root_y_2d, joints_pos_2d, root_vel_2d = \
            self._decompose_2d_motion_coco13_midhip_root(joints_2d)
        joints_rot_2d, joints_vel_2d = self._compute_joint_features_2d_coco13(joints_2d)

        return np.concatenate(
            [root_vel_2d, root_y_2d, joints_pos_2d, joints_rot_2d, joints_vel_2d], axis=-1)

    def _build_global_joints(self, root_y_position, joints_positions,
                              root_linear_velocity, vel_scale=1.0):
        T = joints_positions.shape[0]
        n_joints_no_hips = 21
        joints_local = joints_positions.reshape(T, n_joints_no_hips, 3).copy()

        root_delta = root_linear_velocity * vel_scale
        root_pos_xz = np.cumsum(root_delta, axis=0)
        root_pos_xz = root_pos_xz - root_pos_xz[0:1]

        hips_global = np.zeros((T, 3), dtype=joints_positions.dtype)
        hips_global[:, 0] = root_pos_xz[:, 0]
        hips_global[:, 1] = root_y_position[:, 0]
        hips_global[:, 2] = root_pos_xz[:, 1]

        joints_global_no_hips = joints_local.copy()
        joints_global_no_hips[:, :, 0] += root_pos_xz[:, 0:1]
        joints_global_no_hips[:, :, 2] += root_pos_xz[:, 1:2]

        joints_global_all = np.zeros((T, n_joints_no_hips + 1, 3),
                                     dtype=joints_positions.dtype)
        joints_global_all[:, 0, :] = hips_global
        joints_global_all[:, 1:, :] = joints_global_no_hips
        return joints_global_all

    def _convert_smpl22_to_coco(self, smpl_keypoints):
        SMPL22 = ['pelvis', 'left_hip_extra', 'right_hip_extra', 'spine_1',
                  'left_knee', 'right_knee', 'spine_2', 'left_ankle', 'right_ankle',
                  'spine_3', 'left_foot', 'right_foot', 'neck', 'left_collar',
                  'right_collar', 'nose', 'left_shoulder', 'right_shoulder',
                  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        COCO = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip_extra', 'right_hip_extra',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        coco_kps = np.zeros((smpl_keypoints.shape[0], len(COCO), 3))
        for t in range(smpl_keypoints.shape[0]):
            for idx, joint in enumerate(smpl_keypoints[t]):
                coco_idx = COCO.index(SMPL22[idx]) if SMPL22[idx] in COCO else -1
                if coco_idx != -1:
                    coco_kps[t, coco_idx] = joint
        return coco_kps

    def _build_2D_joints(self, joints_global, yaw_deg=0.0, pitch_deg=0.0, invert_y=True):
        yaw   = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                       [ 0,           1, 0           ],
                       [-np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1,             0,              0           ],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch),  np.cos(pitch)]])
        R = Rx @ Ry
        center = joints_global.reshape(-1, 3).mean(axis=0)
        joints_cam = (joints_global - center) @ R.T
        x = joints_cam[..., 0]
        y = -joints_cam[..., 1] if invert_y else joints_cam[..., 1]
        return np.stack([x, y], axis=-1), R

    def _normalize_2d_coco13_midhip(self, joints_2d, eps=1e-8, q=99):
        COCO13 = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip_extra', 'right_hip_extra',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        joints_2d = np.asarray(joints_2d)
        name2idx = {n: i for i, n in enumerate(COCO13)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']
        root_pos  = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])
        joints_rel = joints_2d - root_pos[:, None, :]
        abs_xy = np.abs(joints_rel).reshape(-1, 2)
        s = max(np.percentile(abs_xy[:, 0], q), np.percentile(abs_xy[:, 1], q), eps)
        return root_pos, joints_rel, s

    def _decompose_2d_motion_coco13_midhip_root(self, joints_2d):
        root_pos, joints_rel, s = self._normalize_2d_coco13_midhip(joints_2d)
        root_y_2d = (root_pos[:, 1:2] / s).astype(np.float32)
        root_y_2d = root_y_2d - root_y_2d[0:1]
        joints_pos_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)
        root_norm = (root_pos / s).astype(np.float32)
        root_vel_2d = np.zeros_like(root_norm)
        root_vel_2d[1:] = root_norm[1:] - root_norm[:-1]
        return root_y_2d, joints_pos_2d, root_vel_2d

    def _compute_joint_features_2d_coco13(self, joints_2d):
        _, joints_rel, s = self._normalize_2d_coco13_midhip(joints_2d)
        joints_rel_norm = (joints_rel / s).astype(np.float32)
        rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)
        vel = np.zeros_like(joints_rel_norm)
        vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
        vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)
        return rot, vel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = TrainVQTokenizerOptions()
    parser.parser.add_argument(
        '--recon_ckpt',
        type=str,
        required=True,
        help='Path to full 2D Reconstruction VQ-VAE checkpoint (.tar). '
             'encoder + quantizer + decoder are all loaded from this file.',
    )
    opt = parser.parse()
    opt.is_train = False

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    if opt.dataset_name == 't2m':
        opt.data_root = str(ROOT_DIR / 'dataset' / 'HumanML3D')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = str(ROOT_DIR / 'dataset' / 'KIT-ML')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    mgpt_meta_dir = ROOT_DIR / 'checkpoints' / opt.dataset_name / 'VQVAEV3_CB1024_CMT_H1024_NRES3' / 'meta'
    mean    = np.load(str(mgpt_meta_dir / 'mean.npy'))
    std     = np.load(str(mgpt_meta_dir / 'std.npy'))
    mean_2d = np.load(str(mgpt_meta_dir / 'mean_2d_coco_normalized.npy'))
    std_2d  = np.load(str(mgpt_meta_dir / 'std_2d_coco_normalized.npy'))

    all_split_file = pjoin(opt.data_root, 'all.txt')

    # Load full 2D Reconstruction VQ-VAE (encoder + quantizer + decoder)
    vqvae = VQVae(
        nfeats=263,
        quantizer="ema_reset",
        code_num=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        norm="none",
        activation="relu",
    ).to(opt.device)

    print(f'Loading 2D Reconstruction checkpoint from {opt.recon_ckpt}')
    ckpt = torch.load(opt.recon_ckpt, map_location='cpu', weights_only=False)
    recon_state = ckpt.get('model_state_dict', ckpt)
    vqvae.load_state_dict(recon_state)
    vqvae.eval()

    all_dataset = Motion2DTokenizeDataset(opt, mean, std, mean_2d, std_2d, all_split_file)
    all_loader = DataLoader(all_dataset, batch_size=1, num_workers=1, pin_memory=True)

    token_data_dir = pjoin(opt.data_root, opt.name)
    os.makedirs(token_data_dir, exist_ok=True)
    print(f'Token output directory: {token_data_dir}')

    num_replics = 5
    opt.unit_length = 4

    with torch.no_grad():
        for e in range(num_replics):
            print(f'Replication {e + 1}/{num_replics}')
            for i, batch in enumerate(tqdm(all_loader)):
                motion_2d, name = batch
                motion_2d = motion_2d.detach().float()
                pad = torch.zeros(
                    motion_2d.shape[0], motion_2d.shape[1],
                    263 - motion_2d.shape[2])
                motion_2d = torch.cat([motion_2d, pad], dim=-1).to(opt.device)

                indices, _ = vqvae.encode(motion_2d)
                indices = [str(t) for t in indices[0].cpu().numpy().tolist()]
                with cs.open(pjoin(token_data_dir, '%s.txt' % name[0]), 'a+') as f:
                    if e == num_replics - 1:
                        f.write(' '.join(indices))
                    else:
                        f.write(' '.join(indices) + '\n')
