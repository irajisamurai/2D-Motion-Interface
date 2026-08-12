import csv
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

import codecs as cs
import random
from os.path import join as pjoin

import numpy as np
import torch
import models.vqvae as vqvae
from options import option
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(args, run_name, group):
    if wandb is None:
        print("wandb is not installed. Training metrics will not be logged to wandb.")
        return None

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "2DMG-MotionLLM"),
        group=os.environ.get("WANDB_GROUP", group),
        name=os.environ.get("WANDB_RUN_NAME", run_name),
        config={
            "dataname": args.dataname,
            "vqvae_pth": args.vqvae_pth,
            "code_dim": args.code_dim,
            "nb_code": args.nb_code,
            "mu": args.mu,
            "down_t": args.down_t,
            "stride_t": args.stride_t,
            "width": args.width,
            "depth": args.depth,
            "dilation_growth_rate": args.dilation_growth_rate,
            "output_emb_width": args.output_emb_width,
            "vq_act": args.vq_act,
            "seed": args.seed,
            "window_size": args.window_size,
            "quantizer": args.quantizer,
            "quantbeta": args.quantbeta,
            "epochs": 3000,
            "train_batch_size": 64,
            "val_batch_size": 32,
            "eval_interval": 10,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
    )


def save_checkpoint(checkpoint_path, epoch, model, optimizer, avg_train_loss,
                    avg_val_loss, same_code_index_ratio):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "same_code_index_ratio": same_code_index_ratio,
        },
        checkpoint_path,
    )



args = option.get_args_parser()
args = args.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)



def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def humanml3d_collate_2d(batch):
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][6] is None else True

    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[6], reverse=True)

    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "motion_2d":
        collate_tensors([torch.tensor(b[2]).float() for b in notnone_batches]),
        "length": [b[3] for b in notnone_batches],
    }

    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[8] for b in notnone_batches],
        })
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[4]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[5]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[6]) for b in notnone_batches]),
            "tokens": [b[7] for b in notnone_batches],
        })

    if len(notnone_batches[0]) == 10:
        adapted_batch.update({"tasks": [b[9] for b in notnone_batches]})

    return adapted_batch


class VQMotion_test_Dataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias=5, window_size=64, unit_length=8):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.dataset_name = dataset_name
        min_motion_len = 40 if dataset_name == 't2m' else 24

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            self.meta_dir = './checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = './checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        train_split_file = pjoin(self.data_root, 'test.txt')
        val_split_file = pjoin(self.data_root, 'val.txt')

        data_dict = {}
        id_list = []
        with cs.open(train_split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'name': name
                }
                new_name_list.append(name)
                length_list.append(len(motion))
            except Exception:
                pass

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list
        self.mean_2d = np.load(ROOT_DIR / f"{dataset_name}_motion_2d_mean.npy")
        self.std_2d = np.load(ROOT_DIR / f"{dataset_name}_motion_2d_std.npy")

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion_2d = self.create_2d_joints_from_features(motion)
        motion = motion[idx:idx + m_length]
        motion_2d = motion_2d[idx:idx + m_length]
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        return None, motion, motion_2d, m_length, None, None, None, None

    def create_2d_joints_from_features(self, motion):
        root_y_position = motion[:, 3].reshape(-1, 1)
        joints_positions = motion[:, 4:4 + 21 * 3]
        root_linear_velocity = motion[:, 1:3]
        joints_global = self.build_global_joints(
            root_y_position,
            joints_positions,
            root_linear_velocity,
            vel_scale=1.0,
        )
        joints_global = self.convert_smpl22_to_coco(joints_global)
        joints_global = np.concatenate([
            joints_global[:, 0, :].reshape(joints_global.shape[0], 1,
                                           joints_global.shape[2]),
            joints_global[:, 5:, :]
        ],
                                       axis=1)

        random_yaw = random.choice([i for i in range(-180, 180, 1)])
        random_pitch = random.choice([i for i in range(0, 60, 1)])

        joints_2d, _ = self.build_2D_joints(
            joints_global,
            yaw_deg=random_yaw,
            pitch_deg=random_pitch,
        )
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(
            joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(
            joints_2d)
        result = np.concatenate(
            [root_vel_2d, root_y_2d, joints_pos_2d, joints_rot_2d, joints_vel_2d],
            axis=-1)
        return result

    def build_global_joints(self,
                            root_y_position,
                            joints_positions,
                            root_linear_velocity,
                            vel_scale=1.0):
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

    def convert_smpl22_to_coco(self, smpl_keypoints):
        SMPL22_KEYPOINTS = [
            'pelvis', 'left_hip_extra', 'right_hip_extra', 'spine_1',
            'left_knee', 'right_knee', 'spine_2', 'left_ankle', 'right_ankle',
            'spine_3', 'left_foot', 'right_foot', 'neck', 'left_collar',
            'right_collar', 'nose', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
        COCO_KEYPOINTS = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip_extra', 'right_hip_extra',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        coco_keypoints = np.zeros(
            (smpl_keypoints.shape[0], len(COCO_KEYPOINTS), 3))
        for t in range(smpl_keypoints.shape[0]):
            for idx, joint in enumerate(smpl_keypoints[t]):
                coco_idx = COCO_KEYPOINTS.index(
                    SMPL22_KEYPOINTS[idx]
                ) if SMPL22_KEYPOINTS[idx] in COCO_KEYPOINTS else -1
                if coco_idx != -1:
                    coco_keypoints[t, coco_idx] = joint
        return coco_keypoints

    def build_2D_joints(self,
                        joints_global,
                        yaw_deg=0.0,
                        pitch_deg=0.0,
                        invert_y=True):
        T, n_joints, _ = joints_global.shape

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)

        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])

        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])

        R = Rx @ Ry

        center = joints_global.reshape(-1, 3).mean(axis=0)

        joints_rel = joints_global - center
        joints_cam = joints_rel @ R.T

        x = joints_cam[..., 0]
        y = joints_cam[..., 1]
        if invert_y:
            y = -y

        joints_2d = np.stack([x, y], axis=-1)

        return joints_2d, R

    def normalize_2d_coco13_midhip(self, joints_2d, eps=1e-8, q=99):
        coco_keypoints = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip_extra',
            'right_hip_extra', 'left_knee', 'right_knee', 'left_ankle',
            'right_ankle'
        ]
        joints_2d = np.asarray(joints_2d)
        T, J, D = joints_2d.shape
        assert (J, D) == (13, 2)

        name2idx = {n: i for i, n in enumerate(coco_keypoints)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']

        root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])

        joints_rel = joints_2d - root_pos[:, None, :]

        abs_xy = np.abs(joints_rel).reshape(-1, 2)
        sx = np.percentile(abs_xy[:, 0], q)
        sy = np.percentile(abs_xy[:, 1], q)
        s = max(sx, sy, eps)

        return root_pos, joints_rel, s

    def decompose_2d_motion_coco13_midhip_root(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d,
                                                                  q=99)

        root_y_position_2d = (root_pos[:, 1:2] / s).astype(np.float32)
        root_y_position_2d = root_y_position_2d - root_y_position_2d[0:1]

        joints_positions_2d = (joints_rel / s).reshape(joints_rel.shape[0],
                                                       -1).astype(np.float32)

        root_norm = (root_pos / s).astype(np.float32)
        root_linear_velocity_2d = np.zeros_like(root_norm)
        root_linear_velocity_2d[1:] = root_norm[1:] - root_norm[:-1]

        return root_y_position_2d, joints_positions_2d, root_linear_velocity_2d

    def compute_joint_features_2d_coco13(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d,
                                                                  q=99)

        joints_rel_norm = (joints_rel / s).astype(np.float32)

        rot = np.arctan2(joints_rel_norm[:, :, 1],
                         joints_rel_norm[:, :, 0]).astype(np.float32)

        vel = np.zeros_like(joints_rel_norm)
        vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
        vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)

        return rot, vel


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias=5, window_size=64, unit_length=8):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.dataset_name = dataset_name
        min_motion_len = 40 if dataset_name == 't2m' else 24

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            self.meta_dir = './checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = './checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        train_split_file = pjoin(self.data_root, 'train.txt')
        val_split_file = pjoin(self.data_root, 'val.txt')

        data_dict = {}
        id_list = []
        with cs.open(train_split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'name': name
                }
                new_name_list.append(name)
                length_list.append(len(motion))
            except Exception:
                pass

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list
        self.mean_2d = np.load(ROOT_DIR / f"{dataset_name}_motion_2d_mean.npy")
        self.std_2d = np.load(ROOT_DIR / f"{dataset_name}_motion_2d_std.npy")

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion_2d = self.create_2d_joints_from_features(motion)
        motion = motion[idx:idx + m_length]
        motion_2d = motion_2d[idx:idx + m_length]
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        return None, motion, motion_2d, m_length, None, None, None, None

    def create_2d_joints_from_features(self, motion):
        root_y_position = motion[:, 3].reshape(-1, 1)
        joints_positions = motion[:, 4:4 + 21 * 3]
        root_linear_velocity = motion[:, 1:3]
        joints_global = self.build_global_joints(
            root_y_position,
            joints_positions,
            root_linear_velocity,
            vel_scale=1.0,
        )
        joints_global = self.convert_smpl22_to_coco(joints_global)
        joints_global = np.concatenate([
            joints_global[:, 0, :].reshape(joints_global.shape[0], 1,
                                           joints_global.shape[2]),
            joints_global[:, 5:, :]
        ],
                                       axis=1)

        random_yaw = random.choice([i for i in range(-180, 180, 1)])
        random_pitch = random.choice([i for i in range(0, 60, 1)])

        joints_2d, _ = self.build_2D_joints(
            joints_global,
            yaw_deg=random_yaw,
            pitch_deg=random_pitch,
        )
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(
            joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(
            joints_2d)
        result = np.concatenate(
            [root_vel_2d, root_y_2d, joints_pos_2d, joints_rot_2d, joints_vel_2d],
            axis=-1)
        return result

    def build_global_joints(self,
                            root_y_position,
                            joints_positions,
                            root_linear_velocity,
                            vel_scale=1.0):
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

    def convert_smpl22_to_coco(self, smpl_keypoints):
        SMPL22_KEYPOINTS = [
            'pelvis', 'left_hip_extra', 'right_hip_extra', 'spine_1',
            'left_knee', 'right_knee', 'spine_2', 'left_ankle', 'right_ankle',
            'spine_3', 'left_foot', 'right_foot', 'neck', 'left_collar',
            'right_collar', 'nose', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
        COCO_KEYPOINTS = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip_extra', 'right_hip_extra',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        coco_keypoints = np.zeros(
            (smpl_keypoints.shape[0], len(COCO_KEYPOINTS), 3))
        for t in range(smpl_keypoints.shape[0]):
            for idx, joint in enumerate(smpl_keypoints[t]):
                coco_idx = COCO_KEYPOINTS.index(
                    SMPL22_KEYPOINTS[idx]
                ) if SMPL22_KEYPOINTS[idx] in COCO_KEYPOINTS else -1
                if coco_idx != -1:
                    coco_keypoints[t, coco_idx] = joint
        return coco_keypoints

    def build_2D_joints(self,
                        joints_global,
                        yaw_deg=0.0,
                        pitch_deg=0.0,
                        invert_y=True):
        T, n_joints, _ = joints_global.shape

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)

        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])

        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])

        R = Rx @ Ry

        center = joints_global.reshape(-1, 3).mean(axis=0)

        joints_rel = joints_global - center
        joints_cam = joints_rel @ R.T

        x = joints_cam[..., 0]
        y = joints_cam[..., 1]
        if invert_y:
            y = -y

        joints_2d = np.stack([x, y], axis=-1)

        return joints_2d, R

    def normalize_2d_coco13_midhip(self, joints_2d, eps=1e-8, q=99):
        coco_keypoints = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip_extra',
            'right_hip_extra', 'left_knee', 'right_knee', 'left_ankle',
            'right_ankle'
        ]
        joints_2d = np.asarray(joints_2d)
        T, J, D = joints_2d.shape
        assert (J, D) == (13, 2)

        name2idx = {n: i for i, n in enumerate(coco_keypoints)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']

        root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])

        joints_rel = joints_2d - root_pos[:, None, :]

        abs_xy = np.abs(joints_rel).reshape(-1, 2)
        sx = np.percentile(abs_xy[:, 0], q)
        sy = np.percentile(abs_xy[:, 1], q)
        s = max(sx, sy, eps)

        return root_pos, joints_rel, s

    def decompose_2d_motion_coco13_midhip_root(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d,
                                                                  q=99)

        root_y_position_2d = (root_pos[:, 1:2] / s).astype(np.float32)
        root_y_position_2d = root_y_position_2d - root_y_position_2d[0:1]

        joints_positions_2d = (joints_rel / s).reshape(joints_rel.shape[0],
                                                       -1).astype(np.float32)

        root_norm = (root_pos / s).astype(np.float32)
        root_linear_velocity_2d = np.zeros_like(root_norm)
        root_linear_velocity_2d[1:] = root_norm[1:] - root_norm[:-1]

        return root_y_position_2d, joints_positions_2d, root_linear_velocity_2d

    def compute_joint_features_2d_coco13(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d,
                                                                  q=99)

        joints_rel_norm = (joints_rel / s).astype(np.float32)

        rot = np.arctan2(joints_rel_norm[:, :, 1],
                         joints_rel_norm[:, :, 0]).astype(np.float32)

        vel = np.zeros_like(joints_rel_norm)
        vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
        vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)

        return rot, vel


dataset = VQMotionDataset(args.dataname, unit_length=2**args.down_t)
test_dataset = VQMotion_test_Dataset(args.dataname, unit_length=2**args.down_t)
train_loader = DataLoader(dataset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=8,
                          drop_last=True,
                          collate_fn=humanml3d_collate_2d)
val_loader = DataLoader(test_dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=8,
                        drop_last=False,
                        collate_fn=humanml3d_collate_2d)


"""all_motion_2d = []
for batch in tqdm(train_loader):
    motion_2d = batch['motion_2d']
    all_motion_2d.append(motion_2d.numpy())
all_motion_2d = np.concatenate(all_motion_2d, axis=0)
motion_2d_mean = np.mean(all_motion_2d, axis=(0,1))
motion_2d_std = np.std(all_motion_2d, axis=(0,1))

save_dir = "./"
np.save(pjoin(save_dir, f'{args.dataname}_motion_2d_mean.npy'), motion_2d_mean)
np.save(pjoin(save_dir, f'{args.dataname}_motion_2d_std.npy'), motion_2d_std)"""



net = vqvae.HumanVQVAE(
    args,
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
)

vqvae_pth = f"./checkpoints/pretrained_vqvae/{args.dataname}.pth"
print('loading checkpoint from {}'.format(vqvae_pth))
ckpt = torch.load(vqvae_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

target = vqvae.HumanVQVAE(
    args,
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
).cuda()



optimizer = torch.optim.AdamW(target.parameters(), lr=1e-4, weight_decay=0)
loss_fn = torch.nn.L1Loss()
run_name = f"2dvq-{args.dataname}-bs64-lr1e-4-seed{args.seed}"
group = "2DEncoder"
wandb_run = init_wandb(args, run_name, group)
checkpoint_dir = ROOT_DIR / "checkpoints" / "2d_vq_train" / args.dataname / f"seed{args.seed}"
os.makedirs(checkpoint_dir, exist_ok=True)
best_same_code_index_ratio = float("-inf")
best_ckpt_path = None

csv_log_path = checkpoint_dir / "training_log.csv"
with open(csv_log_path, "w", newline="") as _f:
    csv.writer(_f).writerow(["epoch", "train_loss", "val_loss", "val_same_code_index_ratio"])



for epoch in range(3000):
    target.train()
    total_train_loss = 0.0
    total_val_loss = 0.0

    for batch in tqdm(train_loader):
        motion = batch['motion'].cuda().transpose(1, 2).float()
        motion_2d = torch.cat([
            batch["motion_2d"],
            torch.zeros(batch["motion_2d"].shape[0], batch["motion_2d"].shape[1],
                        263 - batch["motion_2d"].shape[2])
        ],
                              dim=-1).cuda().transpose(1, 2).float()

        optimizer.zero_grad()

        with torch.no_grad():
            ref_encoded_motion = net.vqvae.encoder(motion)
        pred_encoded_motion = target.vqvae.encoder(motion_2d)
        loss = loss_fn(pred_encoded_motion, ref_encoded_motion)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    if wandb_run is not None:
        wandb_run.log({"train/loss": avg_train_loss}, step=epoch)

    csv_val_loss = ""
    csv_same_code_index_ratio = ""

    if epoch % 10 == 0 and epoch != 0:
        target.eval()
        all_code_idx = []
        all_code_idx_2d = []

        for batch in tqdm(val_loader):
            motion = batch['motion'].cuda().transpose(1, 2).float()
            motion_2d = torch.cat([
                batch["motion_2d"],
                torch.zeros(batch["motion_2d"].shape[0],
                            batch["motion_2d"].shape[1],
                            263 - batch["motion_2d"].shape[2])
            ],
                                  dim=-1).cuda().transpose(1, 2).float()

            with torch.no_grad():
                ref_encoded_motion = net.vqvae.encoder(motion)
                pred_encoded_motion = target.vqvae.encoder(motion_2d)
                loss = loss_fn(pred_encoded_motion, ref_encoded_motion)
                total_val_loss += loss.item()
                x_encoder = ref_encoded_motion
                x_encoder = net.vqvae.postprocess(x_encoder)
                x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])
                code_idx = net.vqvae.quantizer.quantize(x_encoder)
                code_idx = code_idx.reshape(-1)
                all_code_idx.append(code_idx.cpu().numpy())

                x_encoder_2d = pred_encoded_motion
                x_encoder_2d = net.vqvae.postprocess(x_encoder_2d)
                x_encoder_2d = x_encoder_2d.contiguous().view(
                    -1, x_encoder_2d.shape[-1])
                code_idx_2d = net.vqvae.quantizer.quantize(x_encoder_2d)
                code_idx_2d = code_idx_2d.reshape(-1)
                all_code_idx_2d.append(code_idx_2d.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        all_code_idx = np.concatenate(all_code_idx, axis=0)
        all_code_idx_2d = np.concatenate(all_code_idx_2d, axis=0)
        same_idx = (all_code_idx == all_code_idx_2d)
        same_code_index_ratio = float(same_idx.mean())
        csv_val_loss = avg_val_loss
        csv_same_code_index_ratio = same_code_index_ratio

        if wandb_run is not None:
            wandb_run.log(
                {
                    "val/loss": avg_val_loss,
                    "val/same_code_index_ratio": same_code_index_ratio,
                },
                step=epoch,
            )

        if same_code_index_ratio > best_same_code_index_ratio:
            best_same_code_index_ratio = same_code_index_ratio
            if best_ckpt_path is not None and best_ckpt_path.exists():
                os.remove(best_ckpt_path)
            best_ckpt_path = checkpoint_dir / f"best_2dvq_epoch{epoch+1}_ratio{same_code_index_ratio:.4f}.pt"
            save_checkpoint(
                best_ckpt_path,
                epoch + 1,
                target,
                optimizer,
                avg_train_loss,
                avg_val_loss,
                same_code_index_ratio,
            )
            print(f"Saved checkpoint to {best_ckpt_path}")

        print(f"Same code index ratio: {same_code_index_ratio:.4f}")
        print(
            f"Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}"
        )
    else:
        print(f"Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}")

    with open(csv_log_path, "a", newline="") as _f:
        csv.writer(_f).writerow([epoch + 1, avg_train_loss, csv_val_loss, csv_same_code_index_ratio])

if wandb_run is not None:
    wandb_run.finish()
