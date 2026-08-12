import random
import codecs as cs
import numpy as np
import os
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .humanml.dataset_t2m import Text2MotionDataset
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
import torch

class MotionDatasetVQ(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, length = data["motion"], data["length"]

        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]
        motion = (motion - self.mean) / self.std

        return None, motion, length, None, None, None, None,
    
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list

import random
import codecs as cs
import numpy as np
import os
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from src.data.humanml.dataset_t2m import Text2MotionDataset
from src.data.humanml.scripts.motion_process import (process_file, recover_from_ric)
import torch

class MotionDatasetVQ_2D(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d,
        std_2d,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        add_noise = False,
        add_stretch = False,
        sigma= 0.01,
        p_noise= 0.5,
        p_stretch= 0.5,
        stretch_range= [0.8, 1.25],
        alpha= 0.8,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list
        self.mean_2d = mean_2d
        self.std_2d = std_2d,
        self.split = split
        self.add_noise = False
        self.p_noise = p_noise
        self.add_stretch = False
        self.sigma = sigma
        self.p_stretch = p_stretch
        self.stretch_range = stretch_range
        self.alpha = alpha



    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, length = data["motion"], data["length"]

        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]
        motion_2d = self.create_2d_joints_from_features(motion)
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        return None, motion,motion_2d, length, None, None, None, None,
    
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
    
    def add_temporal_noise(self, joints_global, sigma=0.01, alpha=0.8):
        # joints_global: (T, J, 3)
        T, J, _ = joints_global.shape
        eps = np.zeros_like(joints_global)

        for t in range(1, T):
            eps[t] = alpha * eps[t-1] + (1 - alpha) * np.random.randn(J, 3) * sigma
        #eps[:, 0, :] = 0.0   
        return joints_global + eps
    
    def resample_time(self,x, T_new):
        """
        x: (T, ...) numpy
        return: (T_new, ...) numpy
        """
        T = x.shape[0]
        if T_new == T:
            return x

        t_old = np.linspace(0.0, 1.0, T)
        t_new = np.linspace(0.0, 1.0, T_new)

        x_flat = x.reshape(T, -1)
        y_flat = np.empty((T_new, x_flat.shape[1]), dtype=x.dtype)

        for d in range(x_flat.shape[1]):
            y_flat[:, d] = np.interp(t_new, t_old, x_flat[:, d])

        return y_flat.reshape((T_new,) + x.shape[1:])
    
    def time_stretch_to_fixed_len(self,x, T_out, rate):
        """
        """
        T = x.shape[0]
        T_mid = max(2, int(round(T * rate)))
        x_mid = self.resample_time(x, T_mid)
        x_out = self.resample_time(x_mid, T_out)
        return x_out



    def create_2d_joints_from_features(self,
                                       motion):
        root_y_position = motion[:,3].reshape(-1,1)
        joints_positions = motion[:,4:4+21*3]
        root_linear_velocity = motion[:,1:3]
        joints_global = self.build_global_joints(
            root_y_position,
            joints_positions,
            root_linear_velocity,
            vel_scale=1.0,
        )
        if self.split == "train" and self.add_noise and random.random() < self.p_noise:
            joints_global = self.add_temporal_noise(joints_global,sigma=self.sigma, alpha=self.alpha)
        if self.split == "train" and self.add_stretch and random.random() < self.p_stretch:
            rate = random.uniform(self.stretch_range[0], self.stretch_range[1])
            joints_global = self.time_stretch_to_fixed_len(
                joints_global, self.window_size, rate
            )


            
        random_yaw = random.choice([i for i in range(-180,180,1)])
        random_pitch = random.choice([i for i in range(0,60,1)])

        joints_2d, _ = self.build_2D_joints(
        joints_global,
        yaw_deg=random_yaw, pitch_deg=random_pitch,
        )
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_from_hips(joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d(joints_2d)
        result = np.concatenate([
            root_vel_2d,root_y_2d,joints_pos_2d,joints_rot_2d,joints_vel_2d],axis=-1)
        return result
    
    def build_global_joints(self,
                            root_y_position,
                        joints_positions,
                        root_linear_velocity,
                        vel_scale=1.0):
        T = joints_positions.shape[0]
        n_joints_no_hips = 21

        joints_local = joints_positions.reshape(T, n_joints_no_hips, 3).copy()

        root_delta = root_linear_velocity * vel_scale          # (T,2)
        root_pos_xz = np.cumsum(root_delta, axis=0)            # (T,2)
        root_pos_xz = root_pos_xz - root_pos_xz[0:1]

        hips_global = np.zeros((T, 3), dtype=joints_positions.dtype)
        hips_global[:, 0] = root_pos_xz[:, 0]          # x
        hips_global[:, 1] = root_y_position[:, 0]      # y
        hips_global[:, 2] = root_pos_xz[:, 1]          # z

        joints_global_no_hips = joints_local.copy()
        joints_global_no_hips[:, :, 0] += root_pos_xz[:, 0:1]
        joints_global_no_hips[:, :, 2] += root_pos_xz[:, 1:2]

        joints_global_all = np.zeros((T, n_joints_no_hips + 1, 3),
                                    dtype=joints_positions.dtype)
        joints_global_all[:, 0, :] = hips_global              # hips
        joints_global_all[:, 1:, :] = joints_global_no_hips   # 1〜21: leftUpLeg〜rightHand

        return joints_global_all
    
    def build_2D_joints(self,
                        joints_global,
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                    invert_y=True):
        T, n_joints, _ = joints_global.shape

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)

        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [ 0,           1, 0          ],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rx = np.array([
            [1,            0,             0          ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])

        R = Rx @ Ry

        center = joints_global.reshape(-1, 3).mean(axis=0)

        joints_rel = joints_global - center       # (T,22,3)
        joints_cam = joints_rel @ R.T             # (T,22,3)

        x = joints_cam[..., 0]
        y = joints_cam[..., 1]
        if invert_y:
            y = -y

        joints_2d = np.stack([x, y], axis=-1)     # (T,22,2)

        return joints_2d, R
    
    def decompose_2d_motion_from_hips(self,
                                      joints_2d):
        T, J, D = joints_2d.shape
        assert J == 22, f"Expected 22 joints (including hips), got {J}"
        assert D == 2,  f"Expected 2D joints, got {D} dimensions"

        hip_idx = 0

        root_pos_2d = joints_2d[:, hip_idx, :]  # (T,2)

        root_y_position_2d = root_pos_2d[:, 1:2]  # (T,1)

        joints_local_2d = joints_2d - root_pos_2d[:, None, :]  # (T,22,2)

        joints_local_wo_hips = joints_local_2d[:, 1:, :]   # (T,21,2)
        joints_positions_2d = joints_local_wo_hips.reshape(T, -1)  # (T, 21*2) = (T, 42)

        root_linear_velocity_2d = np.zeros_like(root_pos_2d)
        root_linear_velocity_2d[1:] = root_pos_2d[1:] - root_pos_2d[:-1]

        return root_y_position_2d, joints_positions_2d, root_linear_velocity_2d
    
    def compute_joint_features_2d(self,
                                  joints_2d):
        T, J, D = joints_2d.shape
        assert J == 22 and D == 2

        hips = joints_2d[:, 0, :]               # (T,2)
        joints_rel = joints_2d[:, 1:, :] - hips[:, None, :]  # (T,21,2)

        # ---- Rotation angles (atan2)
        rot = np.arctan2(joints_rel[:, :, 1], joints_rel[:, :, 0])  # (T,21)

        # ---- Velocities of joints
        vel = np.zeros_like(joints_rel)
        vel[1:] = joints_rel[1:] - joints_rel[:-1]  # (T,21,2)
        vel = vel.reshape(T, -1)  # (T,42)

        return rot, vel
    
import random
import codecs as cs
import numpy as np
import os
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from src.data.humanml.dataset_t2m import Text2MotionDataset
from src.data.humanml.scripts.motion_process import (process_file, recover_from_ric)
import torch

class MotionDatasetV_2DCOCO(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d_coco,
        std_2d_coco,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list
        self.mean_2d = mean_2d_coco
        self.std_2d = std_2d_coco

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, length = data["motion"], data["length"]

        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]
        motion_2d = self.create_2d_joints_from_features(motion)
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        return None, motion,motion_2d, length, None, None, None, None,
    
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
            
    def create_2d_joints_from_features(self,
                                       motion):
        root_y_position = motion[:,3].reshape(-1,1)
        joints_positions = motion[:,4:4+21*3]
        root_linear_velocity = motion[:,1:3]
        joints_global = self.build_global_joints(
            root_y_position,
            joints_positions,
            root_linear_velocity,
            vel_scale=1.0,
        )
        joints_global = self.convert_smpl22_to_coco(joints_global)
        joints_global = np.concatenate([joints_global[:,0,:].reshape(joints_global.shape[0],1,joints_global.shape[2]), joints_global[:,5:,:]], axis=1)

        random_yaw = random.choice([i for i in range(-180,180,1)])
        random_pitch = random.choice([i for i in range(0,60,1)])

        joints_2d, _ = self.build_2D_joints(
        joints_global,
        yaw_deg=random_yaw, pitch_deg=random_pitch,
        )
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(joints_2d)
        result = np.concatenate([
            root_vel_2d,root_y_2d,joints_pos_2d,joints_rot_2d,joints_vel_2d],axis=-1)
        return result
    
    
    def build_global_joints(self,
                        root_y_position,
                        joints_positions,
                        root_linear_velocity,
                        vel_scale=1.0):
        T = joints_positions.shape[0]
        n_joints_no_hips = 21

        joints_local = joints_positions.reshape(T, n_joints_no_hips, 3).copy()

        root_delta = root_linear_velocity * vel_scale          # (T,2)
        root_pos_xz = np.cumsum(root_delta, axis=0)            # (T,2)
        root_pos_xz = root_pos_xz - root_pos_xz[0:1]

        hips_global = np.zeros((T, 3), dtype=joints_positions.dtype)
        hips_global[:, 0] = root_pos_xz[:, 0]          # x
        hips_global[:, 1] = root_y_position[:, 0]      # y
        hips_global[:, 2] = root_pos_xz[:, 1]          # z

        joints_global_no_hips = joints_local.copy()
        joints_global_no_hips[:, :, 0] += root_pos_xz[:, 0:1]
        joints_global_no_hips[:, :, 2] += root_pos_xz[:, 1:2]

        joints_global_all = np.zeros((T, n_joints_no_hips + 1, 3),
                                    dtype=joints_positions.dtype)
        joints_global_all[:, 0, :] = hips_global              # hips
        joints_global_all[:, 1:, :] = joints_global_no_hips   # 1〜21: leftUpLeg〜rightHand

        return joints_global_all
    def convert_smpl22_to_coco(self,
                            smpl_keypoints):
        SMPL22_KEYPOINTS = [
        'pelvis',
        'left_hip_extra',
        'right_hip_extra',
        'spine_1',
        'left_knee',
        'right_knee',
        'spine_2',
        'left_ankle',
        'right_ankle',
        'spine_3',
        'left_foot',
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist']
        COCO_KEYPOINTS = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip_extra',
        'right_hip_extra',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        ]
        coco_keypoints = np.zeros((smpl_keypoints.shape[0], len(COCO_KEYPOINTS), 3))
        for t in range(smpl_keypoints.shape[0]):
            for idx, joint in enumerate(smpl_keypoints[t]):
                coco_idx = COCO_KEYPOINTS.index(SMPL22_KEYPOINTS[idx]) if SMPL22_KEYPOINTS[idx] in COCO_KEYPOINTS else -1
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

        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [ 0,           1, 0          ],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rx = np.array([
            [1,            0,             0          ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])

        R = Rx @ Ry

        center = joints_global.reshape(-1, 3).mean(axis=0)

        joints_rel = joints_global - center       # (T,22,3)
        joints_cam = joints_rel @ R.T             # (T,22,3)

        x = joints_cam[..., 0]
        y = joints_cam[..., 1]
        if invert_y:
            y = -y

        joints_2d = np.stack([x, y], axis=-1)     # (T,22,2)

        return joints_2d, R
    
    def decompose_2d_motion_coco13_midhip_root(self,
                                            joints_2d):
        """
        root_y_position_2d: (T,1)
        joints_positions_2d: (T, 12*2) = (T,24)
        root_linear_velocity_2d: (T,2)

        """
        coco_keypoints = [
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip_extra',
        'right_hip_extra',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        ]

        joints_2d = np.asarray(joints_2d)
        T, J, D = joints_2d.shape
        assert J == 13, f"Expected 13 joints, got {J}"
        assert D == 2,  f"Expected 2D joints, got {D}"

        name2idx = {n: i for i, n in enumerate(coco_keypoints)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']

        root_pos_2d = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])

        root_y_position_2d = root_pos_2d[:, 1:2]  # (T,1)

        joints_local_2d = joints_2d - root_pos_2d[:, None, :]

        joints_positions_2d = joints_local_2d.reshape(T, -1)  # (T, 13*2) = (T,26)


        joints_with_root = np.concatenate([root_pos_2d[:, None, :], joints_2d], axis=1)  # root, then 13 joints

        joints_local_with_root = joints_with_root - root_pos_2d[:, None, :]

        joints_positions_2d = joints_local_with_root[:, 1:, :].reshape(T, -1)

        root_linear_velocity_2d = np.zeros_like(root_pos_2d)  # (T,2)
        root_linear_velocity_2d[1:] = root_pos_2d[1:] - root_pos_2d[:-1]

        return root_y_position_2d, joints_positions_2d, root_linear_velocity_2d
    
    def compute_joint_features_2d_coco13(self,joints_2d):
        """
        rot: (T,13)
        vel: (T,26)

        root = (left_hip_extra + right_hip_extra)/2
        """
        coco_keypoints = [
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip_extra',
        'right_hip_extra',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        ]
        joints_2d = np.asarray(joints_2d)
        T, J, D = joints_2d.shape
        assert J == 13 and D == 2, f"Expected (T,13,2), got {joints_2d.shape}"

        name2idx = {n: i for i, n in enumerate(coco_keypoints)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']

        root = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])  # (T,2)

        joints_rel = joints_2d - root[:, None, :]

        rot = np.arctan2(joints_rel[:, :, 1], joints_rel[:, :, 0])  # (T,13)

        vel = np.zeros_like(joints_rel)  # (T,13,2)
        vel[1:] = joints_rel[1:] - joints_rel[:-1]
        vel = vel.reshape(T, -1)  # (T,26)

        return rot, vel
    
class MotionDatasetV_2DCOCO_normalized(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d_coco,
        std_2d_coco,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list
        self.mean_2d = mean_2d_coco
        self.std_2d = std_2d_coco

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, length = data["motion"], data["length"]

        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]
        motion_2d = self.create_2d_joints_from_features(motion)
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        return None, motion,motion_2d, length, None, None, None, None,
    
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
            
    def create_2d_joints_from_features(self,
                                       motion):
        root_y_position = motion[:,3].reshape(-1,1)
        joints_positions = motion[:,4:4+21*3]
        root_linear_velocity = motion[:,1:3]
        joints_global = self.build_global_joints(
            root_y_position,
            joints_positions,
            root_linear_velocity,
            vel_scale=1.0,
        )
        joints_global = self.convert_smpl22_to_coco(joints_global)
        joints_global = np.concatenate([joints_global[:,0,:].reshape(joints_global.shape[0],1,joints_global.shape[2]), joints_global[:,5:,:]], axis=1)

        random_yaw = random.choice([i for i in range(-180,180,1)])
        random_pitch = random.choice([i for i in range(0,60,1)])

        joints_2d, _ = self.build_2D_joints(
        joints_global,
        yaw_deg=random_yaw, pitch_deg=random_pitch,
        )
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(joints_2d)
        result = np.concatenate([
            root_vel_2d,root_y_2d,joints_pos_2d,joints_rot_2d,joints_vel_2d],axis=-1)
        return result
    
    
    def build_global_joints(self,
                        root_y_position,
                        joints_positions,
                        root_linear_velocity,
                        vel_scale=1.0):
        T = joints_positions.shape[0]
        n_joints_no_hips = 21

        joints_local = joints_positions.reshape(T, n_joints_no_hips, 3).copy()

        root_delta = root_linear_velocity * vel_scale          # (T,2)
        root_pos_xz = np.cumsum(root_delta, axis=0)            # (T,2)
        root_pos_xz = root_pos_xz - root_pos_xz[0:1]

        hips_global = np.zeros((T, 3), dtype=joints_positions.dtype)
        hips_global[:, 0] = root_pos_xz[:, 0]          # x
        hips_global[:, 1] = root_y_position[:, 0]      # y
        hips_global[:, 2] = root_pos_xz[:, 1]          # z

        joints_global_no_hips = joints_local.copy()
        joints_global_no_hips[:, :, 0] += root_pos_xz[:, 0:1]
        joints_global_no_hips[:, :, 2] += root_pos_xz[:, 1:2]

        joints_global_all = np.zeros((T, n_joints_no_hips + 1, 3),
                                    dtype=joints_positions.dtype)
        joints_global_all[:, 0, :] = hips_global              # hips
        joints_global_all[:, 1:, :] = joints_global_no_hips   # 1〜21: leftUpLeg〜rightHand

        return joints_global_all
    def convert_smpl22_to_coco(self,
                            smpl_keypoints):
        SMPL22_KEYPOINTS = [
        'pelvis',
        'left_hip_extra',
        'right_hip_extra',
        'spine_1',
        'left_knee',
        'right_knee',
        'spine_2',
        'left_ankle',
        'right_ankle',
        'spine_3',
        'left_foot',
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist']
        COCO_KEYPOINTS = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip_extra',
        'right_hip_extra',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        ]
        coco_keypoints = np.zeros((smpl_keypoints.shape[0], len(COCO_KEYPOINTS), 3))
        for t in range(smpl_keypoints.shape[0]):
            for idx, joint in enumerate(smpl_keypoints[t]):
                coco_idx = COCO_KEYPOINTS.index(SMPL22_KEYPOINTS[idx]) if SMPL22_KEYPOINTS[idx] in COCO_KEYPOINTS else -1
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

        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [ 0,           1, 0          ],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rx = np.array([
            [1,            0,             0          ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])

        R = Rx @ Ry

        center = joints_global.reshape(-1, 3).mean(axis=0)

        joints_rel = joints_global - center       # (T,22,3)
        joints_cam = joints_rel @ R.T             # (T,22,3)

        x = joints_cam[..., 0]
        y = joints_cam[..., 1]
        if invert_y:
            y = -y

        joints_2d = np.stack([x, y], axis=-1)     # (T,22,2)

        return joints_2d, R
    
    def normalize_2d_coco13_midhip(self,joints_2d, eps=1e-8, q=99):
        """
        joints_2d: (T,13,2) in pixel
        return:
        root_pos: (T,2)  # pixel root
        joints_rel: (T,13,2)  # root-relative (pixel)
        s: float         # clip scale
        """
        coco_keypoints = [
            'nose','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip_extra','right_hip_extra',
            'left_knee','right_knee','left_ankle','right_ankle',
        ]
        joints_2d = np.asarray(joints_2d)
        T, J, D = joints_2d.shape
        assert (J, D) == (13, 2)

        name2idx = {n: i for i, n in enumerate(coco_keypoints)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']

        # root (T,2)
        root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])

        # root-relative (T,13,2)
        joints_rel = joints_2d - root_pos[:, None, :]

        # clip scale: robust max extent of |x|, |y| over all frames/joints
        abs_xy = np.abs(joints_rel).reshape(-1, 2)  # (T*J,2)
        sx = np.percentile(abs_xy[:, 0], q)
        sy = np.percentile(abs_xy[:, 1], q)
        s = max(sx, sy, eps)

        return root_pos, joints_rel, s
    
    
    def decompose_2d_motion_coco13_midhip_root(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d, q=99)

        root_y_position_2d = (root_pos[:, 1:2] / s).astype(np.float32)
        root_y_position_2d = root_y_position_2d - root_y_position_2d[0:1]


        # joints_positions: root-relative /s, flatten (T,26)
        joints_positions_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)

        # root velocity: compute on scaled root (T,2)
        root_norm = (root_pos / s).astype(np.float32)
        root_linear_velocity_2d = np.zeros_like(root_norm)
        root_linear_velocity_2d[1:] = root_norm[1:] - root_norm[:-1]

        return root_y_position_2d, joints_positions_2d, root_linear_velocity_2d
    
    def compute_joint_features_2d_coco13(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d, q=99)

        joints_rel_norm = (joints_rel / s).astype(np.float32)  # (T,13,2)

        rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)

        # vel: diff of normalized joints_rel (T,13,2) -> flatten (T,26)
        vel = np.zeros_like(joints_rel_norm)
        vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
        vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)

        return rot, vel
    

class MotionDatasetV_2DCOCO_normalized_non_windowsize(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d_coco,
        std_2d_coco,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list
        self.mean_2d = mean_2d_coco
        self.std_2d = std_2d_coco

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data["motion"], data["length"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        motion_2d = self.create_2d_joints_from_features(motion)
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        return None, motion,motion_2d, m_length, None, None, None, None,
    
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
            
    def create_2d_joints_from_features(self,
                                       motion):
        root_y_position = motion[:,3].reshape(-1,1)
        joints_positions = motion[:,4:4+21*3]
        root_linear_velocity = motion[:,1:3]
        joints_global = self.build_global_joints(
            root_y_position,
            joints_positions,
            root_linear_velocity,
            vel_scale=1.0,
        )
        joints_global = self.convert_smpl22_to_coco(joints_global)
        joints_global = np.concatenate([joints_global[:,0,:].reshape(joints_global.shape[0],1,joints_global.shape[2]), joints_global[:,5:,:]], axis=1)

        random_yaw = random.choice([i for i in range(-180,180,1)])
        random_pitch = random.choice([i for i in range(0,60,1)])

        joints_2d, _ = self.build_2D_joints(
        joints_global,
        yaw_deg=random_yaw, pitch_deg=random_pitch,
        )
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(joints_2d)
        result = np.concatenate([
            root_vel_2d,root_y_2d,joints_pos_2d,joints_rot_2d,joints_vel_2d],axis=-1)
        return result
    
    
    def build_global_joints(self,
                        root_y_position,
                        joints_positions,
                        root_linear_velocity,
                        vel_scale=1.0):
        T = joints_positions.shape[0]
        n_joints_no_hips = 21

        joints_local = joints_positions.reshape(T, n_joints_no_hips, 3).copy()

        root_delta = root_linear_velocity * vel_scale          # (T,2)
        root_pos_xz = np.cumsum(root_delta, axis=0)            # (T,2)
        root_pos_xz = root_pos_xz - root_pos_xz[0:1]

        hips_global = np.zeros((T, 3), dtype=joints_positions.dtype)
        hips_global[:, 0] = root_pos_xz[:, 0]          # x
        hips_global[:, 1] = root_y_position[:, 0]      # y
        hips_global[:, 2] = root_pos_xz[:, 1]          # z

        joints_global_no_hips = joints_local.copy()
        joints_global_no_hips[:, :, 0] += root_pos_xz[:, 0:1]
        joints_global_no_hips[:, :, 2] += root_pos_xz[:, 1:2]

        joints_global_all = np.zeros((T, n_joints_no_hips + 1, 3),
                                    dtype=joints_positions.dtype)
        joints_global_all[:, 0, :] = hips_global              # hips
        joints_global_all[:, 1:, :] = joints_global_no_hips   # 1〜21: leftUpLeg〜rightHand

        return joints_global_all
    def convert_smpl22_to_coco(self,
                            smpl_keypoints):
        SMPL22_KEYPOINTS = [
        'pelvis',
        'left_hip_extra',
        'right_hip_extra',
        'spine_1',
        'left_knee',
        'right_knee',
        'spine_2',
        'left_ankle',
        'right_ankle',
        'spine_3',
        'left_foot',
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist']
        COCO_KEYPOINTS = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip_extra',
        'right_hip_extra',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        ]
        coco_keypoints = np.zeros((smpl_keypoints.shape[0], len(COCO_KEYPOINTS), 3))
        for t in range(smpl_keypoints.shape[0]):
            for idx, joint in enumerate(smpl_keypoints[t]):
                coco_idx = COCO_KEYPOINTS.index(SMPL22_KEYPOINTS[idx]) if SMPL22_KEYPOINTS[idx] in COCO_KEYPOINTS else -1
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

        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [ 0,           1, 0          ],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rx = np.array([
            [1,            0,             0          ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])

        R = Rx @ Ry

        center = joints_global.reshape(-1, 3).mean(axis=0)

        joints_rel = joints_global - center       # (T,22,3)
        joints_cam = joints_rel @ R.T             # (T,22,3)

        x = joints_cam[..., 0]
        y = joints_cam[..., 1]
        if invert_y:
            y = -y

        joints_2d = np.stack([x, y], axis=-1)     # (T,22,2)

        return joints_2d, R
    
    def normalize_2d_coco13_midhip(self,joints_2d, eps=1e-8, q=99):
        """
        joints_2d: (T,13,2) in pixel
        return:
        root_pos: (T,2)  # pixel root
        joints_rel: (T,13,2)  # root-relative (pixel)
        s: float         # clip scale
        """
        coco_keypoints = [
            'nose','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip_extra','right_hip_extra',
            'left_knee','right_knee','left_ankle','right_ankle',
        ]
        joints_2d = np.asarray(joints_2d)
        T, J, D = joints_2d.shape
        assert (J, D) == (13, 2)

        name2idx = {n: i for i, n in enumerate(coco_keypoints)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']

        # root (T,2)
        root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])

        # root-relative (T,13,2)
        joints_rel = joints_2d - root_pos[:, None, :]

        # clip scale: robust max extent of |x|, |y| over all frames/joints
        abs_xy = np.abs(joints_rel).reshape(-1, 2)  # (T*J,2)
        sx = np.percentile(abs_xy[:, 0], q)
        sy = np.percentile(abs_xy[:, 1], q)
        s = max(sx, sy, eps)

        return root_pos, joints_rel, s
    
    
    def decompose_2d_motion_coco13_midhip_root(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d, q=99)

        root_y_position_2d = (root_pos[:, 1:2] / s).astype(np.float32)
        root_y_position_2d = root_y_position_2d - root_y_position_2d[0:1]


        # joints_positions: root-relative /s, flatten (T,26)
        joints_positions_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)

        # root velocity: compute on scaled root (T,2)
        root_norm = (root_pos / s).astype(np.float32)
        root_linear_velocity_2d = np.zeros_like(root_norm)
        root_linear_velocity_2d[1:] = root_norm[1:] - root_norm[:-1]

        return root_y_position_2d, joints_positions_2d, root_linear_velocity_2d
    
    def compute_joint_features_2d_coco13(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d, q=99)

        joints_rel_norm = (joints_rel / s).astype(np.float32)  # (T,13,2)

        rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)

        # vel: diff of normalized joints_rel (T,13,2) -> flatten (T,26)
        vel = np.zeros_like(joints_rel_norm)
        vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
        vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)

        return rot, vel
    

class MotionDatasetV_2DCOCO_normalized_non_windowsize(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d_coco,
        std_2d_coco,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list
        self.mean_2d = mean_2d_coco
        self.std_2d = std_2d_coco

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data["motion"], data["length"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        motion_2d = self.create_2d_joints_from_features(motion)
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        return None, motion,motion_2d, m_length, None, None, None, None,
    
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
            
    def create_2d_joints_from_features(self,
                                       motion):
        root_y_position = motion[:,3].reshape(-1,1)
        joints_positions = motion[:,4:4+21*3]
        root_linear_velocity = motion[:,1:3]
        joints_global = self.build_global_joints(
            root_y_position,
            joints_positions,
            root_linear_velocity,
            vel_scale=1.0,
        )
        joints_global = self.convert_smpl22_to_coco(joints_global)
        joints_global = np.concatenate([joints_global[:,0,:].reshape(joints_global.shape[0],1,joints_global.shape[2]), joints_global[:,5:,:]], axis=1)

        random_yaw = random.choice([i for i in range(-180,180,1)])
        random_pitch = random.choice([i for i in range(0,60,1)])

        joints_2d, _ = self.build_2D_joints(
        joints_global,
        yaw_deg=random_yaw, pitch_deg=random_pitch,
        )
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(joints_2d)
        result = np.concatenate([
            root_vel_2d,root_y_2d,joints_pos_2d,joints_rot_2d,joints_vel_2d],axis=-1)
        return result
    
    
    def build_global_joints(self,
                        root_y_position,
                        joints_positions,
                        root_linear_velocity,
                        vel_scale=1.0):
        T = joints_positions.shape[0]
        n_joints_no_hips = 21

        joints_local = joints_positions.reshape(T, n_joints_no_hips, 3).copy()

        root_delta = root_linear_velocity * vel_scale          # (T,2)
        root_pos_xz = np.cumsum(root_delta, axis=0)            # (T,2)
        root_pos_xz = root_pos_xz - root_pos_xz[0:1]

        hips_global = np.zeros((T, 3), dtype=joints_positions.dtype)
        hips_global[:, 0] = root_pos_xz[:, 0]          # x
        hips_global[:, 1] = root_y_position[:, 0]      # y
        hips_global[:, 2] = root_pos_xz[:, 1]          # z

        joints_global_no_hips = joints_local.copy()
        joints_global_no_hips[:, :, 0] += root_pos_xz[:, 0:1]
        joints_global_no_hips[:, :, 2] += root_pos_xz[:, 1:2]

        joints_global_all = np.zeros((T, n_joints_no_hips + 1, 3),
                                    dtype=joints_positions.dtype)
        joints_global_all[:, 0, :] = hips_global              # hips
        joints_global_all[:, 1:, :] = joints_global_no_hips   # 1〜21: leftUpLeg〜rightHand

        return joints_global_all
    def convert_smpl22_to_coco(self,
                            smpl_keypoints):
        SMPL22_KEYPOINTS = [
        'pelvis',
        'left_hip_extra',
        'right_hip_extra',
        'spine_1',
        'left_knee',
        'right_knee',
        'spine_2',
        'left_ankle',
        'right_ankle',
        'spine_3',
        'left_foot',
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist']
        COCO_KEYPOINTS = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip_extra',
        'right_hip_extra',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        ]
        coco_keypoints = np.zeros((smpl_keypoints.shape[0], len(COCO_KEYPOINTS), 3))
        for t in range(smpl_keypoints.shape[0]):
            for idx, joint in enumerate(smpl_keypoints[t]):
                coco_idx = COCO_KEYPOINTS.index(SMPL22_KEYPOINTS[idx]) if SMPL22_KEYPOINTS[idx] in COCO_KEYPOINTS else -1
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

        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [ 0,           1, 0          ],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rx = np.array([
            [1,            0,             0          ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])

        R = Rx @ Ry

        center = joints_global.reshape(-1, 3).mean(axis=0)

        joints_rel = joints_global - center       # (T,22,3)
        joints_cam = joints_rel @ R.T             # (T,22,3)

        x = joints_cam[..., 0]
        y = joints_cam[..., 1]
        if invert_y:
            y = -y

        joints_2d = np.stack([x, y], axis=-1)     # (T,22,2)

        return joints_2d, R
    
    def normalize_2d_coco13_midhip(self,joints_2d, eps=1e-8, q=99):
        """
        joints_2d: (T,13,2) in pixel
        return:
        root_pos: (T,2)  # pixel root
        joints_rel: (T,13,2)  # root-relative (pixel)
        s: float         # clip scale
        """
        coco_keypoints = [
            'nose','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip_extra','right_hip_extra',
            'left_knee','right_knee','left_ankle','right_ankle',
        ]
        joints_2d = np.asarray(joints_2d)
        T, J, D = joints_2d.shape
        assert (J, D) == (13, 2)

        name2idx = {n: i for i, n in enumerate(coco_keypoints)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']

        # root (T,2)
        root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])

        # root-relative (T,13,2)
        joints_rel = joints_2d - root_pos[:, None, :]

        # clip scale: robust max extent of |x|, |y| over all frames/joints
        abs_xy = np.abs(joints_rel).reshape(-1, 2)  # (T*J,2)
        sx = np.percentile(abs_xy[:, 0], q)
        sy = np.percentile(abs_xy[:, 1], q)
        s = max(sx, sy, eps)

        return root_pos, joints_rel, s
    
    
    def decompose_2d_motion_coco13_midhip_root(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d, q=99)

        root_y_position_2d = (root_pos[:, 1:2] / s).astype(np.float32)
        root_y_position_2d = root_y_position_2d - root_y_position_2d[0:1]


        # joints_positions: root-relative /s, flatten (T,26)
        joints_positions_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)

        # root velocity: compute on scaled root (T,2)
        root_norm = (root_pos / s).astype(np.float32)
        root_linear_velocity_2d = np.zeros_like(root_norm)
        root_linear_velocity_2d[1:] = root_norm[1:] - root_norm[:-1]

        return root_y_position_2d, joints_positions_2d, root_linear_velocity_2d
    
    def compute_joint_features_2d_coco13(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d, q=99)

        joints_rel_norm = (joints_rel / s).astype(np.float32)  # (T,13,2)

        rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)

        # vel: diff of normalized joints_rel (T,13,2) -> flatten (T,26)
        vel = np.zeros_like(joints_rel_norm)
        vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
        vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)

        return rot, vel
    
class MotionDatasetV_2DCOCO_normalized_non_windowsize_KIT(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d_coco,
        std_2d_coco,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list
        self.mean_2d = mean_2d_coco
        self.std_2d = std_2d_coco

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data["motion"], data["length"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        motion_2d = self.create_2d_joints_from_features(motion)
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        return None, motion,motion_2d, m_length, None, None, None, None,
    
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
            
    def create_2d_joints_from_features(self,
                                       motion):
        joints_global = recover_from_ric(torch.tensor(motion).float(), joints_num=21)
        joints_global = self.convert_smpl21_to_coco(joints_global)
        joints_global = np.concatenate([joints_global[:,0,:].reshape(joints_global.shape[0],1,joints_global.shape[2]), joints_global[:,5:,:]], axis=1)

        random_yaw = random.choice([i for i in range(-180,180,1)])
        random_pitch = random.choice([i for i in range(0,60,1)])

        joints_2d, _ = self.build_2D_joints(
        joints_global,
        yaw_deg=random_yaw, pitch_deg=random_pitch,
        )
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(joints_2d)
        result = np.concatenate([
            root_vel_2d,root_y_2d,joints_pos_2d,joints_rot_2d,joints_vel_2d],axis=-1)
        return result
    
    
    def convert_smpl21_to_coco(self,
                            smpl_keypoints):
        KIT21_KEYPOINTS = [
        "pelvis",          # 0

        "spine_1",         # 1
        "spine_2",         # 2
        "spine_3",         # 3
        "nose",            # 4

        "left_shoulder",   # 5
        "left_elbow",      # 6
        "left_wrist",      # 7

        "right_shoulder",  # 8
        "right_elbow",     # 9
        "right_wrist",     # 10

        "left_hip",        # 11
        "left_knee",       # 12
        "left_ankle",      # 13
        "left_foot",       # 14
        "left_toe",        # 15

        "right_hip",       # 16
        "right_knee",      # 17
        "right_ankle",     # 18
        "right_foot",      # 19
        "right_toe",       # 20
    ]
        COCO_KEYPOINTS = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        ]
        coco_keypoints = np.zeros((smpl_keypoints.shape[0], len(COCO_KEYPOINTS), 3))
        for t in range(smpl_keypoints.shape[0]):
            for idx, joint in enumerate(smpl_keypoints[t]):
                coco_idx = COCO_KEYPOINTS.index(KIT21_KEYPOINTS[idx]) if KIT21_KEYPOINTS[idx] in COCO_KEYPOINTS else -1
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

        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [ 0,           1, 0          ],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rx = np.array([
            [1,            0,             0          ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])

        R = Rx @ Ry

        center = joints_global.reshape(-1, 3).mean(axis=0)

        joints_rel = joints_global - center       # (T,22,3)
        joints_cam = joints_rel @ R.T             # (T,22,3)

        x = joints_cam[..., 0]
        y = joints_cam[..., 1]
        if invert_y:
            y = -y

        joints_2d = np.stack([x, y], axis=-1)     # (T,22,2)

        return joints_2d, R
    
    def normalize_2d_coco13_midhip(self,joints_2d, eps=1e-8, q=99):
        """
        joints_2d: (T,13,2) in pixel
        return:
        root_pos: (T,2)  # pixel root
        joints_rel: (T,13,2)  # root-relative (pixel)
        s: float         # clip scale
        """
        coco_keypoints = [
            'nose','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip_extra','right_hip_extra',
            'left_knee','right_knee','left_ankle','right_ankle',
        ]
        joints_2d = np.asarray(joints_2d)
        T, J, D = joints_2d.shape
        assert (J, D) == (13, 2)

        name2idx = {n: i for i, n in enumerate(coco_keypoints)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']

        # root (T,2)
        root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])

        # root-relative (T,13,2)
        joints_rel = joints_2d - root_pos[:, None, :]

        # clip scale: robust max extent of |x|, |y| over all frames/joints
        abs_xy = np.abs(joints_rel).reshape(-1, 2)  # (T*J,2)
        sx = np.percentile(abs_xy[:, 0], q)
        sy = np.percentile(abs_xy[:, 1], q)
        s = max(sx, sy, eps)

        return root_pos, joints_rel, s
    
    
    def decompose_2d_motion_coco13_midhip_root(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d, q=99)

        root_y_position_2d = (root_pos[:, 1:2] / s).astype(np.float32)
        root_y_position_2d = root_y_position_2d - root_y_position_2d[0:1]


        # joints_positions: root-relative /s, flatten (T,26)
        joints_positions_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)

        # root velocity: compute on scaled root (T,2)
        root_norm = (root_pos / s).astype(np.float32)
        root_linear_velocity_2d = np.zeros_like(root_norm)
        root_linear_velocity_2d[1:] = root_norm[1:] - root_norm[:-1]

        return root_y_position_2d, joints_positions_2d, root_linear_velocity_2d
    
    def compute_joint_features_2d_coco13(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d, q=99)

        joints_rel_norm = (joints_rel / s).astype(np.float32)  # (T,13,2)

        rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)

        # vel: diff of normalized joints_rel (T,13,2) -> flatten (T,26)
        vel = np.zeros_like(joints_rel_norm)
        vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
        vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)

        return rot, vel