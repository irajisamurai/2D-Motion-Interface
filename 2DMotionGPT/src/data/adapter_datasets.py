"""
Dataset classes for adapter training.

2D adapter classes (VitPose → MotionGPT latent space):
- MotionDataset_vitpose_2DCOCO_normalized: training dataset
- Text2MotionDatasetEval_vitpose_2DCOCO_normalized: evaluation dataset

3D adapter classes (WHAM-estimated 3D → MotionGPT latent space):
- MotionDataset_wham_3d: training dataset
- Text2MotionDatasetEval_wham_3d: evaluation dataset

WHAM data layout:
    <estimated_motion_dir>/adapter_training_WAHA/<motion_id>/<view_id>.npy
    Each .npy: (T, 263) float32 in raw HumanML3D format (same normalization as GT)
"""
import json
import glob
import os
import random

import numpy as np
from os.path import join as pjoin

from src.data.humanml.dataset_t2m import Text2MotionDataset


# ---------------------------------------------------------------------------
# Shared helpers (mixin)
# ---------------------------------------------------------------------------

class _VitPoseMixin:
    """Geometry + feature extraction helpers shared by both dataset classes."""

    def extract_keypoints(self, json_list_path):
        result_dict = {}
        for json_path in json_list_path:
            vid_name = json_path.split("/")[-2]
            if vid_name not in result_dict:
                result_dict[vid_name] = {}
            idx = json_path.split("/")[-1].replace(".json", "")
            data = json.load(open(json_path))
            motions, confs = [], []
            for frame_data in data:
                motions.append(frame_data["instances"][0]["keypoints"])
                confs.append(frame_data["instances"][0]["keypoint_scores"])
            motions = np.array(motions)
            confs = np.array(confs)
            if motions.shape[1] != 17:
                print(f"Skipping {vid_name}: unexpected keypoint count {motions.shape[1]}")
                continue
            result_dict[vid_name][idx] = {"motions": motions, "confs": confs}
        return result_dict

    def create_2d_joints_from_features(self, motion):
        root_y_position = motion[:, 3].reshape(-1, 1)
        joints_positions = motion[:, 4:4 + 21 * 3]
        root_linear_velocity = motion[:, 1:3]
        joints_global = self.build_global_joints(root_y_position, joints_positions, root_linear_velocity)
        joints_global = self.convert_smpl22_to_coco(joints_global)
        joints_global = np.concatenate([
            joints_global[:, 0, :].reshape(joints_global.shape[0], 1, joints_global.shape[2]),
            joints_global[:, 5:, :]
        ], axis=1)
        random_yaw = random.choice(range(-180, 180))
        random_pitch = random.choice(range(0, 60))
        joints_2d, _ = self.build_2D_joints(joints_global, yaw_deg=random_yaw, pitch_deg=random_pitch)
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(joints_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(joints_2d)
        return np.concatenate([root_vel_2d, root_y_2d, joints_pos_2d, joints_rot_2d, joints_vel_2d], axis=-1)

    def build_global_joints(self, root_y_position, joints_positions, root_linear_velocity, vel_scale=1.0):
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
        joints_global_all = np.zeros((T, n_joints_no_hips + 1, 3), dtype=joints_positions.dtype)
        joints_global_all[:, 0, :] = hips_global
        joints_global_all[:, 1:, :] = joints_global_no_hips
        return joints_global_all

    def convert_smpl22_to_coco(self, smpl_keypoints):
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

    def build_2D_joints(self, joints_global, yaw_deg=0.0, pitch_deg=0.0, invert_y=True):
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                       [ 0,           1, 0           ],
                       [-np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1,             0,              0            ],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch),  np.cos(pitch)]])
        R = Rx @ Ry
        center = joints_global.reshape(-1, 3).mean(axis=0)
        joints_cam = (joints_global - center) @ R.T
        x = joints_cam[..., 0]
        y = -joints_cam[..., 1] if invert_y else joints_cam[..., 1]
        return np.stack([x, y], axis=-1), R

    def normalize_2d_coco13_midhip(self, joints_2d, eps=1e-8, q=99):
        COCO13 = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip_extra', 'right_hip_extra',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        joints_2d = np.asarray(joints_2d)
        name2idx = {n: i for i, n in enumerate(COCO13)}
        lhip = name2idx['left_hip_extra']
        rhip = name2idx['right_hip_extra']
        root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])
        joints_rel = joints_2d - root_pos[:, None, :]
        abs_xy = np.abs(joints_rel).reshape(-1, 2)
        s = max(np.percentile(abs_xy[:, 0], q), np.percentile(abs_xy[:, 1], q), eps)
        return root_pos, joints_rel, s

    def decompose_2d_motion_coco13_midhip_root(self, joints_2d):
        root_pos, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d)
        root_y_2d = (root_pos[:, 1:2] / s).astype(np.float32)
        root_y_2d = root_y_2d - root_y_2d[0:1]
        joints_pos_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)
        root_norm = (root_pos / s).astype(np.float32)
        root_vel_2d = np.zeros_like(root_norm)
        root_vel_2d[1:] = root_norm[1:] - root_norm[:-1]
        return root_y_2d, joints_pos_2d, root_vel_2d

    def compute_joint_features_2d_coco13(self, joints_2d):
        _, joints_rel, s = self.normalize_2d_coco13_midhip(joints_2d)
        joints_rel_norm = (joints_rel / s).astype(np.float32)
        rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)
        vel = np.zeros_like(joints_rel_norm)
        vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
        vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)
        return rot, vel

    def _preprocess_estimated(self, estimated_motion_2d, estimated_conf):
        """Convert raw VitPose keypoints → concatenated feature vector."""
        root_y_2d, joints_pos_2d, root_vel_2d = self.decompose_2d_motion_coco13_midhip_root(estimated_motion_2d)
        joints_rot_2d, joints_vel_2d = self.compute_joint_features_2d_coco13(estimated_motion_2d)
        T = root_y_2d.shape[0]
        c = estimated_conf.reshape(T, -1)
        return np.concatenate([root_vel_2d, root_y_2d, joints_pos_2d, joints_rot_2d, joints_vel_2d, c], axis=-1)


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class MotionDataset_vitpose_2DCOCO_normalized(_VitPoseMixin, Text2MotionDataset):
    """Training dataset: HumanML3D motion + VitPose estimated 2D keypoints.

    Returns a 8-tuple compatible with humanml3d_collate_2d:
      (None, motion_3d, estimated_motion_2d_norm, m_length, None, None, None, None)
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d_coco,
        std_2d_coco,
        mean_estimate,
        std_estimate,
        max_motion_length,
        min_motion_length,
        win_size,
        estimated_motion_dir,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny, debug, **kwargs)

        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            if self.data_dict[name]["motion"].shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list

        self.mean_2d = mean_2d_coco
        self.std_2d = std_2d_coco
        self.mean_estimate = mean_estimate
        self.std_estimate = std_estimate

        json_list = glob.glob(os.path.join(estimated_motion_dir, "json", "**", "*.json"), recursive=True)
        print(f"[MotionDataset_vitpose] Found {len(json_list)} JSON files in {estimated_motion_dir}")
        self.estimated_motion_dict = self.extract_keypoints(json_list)

        before_filter = len(self.name_list)
        self.name_list = [n for n in self.name_list if n in self.estimated_motion_dict]
        self.data_dict = {n: self.data_dict[n] for n in self.name_list}
        print(f"[MotionDataset_vitpose] Filtered to {len(self.name_list)}/{before_filter} motions with JSON")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data["motion"], data["length"]

        random_view_idx = random.choice(list(self.estimated_motion_dict[self.name_list[idx]].keys()))
        estimated_motion_2d = self.estimated_motion_dict[self.name_list[idx]][random_view_idx]["motions"]
        estimated_conf = self.estimated_motion_dict[self.name_list[idx]][random_view_idx]["confs"]
        # select COCO-13 subset (drop eyes/ears)
        estimated_motion_2d = estimated_motion_2d[:, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]
        estimated_conf = estimated_conf[:, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        estimated_motion_2d = estimated_motion_2d[idx:idx + m_length]
        estimated_conf = estimated_conf[idx:idx + m_length]

        estimated_feat = self._preprocess_estimated(estimated_motion_2d, estimated_conf)

        motion = (motion - self.mean) / self.std
        estimated_feat = (estimated_feat - self.mean_estimate) / self.std_estimate

        return None, motion, estimated_feat, m_length, None, None, None, None


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

class Text2MotionDatasetEval_vitpose_2DCOCO_normalized(_VitPoseMixin, Text2MotionDataset):
    """Evaluation dataset: HumanML3D motion + VitPose estimated 2D keypoints.

    Returns a 9-tuple compatible with humanml3d_collate_2d (EvalFlag=True):
      (caption, motion_3d, estimated_motion_2d_norm, m_length,
       word_embeddings, pos_one_hots, sent_len, tokens_str, all_captions)
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_2d,
        std_2d,
        mean_estimate,
        std_estimate,
        w_vectorizer,
        estimated_motion_dir,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny, debug, **kwargs)

        self.w_vectorizer = w_vectorizer
        self.mean_2d = mean_2d
        self.std_2d = std_2d
        self.mean_estimate = mean_estimate
        self.std_estimate = std_estimate

        json_list = glob.glob(os.path.join(estimated_motion_dir, "json", "**", "*.json"), recursive=True)
        print(f"[Text2MotionDatasetEval_vitpose] Found {len(json_list)} JSON files in {estimated_motion_dir}")
        self.estimated_motion_dict = self.extract_keypoints(json_list)

        before_filter = len(self.name_list)
        self.name_list = [n for n in self.name_list if n in self.estimated_motion_dict]
        self.data_dict = {n: self.data_dict[n] for n in self.name_list}
        print(f"[Text2MotionDatasetEval_vitpose] Filtered to {len(self.name_list)}/{before_filter} motions with JSON")

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        random_view_idx = random.choice(list(self.estimated_motion_dict[self.name_list[idx]].keys()))
        estimated_motion_2d = self.estimated_motion_dict[self.name_list[idx]][random_view_idx]["motions"]
        estimated_conf = self.estimated_motion_dict[self.name_list[idx]][random_view_idx]["confs"]
        estimated_motion_2d = estimated_motion_2d[:, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]
        estimated_conf = estimated_conf[:, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]
        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        max_text_len = 20
        if len(tokens) < max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            tokens = ["sos/OTHER"] + tokens[:max_text_len] + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots, word_embeddings = [], []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        estimated_motion_2d = estimated_motion_2d[idx:idx + m_length]
        estimated_conf = estimated_conf[idx:idx + m_length]

        estimated_feat = self._preprocess_estimated(estimated_motion_2d, estimated_conf)

        motion = (motion - self.mean) / self.std
        estimated_feat = (estimated_feat - self.mean_estimate) / self.std_estimate

        return caption, motion, estimated_feat, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(tokens), all_captions


# ---------------------------------------------------------------------------
# 3D adapter datasets (WHAM-estimated 3D → MotionGPT latent space)
# ---------------------------------------------------------------------------

def _build_waha_dict(estimated_motion_dir, min_frames):
    """Scan adapter_training_WAHA and return {motion_id: [valid_npy_paths]}."""
    waha_root = os.path.join(estimated_motion_dir, "adapter_training_WAHA")
    waha_dict = {}
    for mid in sorted(os.listdir(waha_root)):
        mid_dir = os.path.join(waha_root, mid)
        if not os.path.isdir(mid_dir):
            continue
        valid_paths = []
        for npy_path in sorted(glob.glob(os.path.join(mid_dir, "*.npy"))):
            arr = np.load(npy_path, mmap_mode='r')
            if arr.shape[0] < min_frames:
                continue
            if np.any(np.isnan(arr)):
                continue
            valid_paths.append(npy_path)
        if valid_paths:
            waha_dict[mid] = valid_paths
    return waha_dict


class MotionDataset_wham_3d(Text2MotionDataset):
    """Training dataset: HumanML3D GT motion + WHAM-estimated 3D motion.

    GT motion is normalized with HumanML3D mean/std.
    WHAM motion is normalized with WHAM-specific mean_wham/std_wham
    (computed from all WAHA files), so its distribution matches GT (std≈1.0).

    Returns an 8-tuple compatible with humanml3d_collate_2d:
      (None, motion_3d_norm, wham_3d_norm, m_length, None, None, None, None)
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_wham,
        std_wham,
        max_motion_length,
        min_motion_length,
        win_size,
        estimated_motion_dir,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny, debug, **kwargs)

        self.mean_wham = mean_wham
        self.std_wham  = std_wham
        self.window_size = win_size

        # remove GT motions shorter than win_size
        name_list = list(self.name_list)
        for name in list(name_list):
            if self.data_dict[name]["motion"].shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list

        # build WAHA lookup: {motion_id: [npy_paths with T >= min_motion_length]}
        self.waha_dict = _build_waha_dict(estimated_motion_dir, min_motion_length)
        print(f"[MotionDataset_wham_3d] Found {len(self.waha_dict)} motion IDs with valid WAHA views")

        # filter to motions that have WAHA data (exact name match — no prefixed entries)
        before = len(self.name_list)
        self.name_list = [n for n in self.name_list if n in self.waha_dict]
        self.data_dict = {n: self.data_dict[n] for n in self.name_list}
        print(f"[MotionDataset_wham_3d] Filtered to {len(self.name_list)}/{before} motions with WAHA data")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length = data["motion"], data["length"]

        # pick a random valid WHAM view
        wham_path = random.choice(self.waha_dict[name])
        wham_motion = np.load(wham_path).astype(np.float32)  # (T_wham, 263)

        # safe temporal window: limited to the shorter of GT and WHAM
        T_valid = min(len(motion), len(wham_motion))

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        # cap m_length to what T_valid allows
        m_length = min(m_length, (T_valid // self.unit_length) * self.unit_length)
        if m_length < self.min_motion_length:
            return None

        start = random.randint(0, T_valid - m_length)
        motion      = motion[start:start + m_length]
        wham_motion = wham_motion[start:start + m_length]

        motion      = (motion      - self.mean)      / self.std
        wham_motion = (wham_motion - self.mean_wham) / self.std_wham

        return None, motion, wham_motion, m_length, None, None, None, None


class Text2MotionDatasetEval_wham_3d(Text2MotionDataset):
    """Evaluation dataset: HumanML3D GT motion + WHAM-estimated 3D motion.

    Returns a 9-tuple compatible with humanml3d_collate_2d (EvalFlag=True):
      (caption, motion_3d_norm, wham_3d_norm, m_length,
       word_embeddings, pos_one_hots, sent_len, tokens_str, all_captions)
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        mean_wham,
        std_wham,
        w_vectorizer,
        estimated_motion_dir,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny, debug, **kwargs)

        self.mean_wham = mean_wham
        self.std_wham  = std_wham
        self.w_vectorizer = w_vectorizer

        # build WAHA lookup
        self.waha_dict = _build_waha_dict(estimated_motion_dir, min_motion_length)
        print(f"[Text2MotionDatasetEval_wham_3d] Found {len(self.waha_dict)} motion IDs with valid WAHA views")

        before = len(self.name_list)
        self.name_list = [n for n in self.name_list if n in self.waha_dict]
        self.data_dict = {n: self.data_dict[n] for n in self.name_list}
        print(f"[Text2MotionDatasetEval_wham_3d] Filtered to {len(self.name_list)}/{before} motions with WAHA data")

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # pick a random valid WHAM view
        wham_path = random.choice(self.waha_dict[name])
        wham_motion = np.load(wham_path).astype(np.float32)  # (T_wham, 263)

        T_valid = min(len(motion), len(wham_motion))

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]
        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        max_text_len = 20
        if len(tokens) < max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            tokens = ["sos/OTHER"] + tokens[:max_text_len] + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots, word_embeddings = [], []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots   = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        m_length = min(m_length, (T_valid // self.unit_length) * self.unit_length)
        if m_length < self.min_motion_length:
            return None

        start = random.randint(0, T_valid - m_length)
        motion      = motion[start:start + m_length]
        wham_motion = wham_motion[start:start + m_length]

        motion      = (motion      - self.mean)      / self.std
        wham_motion = (wham_motion - self.mean_wham) / self.std_wham

        return caption, motion, wham_motion, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(tokens), all_captions
