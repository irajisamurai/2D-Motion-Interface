import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
from pathlib import Path
import random
import json
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate

ROOT_DIR = Path(__file__).resolve().parents[1]


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


# ---- 2D projection helpers (shared by Motion2TextDataset and Motion2MotionScriptDataset) ----

def _build_global_joints(root_y_position, joints_positions, root_linear_velocity, vel_scale=1.0):
    T = joints_positions.shape[0]
    joints_local = joints_positions.reshape(T, 21, 3).copy()
    root_delta = root_linear_velocity * vel_scale
    root_pos_xz = np.cumsum(root_delta, axis=0) - np.cumsum(root_delta, axis=0)[0:1]
    hips_global = np.zeros((T, 3), dtype=joints_positions.dtype)
    hips_global[:, 0] = root_pos_xz[:, 0]
    hips_global[:, 1] = root_y_position[:, 0]
    hips_global[:, 2] = root_pos_xz[:, 1]
    joints_global_no_hips = joints_local.copy()
    joints_global_no_hips[:, :, 0] += root_pos_xz[:, 0:1]
    joints_global_no_hips[:, :, 2] += root_pos_xz[:, 1:2]
    joints_global_all = np.zeros((T, 22, 3), dtype=joints_positions.dtype)
    joints_global_all[:, 0, :] = hips_global
    joints_global_all[:, 1:, :] = joints_global_no_hips
    return joints_global_all


def _convert_smpl22_to_coco(smpl_keypoints):
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


def _build_2D_joints(joints_global, yaw_deg=0.0, pitch_deg=0.0, invert_y=True):
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    R = Rx @ Ry
    center = joints_global.reshape(-1, 3).mean(axis=0)
    joints_cam = (joints_global - center) @ R.T
    x, y = joints_cam[..., 0], joints_cam[..., 1]
    if invert_y:
        y = -y
    return np.stack([x, y], axis=-1), R


def _normalize_2d_coco13_midhip(joints_2d, eps=1e-8, q=99):
    joints_2d = np.asarray(joints_2d)
    lhip, rhip = 7, 8
    root_pos = 0.5 * (joints_2d[:, lhip, :] + joints_2d[:, rhip, :])
    joints_rel = joints_2d - root_pos[:, None, :]
    abs_xy = np.abs(joints_rel).reshape(-1, 2)
    s = max(np.percentile(abs_xy[:, 0], q), np.percentile(abs_xy[:, 1], q), eps)
    return root_pos, joints_rel, s


def _decompose_2d_motion_coco13_midhip_root(joints_2d):
    root_pos, joints_rel, s = _normalize_2d_coco13_midhip(joints_2d)
    root_y_2d = (root_pos[:, 1:2] / s).astype(np.float32)
    root_y_2d = root_y_2d - root_y_2d[0:1]
    joints_pos_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)
    root_norm = (root_pos / s).astype(np.float32)
    root_vel_2d = np.zeros_like(root_norm)
    root_vel_2d[1:] = root_norm[1:] - root_norm[:-1]
    return root_y_2d, joints_pos_2d, root_vel_2d


def _compute_joint_features_2d_coco13(joints_2d):
    _, joints_rel, s = _normalize_2d_coco13_midhip(joints_2d)
    joints_rel_norm = (joints_rel / s).astype(np.float32)
    rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)
    vel = np.zeros_like(joints_rel_norm)
    vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
    return rot, vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)


def _create_2d_joints_from_features(motion):
    joints_global = _build_global_joints(
        motion[:, 3].reshape(-1, 1),
        motion[:, 4:4 + 21 * 3],
        motion[:, 1:3],
    )
    joints_global = _convert_smpl22_to_coco(joints_global)
    joints_global = np.concatenate([
        joints_global[:, 0:1, :],
        joints_global[:, 5:, :],
    ], axis=1)
    joints_2d, _ = _build_2D_joints(
        joints_global,
        yaw_deg=random.choice(range(-180, 180)),
        pitch_deg=random.choice(range(0, 60)),
    )
    root_y_2d, joints_pos_2d, root_vel_2d = _decompose_2d_motion_coco13_midhip_root(joints_2d)
    joints_rot_2d, joints_vel_2d = _compute_joint_features_2d_coco13(joints_2d)
    return np.concatenate([root_vel_2d, root_y_2d, joints_pos_2d, joints_rot_2d, joints_vel_2d], axis=-1)



'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, split, w_vectorizer, feat_bias=5, max_text_len=20, unit_length=4):

        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
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
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = './checkpoints/kit/Decomp_SP001_SM001_H512/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, f'{split}.txt')

        min_motion_len = 40 if self.dataset_name == 't2m' else 24

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name



'''For use of training text-2-motion generative model'''
class Text2MotionDataset_withMotionScript(data.Dataset):
    def __init__(self, dataset_name, split, w_vectorizer, feat_bias=5, max_text_len=20, unit_length=4):

        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.finemotion_text_dir = pjoin(self.data_root, 'finemotion_texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
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
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = './checkpoints/kit/Decomp_SP001_SM001_H512/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        # detailed text for motions
        BPMSD_auto_file = pjoin(self.finemotion_text_dir, 'BPMSD_auto.json')
        with open(BPMSD_auto_file, 'r') as f:
            BPMSD_dict = json.load(f)

        BPMSD_human_file = pjoin(self.finemotion_text_dir, 'BPMSD_human.json')
        with open(BPMSD_human_file, 'r') as f:
            BPMSD_human_dict = json.load(f)
        BPMSD_dict.update(BPMSD_human_dict)

        split_file = pjoin(self.data_root, f'{split}.txt')

        min_motion_len = 40 if self.dataset_name == 't2m' else 24

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['tokens'] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:

                            bodyPart_text_list = BPMSD_dict[name]

                            summary_detail_text_dict = text_dict.copy()
                            summary_detail_text_dict['summary'] = caption
                            summary_detail_text_dict['detail'] = bodyPart_text_list
                            text_data.append(summary_detail_text_dict)

                            flag = True

                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue

                                bodyPart_text_list = BPMSD_dict[name][int(f_tag / 0.5): int(to_tag / 0.5)]

                                text_data_new = []

                                # summary + detail
                                summary_detail_text_dict = text_dict.copy()
                                summary_detail_text_dict['summary'] = caption
                                summary_detail_text_dict['detail'] = bodyPart_text_list
                                text_data_new.append(summary_detail_text_dict)

                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name

                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': text_data_new}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        summary = text_data['summary']
        bodyPart_text_list = text_data['detail'][:]
        tokens = text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Each item in bodyPart_text_list corresponds to 0.5 seconds, i.e., 10 frames of motion.
        # Here, we ensure strict alignment between motion tokens and detailed body part movement descriptions.
        m_length = (m_length // 20) * 20        # 20 is the least common multiple of 10 (corresponds to 1 item in bodyPart_text_list) and 4 (self.unit_length, also the frames of a singel token).
        motion = motion[:m_length]
        bodyPart_text_list = bodyPart_text_list[:int(m_length/10)]

        for i in range(len(bodyPart_text_list)):
            bodyPart_text_item = bodyPart_text_list[i]
            if bodyPart_text_item == "":
                bodyPart_text_list[i] = '<Motionless>'
        long_text = (" <SEP> ").join(bodyPart_text_list)
        detail = long_text

        summary = '### Motion Summary ###\n' + summary
        detail = '### Motion Script ###\n' + detail

        caption = '\n\n' + summary + '\n\n'
        caption = caption + detail

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name



'''For use of training motion-2-text generative model'''
class Motion2TextDataset(data.Dataset):
    def __init__(self, dataset_name, split, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4):

        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
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
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = './checkpoints/kit/Decomp_SP001_SM001_H512/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, f'{split}.txt')

        min_motion_len = 40 if self.dataset_name == 't2m' else 24

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.mean_2d = np.load(ROOT_DIR / f"{dataset_name}_motion_2d_mean.npy")
        self.std_2d = np.load(ROOT_DIR / f"{dataset_name}_motion_2d_std.npy")
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        all_captions = [' '.join(
            [token.split('/')[0] for token in text_dic['tokens']]
        ) for text_dic in text_list]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3


        if len(tokens) < self.max_text_len:         # max_text_len = 20
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion_2d = _create_2d_joints_from_features(motion)
        motion = motion[idx:idx + m_length]
        motion_2d = motion_2d[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
            motion_2d = np.concatenate([motion_2d,
                                        np.zeros((self.max_motion_length - m_length, motion_2d.shape[1]))
                                        ], axis=0)
        motion_2d = np.concatenate([motion_2d,
                                    np.zeros((motion_2d.shape[0], 263 - motion_2d.shape[1]))
                                    ], axis=-1)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, motion_2d, m_length, '_'.join(tokens), name, all_captions



'''For use of training text-2-motion generative model'''
class Motion2MotionScriptDataset(data.Dataset):
    def __init__(self, dataset_name, split):

        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.finemotion_text_dir = pjoin(self.data_root, 'finemotion_texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
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
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = './checkpoints/kit/Decomp_SP001_SM001_H512/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        # detailed text for motions
        BPMSD_auto_file = pjoin(self.finemotion_text_dir, 'BPMSD_auto.json')
        with open(BPMSD_auto_file, 'r') as f:
            BPMSD_dict = json.load(f)

        BPMSD_human_file = pjoin(self.finemotion_text_dir, 'BPMSD_human.json')
        with open(BPMSD_human_file, 'r') as f:
            BPMSD_human_dict = json.load(f)
        BPMSD_dict.update(BPMSD_human_dict)

        split_file = pjoin(self.data_root, f'{split}.txt')

        min_motion_len = 40 if self.dataset_name == 't2m' else 24

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['tokens'] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:

                            bodyPart_text_list = BPMSD_dict[name]

                            summary_detail_text_dict = text_dict.copy()
                            summary_detail_text_dict['summary'] = caption
                            summary_detail_text_dict['detail'] = bodyPart_text_list
                            text_data.append(summary_detail_text_dict)

                            flag = True

                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue

                                bodyPart_text_list = BPMSD_dict[name][int(f_tag / 0.5): int(to_tag / 0.5)]

                                text_data_new = []

                                # summary + detail
                                summary_detail_text_dict = text_dict.copy()
                                summary_detail_text_dict['summary'] = caption
                                summary_detail_text_dict['detail'] = bodyPart_text_list
                                text_data_new.append(summary_detail_text_dict)

                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name

                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': text_data_new}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.mean_2d = np.load(ROOT_DIR / f"{dataset_name}_motion_2d_mean.npy")
        self.std_2d = np.load(ROOT_DIR / f"{dataset_name}_motion_2d_std.npy")
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        text_data = random.choice(text_list)
        bodyPart_text_list = text_data['detail'][:]

        # Each item in bodyPart_text_list corresponds to 0.5 seconds, i.e., 10 frames of motion.
        # Here, we ensure strict alignment between motion tokens and detailed body part movement descriptions.
        m_length = (m_length // 20) * 20        # 20 is the least common multiple of 10 (corresponds to 1 item in bodyPart_text_list) and 4 (self.unit_length, also the frames of a singel token).
        motion = motion[:m_length]
        motion_2d = _create_2d_joints_from_features(motion)
        bodyPart_text_list = bodyPart_text_list[:int(m_length/10)]

        for i in range(len(bodyPart_text_list)):
            bodyPart_text_item = bodyPart_text_list[i]
            if bodyPart_text_item == "":
                bodyPart_text_list[i] = '<Motionless>'
        long_text = (" <SEP> ").join(bodyPart_text_list)
        detail = long_text
        caption = detail

        "Z Normalization"
        motion = (motion - self.mean) / self.std
        motion_2d = (motion_2d - self.mean_2d) / self.std_2d

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
            motion_2d = np.concatenate([motion_2d,
                                        np.zeros((self.max_motion_length - m_length, motion_2d.shape[1]))
                                        ], axis=0)
        motion_2d = np.concatenate([motion_2d,
                                    np.zeros((motion_2d.shape[0], 263 - motion_2d.shape[1]))
                                    ], axis=-1)

        return caption, motion, motion_2d, m_length, name





def DATALoader(dataset_name, split,
               batch_size, w_vectorizer,
               num_workers=8, unit_length=4):
    val_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(dataset_name, split, w_vectorizer, unit_length=unit_length),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True)
    return val_loader



def DATALoader_tdt2m(dataset_name, split,
               batch_size, w_vectorizer,
               num_workers=8, unit_length=4):
    val_loader = torch.utils.data.DataLoader(
        Text2MotionDataset_withMotionScript(dataset_name, split, w_vectorizer, unit_length=unit_length),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True)
    return val_loader



def M2T_DATALoader(dataset_name, split,
                batch_size, w_vectorizer,
               num_workers=8, unit_length=4):
    val_loader = torch.utils.data.DataLoader(
        Motion2TextDataset(dataset_name, split, w_vectorizer, unit_length=unit_length),
        batch_size,
        shuffle = True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True)
    return val_loader



def M2DT_DATALoader(dataset_name, split,
                batch_size,
               num_workers=8):
    val_loader = torch.utils.data.DataLoader(
        Motion2MotionScriptDataset(dataset_name, split),
        batch_size,
        shuffle = True,
        num_workers=num_workers,
        drop_last=True)
    return val_loader



def cycle(iterable):
    while True:
        for x in iterable:
            yield x
