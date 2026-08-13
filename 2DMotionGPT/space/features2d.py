"""Standalone 2D keypoint -> feature conversion.

Extracted verbatim from src.data.adapter_datasets._VitPoseMixin so the demo
carries no dependency on Text2MotionDataset / HumanML3D / glove.

Two feature layouts are produced:
  * 81-dim "estimated" features  (with per-joint confidence) -> adapter input
  * 68-dim features             (without confidence)         -> adapter-less input
"""

import json

import numpy as np

# COCO-17 -> COCO-13: drop eyes/ears (indices 1,2,3,4)
COCO17_TO_COCO13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

COCO13 = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip_extra', 'right_hip_extra',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

_LHIP = COCO13.index('left_hip_extra')
_RHIP = COCO13.index('right_hip_extra')


def load_vitpose_json(path):
    """<id>.json -> (keypoints (T,17,2), confidences (T,17)) as float32."""
    data = json.load(open(path))
    kp = np.array([f["instances"][0]["keypoints"] for f in data], dtype=np.float32)
    cf = np.array([f["instances"][0]["keypoint_scores"] for f in data], dtype=np.float32)
    return kp, cf


def to_coco13(kp, cf):
    return kp[:, COCO17_TO_COCO13, :], cf[:, COCO17_TO_COCO13]


def normalize_2d_coco13_midhip(joints_2d, eps=1e-8, q=99):
    """Mid-hip-centred, scale-normalised 2D joints. Scale makes this
    resolution-independent, so raw pixel coordinates are fine as input."""
    joints_2d = np.asarray(joints_2d)
    root_pos = 0.5 * (joints_2d[:, _LHIP, :] + joints_2d[:, _RHIP, :])
    joints_rel = joints_2d - root_pos[:, None, :]
    abs_xy = np.abs(joints_rel).reshape(-1, 2)
    s = max(np.percentile(abs_xy[:, 0], q), np.percentile(abs_xy[:, 1], q), eps)
    return root_pos, joints_rel, s


def decompose_2d_motion_coco13_midhip_root(joints_2d):
    root_pos, joints_rel, s = normalize_2d_coco13_midhip(joints_2d)
    root_y_2d = (root_pos[:, 1:2] / s).astype(np.float32)
    root_y_2d = root_y_2d - root_y_2d[0:1]
    joints_pos_2d = (joints_rel / s).reshape(joints_rel.shape[0], -1).astype(np.float32)
    root_norm = (root_pos / s).astype(np.float32)
    root_vel_2d = np.zeros_like(root_norm)
    root_vel_2d[1:] = root_norm[1:] - root_norm[:-1]
    return root_y_2d, joints_pos_2d, root_vel_2d


def compute_joint_features_2d_coco13(joints_2d):
    _, joints_rel, s = normalize_2d_coco13_midhip(joints_2d)
    joints_rel_norm = (joints_rel / s).astype(np.float32)
    rot = np.arctan2(joints_rel_norm[:, :, 1], joints_rel_norm[:, :, 0]).astype(np.float32)
    vel = np.zeros_like(joints_rel_norm)
    vel[1:] = joints_rel_norm[1:] - joints_rel_norm[:-1]
    vel = vel.reshape(joints_rel_norm.shape[0], -1).astype(np.float32)
    return rot, vel


def feature_68(joints_2d):
    """[root_vel(2), root_y(1), joints_pos(26), joints_rot(13), joints_vel(26)]"""
    root_y, joints_pos, root_vel = decompose_2d_motion_coco13_midhip_root(joints_2d)
    joints_rot, joints_vel = compute_joint_features_2d_coco13(joints_2d)
    return np.concatenate([root_vel, root_y, joints_pos, joints_rot, joints_vel], axis=-1)


def feature_81(joints_2d, conf):
    """feature_68 with per-joint confidence appended (adapter input)."""
    feat = feature_68(joints_2d)
    c = np.asarray(conf).reshape(feat.shape[0], -1)
    return np.concatenate([feat, c], axis=-1)
