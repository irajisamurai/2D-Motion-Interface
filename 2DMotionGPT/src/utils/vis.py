import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

JP = {
    "leftUpLeg": 0, "rightUpLeg": 1, "spine": 2,
    "leftLeg": 3, "rightLeg": 4, "spine1": 5,
    "leftFoot": 6, "rightFoot": 7, "spine2": 8,
    "leftToeBase": 9, "rightToeBase": 10, "neck": 11,
    "leftShoulder": 12, "rightShoulder": 13, "head": 14,
    "leftArm": 15, "rightArm": 16, "leftForeArm": 17,
    "rightForeArm": 18, "leftHand": 19, "rightHand": 20,
}

SKELETON_EDGES = [
    (JP["spine"], JP["spine1"]), (JP["spine1"], JP["spine2"]),
    (JP["spine2"], JP["neck"]), (JP["neck"], JP["head"]),

    (JP["neck"], JP["leftShoulder"]), (JP["leftShoulder"], JP["leftArm"]),
    (JP["leftArm"], JP["leftForeArm"]), (JP["leftForeArm"], JP["leftHand"]),

    (JP["neck"], JP["rightShoulder"]), (JP["rightShoulder"], JP["rightArm"]),
    (JP["rightArm"], JP["rightForeArm"]), (JP["rightForeArm"], JP["rightHand"]),

    (JP["spine"], JP["leftUpLeg"]), (JP["leftUpLeg"], JP["leftLeg"]),
    (JP["leftLeg"], JP["leftFoot"]), (JP["leftFoot"], JP["leftToeBase"]),

    (JP["spine"], JP["rightUpLeg"]), (JP["rightUpLeg"], JP["rightLeg"]),
    (JP["rightLeg"], JP["rightFoot"]), (JP["rightFoot"], JP["rightToeBase"]),
]


def save_mp4_motion_3d_with_root(root_y_position,
                                 joints_positions,
                                 root_linear_velocity,
                                 save_path="motion_3d_root.mp4",
                                 fps=20,
                                 step=1):
    """

    root_y_position     : (T, 1)
    joints_positions    : (T, 63) = 21 * 3
    root_linear_velocity: (T, 2) = [vx, vz]
    """
    T = joints_positions.shape[0]
    n_joints = 21
    assert root_linear_velocity.shape[0] == T

    # (T, 63) → (T, 21, 3)
    joints = joints_positions.reshape(T, n_joints, 3).copy()

    joints[:, :, 1] += root_y_position

    dt = 1.0
    root_pos_xz = np.cumsum(root_linear_velocity * dt, axis=0)  # (T, 2)

    root_pos_xz = root_pos_xz - root_pos_xz[0:1]

    joints[:, :, 0] += root_pos_xz[:, 0:1]  # x
    joints[:, :, 2] += root_pos_xz[:, 1:1+1]  # z

    all_xyz = joints.reshape(-1, 3)
    max_range = np.max(np.ptp(all_xyz, axis=0)) / 2.0
    mid = all_xyz.mean(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter([], [], [], s=25)
    lines = [ax.plot([], [], [], lw=2)[0] for _ in SKELETON_EDGES]

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[2] - max_range, mid[2] + max_range)
    ax.set_zlim(mid[1] - max_range, mid[1] + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.invert_yaxis()

    def update(frame):
        f = frame * step
        if f >= T:
            f = T - 1
        xyz = joints[f]

        scat._offsets3d = (xyz[:, 0], xyz[:, 2], xyz[:, 1])

        for line, (a, b) in zip(lines, SKELETON_EDGES):
            line.set_data([xyz[a, 0], xyz[b, 0]],
                          [xyz[a, 2], xyz[b, 2]])
            line.set_3d_properties([xyz[a, 1], xyz[b, 1]])

        ax.set_title(f"Frame {f}/{T-1}")
        return [scat] + lines

    n_frames = (T + step - 1) // step
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"Saved 3D motion with root to: {save_path}")

def reconstruct_2d_motion_to_hips(root_y_position_2d,
                                  joints_positions_2d,
                                  root_linear_velocity_2d,
                                  root_x0: float = 0.0):
    """

    Parameters
    ----------
    root_y_position_2d : (T,1)
    joints_positions_2d : (T, 21*2) = (T,42)
    root_linear_velocity_2d : (T,2)
    root_x0 : float, default 0.0

    Returns
    -------
    joints_2d : (T,22,2)
    """
    T = root_y_position_2d.shape[0]
    assert root_y_position_2d.shape == (T, 1), f"Expected (T,1) for root_y_position_2d, got {root_y_position_2d.shape}"
    assert joints_positions_2d.shape == (T, 21 * 2), f"Expected (T,42) for joints_positions_2d, got {joints_positions_2d.shape}"
    assert root_linear_velocity_2d.shape == (T, 2), f"Expected (T,2) for root_linear_velocity_2d, got {root_linear_velocity_2d.shape}"

    hip_idx = 0

    root_pos_2d = np.cumsum(root_linear_velocity_2d, axis=0)  # (T,2)

    offset = np.array([root_x0, root_y_position_2d[0, 0]], dtype=root_pos_2d.dtype)
    root_pos_2d += offset

    root_pos_2d[:, 1] = root_y_position_2d[:, 0]

    # joints_positions_2d: (T,42) → (T,21,2)
    joints_local_wo_hips = joints_positions_2d.reshape(T, 21, 2)  # (T,21,2)

    joints_local_2d = np.zeros((T, 22, 2), dtype=joints_local_wo_hips.dtype)  # (T,22,2)
    joints_local_2d[:, 1:, :] = joints_local_wo_hips

    joints_2d = joints_local_2d + root_pos_2d[:, None, :]  # (T,22,2)

    return joints_2d

def save_2d_view_mp4(joints_global,
                     save_path="view_front.mp4",
                     fps=20,
                     step=1,
                     yaw_deg=0.0,
                     pitch_deg=0.0,
                     invert_y=True):
    """
    """
    T, n_joints, _ = joints_global.shape
    assert n_joints == 22, f"Expected 22 joints (including hips), got {n_joints}"

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0,           1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    Rx = np.array([
        [1,            0,             0          ],
        [0,  np.cos(pitch), -np.sin(pitch)],
        [0,  np.sin(pitch),  np.cos(pitch)]
    ])

    R = Rx @ Ry

    all_xyz = joints_global.reshape(-1, 3)
    center = all_xyz.mean(axis=0)

    all_rel = all_xyz - center
    all_cam = all_rel @ R.T
    x_all = all_cam[:, 0]
    y_all = all_cam[:, 1]
    max_range = max(x_all.max() - x_all.min(),
                    y_all.max() - y_all.min()) / 2.0
    mid_x = (x_all.max() + x_all.min()) * 0.5
    mid_y = (y_all.max() + y_all.min()) * 0.5

    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=25)
    lines = [ax.plot([], [], lw=2)[0] for _ in SKELETON_EDGES]

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X_cam")
    ax.set_ylabel("Y_cam")

    if invert_y:
        ax.invert_yaxis()

    def update(frame):
        f = frame * step
        if f >= T:
            f = T - 1
        xyz = joints_global[f]              # (22,3)
        xyz_rel = xyz - center
        xyz_cam = xyz_rel @ R.T

        x = xyz_cam[:, 0]
        y = xyz_cam[:, 1]

        scat.set_offsets(np.stack([x, y], axis=1))

        for line, (a, b) in zip(lines, SKELETON_EDGES):
            line.set_data([x[a], x[b]], [y[a], y[b]])

        ax.set_title(f"Frame {f}/{T-1}  yaw={yaw_deg}, pitch={pitch_deg}")
        return [scat] + lines

    n_frames = (T + step - 1) // step
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"Saved 2D view to: {save_path}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ===== 3D skeleton (21 joints, no hips) =====
JP = {
    "leftUpLeg": 0, "rightUpLeg": 1, "spine": 2,
    "leftLeg": 3, "rightLeg": 4, "spine1": 5,
    "leftFoot": 6, "rightFoot": 7, "spine2": 8,
    "leftToeBase": 9, "rightToeBase": 10, "neck": 11,
    "leftShoulder": 12, "rightShoulder": 13, "head": 14,
    "leftArm": 15, "rightArm": 16, "leftForeArm": 17,
    "rightForeArm": 18, "leftHand": 19, "rightHand": 20,
}

SKELETON_EDGES_3D = [
    (JP["spine"], JP["spine1"]), (JP["spine1"], JP["spine2"]),
    (JP["spine2"], JP["neck"]), (JP["neck"], JP["head"]),
    (JP["neck"], JP["leftShoulder"]), (JP["leftShoulder"], JP["leftArm"]),
    (JP["leftArm"], JP["leftForeArm"]), (JP["leftForeArm"], JP["leftHand"]),
    (JP["neck"], JP["rightShoulder"]), (JP["rightShoulder"], JP["rightArm"]),
    (JP["rightArm"], JP["rightForeArm"]), (JP["rightForeArm"], JP["rightHand"]),
    (JP["spine"], JP["leftUpLeg"]), (JP["leftUpLeg"], JP["leftLeg"]),
    (JP["leftLeg"], JP["leftFoot"]), (JP["leftFoot"], JP["leftToeBase"]),
    (JP["spine"], JP["rightUpLeg"]), (JP["rightUpLeg"], JP["rightLeg"]),
    (JP["rightLeg"], JP["rightFoot"]), (JP["rightFoot"], JP["rightToeBase"]),
]

# ===== 2D skeleton (22 joints, with hips=0) =====
SMPL_JOINTS_2D = {
    'hips' : 0,
    'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3,
    'leftLeg' : 4, 'rightLeg' : 5, 'spine1' : 6,
    'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9,
    'leftToeBase' : 10, 'rightToeBase' : 11,
    'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14,
    'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
    'leftForeArm' : 18, 'rightForeArm' : 19,
    'leftHand' : 20, 'rightHand' : 21
}

SKELETON_EDGES_2D = [
    (SMPL_JOINTS_2D['hips'],  SMPL_JOINTS_2D['spine']),
    (SMPL_JOINTS_2D['spine'], SMPL_JOINTS_2D['spine1']),
    (SMPL_JOINTS_2D['spine1'], SMPL_JOINTS_2D['spine2']),
    (SMPL_JOINTS_2D['spine2'], SMPL_JOINTS_2D['neck']),
    (SMPL_JOINTS_2D['neck'], SMPL_JOINTS_2D['head']),
    (SMPL_JOINTS_2D['neck'], SMPL_JOINTS_2D['leftShoulder']),
    (SMPL_JOINTS_2D['leftShoulder'], SMPL_JOINTS_2D['leftArm']),
    (SMPL_JOINTS_2D['leftArm'], SMPL_JOINTS_2D['leftForeArm']),
    (SMPL_JOINTS_2D['leftForeArm'], SMPL_JOINTS_2D['leftHand']),
    (SMPL_JOINTS_2D['neck'], SMPL_JOINTS_2D['rightShoulder']),
    (SMPL_JOINTS_2D['rightShoulder'], SMPL_JOINTS_2D['rightArm']),
    (SMPL_JOINTS_2D['rightArm'], SMPL_JOINTS_2D['rightForeArm']),
    (SMPL_JOINTS_2D['rightForeArm'], SMPL_JOINTS_2D['rightHand']),
    (SMPL_JOINTS_2D['hips'], SMPL_JOINTS_2D['leftUpLeg']),
    (SMPL_JOINTS_2D['leftUpLeg'], SMPL_JOINTS_2D['leftLeg']),
    (SMPL_JOINTS_2D['leftLeg'], SMPL_JOINTS_2D['leftFoot']),
    (SMPL_JOINTS_2D['leftFoot'], SMPL_JOINTS_2D['leftToeBase']),
    (SMPL_JOINTS_2D['hips'], SMPL_JOINTS_2D['rightUpLeg']),
    (SMPL_JOINTS_2D['rightUpLeg'], SMPL_JOINTS_2D['rightLeg']),
    (SMPL_JOINTS_2D['rightLeg'], SMPL_JOINTS_2D['rightFoot']),
    (SMPL_JOINTS_2D['rightFoot'], SMPL_JOINTS_2D['rightToeBase']),
]


def _feats3d_to_joints21_with_root(sample_motion_3d):
    """
    sample_motion_3d: (T, D) where you use:
      root_y = [:,3]
      root_vel_xz = [:,1:3]  (vx, vz)
      joints_pos = [:,4:4+63] (21*3)
    return: joints (T,21,3) in global coordinates (root translation applied)
    """
    root_y_position = sample_motion_3d[:, 3].reshape(-1, 1)          # (T,1)
    joints_positions = sample_motion_3d[:, 4:4+21*3]                 # (T,63)
    root_linear_velocity = sample_motion_3d[:, 1:3]                  # (T,2)

    T = joints_positions.shape[0]
    joints = joints_positions.reshape(T, 21, 3).copy()

    # add root Y
    joints[:, :, 1] += root_y_position

    # integrate root xz
    root_pos_xz = np.cumsum(root_linear_velocity, axis=0)
    root_pos_xz = root_pos_xz - root_pos_xz[0:1]

    joints[:, :, 0] += root_pos_xz[:, 0:1]
    joints[:, :, 2] += root_pos_xz[:, 1:2]
    return joints


def _motion2d_to_joints2d(motion_2d, mean_2d=None, std_2d=None, is_normalized=True):
    """
    motion_2d: (T,108) = [root_vel(2), root_y(1), joints_pos(42), joints_rot(21), joints_vel(42)]
    return joints_2d: (T,22,2)
    """
    if is_normalized:
        assert mean_2d is not None and std_2d is not None
        motion_2d = motion_2d * std_2d + mean_2d

    root_vel_2d   = motion_2d[:, 0:2]
    root_y_2d     = motion_2d[:, 2:3]
    joints_pos_2d = motion_2d[:, 3:3+42]

    T = motion_2d.shape[0]
    root_pos_2d = np.cumsum(root_vel_2d, axis=0)
    root_pos_2d = root_pos_2d - root_pos_2d[0:1]
    root_pos_2d[:, 1:2] = root_y_2d

    joints_local = joints_pos_2d.reshape(T, 21, 2)
    joints_2d = np.zeros((T, 22, 2), dtype=motion_2d.dtype)
    joints_2d[:, 0, :] = root_pos_2d
    joints_2d[:, 1:, :] = root_pos_2d[:, None, :] + joints_local
    return joints_2d


def save_triplet_mp4(gt_motion_3d,
                     in_motion_2d,
                     recon_motion_3d,
                     save_path="triplet.mp4",
                     mean_2d=None,
                     std_2d=None,
                     in_2d_is_normalized=True,
                     fps=20,
                     step=1,
                     invert_y_2d=True,
                     elev=15, azim=-70,
                     pad_ratio=0.15):
    """
    """

    # --- build joints arrays ---
    gt_j3d   = _feats3d_to_joints21_with_root(gt_motion_3d)        # (T,21,3)
    rec_j3d  = _feats3d_to_joints21_with_root(recon_motion_3d)     # (T,21,3)

    if in_motion_2d.ndim == 3 and in_motion_2d.shape[1:] == (22, 2):
        in_j2d = in_motion_2d
    else:
        in_j2d = _motion2d_to_joints2d(in_motion_2d, mean_2d, std_2d, is_normalized=in_2d_is_normalized)

    # --- align length ---
    T = min(gt_j3d.shape[0], in_j2d.shape[0], rec_j3d.shape[0])
    gt_j3d  = gt_j3d[:T]
    in_j2d  = in_j2d[:T]
    rec_j3d = rec_j3d[:T]

    # --- fixed axis ranges ---
    # 3D range: use both gt+recon for fair comparison
    all_3d = np.concatenate([gt_j3d.reshape(-1,3), rec_j3d.reshape(-1,3)], axis=0)
    mid3 = all_3d.mean(axis=0)
    max_range3 = np.max(np.ptp(all_3d, axis=0)) / 2.0
    max_range3 = max_range3 * (1.0 + pad_ratio)

    # 2D range
    x_all = in_j2d[..., 0].reshape(-1)
    y_all = in_j2d[..., 1].reshape(-1)
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    x_pad = (x_max - x_min) * pad_ratio + 1e-9
    y_pad = (y_max - y_min) * pad_ratio + 1e-9

    # --- figure ---
    fig = plt.figure(figsize=(15, 5))
    axL = fig.add_subplot(1, 3, 1, projection="3d")
    axM = fig.add_subplot(1, 3, 2)
    axR = fig.add_subplot(1, 3, 3, projection="3d")

    # set 3D view
    for ax in (axL, axR):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(mid3[0] - max_range3, mid3[0] + max_range3)
        ax.set_ylim(mid3[2] - max_range3, mid3[2] + max_range3)  # Z shown on Y-axis label
        ax.set_zlim(mid3[1] - max_range3, mid3[1] + max_range3)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.invert_yaxis()

    # 2D axis
    axM.set_xlim(x_min - x_pad, x_max + x_pad)
    axM.set_ylim(y_min - y_pad, y_max + y_pad)
    axM.set_aspect("equal", adjustable="box")
    axM.set_xlabel("x")
    axM.set_ylabel("y")
    if invert_y_2d:
        axM.invert_yaxis()

    # artists: left 3D
    scatL = axL.scatter([], [], [], s=25)
    linesL = [axL.plot([], [], [], lw=2)[0] for _ in SKELETON_EDGES_3D]
    axL.set_title("GT 3D")

    # artists: middle 2D
    scatM = axM.scatter([], [], s=25)
    linesM = [axM.plot([], [], lw=2)[0] for _ in SKELETON_EDGES_2D]
    axM.set_title("Input 2D")

    # artists: right 3D
    scatR = axR.scatter([], [], [], s=25)
    linesR = [axR.plot([], [], [], lw=2)[0] for _ in SKELETON_EDGES_3D]
    axR.set_title("Recon 3D")

    def update(frame_idx):
        f = frame_idx * step
        if f >= T:
            f = T - 1

        # --- GT 3D ---
        xyz = gt_j3d[f]  # (21,3)
        scatL._offsets3d = (xyz[:,0], xyz[:,2], xyz[:,1])
        for line, (a,b) in zip(linesL, SKELETON_EDGES_3D):
            line.set_data([xyz[a,0], xyz[b,0]], [xyz[a,2], xyz[b,2]])
            line.set_3d_properties([xyz[a,1], xyz[b,1]])

        # --- Input 2D ---
        xy = in_j2d[f]   # (22,2)
        x, y = xy[:,0], xy[:,1]
        scatM.set_offsets(np.stack([x,y], axis=1))
        for line, (a,b) in zip(linesM, SKELETON_EDGES_2D):
            line.set_data([x[a], x[b]], [y[a], y[b]])

        # --- Recon 3D ---
        xyz = rec_j3d[f]
        scatR._offsets3d = (xyz[:,0], xyz[:,2], xyz[:,1])
        for line, (a,b) in zip(linesR, SKELETON_EDGES_3D):
            line.set_data([xyz[a,0], xyz[b,0]], [xyz[a,2], xyz[b,2]])
            line.set_3d_properties([xyz[a,1], xyz[b,1]])

        fig.suptitle(f"Frame {f}/{T-1}", y=0.98)
        return [scatL, scatM, scatR] + linesL + linesM + linesR

    n_frames = (T + step - 1) // step
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"Saved triplet mp4 to: {save_path}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =========================
# COCO13 definition & edges
# =========================
COCO13_KEYPOINTS = [
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

def coco13_edges():
    n2i = {n:i for i,n in enumerate(COCO13_KEYPOINTS)}
    edges_name = [
        # arms
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        # legs
        ('left_hip_extra', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip_extra', 'right_knee'),
        ('right_knee', 'right_ankle'),
        # torso
        ('left_shoulder', 'right_shoulder'),
        ('left_hip_extra', 'right_hip_extra'),
        ('left_shoulder', 'left_hip_extra'),
        ('right_shoulder', 'right_hip_extra'),
        # head-ish
        ('nose', 'left_shoulder'),
        ('nose', 'right_shoulder'),
    ]
    return [(n2i[a], n2i[b]) for a,b in edges_name]

SKELETON_EDGES_2D_COCO13 = coco13_edges()


# =========================
# (T,68) -> (T,13,2) recon
# =========================
def reconstruct_coco2d_from_features(result,mean_2d=None, std_2d=None, is_normalized=True, root_init_xy=None,):
    """
    result: (T,68) = [root_vel(2), root_y(1), joints_pos(26), rot(13), vel(26)]
    return: joints_2d (T,13,2)

    """
    if is_normalized:
        assert mean_2d is not None and std_2d is not None
        result = result * std_2d + mean_2d
    result = np.asarray(result)
    T, D = result.shape
    assert D == 68, f"Expected feature dim 68, got {D}"

    idx = 0
    root_vel = result[:, idx:idx+2]; idx += 2            # (T,2)
    root_y   = result[:, idx:idx+1]; idx += 1            # (T,1)
    joints_pos = result[:, idx:idx+26]; idx += 26        # (T,26)
    # rot = result[:, idx:idx+13]; idx += 13             # (T,13)  # not needed
    # vel = result[:, idx:idx+26]                        # (T,26)  # not needed

    root_xy = np.zeros((T, 2), dtype=result.dtype)
    if root_init_xy is None:
        root_xy[0, 0] = 0.0
        root_xy[0, 1] = root_y[0, 0]
    else:
        root_xy[0] = root_init_xy

    for t in range(1, T):
        root_xy[t] = root_xy[t-1] + root_vel[t]

    root_xy[:, 1] = root_y[:, 0]

    joints_rel = joints_pos.reshape(T, 13, 2)            # (T,13,2)
    joints_2d  = joints_rel + root_xy[:, None, :]        # (T,13,2)
    return joints_2d


# ============================================================
# ============================================================
def save_triplet_mp4_coco68(
    gt_motion_3d,
    in_motion_2d,
    recon_motion_3d,
    save_path="triplet.mp4",
    mean_2d=None,
    std_2d=None,
    in_2d_is_normalized=True,
    fps=20,
    step=1,
    invert_y_2d=True,
    elev=15, azim=-70,
    pad_ratio=0.15,
    root_init_xy=None,
):
    """

    """

    # --- build joints arrays (3D) ---
    gt_j3d  = _feats3d_to_joints21_with_root(gt_motion_3d)     # (T,21,3)
    rec_j3d = _feats3d_to_joints21_with_root(recon_motion_3d)  # (T,21,3)

    # --- build joints arrays (2D input) ---
    if in_motion_2d.ndim == 3 and in_motion_2d.shape[1:] == (22, 2):
        in_j2d = in_motion_2d
        edges_2d = SKELETON_EDGES_2D
        input_title = "Input 2D (22j)"
    elif in_motion_2d.ndim == 2 and in_motion_2d.shape[1] == 68:
        in_j2d = reconstruct_coco2d_from_features(in_motion_2d, mean_2d=mean_2d, std_2d=std_2d, is_normalized=in_2d_is_normalized, root_init_xy=root_init_xy)  # (T,13,2)
        edges_2d = SKELETON_EDGES_2D_COCO13
        input_title = "Input 2D (COCO13 from 68D)"
    else:
        in_j2d = _motion2d_to_joints2d(in_motion_2d, mean_2d, std_2d, is_normalized=in_2d_is_normalized)
        edges_2d = SKELETON_EDGES_2D
        input_title = "Input 2D"

    # --- align length ---
    T = min(gt_j3d.shape[0], in_j2d.shape[0], rec_j3d.shape[0])
    gt_j3d  = gt_j3d[:T]
    in_j2d  = in_j2d[:T]
    rec_j3d = rec_j3d[:T]

    # --- root-relative centering: remove XZ root translation so range is tight ---
    xz_mean_gt  = gt_j3d.mean(axis=1, keepdims=True).copy();  xz_mean_gt[:, :, 1]  = 0
    xz_mean_rec = rec_j3d.mean(axis=1, keepdims=True).copy(); xz_mean_rec[:, :, 1] = 0
    gt_j3d_vis  = gt_j3d  - xz_mean_gt
    rec_j3d_vis = rec_j3d - xz_mean_rec
    in_j2d_vis  = in_j2d  - in_j2d.mean(axis=1, keepdims=True)

    # --- axis ranges (body-size, not trajectory-size) ---
    all_3d = np.concatenate([gt_j3d_vis.reshape(-1, 3), rec_j3d_vis.reshape(-1, 3)], axis=0)
    mid3 = all_3d.mean(axis=0)
    max_range3 = np.max(np.ptp(all_3d, axis=0)) / 2.0
    max_range3 = max_range3 * (1.0 + pad_ratio)

    x_all = in_j2d_vis[..., 0].reshape(-1)
    y_all = in_j2d_vis[..., 1].reshape(-1)
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    x_pad = (x_max - x_min) * pad_ratio + 1e-9
    y_pad = (y_max - y_min) * pad_ratio + 1e-9

    # --- figure ---
    fig = plt.figure(figsize=(15, 5))
    axL = fig.add_subplot(1, 3, 1, projection="3d")
    axM = fig.add_subplot(1, 3, 2)
    axR = fig.add_subplot(1, 3, 3, projection="3d")

    # set 3D view
    for ax in (axL, axR):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(mid3[0] - max_range3, mid3[0] + max_range3)
        ax.set_ylim(mid3[2] - max_range3, mid3[2] + max_range3)
        ax.set_zlim(mid3[1] - max_range3, mid3[1] + max_range3)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.invert_yaxis()

    # 2D axis
    axM.set_xlim(x_min - x_pad, x_max + x_pad)
    axM.set_ylim(y_min - y_pad, y_max + y_pad)
    axM.set_aspect("equal", adjustable="box")
    axM.set_xlabel("x")
    axM.set_ylabel("y")
    if invert_y_2d:
        axM.invert_yaxis()

    # artists: left 3D
    scatL = axL.scatter([], [], [], s=25)
    linesL = [axL.plot([], [], [], lw=2)[0] for _ in SKELETON_EDGES_3D]
    axL.set_title("GT 3D")

    # artists: middle 2D
    scatM = axM.scatter([], [], s=25)
    linesM = [axM.plot([], [], lw=2)[0] for _ in edges_2d]
    axM.set_title(input_title)

    # artists: right 3D
    scatR = axR.scatter([], [], [], s=25)
    linesR = [axR.plot([], [], [], lw=2)[0] for _ in SKELETON_EDGES_3D]
    axR.set_title("Recon 3D")

    def update(frame_idx):
        f = frame_idx * step
        if f >= T:
            f = T - 1

        # --- GT 3D (root-relative) ---
        xyz = gt_j3d_vis[f]
        scatL._offsets3d = (xyz[:, 0], xyz[:, 2], xyz[:, 1])
        for line, (a, b) in zip(linesL, SKELETON_EDGES_3D):
            line.set_data([xyz[a, 0], xyz[b, 0]], [xyz[a, 2], xyz[b, 2]])
            line.set_3d_properties([xyz[a, 1], xyz[b, 1]])

        # --- Input 2D (root-relative) ---
        xy = in_j2d_vis[f]
        x, y = xy[:, 0], xy[:, 1]
        scatM.set_offsets(np.stack([x, y], axis=1))
        for line, (a, b) in zip(linesM, edges_2d):
            line.set_data([x[a], x[b]], [y[a], y[b]])

        # --- Recon 3D (root-relative) ---
        xyz = rec_j3d_vis[f]
        scatR._offsets3d = (xyz[:, 0], xyz[:, 2], xyz[:, 1])
        for line, (a, b) in zip(linesR, SKELETON_EDGES_3D):
            line.set_data([xyz[a, 0], xyz[b, 0]], [xyz[a, 2], xyz[b, 2]])
            line.set_3d_properties([xyz[a, 1], xyz[b, 1]])

        fig.suptitle(f"Frame {f}/{T-1}", y=0.98)
        return [scatL, scatM, scatR] + linesL + linesM + linesR

    n_frames = (T + step - 1) // step
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"Saved triplet mp4 to: {save_path}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import textwrap

# =========================
# COCO13 definition & edges
# =========================
COCO13_KEYPOINTS = [
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

def coco13_edges():
    n2i = {n: i for i, n in enumerate(COCO13_KEYPOINTS)}
    edges_name = [
        # arms
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        # legs
        ('left_hip_extra', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip_extra', 'right_knee'),
        ('right_knee', 'right_ankle'),
        # torso
        ('left_shoulder', 'right_shoulder'),
        ('left_hip_extra', 'right_hip_extra'),
        ('left_shoulder', 'left_hip_extra'),
        ('right_shoulder', 'right_hip_extra'),
        # head-ish
        ('nose', 'left_shoulder'),
        ('nose', 'right_shoulder'),
    ]
    return [(n2i[a], n2i[b]) for a, b in edges_name]

SKELETON_EDGES_2D_COCO13 = coco13_edges()


# =========================
# (T,68) -> (T,13,2) recon
# =========================

def save_triplet_mp4_with_gt_and_pred_caption(
    gt_motion_3d,
    in_motion_2d,
    gt_caption_text,                 # str or list[str] length T
    pred_caption_text,               # str or list[str] length T
    save_path="triplet_gt_pred_caption.mp4",
    mean_2d=None,
    std_2d=None,
    in_2d_is_normalized=True,
    fps=20,
    step=1,
    invert_y_2d=True,
    elev=15, azim=-70,
    pad_ratio=0.15,
    caption_wrap=52,
    caption_fontsize=11,
    root_init_xy=None,
):
    """
    """

    # --- build joints arrays ---
    gt_j3d = _feats3d_to_joints21_with_root(gt_motion_3d)

    if in_motion_2d.ndim == 3 and in_motion_2d.shape[1:] == (22, 2):
        in_j2d = in_motion_2d
        edges_2d = SKELETON_EDGES_2D
        input_title = "Input 2D (22j)"
    elif in_motion_2d.ndim == 2 and in_motion_2d.shape[1] == 68:
        in_j2d = reconstruct_coco2d_from_features(in_motion_2d, mean_2d=mean_2d, std_2d=std_2d, is_normalized=in_2d_is_normalized, root_init_xy=root_init_xy)  # (T,13,2)
        edges_2d = SKELETON_EDGES_2D_COCO13
        input_title = "Input 2D (COCO13 from 68D)"
    else:
        in_j2d = _motion2d_to_joints2d(in_motion_2d, mean_2d, std_2d, is_normalized=in_2d_is_normalized)
        edges_2d = SKELETON_EDGES_2D
        input_title = "Input 2D"

    # --- align length ---
    T = min(gt_j3d.shape[0], in_j2d.shape[0])
    gt_j3d = gt_j3d[:T]
    in_j2d = in_j2d[:T]

    # --- helpers: caption getter ---
    def _get_caption(cap, f):
        if isinstance(cap, (list, tuple, np.ndarray)):
            if len(cap) == 0:
                return ""
            idx = min(max(int(f), 0), len(cap) - 1)
            return cap[idx]
        return cap if cap is not None else ""

    def _wrap(s):
        s = "" if s is None else str(s)
        return textwrap.fill(s, width=caption_wrap)

    # --- fixed axis ranges ---
    all_3d = gt_j3d.reshape(-1, 3)
    mid3 = all_3d.mean(axis=0)
    max_range3 = np.max(np.ptp(all_3d, axis=0)) / 2.0
    max_range3 = max_range3 * (1.0 + pad_ratio)

    x_all = in_j2d[..., 0].reshape(-1)
    y_all = in_j2d[..., 1].reshape(-1)
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    x_pad = (x_max - x_min) * pad_ratio + 1e-9
    y_pad = (y_max - y_min) * pad_ratio + 1e-9

    # --- figure ---
    fig = plt.figure(figsize=(15, 5))
    axL = fig.add_subplot(1, 3, 1, projection="3d")
    axM = fig.add_subplot(1, 3, 2)
    axR = fig.add_subplot(1, 3, 3)   # text only

    # left 3D view
    axL.view_init(elev=elev, azim=azim)
    axL.set_xlim(mid3[0] - max_range3, mid3[0] + max_range3)
    axL.set_ylim(mid3[2] - max_range3, mid3[2] + max_range3)
    axL.set_zlim(mid3[1] - max_range3, mid3[1] + max_range3)
    axL.set_xlabel("X")
    axL.set_ylabel("Z")
    axL.set_zlabel("Y")
    axL.invert_yaxis()
    axL.set_title("GT 3D")

    # middle 2D axis
    axM.set_xlim(x_min - x_pad, x_max + x_pad)
    axM.set_ylim(y_min - y_pad, y_max + y_pad)
    axM.set_aspect("equal", adjustable="box")
    axM.set_xlabel("x")
    axM.set_ylabel("y")
    if invert_y_2d:
        axM.invert_yaxis()
    axM.set_title(input_title)

    # right captions axis
    axR.axis("off")
    axR.set_title("Captions")

    # artists: left 3D
    scatL = axL.scatter([], [], [], s=25)
    linesL = [axL.plot([], [], [], lw=2)[0] for _ in SKELETON_EDGES_3D]

    # artists: middle 2D
    scatM = axM.scatter([], [], s=25)
    linesM = [axM.plot([], [], lw=2)[0] for _ in edges_2d]

    # text blocks
    axR.text(0.02, 0.98, "GT:", transform=axR.transAxes, va="top", ha="left",
             fontsize=caption_fontsize+1, fontweight="bold")
    axR.text(0.02, 0.48, "Pred:", transform=axR.transAxes, va="top", ha="left",
             fontsize=caption_fontsize+1, fontweight="bold")

    gt_text_artist = axR.text(
        0.02, 0.92, _wrap(_get_caption(gt_caption_text, 0)),
        transform=axR.transAxes, va="top", ha="left",
        fontsize=caption_fontsize
    )
    pred_text_artist = axR.text(
        0.02, 0.42, _wrap(_get_caption(pred_caption_text, 0)),
        transform=axR.transAxes, va="top", ha="left",
        fontsize=caption_fontsize
    )

    def update(frame_idx):
        f = frame_idx * step
        if f >= T:
            f = T - 1

        # --- GT 3D ---
        xyz = gt_j3d[f]  # (21,3)
        scatL._offsets3d = (xyz[:, 0], xyz[:, 2], xyz[:, 1])
        for line, (a, b) in zip(linesL, SKELETON_EDGES_3D):
            line.set_data([xyz[a, 0], xyz[b, 0]], [xyz[a, 2], xyz[b, 2]])
            line.set_3d_properties([xyz[a, 1], xyz[b, 1]])

        # --- Input 2D ---
        xy = in_j2d[f]  # (13,2) or (22,2)
        x, y = xy[:, 0], xy[:, 1]
        scatM.set_offsets(np.stack([x, y], axis=1))
        for line, (a, b) in zip(linesM, edges_2d):
            line.set_data([x[a], x[b]], [y[a], y[b]])

        # --- captions ---
        gt_text_artist.set_text(_wrap(_get_caption(gt_caption_text, f)))
        pred_text_artist.set_text(_wrap(_get_caption(pred_caption_text, f)))

        fig.suptitle(f"Frame {f}/{T-1}", y=0.98)

        return ([scatL, scatM, gt_text_artist, pred_text_artist]
                + linesL + linesM)

    n_frames = (T + step - 1) // step
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"Saved mp4 to: {save_path}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# - SKELETON_EDGES_3D
# - _feats3d_to_joints21_with_root
# - _motion2d_to_joints2d


def save_pair_mp4_coco68(
    in_motion_2d,
    recon_motion_3d,
    save_path="pair.mp4",
    mean_2d=None,
    std_2d=None,
    in_2d_is_normalized=True,
    fps=20,
    step=1,
    invert_y_2d=True,
    elev=15, azim=-70,
    pad_ratio=0.15,
    root_init_xy=None,
):
    """

    """

    # --- build joints arrays (3D) ---
    rec_j3d = _feats3d_to_joints21_with_root(recon_motion_3d)  # (T,21,3)

    # --- build joints arrays (2D input) ---
    if in_motion_2d.ndim == 3 and in_motion_2d.shape[1:] == (22, 2):
        in_j2d = in_motion_2d
        edges_2d = SKELETON_EDGES_2D
        input_title = "Input 2D (22j)"
    elif in_motion_2d.ndim == 2 and in_motion_2d.shape[1] == 68:
        in_j2d = reconstruct_coco2d_from_features(
            in_motion_2d,
            mean_2d=mean_2d,
            std_2d=std_2d,
            is_normalized=in_2d_is_normalized,
            root_init_xy=root_init_xy
        )  # (T,13,2)
        edges_2d = SKELETON_EDGES_2D_COCO13
        input_title = "Input 2D (COCO13 from 68D)"
    else:
        in_j2d = _motion2d_to_joints2d(
            in_motion_2d, mean_2d, std_2d, is_normalized=in_2d_is_normalized
        )
        edges_2d = SKELETON_EDGES_2D
        input_title = "Input 2D"

    # --- align length ---
    T = min(in_j2d.shape[0], rec_j3d.shape[0])
    in_j2d = in_j2d[:T]
    rec_j3d = rec_j3d[:T]

    # --- root-relative centering: remove XZ root translation so range is tight ---
    # 3D: subtract per-frame XZ centroid (keep Y/height)
    xz_mean = rec_j3d.mean(axis=1, keepdims=True).copy()
    xz_mean[:, :, 1] = 0  # leave Y intact
    rec_j3d_vis = rec_j3d - xz_mean

    # 2D: subtract per-frame centroid
    in_j2d_vis = in_j2d - in_j2d.mean(axis=1, keepdims=True)

    # --- axis ranges (now body-size, not trajectory-size) ---
    all_3d = rec_j3d_vis.reshape(-1, 3)
    mid3 = all_3d.mean(axis=0)
    max_range3 = np.max(np.ptp(all_3d, axis=0)) / 2.0
    max_range3 = max_range3 * (1.0 + pad_ratio)

    x_all = in_j2d_vis[..., 0].reshape(-1)
    y_all = in_j2d_vis[..., 1].reshape(-1)
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    x_pad = (x_max - x_min) * pad_ratio + 1e-9
    y_pad = (y_max - y_min) * pad_ratio + 1e-9

    # --- figure ---
    fig = plt.figure(figsize=(10, 5))
    axL = fig.add_subplot(1, 2, 1)                  # 2D
    axR = fig.add_subplot(1, 2, 2, projection="3d") # 3D

    # right 3D view
    axR.view_init(elev=elev, azim=azim)
    axR.set_xlim(mid3[0] - max_range3, mid3[0] + max_range3)
    axR.set_ylim(mid3[2] - max_range3, mid3[2] + max_range3)
    axR.set_zlim(mid3[1] - max_range3, mid3[1] + max_range3)
    axR.set_xlabel("X")
    axR.set_ylabel("Z")
    axR.set_zlabel("Y")
    axR.invert_yaxis()

    # left 2D axis
    axL.set_xlim(x_min - x_pad, x_max + x_pad)
    axL.set_ylim(y_min - y_pad, y_max + y_pad)
    axL.set_aspect("equal", adjustable="box")
    axL.set_xlabel("x")
    axL.set_ylabel("y")
    if invert_y_2d:
        axL.invert_yaxis()

    # artists: left 2D
    scatL = axL.scatter([], [], s=25)
    linesL = [axL.plot([], [], lw=2)[0] for _ in edges_2d]
    axL.set_title(input_title)

    # artists: right 3D
    scatR = axR.scatter([], [], [], s=25)
    linesR = [axR.plot([], [], [], lw=2)[0] for _ in SKELETON_EDGES_3D]
    axR.set_title("Recon 3D")

    def update(frame_idx):
        f = frame_idx * step
        if f >= T:
            f = T - 1

        # --- Input 2D (root-relative) ---
        xy = in_j2d_vis[f]
        x, y = xy[:, 0], xy[:, 1]
        scatL.set_offsets(np.stack([x, y], axis=1))
        for line, (a, b) in zip(linesL, edges_2d):
            line.set_data([x[a], x[b]], [y[a], y[b]])

        # --- Recon 3D (root-relative) ---
        xyz = rec_j3d_vis[f]
        scatR._offsets3d = (xyz[:, 0], xyz[:, 2], xyz[:, 1])
        for line, (a, b) in zip(linesR, SKELETON_EDGES_3D):
            line.set_data([xyz[a, 0], xyz[b, 0]], [xyz[a, 2], xyz[b, 2]])
            line.set_3d_properties([xyz[a, 1], xyz[b, 1]])

        fig.suptitle(f"Frame {f}/{T-1}", y=0.98)
        return [scatL, scatR] + linesL + linesR

    n_frames = (T + step - 1) // step
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = FFMpegWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"Saved pair mp4 to: {save_path}")
