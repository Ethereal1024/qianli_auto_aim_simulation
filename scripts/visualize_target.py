import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from diff_env.target import TargetMotion
from tools.math import geometry as gmt
from visualize import Animator, CameraView, WorldView

# ------------------------------------------------------------------- scene ---
DISTANCE = 3.0  # m, target straight ahead of the camera
HEIGHT = 0.5  # m, target and camera height
RADIUS = 0.3  # m, armor ring radius
YAW = 0.0  # deg, target heading at t = 0
YAW_RATE = 8.0
CAM_YAW = 0.0  # deg, positive looks left
FX = 4000.0  # px, focal length
IMAGE_W, IMAGE_H = 1280, 1024  # px, image size
SECONDS = 1000.0  # s, clip length
FPS = 60.0  # frame slots per second of clip time

ARMOR_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")  # A0, A1, A2, A3
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")  # x, y, z
N_ARMORS = len(ARMOR_COLORS)

# ---------------------------------------------------------------- geometry ---
CAM_POS = np.array([0.0, 0.0, HEIGHT])
BODY_POS = np.array([DISTANCE, 0.0, HEIGHT])
R_CW = gmt.euler_to_rotmat(
    np.array(CAM_YAW - math.pi / 2), np.array(0), np.array(-math.pi / 2)
)
N_FRAMES = max(2, round(SECONDS * FPS))


track = TargetMotion()(
    torch.linspace(0.0, SECONDS, N_FRAMES).unsqueeze(0),
    "complex",
    p0=torch.tensor([[DISTANCE, 0.0, HEIGHT]]),
    yaw_0=torch.tensor([math.radians(YAW)]),
    yaw_rate=YAW_RATE,
    yaw_rate_amp=0.5,
    yaw_rate_freq=0.3,
    n_armors=N_ARMORS,
    radius=RADIUS,
    sinus_amp=torch.tensor([[0.5, 0.3, 0.2]]),  # 上下左右摆动
    sinus_freq=torch.tensor([[0.2, 0.5, 0.3]]),
    pitch_amp=0.2,
    pitch_freq=0.5,
)
corners = track["corners"][0].numpy()  # (frames, armors, 4, 3)
centers = track["centers"][0].numpy()
body_pos = track["body_positions"][0].numpy()
body_rot = track["body_rotations"][0].numpy()
armor_rot = track["armor_rotations"][0].numpy()

sql = centers.shape[0]

# ------------------------------------------------------ figure and artists ---
fig = plt.figure(figsize=(16.5, 5.2))
ax_world = fig.add_subplot(1, 3, 1, projection="3d")
ax_armor = fig.add_subplot(1, 3, 2, projection="3d")
ax_image = fig.add_subplot(1, 3, 3)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.10, top=0.85, wspace=0.15)
fig.suptitle("Visualize")

camera_view = CameraView(
    ax=ax_image,
    armor_corners=corners,
    armor_centers=centers,
    cam_rots=np.broadcast_to(R_CW, (sql, 3, 3)),
    cam_trans=np.broadcast_to(CAM_POS[..., None].T, (sql, 3)),
    fx=FX,
)

world_view = WorldView(
    ax=ax_world,
    armor_corners=corners,
    frames_metas=[
        ("W", 0.5),
        ("C", 0.3),
        ("B", 0.25),
    ],
    frames_rots=[
        np.broadcast_to(np.eye(3), (sql, 3, 3)),
        np.broadcast_to(R_CW, (sql, 3, 3)),
        body_rot,
    ],
    frames_trans=[
        np.broadcast_to(np.zeros(3), (sql, 3)),
        np.broadcast_to(CAM_POS, (sql, 3)),
        body_pos,
    ],
    armor_colors=("#666666", "#666666", "#666666", "#666666"),
)

armor_view = WorldView(
    ax=ax_armor,
    armor_corners=corners,
    frames_metas=[("B", 0.25), *[(f"A{m}", 0.1) for m in range(N_ARMORS)]],
    frames_rots=[body_rot, *[armor_rot[:, m] for m in range(N_ARMORS)]],
    frames_trans=[body_pos, *[centers[:, m] for m in range(N_ARMORS)]],
)

animator = Animator(
    fig=fig,
    views=[camera_view, world_view, armor_view],
    fps=FPS
)

animator.play(SECONDS)