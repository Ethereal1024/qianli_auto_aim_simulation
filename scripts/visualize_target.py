#!/usr/bin/env python3
"""Live animation of a diff_env target seen from a fixed camera.

Three panels: world/body/camera frames, armor frames up close, camera image.
Change the numbers under "scene" and run the file.

world : +X forward, +Y left, +Z up (m)
camera: +Z depth, +X right, +Y down (px)
frame axes: x red, y green, z blue

Artists are built once and later frames only rewrite their data, so every panel
keeps one fixed scale.  The frame index comes from the wall clock, so a slow
redraw drops frames instead of slowing the motion down, and blitting repaints
only the artists that move (~14 ms per frame instead of ~95 ms for a full one).
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D

from diff_env.target import TargetMotion
from visualize import CameraView


def camera_rotation(yaw_deg):
    """world -> camera rotation; positive yaw looks left."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    return base @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def closed(points):
    """repeat the first point, so a polygon line ends where it starts."""
    return np.vstack([points, points[:1]])


def set_cube(ax, points, pad):
    """equal-aspect box around points, fixed for the whole clip."""
    lo, hi = points.reshape(-1, 3).min(0), points.reshape(-1, 3).max(0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 + pad
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_autoscale_on(False)


def line3d(ax, color, lw=2.0):
    """an empty 3D line; plot() turns the Line2D it made into a Line3D."""
    line = ax.plot([], [], [], color=color, lw=lw)[0]
    assert isinstance(line, Line3D)
    return line


def text3d(ax, label):
    """a 3D text label at the origin; text() turns it into a Text3D."""
    text = ax.text(0, 0, 0, label, fontsize=8)
    assert isinstance(text, Text3D)
    return text


def triad(ax, label, length):
    """an empty frame: three axis lines `length` metres long plus a label."""
    lines = [line3d(ax, c) for c in AXIS_COLORS]
    return lines, text3d(ax, label), length


def place(frame, origin, rot):
    """move one frame to origin, rotated by rot (its columns are the axes)."""
    lines, label, length = frame
    for line, end in zip(lines, origin + length * rot.T):
        line.set_data_3d(*np.vstack([origin, end]).T)
    label.set_position_3d(origin)
    
    
# ------------------------------------------------------------------- scene ---
DISTANCE = 3.0                    # m, target straight ahead of the camera
HEIGHT = 0.5                      # m, target and camera height
RADIUS = 0.3                      # m, armor ring radius
YAW = 0.0                         # deg, target heading at t = 0
YAW_RATE = 5.0
CAM_YAW = 0.0                     # deg, positive looks left
FX = 4000.0                        # px, focal length
IMAGE_W, IMAGE_H = 1280, 1024     # px, image size
SECONDS = 1000.0                    # s, clip length
FPS = 60.0                        # frame slots per second of clip time

ARMOR_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")    # A0, A1, A2, A3
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")                # x, y, z
N_ARMORS = len(ARMOR_COLORS)

# ---------------------------------------------------------------- geometry ---
CAM_POS = np.array([0.0, 0.0, HEIGHT])
BODY_POS = np.array([DISTANCE, 0.0, HEIGHT])
R_WC = camera_rotation(CAM_YAW)
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
    sinus_amp=torch.tensor([[0.5, 0.3, 0.2]]),   # 上下左右摆动
    sinus_freq=torch.tensor([[0.2, 0.5, 0.3]]),
    pitch_amp=0.2,
    pitch_freq=0.5
)
corners = track["corners"][0].numpy()              # (frames, armors, 4, 3)
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

camera_view = CameraView(
    ax=ax_image,
    corners=corners,
    centers=centers,
    rots_wc=np.broadcast_to(R_WC.T, (sql, 3, 3)),
    trans_wc=np.broadcast_to(CAM_POS[..., None].T, (sql, 3))
)

world_frames = [triad(ax_world, "W", 0.5), triad(ax_world, "C", 0.3),
                triad(ax_world, "B", 0.25)]
armor_frames = [triad(ax_armor, "B", 0.2)] + [
    triad(ax_armor, f"A{m}", 0.1) for m in range(N_ARMORS)]
world_plates = [line3d(ax_world, "#666666", 1.0) for _ in range(N_ARMORS)]
armor_plates = [line3d(ax_armor, c, 1.2) for c in ARMOR_COLORS]

# limits from every frame of the track: the scale never moves
cubes = np.vstack([corners.reshape(-1, 3), CAM_POS, np.zeros(3)])
set_cube(ax_world, cubes, 0.1)
set_cube(ax_armor, np.vstack([corners.reshape(-1, 3), BODY_POS]), 0.15)


for ax in (ax_world, ax_armor):
    ax.set_xlabel("X fwd (m)")
    ax.set_ylabel("Y left (m)")
    ax.set_zlabel("Z up (m)")
ax_world.set_title("world, camera and body frames")
ax_armor.set_title("armor frames")
fig.suptitle("W world, C camera, B body, A# armor;  x red, y green, z blue",
             fontsize=9)


def update(index):
    """point every artist at frame index; nothing is created or cleared."""
    plate = corners[index]                         # (armors, 4, 3)

    place(world_frames[0], np.zeros(3), np.eye(3))
    place(world_frames[1], CAM_POS, R_WC.T)
    place(world_frames[2], body_pos[index], body_rot[index])
    place(armor_frames[0], body_pos[index], body_rot[index])
    for m in range(N_ARMORS):
        loop = closed(plate[m]).T                  # (3, 5), back to the start
        place(armor_frames[m + 1], centers[index, m], armor_rot[index, m])
        world_plates[m].set_data_3d(*loop)
        armor_plates[m].set_data_3d(*loop)

    camera_view.update(index)


# ---------------------------------------------------------------- playback ---
update(0)
start = time.perf_counter()

# the artists that move; blitting repaints only these over a cached background
moving = ([a for lines, text, _ in world_frames + armor_frames
           for a in (*lines, text)]
          + world_plates + armor_plates + camera_view.fills + camera_view.edges + camera_view.tags)


def animate(_tick):
    index = min(int((time.perf_counter() - start) / SECONDS * N_FRAMES),
                N_FRAMES - 1)
    update(index)
    if index == N_FRAMES - 1 and animation.event_source is not None:
        animation.event_source.stop()
    return moving


animation = FuncAnimation(fig, animate, interval=1000.0 / FPS, repeat=False,
                          blit=True, cache_frame_data=False)
plt.show()
