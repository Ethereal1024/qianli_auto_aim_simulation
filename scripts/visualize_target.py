#!/usr/bin/env python3
"""Draw a diff_env target, its coordinate frames and the camera projection.

world : +X forward, +Y left, +Z up (metres)
camera: +Z depth, +X right, +Y down (pixels)
frame axes: x red, y green, z blue
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from diff_env.target import TargetMotion

ARMOR_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")   # x red, y green, z blue


def camera_rotation(yaw_deg: float) -> np.ndarray:
    """R_wc with camera = R_wc @ (world - cam_pos); positive yaw looks left."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    return base @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def project(points, R_wc, cam_pos, fx, fy, cx, cy):
    """world points [..., 3] -> pixel u, v and camera depth."""
    cam = (points - cam_pos) @ R_wc.T
    u = fx * cam[..., 0] / cam[..., 2] + cx
    v = cy + fy * cam[..., 1] / cam[..., 2]
    return u, v, cam[..., 2]


def draw_frame(ax, origin, R, length, label=""):
    """Draw a frame: the columns of R are its x, y, z axes in world."""
    for i, color in enumerate(AXIS_COLORS):
        ax.plot(*np.vstack([origin, origin + length * R[:, i]]).T, color=color, lw=2)
    if label:
        ax.text(*origin, label, fontsize=8)


def set_limits(ax, points, pad=0.1):
    """Equal-aspect cube around the points."""
    pts = np.reshape(points, (-1, 3))
    lo, hi = pts.min(0), pts.max(0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 + pad
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def label_axes(ax):
    ax.set_xlabel("X fwd (m)")
    ax.set_ylabel("Y left (m)")
    ax.set_zlabel("Z up (m)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="scripts/target_vis.png")
    p.add_argument("--distance", type=float, default=3.0, help="target distance (m)")
    p.add_argument("--height", type=float, default=0.5, help="target/camera height (m)")
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--yaw", type=float, default=0.0, help="target yaw (deg)")
    p.add_argument("--cam-yaw", type=float, default=0.0,
                   help="camera yaw (deg), positive looks left")
    p.add_argument("--fx", type=float, default=800.0)
    p.add_argument("--image-size", type=int, nargs=2, default=[1280, 1024],
                   metavar=("W", "H"))
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args(argv)

    # one snapshot of the target
    out = TargetMotion()(
        torch.zeros(1, 1),
        "complex",
        p0=torch.tensor([[args.distance, 0.0, args.height]]),
        yaw_0=torch.tensor([math.radians(args.yaw)]),
        n_armors=4,
        radius=torch.tensor([args.radius]),
    )
    centers = out["centers"][0, 0].numpy()
    corners = out["corners"][0, 0].numpy()          # (M, 4, 3)
    body_rot = out["body_rotations"][0, 0].numpy()
    armor_rot = out["armor_rotations"][0, 0].numpy()

    w, h = args.image_size
    cam_pos = np.array([0.0, 0.0, args.height])
    body = np.array([args.distance, 0.0, args.height])
    R_wc = camera_rotation(args.cam_yaw)
    u, v, depth = project(corners, R_wc, cam_pos, args.fx, args.fx, w / 2, h / 2)
    assert (depth > 0).all(), "target is behind the camera"

    # a plate faces the camera when its normal points towards it
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    front = (normal * (cam_pos - centers)).sum(-1) > 0

    fig = plt.figure(figsize=(16.5, 5.2))

    # world / camera / body frames, true scale
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw_frame(ax, np.zeros(3), np.eye(3), 0.5, "W")
    draw_frame(ax, cam_pos, R_wc.T, 0.3, "C")
    draw_frame(ax, body, body_rot, 0.25, "B")
    ax.plot(*np.vstack([cam_pos, body]).T, "k--", lw=1.0)
    for m in range(len(corners)):
        ax.plot(*np.vstack([corners[m], corners[m, 0]]).T, color="#666666", lw=1.0)
    set_limits(ax, np.vstack([corners.reshape(-1, 3), cam_pos, np.zeros(3)]))
    label_axes(ax)
    ax.set_title("world, camera and body frames")

    # body and armor frames, close up
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    draw_frame(ax2, body, body_rot, 0.2, "B")
    for m in range(len(corners)):
        ax2.plot(*np.vstack([corners[m], corners[m, 0]]).T,
                 color=ARMOR_COLORS[m % len(ARMOR_COLORS)], lw=1.2)
        draw_frame(ax2, centers[m], armor_rot[m], 0.1, f"A{m}")
    set_limits(ax2, np.vstack([corners.reshape(-1, 3), body]), pad=0.15)
    label_axes(ax2)
    ax2.set_title("armor frames")

    # camera view, zoomed on the target; solid = front face, dotted = back
    ax_img = fig.add_subplot(1, 3, 3)
    for m in range(len(corners)):
        loop_u = np.append(u[m], u[m, 0])
        loop_v = np.append(v[m], v[m, 0])
        color = ARMOR_COLORS[m % len(ARMOR_COLORS)]
        if front[m]:
            ax_img.fill(loop_u, loop_v, color=color, alpha=0.35)
        ax_img.plot(loop_u, loop_v, "-" if front[m] else ":", color=color, lw=1.2)
        ax_img.annotate(f"A{m}", (u[m].mean(), v[m].mean()),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    pad = 0.1 * max(np.ptp(u), np.ptp(v)) + 10
    ax_img.set_xlim(u.min() - pad, u.max() + pad)
    ax_img.set_ylim(v.max() + pad, v.min() - pad)     # v grows downwards
    ax_img.set_aspect("equal")
    ax_img.grid(alpha=0.3)
    ax_img.set_title(f"camera view at {args.distance:.1f} m "
                     f"({w}x{h}, fx={args.fx:.0f}, zoom)")

    fig.suptitle("W world, C camera, B body, A# armor;  x red, y green, z blue",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
