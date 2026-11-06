from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from diff_env.ballistic import Ballistic
from diff_env.camera import Camera
from tools.math import geometry as gmt
from diff_env.gimbal import Gimbal
from diff_env.target import TargetMotion

R_GC_DEFAULT = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class AutoAimPolicy(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def init_state(self, bsz: int, device: torch.device, dtype: torch.dtype) -> Any: ...

    @abstractmethod
    def forward(
        self,
        keypoints: torch.Tensor,
        gimbal_vel: torch.Tensor,
        dt: torch.Tensor,
        state=None,
    ) -> Any: ...


class SimpleProportionalPolicy(AutoAimPolicy):
    def __init__(self, gain: float = 10.0):
        super().__init__()
        self.gain = gain

    def init_state(self, bsz: int, device: torch.device, dtype: torch.dtype):
        return None

    def forward(
        self,
        keypoints: torch.Tensor,
        gimbal_vel: torch.Tensor,
        dt: torch.Tensor,
        state=None,
    ):
        cx = keypoints[..., 0::2].mean(-1, keepdim=True)
        cy = keypoints[..., 1::2].mean(-1, keepdim=True)

        yaw_vel_cmd = torch.tanh(-self.gain * cx)
        pitch_vel_cmd = torch.tanh(-self.gain * cy)
        return yaw_vel_cmd, pitch_vel_cmd


class DiffSimulator(nn.Module):
    def __init__(
        self,
        cam_rotation: torch.Tensor | None = None,
        cam_offset: torch.Tensor | None = None,
        max_yaw_vel: float = 10.0,
        max_pitch_vel: float = 8.0,
        bullet_speed: float = 25.0,
        gravity: float = 9.81,
        muzzle_len: float = 0.15,
        armor_width: float = 0.12,
        armor_height: float = 0.06,
        occlusion_factor: float = 0.0,
    ):
        super().__init__()
        self.gimbal = Gimbal(max_yaw_vel, max_pitch_vel)
        self.target = TargetMotion(armor_width, armor_height)
        self.camera = Camera()
        self.ballistic = Ballistic(bullet_speed, gravity)
        self.muzzle_len = muzzle_len
        self.occlusion_factor = occlusion_factor
        self.armor_size = armor_height * armor_width

        r_gc_src = R_GC_DEFAULT if cam_rotation is None else cam_rotation
        self.r_gc: torch.Tensor
        self.register_buffer("r_gc", r_gc_src.clone())

        t_gc_src = torch.zeros(3) if cam_offset is None else cam_offset
        self.t_gc: torch.Tensor
        self.register_buffer("t_gc", t_gc_src.clone())

    def _mask_state(
        self,
        old_state: torch.Tensor | list | tuple,
        new_state: torch.Tensor | list | tuple,
        mask: torch.Tensor,
    ):
        if old_state is None:
            return new_state
        if isinstance(new_state, torch.Tensor) and isinstance(old_state, torch.Tensor):
            view_shape = (-1,) + (1,) * (new_state.dim() - 1)
            return torch.where(mask.view(*view_shape), new_state, old_state)
        if isinstance(new_state, (tuple, list)) and isinstance(
            old_state, (tuple, list)
        ):
            return type(new_state)(
                self._mask_state(o, n, mask) for o, n in zip(old_state, new_state)
            )
        return new_state

    def _policy_step(
        self,
        policy: AutoAimPolicy,
        kp: torch.Tensor,
        gimbal_vel: float,
        dt: float,
        state: torch.Tensor | list | tuple,
        mask: torch.Tensor,
    ):
        if not mask.any():
            zero = torch.zeros(kp.shape[0], 1, device=kp.device, dtype=kp.dtype)
            return zero, zero, state
        yr, pr, state_new = policy(kp, gimbal_vel, dt, state)
        yr = torch.where(mask.unsqueeze(-1), yr, torch.zeros_like(yr))
        pr = torch.where(mask.unsqueeze(-1), pr, torch.zeros_like(pr))
        state = self._mask_state(state, state_new, mask)
        return yr, pr, state

    def _world_to_gimbal(
        self,
        points_world: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ):
        zeros = torch.zeros_like(yaw)
        r_wg = gmt.euler_to_rotmat(yaw, -pitch, zeros)
        r_gw = r_wg.transpose(-1, -2)
        return torch.einsum("bij,bnj->bni", r_gw, points_world)

    def _gimbal_to_world(
        self,
        points_gimbal: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ):
        zeros = torch.zeros_like(yaw)
        r_wg = gmt.euler_to_rotmat(yaw, -pitch, zeros)
        return torch.einsum("bij,bnj->bni", r_wg, points_gimbal)

    def forward(
        self,
        t: torch.Tensor,
        target_params: dict,
        init_yaw: torch.Tensor,
        init_pitch: torch.Tensor,
        policy: AutoAimPolicy,
    ):
        if policy is None:
            policy = SimpleProportionalPolicy()

        bsz, sql = t.shape
        device = t.device
        dtype = t.dtype

        dt = torch.zeros_like(t)
        if sql > 1:
            dt[:, 1:] = t[:, 1:] - t[:, :-1]

        # target
        tgt = self.target(t, mode="complex", **target_params)
        corners_world: torch.Tensor = tgt["corners"]  # [bsz, sql, n_armors, 4, 3]
        centets_world: torch.Tensor = tgt["centers"]  # [bsz, sql, n_armors, 3]
        n_armors = corners_world.shape[2]

        # gimbal
        yaw = init_yaw.clone()
        pitch = init_pitch.clone()
        yaw_vel = torch.zeros(bsz, device=device, dtype=dtype)
        pitch_vel = torch.zeros(bsz, device=device, dtype=dtype)

        state = policy.init_state(bsz, device, dtype)

        logs: dict = {
            "keypoints": [],
            "valid": [],
            "gimbal_yaw": [],
            "gimbal_pitch": [],
            "yaw_vel_cmd": [],
            "pitch_vel_cmd": [],
            "hit_dist": [],
            "t_fly": [],
        }

        r_gc = self.r_gc.unsqueeze(0).expand(bsz, -1, -1)
        arange_bsz = torch.arange(bsz, device=device)
        idx = torch.arange(sql, device=device)

        for k in range(sql):
            cnr_w = corners_world[:, k]  # [bsz, n_armors, 4, 3]
            ctr_w = centets_world[:, k]  # [bsz, n_armors, 3]

            cnr_g = self._world_to_gimbal(
                cnr_w.reshape(bsz, n_armors * 4, 3), yaw, pitch
            ).reshape(bsz, n_armors, 4, 3)
            ctr_g = self._world_to_gimbal(ctr_w, yaw, pitch)

            kp, valid = self.camera(cnr_g.reshape(bsz, n_armors * 4, 3), r_gc)
            kp = kp.reshape(bsz, n_armors, 4, 2)
            valid = valid.reshape(bsz, n_armors, 4)
            armor_visible = valid.all(dim=-1)  # [bsz, n_armors]

            r_wg: torch.Tensor = gmt.euler_to_rotmat(yaw, -pitch, 0)
            t_wc = r_wg @ self.t_gc

            normal = torch.cross(
                cnr_w[..., 1, :] - cnr_w[..., 0, :],
                cnr_w[..., 3, :] - cnr_w[..., 0, :],
            )                     
            diff = t_wc.unsqueeze(1) - ctr_w
            drct_dot_prod = (normal * diff).sum(-1)
            
            armor_front = drct_dot_prod > 0  # [bsz, n_armors]
            armor_visible = armor_visible & armor_front
            cost = 1 - (drct_dot_prod / (diff.norm().clamp(min=1e-6) * self.armor_size)).clamp(0, 1)
            
            if self.training and self.occlusion_factor > 0:
                keep = torch.rand_like(armor_visible.float()) >= self.occlusion_factor * cost
                armor_visible = armor_visible & keep

            # gather 到 [B, 8] 的 kp
            kp_flat = kp.reshape(bsz, n_armors, 8)
            kp_first = kp_flat[arange_bsz, first_idx.clamp(max=n_armors - 1)]
            kp_second = kp_flat[arange_bsz, second_idx.clamp(max=n_armors - 1)]

            gimbal_vel = torch.stack([yaw_vel, pitch_vel], dim=-1)  # [B, 2]
            dt_k = dt[:, k].unsqueeze(-1)  # [B, 1]

            # 第一块：dt 正常
            yr0, pr0, state = self._policy_step(
                policy,
                kp_first,
                gimbal_vel,
                dt_k,
                state,
                has_first,
            )

            # 第二块：有第一块时 dt=0，否则（只有一块时）dt 正常
            dt_second = torch.where(
                has_first,
                torch.zeros_like(dt_k),
                dt_k,
            )
            yr1, pr1, state = self._policy_step(
                policy,
                kp_second,
                gimbal_vel,
                dt_second,
                state,
                has_second,
            )

            # 合并输出：两块平均，一块直取，零块为 0
            both = has_first & has_second
            yaw_cmd = torch.where(
                both,
                (yr0 + yr1) / 2,
                torch.where(has_first, yr0, yr1),
            )
            pitch_cmd = torch.where(
                both,
                (pr0 + pr1) / 2,
                torch.where(has_first, pr0, pr1),
            )
            has_any = has_first | has_second
            yaw_cmd = torch.where(has_any, yaw_cmd, torch.zeros_like(yaw_cmd))
            pitch_cmd = torch.where(has_any, pitch_cmd, torch.zeros_like(pitch_cmd))

            # 云台积分
            yaw, pitch, yaw_vel, pitch_vel = self.gimbal(
                yaw,
                pitch,
                yaw_cmd.squeeze(-1),
                pitch_cmd.squeeze(-1),
                dt[:, k],
            )

            # 枪口世界位置
            muzzle_local = torch.stack(
                [
                    torch.full_like(yaw, self.muzzle_length),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                ],
                dim=-1,
            ).unsqueeze(
                1
            )  # [B, 1, 3]
            muzzle_world = self._gimbal_to_world(muzzle_local, yaw, pitch).squeeze(
                1
            )  # [B, 3]

            # 弹道：用第一块可见装甲板中心，无可见时用中心均值
            ctr_w_flat = ctr_w  # [B, M, 3]
            target_w = ctr_w_flat[arange_B, first_idx.clamp(max=M - 1)]
            target_w = torch.where(
                has_first.unsqueeze(-1),
                target_w,
                ctr_w_flat.mean(dim=1),
            )
            hit_pos, t_fly, hit_dist = self.ballistic(
                yaw,
                pitch,
                target_w,
                muzzle_pos=muzzle_world,
            )

            logs["keypoints"].append(kp)
            logs["valid"].append(armor_valid)
            logs["gimbal_yaw"].append(yaw)
            logs["gimbal_pitch"].append(pitch)
            logs["yaw_vel_cmd"].append(yaw_cmd.squeeze(-1))
            logs["pitch_vel_cmd"].append(pitch_cmd.squeeze(-1))
            logs["hit_dist"].append(hit_dist)
            logs["t_fly"].append(t_fly)

        for key, _ in logs.values():
            logs[key] = torch.stack(logs[key], dim=1)

        return logs
