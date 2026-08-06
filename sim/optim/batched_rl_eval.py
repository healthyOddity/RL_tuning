"""Batched evaluator for truck_trailer RL tuning actions."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from optim.train import tracking_loss
from optim.train_batch import (
    RL_ACTION_BOUNDS,
    batched_tracking_loss,
    build_batched_rl_controllers,
    run_simulation_batch,
)
from sim_loop import run_simulation


def clip_rl_actions(actions) -> torch.Tensor:
    action_t = torch.as_tensor(actions, dtype=torch.float32)
    if action_t.dim() != 2 or action_t.shape[1] != 11:
        raise ValueError(f"actions must have shape [B, 11], got {tuple(action_t.shape)}")
    return torch.clamp(action_t, -RL_ACTION_BOUNDS, RL_ACTION_BOUNDS)


def _apply_action_to_scalar_controllers(lat_ctrl, lon_ctrl, action: torch.Tensor):
    base_t2 = lat_ctrl.T2_y.data.clone()
    base_t3 = lat_ctrl.T3_y.data.clone()
    base_t4 = lat_ctrl.T4_y.data.clone()
    base_t6 = lat_ctrl.T6_y.data.clone()

    lat_ctrl.T2_y.data = base_t2 * (1.0 + float(action[0]))
    lat_ctrl.T3_y.data = base_t3 * (1.0 + float(action[1]))
    lat_ctrl.T4_y.data = base_t4 * (1.0 + float(action[2]))
    lat_ctrl.T6_y.data = base_t6 * (1.0 + float(action[3]))

    lon_ctrl.station_kp.data = torch.tensor(max(0.0, lon_ctrl.station_kp.item() + float(action[4])))
    lon_ctrl.station_ki.data = torch.tensor(max(0.0, lon_ctrl.station_ki.item() + float(action[5])))
    lon_ctrl.low_speed_kp.data = torch.tensor(max(0.0, lon_ctrl.low_speed_kp.item() + float(action[6])))
    lon_ctrl.low_speed_ki.data = torch.tensor(max(0.0, lon_ctrl.low_speed_ki.item() + float(action[7])))
    lon_ctrl.high_speed_kp.data = torch.tensor(max(0.0, lon_ctrl.high_speed_kp.item() + float(action[8])))
    lon_ctrl.high_speed_ki.data = torch.tensor(max(0.0, lon_ctrl.high_speed_ki.item() + float(action[9])))
    switch_speed = lon_ctrl.switch_speed.item() + float(action[10])
    lon_ctrl.switch_speed.data = torch.tensor(max(0.5, min(10.0, switch_speed)))


def _action_l2_penalties(cfg: dict, actions: torch.Tensor) -> torch.Tensor:
    baseline_lat = LatControllerTruck(cfg, differentiable=False)
    baseline_lon = LonController(cfg, differentiable=False)

    l2 = torch.zeros(actions.shape[0], dtype=torch.float32)
    for col, name in enumerate(['T2_y', 'T3_y', 'T4_y', 'T6_y']):
        base = getattr(baseline_lat, name).detach()
        delta = base.unsqueeze(0) * actions[:, col:col + 1]
        l2 += (delta ** 2).sum(dim=1)

    lon_specs = [
        ('station_kp', 4, 0.0, None),
        ('station_ki', 5, 0.0, None),
        ('low_speed_kp', 6, 0.0, None),
        ('low_speed_ki', 7, 0.0, None),
        ('high_speed_kp', 8, 0.0, None),
        ('high_speed_ki', 9, 0.0, None),
        ('switch_speed', 10, 0.5, 10.0),
    ]
    for name, col, min_value, max_value in lon_specs:
        base = getattr(baseline_lon, name).detach()
        tuned = torch.clamp(base + actions[:, col], min=float(min_value))
        if max_value is not None:
            tuned = torch.clamp(tuned, max=float(max_value))
        l2 += (tuned - base) ** 2
    return l2


def _rewards_from_losses(raw_losses, trajectory_keys, actions, cfg, baseline_losses, norm_floor):
    losses = torch.as_tensor(raw_losses, dtype=torch.float32)
    baselines = torch.tensor(
        [float(baseline_losses[key]) for key in trajectory_keys],
        dtype=torch.float32,
    )
    norm = torch.maximum(torch.sqrt(baselines), torch.tensor(float(norm_floor)))
    normalized = losses / norm
    l2 = _action_l2_penalties(cfg, actions)
    rewards = -(normalized + 0.01 * l2)
    return normalized.numpy(), rewards.numpy(), l2.numpy()


def evaluate_batched_actions(
    trajectories,
    trajectory_keys,
    actions,
    cfg,
    baseline_losses,
    norm_floor,
):
    actions = clip_rl_actions(actions)
    if len(trajectories) != actions.shape[0] or len(trajectory_keys) != actions.shape[0]:
        raise ValueError('trajectories, trajectory_keys, and actions must have the same batch size')

    lat_ctrl, lon_ctrl = build_batched_rl_controllers(cfg, actions)
    history = run_simulation_batch(
        list(trajectories), cfg=cfg, lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
        tbptt_k=0, hard_mode=True)
    ref_speeds = torch.tensor([traj[0].v for traj in trajectories], dtype=torch.float32)
    raw_losses = batched_tracking_loss(history, ref_speeds).detach().cpu()
    normalized, rewards, l2 = _rewards_from_losses(
        raw_losses, trajectory_keys, actions, cfg, baseline_losses, norm_floor)
    return {
        'trajectory_keys': list(trajectory_keys),
        'raw_losses': raw_losses.numpy(),
        'normalized_losses': normalized,
        'action_l2_penalties': l2,
        'rewards': rewards,
    }


def evaluate_scalar_actions(
    trajectories,
    trajectory_keys,
    actions,
    cfg,
    baseline_losses,
    norm_floor,
):
    actions = clip_rl_actions(actions)
    if len(trajectories) != actions.shape[0] or len(trajectory_keys) != actions.shape[0]:
        raise ValueError('trajectories, trajectory_keys, and actions must have the same batch size')

    raw_losses = []
    for traj, action in zip(trajectories, actions):
        lat_ctrl = LatControllerTruck(cfg, differentiable=False)
        lon_ctrl = LonController(cfg, differentiable=False)
        _apply_action_to_scalar_controllers(lat_ctrl, lon_ctrl, action)
        history = run_simulation(
            traj, init_speed=traj[0].v,
            init_x=traj[0].x, init_y=traj[0].y, init_yaw=traj[0].theta,
            cfg=cfg, lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
            differentiable=False, tbptt_k=0)
        tensor_history = [
            {key: torch.tensor(value, dtype=torch.float32) for key, value in row.items()}
            for row in history
        ]
        raw_losses.append(float(tracking_loss(
            tensor_history, ref_speed=traj[0].v,
            w_lat=10.0, w_head=8.0, w_speed=3.0,
            w_steer_rate=0.05, w_acc_rate=0.01)))

    raw_loss_t = torch.tensor(raw_losses, dtype=torch.float32)
    normalized, rewards, l2 = _rewards_from_losses(
        raw_loss_t, trajectory_keys, actions, cfg, baseline_losses, norm_floor)
    return {
        'trajectory_keys': list(trajectory_keys),
        'raw_losses': raw_loss_t.numpy(),
        'normalized_losses': normalized,
        'action_l2_penalties': l2,
        'rewards': rewards,
    }
