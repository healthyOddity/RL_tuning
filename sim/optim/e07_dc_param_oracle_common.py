"""Shared E07 DC-parameter oracle utilities."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.trajectory import expand_trajectories, generate_circle, generate_lane_change
from optim.rl_env import extract_geometric_features
from optim.train import DiffControllerParams, tracking_loss
from sim_loop import run_simulation


DEFAULT_DC_CONFIG = (
    'results/training/truck_trailer/20260608_203406_mlp0525/'
    'tuned_4740dec_20260608_203243.yaml'
)
PARAM_VECTOR_DIM = 35


def materialize_standard_trajectories(max_trajectories: int | None = None):
    expanded = expand_trajectories(None)
    if max_trajectories is not None:
        expanded = expanded[:int(max_trajectories)]
    return [(key, gen()) for key, _label, gen in expanded]


def materialize_smoke_trajectories():
    return [
        ('circle_fixture', generate_circle(radius=50.0, speed=5.0,
                                           arc_angle=0.15)),
        ('lane_change_fixture', generate_lane_change(
            lane_width=3.5, change_length=25.0, speed=5.0,
            lead_in=5.0, lead_out=5.0)),
    ]


def _parameter_entries(params: DiffControllerParams):
    entries = []
    offset = 0
    for name, tensor in params.named_parameters():
        n = tensor.numel()
        entries.append((name, tuple(tensor.shape), offset, offset + n))
        offset += n
    return entries


def parameter_names(cfg: dict) -> list[str]:
    params = DiffControllerParams(cfg)
    names = []
    for name, tensor in params.named_parameters():
        flat_n = tensor.numel()
        if flat_n == 1:
            names.append(name)
        else:
            names.extend([f'{name}[{i}]' for i in range(flat_n)])
    return names


def flatten_params(params: DiffControllerParams) -> np.ndarray:
    values = [
        tensor.detach().reshape(-1).cpu().numpy()
        for _name, tensor in params.named_parameters()
    ]
    return np.concatenate(values).astype(np.float32)


def project_params_(params: DiffControllerParams,
                    fallback: dict[str, torch.Tensor] | None = None) -> None:
    with torch.no_grad():
        for name, p in params.named_parameters():
            if fallback is not None:
                finite = torch.isfinite(p)
                p.copy_(torch.where(finite, p, fallback[name]))
            else:
                p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0))
            if name in ('lon_ctrl.station_kp', 'lon_ctrl.station_ki',
                        'lon_ctrl.low_speed_kp', 'lon_ctrl.low_speed_ki',
                        'lon_ctrl.high_speed_kp', 'lon_ctrl.high_speed_ki'):
                p.clamp_(min=0.0)
            elif name == 'lon_ctrl.switch_speed':
                p.clamp_(min=0.5, max=10.0)
            elif name in ('lat_ctrl.T2_y', 'lat_ctrl.T3_y',
                          'lat_ctrl.T4_y', 'lat_ctrl.T6_y'):
                p.clamp_(min=0.0)


def apply_param_vector(params: DiffControllerParams, vector: np.ndarray) -> None:
    vector_t = torch.tensor(vector, dtype=torch.float32)
    entries = _parameter_entries(params)
    if len(vector_t) != entries[-1][3]:
        raise ValueError(f'expected {entries[-1][3]} params, got {len(vector_t)}')
    with torch.no_grad():
        for name, shape, start, end in entries:
            module = params
            parts = name.split('.')
            for part in parts[:-1]:
                module = getattr(module, part)
            tensor = getattr(module, parts[-1])
            tensor.copy_(vector_t[start:end].reshape(shape))
        project_params_(params)


def baseline_param_vector(cfg: dict) -> np.ndarray:
    return flatten_params(DiffControllerParams(cfg))


def observation_for_trajectory(traj, cfg: dict) -> np.ndarray:
    return np.concatenate([
        extract_geometric_features(traj),
        baseline_param_vector(cfg),
    ]).astype(np.float32)


def _loss_for_params(traj, cfg: dict,
                     param_vector: np.ndarray | None,
                     differentiable: bool,
                     tbptt_k: int = 0) -> float:
    params = DiffControllerParams(cfg)
    if param_vector is not None:
        apply_param_vector(params, param_vector)
    simulation_kwargs = {
        'trajectory': traj,
        'init_speed': traj[0].v,
        'init_x': traj[0].x,
        'init_y': traj[0].y,
        'init_yaw': traj[0].theta,
        'differentiable': differentiable,
        'tbptt_k': tbptt_k if differentiable else 0,
    }
    if differentiable:
        simulation_kwargs.update({
            'cfg': params.cfg,
            'lat_ctrl': params.lat_ctrl,
            'lon_ctrl': params.lon_ctrl,
        })
    else:
        # Export the candidate params, then let run_simulation construct the
        # deployed non-differentiable controllers from that config.
        simulation_kwargs['cfg'] = params.to_config_dict()
    history = run_simulation(**simulation_kwargs)
    if not differentiable:
        history = [
            {
                key: (
                    value if isinstance(value, torch.Tensor)
                    else torch.tensor(value, dtype=torch.float32)
                )
                for key, value in row.items()
            }
            for row in history
        ]
    return float(tracking_loss(
        history,
        ref_speed=None,
        w_lat=10.0,
        w_head=8.0,
        w_speed=3.0,
        w_steer_rate=0.05,
        w_acc_rate=0.01,
    ).detach().item())


def soft_loss_for_params(traj, cfg: dict,
                         param_vector: np.ndarray | None = None,
                         tbptt_k: int = 150) -> float:
    """Evaluate the differentiable DC training surrogate."""
    return _loss_for_params(
        traj, cfg, param_vector, differentiable=True, tbptt_k=tbptt_k)


def hard_loss_for_params(traj, cfg: dict,
                         param_vector: np.ndarray | None = None) -> float:
    """Evaluate the deployed scalar DC closed loop."""
    return _loss_for_params(
        traj, cfg, param_vector, differentiable=False, tbptt_k=0)


def scalar_loss_for_params(traj, cfg: dict,
                           param_vector: np.ndarray | None = None) -> float:
    """Backward-compatible name for the formal hard scalar evaluation."""
    return hard_loss_for_params(traj, cfg, param_vector)
