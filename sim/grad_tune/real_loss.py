from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from grad_tune.data_schema import GradTuneSample
from grad_tune.observability import GRADIENT_EXPRESSION


@dataclass
class GradTuneLossWeights:
    w_lat: float = 10.0
    w_head: float = 8.0
    w_speed: float = 3.0
    w_steer_rate: float = 0.05
    w_acc_rate: float = 0.01


def _stack_history(history: list[dict], key: str) -> torch.Tensor:
    vals = [h[key] if isinstance(h[key], torch.Tensor)
            else torch.tensor(float(h[key])) for h in history]
    return torch.stack(vals)


def _record_tensor(values: np.ndarray, n: int,
                   like: torch.Tensor | None = None) -> torch.Tensor:
    arr = values[:n].astype(float)
    dtype = like.dtype if like is not None else torch.float32
    return torch.tensor(arr, dtype=dtype).detach()


def _frenet_speed_error(v_sim: torch.Tensor, head_sim: torch.Tensor,
                        lat_sim: torch.Tensor, ref_v_values,
                        ref_kappa_values,
                        n: int) -> tuple[torch.Tensor, torch.Tensor]:
    ref_v = _record_tensor(ref_v_values, n, v_sim)
    ref_kappa = _record_tensor(ref_kappa_values, n, v_sim)
    denom = 1.0 - ref_kappa * lat_sim
    denom_safe = torch.clamp(denom, min=0.2, max=5.0)
    s_dot = v_sim * torch.cos(head_sim) / denom_safe
    return ref_v - s_dot, denom_safe


def compute_real_report_loss(sample: GradTuneSample,
                             weights: GradTuneLossWeights) -> tuple[float, dict]:
    lat = sample.lateral_error
    head = sample.heading_error
    speed = sample.speed_error
    lat_mse = float(np.mean(lat * lat))
    head_mse = float(np.mean(head * head))
    speed_mse = float(np.mean(speed * speed))
    loss = (weights.w_lat * lat_mse
            + weights.w_head * head_mse
            + weights.w_speed * speed_mse)
    return loss, {
        'lat_rmse': float(np.sqrt(lat_mse)),
        'head_rmse': float(np.sqrt(head_mse)),
        'speed_rmse': float(np.sqrt(speed_mse)),
        'loss_lat': weights.w_lat * lat_mse,
        'loss_head': weights.w_head * head_mse,
        'loss_speed': weights.w_speed * speed_mse,
    }


def compute_backward_loss(history: list[dict], sample: GradTuneSample,
                          weights: GradTuneLossWeights,
                          trajectory=None) -> tuple[torch.Tensor, dict]:
    n = min(len(history), sample.n_steps)
    if n < 2:
        raise ValueError("Need at least 2 simulation steps for grad tune loss")

    lat_sim = _stack_history(history[:n], 'lateral_error')
    head_sim = _stack_history(history[:n], 'heading_error')
    v_sim = _stack_history(history[:n], 'v')
    steer_sim = _stack_history(history[:n], 'steer')
    acc_sim = _stack_history(history[:n], 'acc')

    lat_real = _record_tensor(sample.lateral_error, n, lat_sim)
    head_real = _record_tensor(sample.heading_error, n, head_sim)
    speed_real = _record_tensor(sample.speed_error, n, v_sim)
    if trajectory is None:
        ref_v_values = sample.ref_v
        ref_kappa_values = sample.ref_kappa
        refline_source_for_speed = 'csv'
    else:
        ref_v_values = np.array([p.v for p in trajectory[:n]], dtype=float)
        ref_kappa_values = np.array([p.kappa for p in trajectory[:n]], dtype=float)
        refline_source_for_speed = 'trajectory'
    speed_sim, speed_denom = _frenet_speed_error(
        v_sim, head_sim, lat_sim, ref_v_values, ref_kappa_values, n)
    tracking = (
        2.0 * weights.w_lat * lat_real * lat_sim
        + 2.0 * weights.w_head * head_real * head_sim
        + 2.0 * weights.w_speed * speed_real * speed_sim
    ).mean()

    steer_rate_mse = torch.tensor(0.0, dtype=tracking.dtype)
    acc_rate_mse = torch.tensor(0.0, dtype=tracking.dtype)
    if n > 1:
        steer_rate_mse = ((steer_sim[1:] - steer_sim[:-1]) ** 2).mean()
        acc_rate_mse = ((acc_sim[1:] - acc_sim[:-1]) ** 2).mean()

    loss = (tracking
            + weights.w_steer_rate * steer_rate_mse
            + weights.w_acc_rate * acc_rate_mse)

    return loss, {
        'n_steps': n,
        'speed_error_mode': 'frenet_ref_v_minus_s_dot',
        'speed_refline_source': refline_source_for_speed,
        'speed_denom_min': float(speed_denom.detach().min().item()),
        'speed_denom_max': float(speed_denom.detach().max().item()),
        'tracking_backward': float(tracking.detach().item()),
        'steer_rate_mse': float(steer_rate_mse.detach().item()),
        'acc_rate_mse': float(acc_rate_mse.detach().item()),
        'gradient_expression': GRADIENT_EXPRESSION,
    }
