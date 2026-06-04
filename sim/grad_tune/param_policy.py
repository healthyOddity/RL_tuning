from __future__ import annotations

import math

import torch

from grad_tune.observability import UPDATE_RULE, tensor_values


PARAM_MODES = {
    'lat_offset': ('lat_ctrl.',),
    'lon_speed': ('lon_ctrl.',),
    'all': ('lat_ctrl.', 'lon_ctrl.'),
}


def _matches_mode(name: str, param_mode: str) -> bool:
    if param_mode not in PARAM_MODES:
        raise ValueError(f"Unknown param_mode '{param_mode}'. "
                         f"Available: {sorted(PARAM_MODES)}")
    return any(name.startswith(prefix) for prefix in PARAM_MODES[param_mode])


def freeze_except(params, param_mode: str) -> None:
    for name, p in params.named_parameters():
        p.requires_grad_(_matches_mode(name, param_mode))


def collect_open_parameters(params, param_mode: str):
    return [(name, p) for name, p in params.named_parameters()
            if _matches_mode(name, param_mode) and p.requires_grad]


def _sanitize_grad(grad: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def _project_physical_bounds(name: str, p: torch.Tensor) -> None:
    if name in (
        'lon_ctrl.station_kp', 'lon_ctrl.station_ki',
        'lon_ctrl.low_speed_kp', 'lon_ctrl.low_speed_ki',
        'lon_ctrl.high_speed_kp', 'lon_ctrl.high_speed_ki',
    ):
        p.clamp_(min=0.0)
    elif name == 'lon_ctrl.switch_speed':
        p.clamp_(min=0.5, max=10.0)
    elif name in ('lat_ctrl.T2_y', 'lat_ctrl.T3_y',
                  'lat_ctrl.T4_y', 'lat_ctrl.T6_y'):
        p.clamp_(min=0.0)


def apply_gradient_step(params, param_mode: str, lr: float,
                        max_delta_ratio: float = 0.05,
                        grad_clip: float | None = 10.0) -> dict:
    open_params = collect_open_parameters(params, param_mode)
    if not open_params:
        raise ValueError(f"No open parameters for mode '{param_mode}'")

    grads = [p.grad for _name, p in open_params if p.grad is not None]
    if not grads:
        raise ValueError("No gradients found for open parameters")
    for grad in grads:
        if not torch.isfinite(grad).all():
            raise ValueError("Non-finite gradient detected")

    grad_vec = torch.cat([g.detach().double().reshape(-1) for g in grads])
    grad_norm = float(torch.linalg.vector_norm(grad_vec).item())
    if not math.isfinite(grad_norm):
        raise ValueError("Gradient norm is non-finite")
    clip_scale = 1.0
    if grad_clip is not None and grad_norm > float(grad_clip) > 0.0:
        clip_scale = float(grad_clip) / (grad_norm + 1e-12)

    updated = []
    with torch.no_grad():
        for name, p in open_params:
            if p.grad is None:
                continue
            before = p.detach().clone()
            grad = _sanitize_grad(p.grad.detach()) * clip_scale
            raw_delta = -float(lr) * grad
            max_delta = torch.maximum(
                before.abs() * float(max_delta_ratio),
                torch.full_like(before, 1e-6))
            delta = torch.clamp(raw_delta, -max_delta, max_delta)
            p.add_(delta)
            _project_physical_bounds(name, p)
            actual_delta = p.detach() - before
            nonzero = before.abs() > 1e-6
            if bool(nonzero.any()):
                delta_pct_max = float(
                    (actual_delta.abs()[nonzero] / before.abs()[nonzero])
                    .max().item() * 100.0)
            else:
                delta_pct_max = None
            updated.append({
                'name': name,
                'shape': list(p.shape),
                'before_mean': float(before.float().mean().item()),
                'after_mean': float(p.detach().float().mean().item()),
                'delta_abs_max': float(actual_delta.abs().max().item()),
                'delta_pct_max': delta_pct_max,
                'zero_baseline_abs_delta_max': float(
                    actual_delta.abs()[~nonzero].max().item())
                if bool((~nonzero).any()) else 0.0,
                'before_values': tensor_values(before),
                'grad_values': tensor_values(p.grad.detach()),
                'clipped_grad_values': tensor_values(grad),
                'raw_delta_values': tensor_values(raw_delta),
                'bounded_delta_values': tensor_values(delta),
                'after_values': tensor_values(p.detach()),
                'projection_delta_values': tensor_values(actual_delta - delta),
            })

    return {
        'param_mode': param_mode,
        'lr': lr,
        'max_delta_ratio': max_delta_ratio,
        'grad_norm': grad_norm,
        'grad_clip': grad_clip,
        'clip_scale': clip_scale,
        'update_rule': UPDATE_RULE,
        'updated': updated,
    }
