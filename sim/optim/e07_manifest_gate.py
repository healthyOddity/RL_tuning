"""E07 v2 manifest gating utilities.

This module keeps fast geometry gating separate from slower vehicle simulation
checks.  The output manifests are ordinary E07 manifests, so downstream oracle
and adapter scripts can consume the accepted training manifest unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

from optim.e07_trajectory_manifest import materialize_e07_specs, write_e07_manifest


@dataclass(frozen=True)
class GeometryGateLimits:
    max_abs_curvature: float = 0.09
    max_lat_accel: float = 2.2
    max_abs_accel: float = 1.5


def _trajectory_arrays(traj):
    return {
        'x': np.asarray([p.x for p in traj], dtype=np.float64),
        'y': np.asarray([p.y for p in traj], dtype=np.float64),
        'theta': np.asarray([p.theta for p in traj], dtype=np.float64),
        'kappa': np.asarray([p.kappa for p in traj], dtype=np.float64),
        'v': np.asarray([p.v for p in traj], dtype=np.float64),
        'a': np.asarray([p.a for p in traj], dtype=np.float64),
        's': np.asarray([p.s for p in traj], dtype=np.float64),
        't': np.asarray([p.t for p in traj], dtype=np.float64),
    }


def geometry_metrics_for_spec(spec: dict) -> dict:
    key, traj = materialize_e07_specs([spec])[0]
    arr = _trajectory_arrays(traj)
    lat_accel = np.abs(arr['kappa'] * arr['v'] * arr['v'])
    return {
        'key': key,
        'point_count': int(len(traj)),
        'finite': bool(all(np.isfinite(values).all() for values in arr.values())),
        's_monotonic': bool(np.all(np.diff(arr['s']) >= -1e-9)),
        't_monotonic': bool(np.all(np.diff(arr['t']) >= -1e-9)),
        'min_speed': float(arr['v'].min()) if len(traj) else 0.0,
        'max_abs_curvature': float(np.abs(arr['kappa']).max()) if len(traj) else 0.0,
        'max_lat_accel': float(lat_accel.max()) if len(traj) else 0.0,
        'max_abs_accel': float(np.abs(arr['a']).max()) if len(traj) else 0.0,
    }


def geometry_rejection_reasons(metrics: dict,
                               limits: GeometryGateLimits | None = None) -> list[str]:
    limits = limits or GeometryGateLimits()
    reasons = []
    if not metrics['finite']:
        reasons.append('nonfinite')
    if not metrics['s_monotonic']:
        reasons.append('s_not_monotonic')
    if not metrics['t_monotonic']:
        reasons.append('t_not_monotonic')
    if metrics['min_speed'] < -1e-9:
        reasons.append('negative_speed')
    if (metrics['max_abs_curvature'] > limits.max_abs_curvature
            or metrics['max_lat_accel'] > limits.max_lat_accel):
        reasons.append('speed_curvature')
    if metrics['max_abs_accel'] > limits.max_abs_accel:
        reasons.append('acceleration')
    return reasons


def gate_specs_by_geometry(specs: list[dict],
                           limits: GeometryGateLimits | None = None):
    accepted = []
    rejected = []
    results = []
    for spec in specs:
        metrics = geometry_metrics_for_spec(spec)
        reasons = geometry_rejection_reasons(metrics, limits=limits)
        is_core = spec.get('gate_role') == 'core'
        status = 'accepted' if not reasons else 'rejected'
        if is_core and reasons:
            status = 'accepted_core_override'
        row = {
            **metrics,
            'trajectory_type': spec['type'],
            'gate_role': spec.get('gate_role'),
            'gate_status': status,
            'reasons': reasons,
        }
        results.append(row)
        if reasons and not is_core:
            rejected.append(spec)
        else:
            accepted.append(spec)
    return accepted, rejected, results


def write_gate_outputs(accepted: list[dict], rejected: list[dict],
                       results: list[dict], output_dir: str,
                       preset: str, seed: int,
                       extra_summary: dict | None = None) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    train_manifest = os.path.join(output_dir, 'e07_v2_train_manifest.yaml')
    rejected_manifest = os.path.join(output_dir, 'e07_v2_rejected_manifest.yaml')
    summary_path = os.path.join(output_dir, 'e07_v2_gate_summary.yaml')

    write_e07_manifest(accepted, train_manifest, preset=preset, seed=seed)
    write_e07_manifest(rejected, rejected_manifest, preset=f'{preset}_rejected',
                       seed=seed)

    summary = {
        'preset': preset,
        'seed': int(seed),
        'accepted_count': len(accepted),
        'rejected_count': len(rejected),
        'results': results,
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    return {
        'train_manifest': train_manifest,
        'rejected_manifest': rejected_manifest,
        'summary': summary_path,
    }
