"""Evaluation source adapters for RL evaluation.

This module keeps non-standard evaluation inputs out of ``rl_evaluate.py``.
Each adapter returns ``EvalScenario`` objects so the evaluator can treat
standard trajectories, E07 manifest trajectories, and real-record reflines
with the same loop.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from common import TrajectoryPoint
from model.trajectory import expand_trajectories
from optim.e07_trajectory_manifest import (
    load_e07_manifest,
    materialize_e07_specs,
    metadata_by_key,
)
from optim.post_training import _build_eval_scenarios


@dataclass(frozen=True)
class EvalScenario:
    key: str
    label: str
    make_trajectory: Callable[[], list[TrajectoryPoint]]
    metadata: dict


def _copy_traj(traj: list[TrajectoryPoint]) -> list[TrajectoryPoint]:
    return [
        TrajectoryPoint(
            x=p.x, y=p.y, theta=p.theta, kappa=p.kappa,
            v=p.v, a=p.a, s=p.s, t=p.t,
        )
        for p in traj
    ]


def build_standard_scenarios(trajectory_types=None,
                             include_park_route: bool = False) -> list[EvalScenario]:
    raw = (_build_eval_scenarios(trajectory_types)
           if include_park_route else expand_trajectories(trajectory_types))
    scenarios = []
    for key, label, gen in raw:
        scenarios.append(EvalScenario(
            key=key,
            label=label,
            make_trajectory=gen,
            metadata={'source': 'standard'},
        ))
    return scenarios


def build_manifest_scenarios(path: str,
                             max_scenarios: int | None = None) -> list[EvalScenario]:
    specs = load_e07_manifest(path)
    if max_scenarios is not None:
        specs = specs[:max_scenarios]
    materialized = materialize_e07_specs(specs)
    metadata = metadata_by_key(specs)
    scenarios = []
    for key, traj in materialized:
        meta = {
            'source': 'trajectory_manifest',
            'manifest_path': path,
            **metadata.get(key, {}),
        }
        scenarios.append(EvalScenario(
            key=key,
            label=key,
            make_trajectory=lambda traj=traj: _copy_traj(traj),
            metadata=meta,
        ))
    return scenarios


def _to_float(value):
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_numeric(row: dict, names: list[str]):
    for name in names:
        if name in row:
            value = _to_float(row.get(name))
            if value is not None:
                return value
    return None


def _unwrap_degrees(theta: np.ndarray) -> np.ndarray:
    if len(theta) == 0:
        return theta
    if np.nanmax(np.abs(theta)) > 2 * math.pi:
        return np.deg2rad(theta)
    return theta


def _derive_s(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(xs) == 0:
        return np.array([], dtype=float)
    ds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def _derive_kappa(theta: np.ndarray, s: np.ndarray) -> np.ndarray:
    if len(theta) < 2:
        return np.zeros_like(theta)
    theta_unwrapped = np.unwrap(theta)
    if np.any(np.diff(s) <= 0.0):
        raise ValueError('record-refline arc length must be strictly increasing')
    return np.gradient(theta_unwrapped, s, edge_order=1)


def _derive_acc(vs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    if len(vs) < 2:
        return np.zeros_like(vs)
    if np.any(np.diff(ts) <= 0.0):
        raise ValueError('record-refline time must be strictly increasing')
    return np.gradient(vs, ts, edge_order=1)


def build_real_csv_scenario(path: str, key: str = 'record',
                            max_steps: int | None = None,
                            min_speed_mps: float = 0.5) -> EvalScenario:
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    xs, ys, thetas, speeds, times = [], [], [], [], []
    skipped_disabled = 0
    skipped_stationary = 0
    skipped_invalid = 0
    skipped_nonmonotonic = 0
    skipped_duplicate_position = 0
    for idx, row in enumerate(rows):
        controller_enable = _first_numeric(row, ['controller_enable'])
        if controller_enable is not None and controller_enable != 1.0:
            skipped_disabled += 1
            continue

        x = _first_numeric(row, ['position_enu.x', 'X_m', 'x'])
        y = _first_numeric(row, ['position_enu.y', 'Y_m', 'y'])
        theta = _first_numeric(row, [
            'heading', 'Yaw_deg', 'euler_angles.z', 'yaw_deg', 'yaw',
        ])
        speed_kph = _first_numeric(row, [
            'VehicleInfoBDData.BDCFF10D0_VCU_VehicleSpeed',
            'vehicle_speed_kph',
            'speed_kph',
        ])
        speed = (speed_kph / 3.6 if speed_kph is not None else
                 _first_numeric(row, ['Vx_mps', 'v', 'speed_mps']))
        t = _first_numeric(row, ['timestamp', 'Time_s', 't'])
        if t is None:
            t = _first_numeric(row, ['timestamp_ms'])
            if t is not None:
                t = t / 1000.0
        if t is None:
            t = idx * 0.02

        values = [x, y, theta, speed, t]
        if any(value is None or not math.isfinite(value) for value in values):
            skipped_invalid += 1
            continue
        if speed < min_speed_mps:
            skipped_stationary += 1
            continue
        if times and t <= times[-1]:
            skipped_nonmonotonic += 1
            continue
        if xs and math.hypot(x - xs[-1], y - ys[-1]) <= 1e-6:
            skipped_duplicate_position += 1
            continue

        xs.append(x)
        ys.append(y)
        thetas.append(theta)
        speeds.append(speed)
        times.append(t)
        if max_steps is not None and len(xs) >= max_steps:
            break

    if len(xs) < 2:
        raise ValueError(f'Not enough valid real CSV rows: {path}')

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    theta_arr = _unwrap_degrees(np.asarray(thetas, dtype=float))
    speed_arr = np.asarray(speeds, dtype=float)
    time_arr = np.asarray(times, dtype=float)
    time_arr = time_arr - time_arr[0]
    s_arr = _derive_s(xs_arr, ys_arr)
    kappa_arr = _derive_kappa(theta_arr, s_arr)
    acc_arr = _derive_acc(speed_arr, time_arr)

    traj = [
        TrajectoryPoint(
            x=float(xs_arr[i]), y=float(ys_arr[i]),
            theta=float(theta_arr[i]), kappa=float(kappa_arr[i]),
            v=float(speed_arr[i]), a=float(acc_arr[i]),
            s=float(s_arr[i]), t=float(time_arr[i]),
        )
        for i in range(len(xs_arr))
    ]
    label = os.path.splitext(os.path.basename(path))[0]
    return EvalScenario(
        key=key,
        label=label,
        make_trajectory=lambda traj=traj: _copy_traj(traj),
        metadata={
            'source': 'real_csv_record_refline',
            'csv_path': path,
            'record_key': key,
            'raw_row_count': len(rows),
            'accepted_row_count': len(traj),
            'duration_s': float(time_arr[-1]),
            'trajectory_length_m': float(s_arr[-1]),
            'filter_counts': {
                'disabled': skipped_disabled,
                'stationary': skipped_stationary,
                'invalid': skipped_invalid,
                'nonmonotonic_time': skipped_nonmonotonic,
                'duplicate_position': skipped_duplicate_position,
            },
        },
    )


def build_real_csv_manifest_scenarios(path: str, role: str | None = None,
                                      max_scenarios: int | None = None
                                      ) -> list[EvalScenario]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding='utf-8') as stream:
        payload = yaml.safe_load(stream) or {}
    items = payload.get('trajectories') or []
    scenarios = []
    for item in items:
        roles = [str(value) for value in item.get('roles', [])]
        if role is not None and role not in roles:
            continue
        csv_path = Path(str(item['csv_path']))
        if not csv_path.is_absolute():
            csv_path = (manifest_path.parent / csv_path).resolve()
        scenario = build_real_csv_scenario(
            str(csv_path), key=str(item['key']))
        scenarios.append(EvalScenario(
            key=scenario.key,
            label=scenario.label,
            make_trajectory=scenario.make_trajectory,
            metadata={
                **scenario.metadata,
                'record_manifest_path': str(manifest_path),
                'manifest_roles': roles,
                'manifest_quality': {
                    key: value for key, value in item.items()
                    if key not in {'key', 'csv_path', 'roles'}
                },
            },
        ))
        if max_scenarios is not None and len(scenarios) >= max_scenarios:
            break
    return scenarios


def apply_window_policy(scenarios: list[EvalScenario],
                        policy: str) -> list[EvalScenario]:
    if policy == 'all':
        return list(scenarios)
    if policy == 'id_only':
        return [
            scenario for scenario in scenarios
            if scenario.metadata.get('source') == 'standard'
        ]
    if policy == 'ood_only':
        return [
            scenario for scenario in scenarios
            if scenario.metadata.get('source') != 'standard'
        ]
    raise ValueError(f'Unknown eval window policy: {policy}')
