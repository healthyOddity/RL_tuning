from __future__ import annotations

import numpy as np

from common import TrajectoryPoint
from grad_tune.data_schema import GradTuneSample


def _arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=float)
    ds = np.hypot(np.diff(x.astype(float)), np.diff(y.astype(float)))
    return np.concatenate([[0.0], np.cumsum(ds)])


def _curvature_from_yaw(yaw: np.ndarray, s: np.ndarray) -> np.ndarray:
    if len(yaw) < 3:
        return np.zeros_like(yaw, dtype=float)
    yaw_unwrapped = np.unwrap(yaw.astype(float))
    kappa = np.zeros_like(yaw_unwrapped, dtype=float)
    for i in range(1, len(yaw_unwrapped) - 1):
        ds = s[i + 1] - s[i - 1]
        if abs(ds) > 1e-6:
            kappa[i] = (yaw_unwrapped[i + 1] - yaw_unwrapped[i - 1]) / ds
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def _acc_from_speed(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(v) < 2:
        return np.zeros_like(v, dtype=float)
    return np.gradient(v.astype(float), t.astype(float), edge_order=1)


def build_csv_trajectory_points(sample: GradTuneSample) -> list[TrajectoryPoint]:
    if sample.n_steps < 2:
        raise ValueError("Refline needs at least 2 points")
    return [
        TrajectoryPoint(
            x=float(sample.ref_x[i]),
            y=float(sample.ref_y[i]),
            theta=float(sample.ref_yaw[i]),
            kappa=float(sample.ref_kappa[i]),
            v=float(sample.ref_v[i]),
            a=float(sample.ref_a[i]),
            s=float(sample.ref_s[i]),
            t=float(sample.time[i]),
        )
        for i in range(sample.n_steps)
    ]


def build_record_trajectory_points(sample: GradTuneSample) -> list[TrajectoryPoint]:
    if sample.n_steps < 2:
        raise ValueError("Refline needs at least 2 points")
    s = _arc_length(sample.x, sample.y)
    kappa = _curvature_from_yaw(sample.yaw, s)
    acc = _acc_from_speed(sample.ego_v, sample.time)
    return [
        TrajectoryPoint(
            x=float(sample.x[i]),
            y=float(sample.y[i]),
            theta=float(sample.yaw[i]),
            kappa=float(kappa[i]),
            v=float(sample.ego_v[i]),
            a=float(acc[i]),
            s=float(s[i]),
            t=float(sample.time[i]),
        )
        for i in range(sample.n_steps)
    ]


def build_trajectory_points(sample: GradTuneSample,
                            source: str = 'record') -> list[TrajectoryPoint]:
    if source == 'record':
        return build_record_trajectory_points(sample)
    if source == 'csv':
        return build_csv_trajectory_points(sample)
    raise ValueError("Unknown refline source. Expected 'record' or 'csv'.")
