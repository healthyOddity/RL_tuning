from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


COLUMN_ALIASES = {
    'timestamp': ('timestamp', 'Time_s'),
    'controller_enable': ('controller_enable',),
    'x': ('x', 'position_enu.x', 'X_m'),
    'y': ('y', 'position_enu.y', 'Y_m'),
    'yaw': ('yaw', 'euler_angles.z', 'Yaw_deg'),
    'ego_v': ('ego_v', 'debug.simple_lon_debug.current_speed', 'Vx_mps'),
    'lateral_error': (
        'lateral_error', 'debug.simple_lat_debug.lateral_error'),
    'heading_error': (
        'heading_error', 'debug.simple_lat_debug.heading_error'),
    'speed_error': (
        'debug.simple_lon_debug.speed_error', 'speed_error'),
    'ref_x': ('ref_x',),
    'ref_y': ('ref_y',),
    'ref_yaw': ('ref_yaw', 'ref_theta'),
    'ref_kappa': ('ref_kappa',),
    'ref_s': ('ref_s',),
    'ref_v': ('ref_v',),
    'ref_a': ('ref_a',),
    'event_takeover': ('event_takeover',),
}

REQUIRED_FIELDS = (
    'timestamp', 'controller_enable', 'x', 'y', 'yaw', 'ego_v',
    'lateral_error', 'heading_error', 'speed_error',
    'ref_x', 'ref_y', 'ref_yaw', 'ref_kappa', 'ref_s', 'ref_v', 'ref_a',
)


@dataclass
class GradTuneSample:
    source_path: str
    time: np.ndarray
    init_x: float
    init_y: float
    init_yaw: float
    init_v: float
    yaw: np.ndarray
    ego_v: np.ndarray
    x: np.ndarray
    y: np.ndarray
    lateral_error: np.ndarray
    heading_error: np.ndarray
    speed_error: np.ndarray
    ref_x: np.ndarray
    ref_y: np.ndarray
    ref_yaw: np.ndarray
    ref_kappa: np.ndarray
    ref_s: np.ndarray
    ref_v: np.ndarray
    ref_a: np.ndarray
    window_start: float
    window_end: float
    preprocess_report: dict | None = None

    @property
    def n_steps(self) -> int:
        return int(len(self.time))


def resolve_column(columns, field: str) -> str | None:
    for candidate in COLUMN_ALIASES[field]:
        if candidate in columns:
            return candidate
    return None


def validate_columns(df) -> None:
    missing = []
    for field in REQUIRED_FIELDS:
        if resolve_column(df.columns, field) is None:
            missing.append(f"{field} ({' or '.join(COLUMN_ALIASES[field])})")
    if missing:
        raise ValueError("Missing required grad_tune columns: "
                         + ', '.join(missing))


def selected_columns(df) -> dict[str, str]:
    validate_columns(df)
    return {field: resolve_column(df.columns, field)
            for field in REQUIRED_FIELDS
            if resolve_column(df.columns, field) is not None}


def maybe_degrees_to_radians(values: np.ndarray) -> np.ndarray:
    arr = values.astype(float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(np.abs(finite)) > (2.0 * np.pi + 0.1):
        return np.deg2rad(arr)
    return arr


def default_run_id(source_path: str | Path) -> str:
    stem = Path(source_path).stem if source_path else 'grad_tune'
    return stem.replace('.', '_')
