"""Approximate jittery E09 real drives with smooth deployment reference lines.

The real drive remains the source of the route geometry and speed profile.  Only
the spatial reference line is fitted; heading and curvature are then recomputed
from that fitted line so the agent does not observe localization-scale jitter.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import UnivariateSpline


def _count_peaks(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    threshold = np.mean(values) * 0.3 if np.mean(values) > 1e-9 else 0.001
    peaks = sum(
        values[index] > values[index - 1]
        and values[index] > values[index + 1]
        and values[index] > threshold
        for index in range(1, len(values) - 1)
    )
    return int(max(peaks, 1)) if np.max(values) > 1e-9 else 0


def _count_sign_changes(values: np.ndarray) -> int:
    return int(np.sum(values[1:] * values[:-1] < 0.0))


def _curvature_from_heading(heading_rad: np.ndarray,
                            distance_m: np.ndarray) -> np.ndarray:
    return np.gradient(np.unwrap(heading_rad), distance_m, edge_order=1)


def smooth_reference_xy(x: np.ndarray, y: np.ndarray,
                        fit_rms_tolerance_m: float = 0.02) -> dict:
    """Fit x(s), y(s) smoothing splines and derive a consistent reference line."""
    if len(x) < 4:
        raise ValueError('reference line needs at least four points')
    if fit_rms_tolerance_m <= 0.0:
        raise ValueError('fit_rms_tolerance_m must be positive')

    raw_steps = np.hypot(np.diff(x), np.diff(y))
    if np.any(raw_steps <= 1e-6):
        raise ValueError('reference line contains duplicate positions')
    raw_s = np.concatenate([[0.0], np.cumsum(raw_steps)])
    smoothing_budget = len(raw_s) * fit_rms_tolerance_m ** 2
    spline_x = UnivariateSpline(raw_s, x, k=3, s=smoothing_budget)
    spline_y = UnivariateSpline(raw_s, y, k=3, s=smoothing_budget)

    smooth_x = spline_x(raw_s)
    smooth_y = spline_y(raw_s)
    dx = spline_x.derivative(1)(raw_s)
    dy = spline_y.derivative(1)(raw_s)
    heading_rad = np.unwrap(np.arctan2(dy, dx))
    smooth_steps = np.hypot(np.diff(smooth_x), np.diff(smooth_y))
    if np.any(smooth_steps <= 1e-6):
        raise ValueError('smoothing produced duplicate positions')
    smooth_s = np.concatenate([[0.0], np.cumsum(smooth_steps)])
    curvature = _curvature_from_heading(heading_rad, smooth_s)
    displacement = np.hypot(smooth_x - x, smooth_y - y)

    return {
        'x': smooth_x,
        'y': smooth_y,
        'heading_rad': heading_rad,
        'curvature': curvature,
        's': smooth_s,
        'raw_length_m': float(raw_s[-1]),
        'smooth_length_m': float(smooth_s[-1]),
        'rms_deviation_m': float(np.sqrt(np.mean(displacement ** 2))),
        'max_deviation_m': float(np.max(displacement)),
        'start_deviation_m': float(displacement[0]),
        'end_deviation_m': float(displacement[-1]),
    }


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f'missing required column: {column}')
    values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'column contains non-finite values: {column}')
    return values


def smooth_reference_frame(frame: pd.DataFrame,
                           fit_rms_tolerance_m: float = 0.02) -> tuple[pd.DataFrame, dict]:
    x = _numeric(frame, 'X_m')
    y = _numeric(frame, 'Y_m')
    raw_heading_deg = _numeric(frame, 'Yaw_deg')
    result = smooth_reference_xy(x, y, fit_rms_tolerance_m)

    smooth = frame.copy()
    smooth['X_m'] = result['x']
    smooth['Y_m'] = result['y']
    smooth['Yaw_deg'] = np.rad2deg(result['heading_rad'])
    for target, values in (
        ('position_enu.x', result['x']),
        ('position_enu.y', result['y']),
        ('ins_heading_deg', np.rad2deg(result['heading_rad'])),
    ):
        if target in smooth.columns:
            smooth[target] = values
    smooth['ref_s_smooth_m'] = result['s']
    smooth['ref_kappa_smooth_1pm'] = result['curvature']

    raw_steps = np.hypot(np.diff(x), np.diff(y))
    raw_s = np.concatenate([[0.0], np.cumsum(raw_steps)])
    raw_kappa = _curvature_from_heading(np.deg2rad(raw_heading_deg), raw_s)
    result.update({
        'length_ratio': result['smooth_length_m'] / max(result['raw_length_m'], 1e-9),
        'raw_curvature_peak_count': _count_peaks(np.abs(raw_kappa)),
        'smooth_curvature_peak_count': _count_peaks(np.abs(result['curvature'])),
        'raw_curvature_sign_change_count': _count_sign_changes(raw_kappa),
        'smooth_curvature_sign_change_count': _count_sign_changes(result['curvature']),
    })
    metrics = {key: value for key, value in result.items()
               if not isinstance(value, np.ndarray)}
    return smooth, metrics


def smooth_manifest(input_manifest: str, output_dir: str,
                    fit_rms_tolerance_m: float = 0.02,
                    max_deviation_m: float = 0.30) -> dict:
    source_path = Path(input_manifest).resolve()
    output_root = Path(output_dir).resolve()
    with source_path.open(encoding='utf-8') as stream:
        source = yaml.safe_load(stream) or {}

    rows = []
    for item in source.get('trajectories', []):
        csv_path = Path(str(item['csv_path']))
        if not csv_path.is_absolute():
            csv_path = (source_path.parent / csv_path).resolve()
        frame = pd.read_csv(csv_path)
        smooth, metrics = smooth_reference_frame(
            frame, fit_rms_tolerance_m=fit_rms_tolerance_m)
        if metrics['max_deviation_m'] > max_deviation_m:
            raise ValueError(
                f"{item['key']} max deviation {metrics['max_deviation_m']:.3f} m "
                f'exceeds gate {max_deviation_m:.3f} m')

        session_id = str(item.get('session_id') or csv_path.parent.parent.name)
        output_csv = (output_root / 'items' / session_id /
                      f"{item['key']}_smooth_reference.csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        smooth.to_csv(output_csv, index=False)
        rows.append({
            **item,
            'csv_path': os.path.relpath(output_csv, output_root),
            'source_csv_path': os.path.relpath(csv_path, output_root),
            'reference_kind': 'smooth_approximation_of_real_drive',
            'smoothing_metrics': metrics,
        })

    payload = {
        'schema_version': 1,
        'source': 'e09_smooth_reference_from_real_drive',
        'source_manifest': os.path.relpath(source_path, output_root),
        'trajectory_count': len(rows),
        'smoothing': {
            'method': 'cubic_univariate_spline_xy_over_raw_arc_length',
            'fit_rms_tolerance_m_per_axis': float(fit_rms_tolerance_m),
            'max_deviation_gate_m': float(max_deviation_m),
            'speed_profile': 'preserve_real_Vx_mps',
            'heading': 'recomputed_from_smooth_xy_tangent',
            'curvature': 'recomputed_from_smooth_heading_over_smooth_arc_length',
        },
        'trajectories': rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / 'e09_smooth_reference_manifest.yaml'
    with manifest_path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fit-rms-tolerance-m', type=float, default=0.02)
    parser.add_argument('--max-deviation-m', type=float, default=0.30)
    args = parser.parse_args()
    payload = smooth_manifest(
        args.input_manifest, args.output_dir,
        fit_rms_tolerance_m=args.fit_rms_tolerance_m,
        max_deviation_m=args.max_deviation_m,
    )
    print(f"generated={payload['trajectory_count']}")


if __name__ == '__main__':
    main()
