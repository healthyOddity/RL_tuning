from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from grad_tune.data_schema import (
    COLUMN_ALIASES,
    GradTuneSample,
    maybe_degrees_to_radians,
    resolve_column,
    selected_columns,
    validate_columns,
)
from grad_tune.preprocess import select_valid_segment


def _as_float_array(series) -> np.ndarray:
    return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)


def _longest_true_segment(mask: np.ndarray) -> tuple[int, int] | None:
    best = None
    start = None
    for idx, ok in enumerate(mask):
        if ok and start is None:
            start = idx
        elif not ok and start is not None:
            end = idx
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
            start = None
    if start is not None:
        end = len(mask)
        if best is None or (end - start) > (best[1] - best[0]):
            best = (start, end)
    return best


def _interp(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    finite = np.isfinite(y_src)
    if finite.sum() < 2:
        raise ValueError("Not enough finite samples for interpolation")
    return np.interp(t_dst, t_src[finite], y_src[finite])


def _canonical_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = selected_columns(df)
    arrays = {field: _as_float_array(df[col]) for field, col in cols.items()}
    takeover_col = resolve_column(df.columns, 'event_takeover')
    if takeover_col is not None:
        arrays['event_takeover'] = _as_float_array(df[takeover_col])
    return arrays


def dataframe_to_sample(df: pd.DataFrame, dt: float = 0.02,
                        source_path: str = '<dataframe>',
                        min_duration_s: float = 2.0,
                        max_duration_s: float | None = 6.0) -> GradTuneSample:
    validate_columns(df)
    arrays = _canonical_arrays(df)

    t = arrays['timestamp']
    enabled = arrays['controller_enable'] == 1
    if 'event_takeover' in arrays:
        enabled = enabled & (arrays['event_takeover'] == 0)

    dt_raw = np.diff(t, prepend=t[0])
    dt_ok = np.ones_like(enabled, dtype=bool)
    dt_ok[1:] = (dt_raw[1:] >= 0.005) & (dt_raw[1:] <= 0.039)
    finite_core = np.isfinite(t) & np.isfinite(arrays['x']) & np.isfinite(arrays['y'])
    mask = enabled & dt_ok & finite_core

    segment = _longest_true_segment(mask)
    if segment is None:
        raise ValueError("No valid controller_enable segment found")
    start, end = segment
    if end - start < 2:
        raise ValueError("Valid segment is too short")

    seg_t = t[start:end]
    if seg_t[-1] - seg_t[0] < min_duration_s:
        raise ValueError(
            f"Valid segment duration {seg_t[-1] - seg_t[0]:.3f}s is shorter "
            f"than required {min_duration_s:.3f}s")

    duration = seg_t[-1] - seg_t[0]
    if max_duration_s is not None:
        duration = min(duration, float(max_duration_s))
    t_out = np.arange(0.0, duration + 0.5 * dt, dt)
    t_abs = seg_t[0] + t_out

    out = {}
    for field in (
        'x', 'y', 'yaw', 'ego_v', 'lateral_error', 'heading_error',
        'speed_error', 'ref_x', 'ref_y', 'ref_yaw', 'ref_kappa', 'ref_s',
        'ref_v', 'ref_a',
    ):
        out[field] = _interp(t[start:end], arrays[field][start:end], t_abs)

    yaw = maybe_degrees_to_radians(out['yaw'])
    ref_yaw = maybe_degrees_to_radians(out['ref_yaw'])

    return GradTuneSample(
        source_path=str(source_path),
        time=t_out,
        init_x=float(out['x'][0]),
        init_y=float(out['y'][0]),
        init_yaw=float(yaw[0]),
        init_v=float(out['ego_v'][0]),
        yaw=yaw,
        ego_v=out['ego_v'],
        x=out['x'],
        y=out['y'],
        lateral_error=out['lateral_error'],
        heading_error=out['heading_error'],
        speed_error=out['speed_error'],
        ref_x=out['ref_x'],
        ref_y=out['ref_y'],
        ref_yaw=ref_yaw,
        ref_kappa=out['ref_kappa'],
        ref_s=out['ref_s'],
        ref_v=out['ref_v'],
        ref_a=out['ref_a'],
        window_start=float(seg_t[0]),
        window_end=float(seg_t[0] + duration),
    )


def load_csv_sample(input_csv: str | Path, cfg: dict,
                    min_duration_s: float = 2.0,
                    max_duration_s: float | None = 6.0,
                    preprocess_mode: str = 'combined',
                    min_motion_speed: float = 0.05,
                    zero_threshold: float = 1e-7,
                    vy_jump_threshold: float = 2.0,
                    jump_dilate_frames: int = 3) -> GradTuneSample:
    path = Path(input_csv)
    df = pd.read_csv(path)
    dt = float(cfg.get('simulation', {}).get('dt', 0.02))
    df, preprocess_report = select_valid_segment(
        df, mode=preprocess_mode, min_motion_speed=min_motion_speed,
        zero_threshold=zero_threshold, min_duration_s=min_duration_s,
        vy_jump_threshold=vy_jump_threshold,
        jump_dilate_frames=jump_dilate_frames, dt=dt)
    sample = dataframe_to_sample(
        df, dt=dt, source_path=str(path),
        min_duration_s=min_duration_s, max_duration_s=max_duration_s)
    sample.preprocess_report = preprocess_report.__dict__
    return sample


def record_dir_to_csv(record_dir: str | Path, output_root: str | Path | None = None):
    input_path = Path(record_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_file():
        record_files = [input_path]
    else:
        record_files = sorted(
            p for p in input_path.rglob('*')
            if p.is_file() and (p.suffix == '.record' or '.record.' in p.name))
    if not record_files:
        raise ValueError(f"No record files found in {input_path}")

    sim_dir = Path(__file__).resolve().parents[1]
    repo_dir = sim_dir.parent
    script = repo_dir / 'data_process_truck' / 'data_process' / 'extract_topic_truck_trq.py'
    if not script.exists():
        raise FileNotFoundError(script)

    output_dir = Path(output_root) if output_root else input_path.parent / '0_interpolated'
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    for record_file in record_files:
        before = set(output_dir.glob('*_interpolated.csv'))
        cmd = [
            sys.executable, str(script), str(record_file),
            '--output-dir', str(output_dir),
        ]
        result = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "record extraction failed for "
                f"{record_file}: {result.stderr or result.stdout}")
        after = set(output_dir.glob('*_interpolated.csv'))
        new_files = sorted(after - before)
        if new_files:
            produced.extend(new_files)
        else:
            expected = output_dir / f"{record_file.name.replace('.record', '')}_interpolated.csv"
            if expected.exists():
                produced.append(expected)

    produced = sorted(set(produced))
    if not produced:
        raise RuntimeError(f"No interpolated CSV produced in {output_dir}")
    if len(produced) > 1:
        raise ValueError(
            "Multiple interpolated CSV files were produced. Use --input-csv "
            f"to select one explicitly: {[str(p) for p in produced]}")
    return produced[0]
