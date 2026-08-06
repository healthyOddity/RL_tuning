"""Build a quality-gated E09-B/D manifest from data_process_general segments."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _segment_metadata(path: Path) -> dict:
    frame = pd.read_csv(path)
    required = ['Time_s', 'X_m', 'Y_m', 'Yaw_deg', 'Vx_mps']
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f'{path} missing columns: {missing}')
    values = frame[required].apply(pd.to_numeric, errors='coerce')
    finite = bool(np.isfinite(values.to_numpy()).all())
    time_s = values['Time_s'].to_numpy(dtype=float)
    x = values['X_m'].to_numpy(dtype=float)
    y = values['Y_m'].to_numpy(dtype=float)
    speed = values['Vx_mps'].to_numpy(dtype=float)
    distance_steps = np.hypot(np.diff(x), np.diff(y))
    dt = np.diff(time_s)
    return {
        'row_count': int(len(frame)),
        'duration_s': float(time_s[-1] - time_s[0]),
        'length_m': float(distance_steps.sum()),
        'speed_mean_mps': float(speed.mean()),
        'speed_std_mps': float(speed.std()),
        'finite': finite,
        'duplicate_position_count': int(np.sum(distance_steps <= 1e-6)),
        'bad_dt_count': int(np.sum((dt < 0.019) | (dt > 0.021))),
    }


def build_manifest(run_dir: str, output_path: str,
                   min_duration_s: float = 10.0,
                   min_length_m: float = 80.0,
                   one_shot_max_length_m: float = 280.0,
                   rolling_min_length_m: float = 200.0) -> dict:
    root = Path(run_dir).resolve()
    output = Path(output_path).resolve()
    paths = sorted(root.glob('items/*/filtered_segments/*_segment_*.csv'))
    trajectories = []
    rejected = []
    for path in paths:
        metadata = _segment_metadata(path)
        reasons = []
        if not metadata['finite']:
            reasons.append('non_finite')
        if metadata['duplicate_position_count']:
            reasons.append('duplicate_position')
        if metadata['bad_dt_count']:
            reasons.append('bad_dt')
        if metadata['duration_s'] < min_duration_s:
            reasons.append('short_duration')
        if metadata['length_m'] < min_length_m:
            reasons.append('short_length')
        key = path.stem
        row = {
            'key': key,
            'csv_path': os.path.relpath(path, output.parent),
            'session_id': path.parents[1].name,
            **metadata,
        }
        if reasons:
            row['reject_reasons'] = reasons
            rejected.append(row)
            continue
        roles = []
        if metadata['length_m'] <= one_shot_max_length_m:
            roles.append('one_shot')
        if metadata['length_m'] >= rolling_min_length_m:
            roles.append('rolling')
        row['roles'] = roles
        trajectories.append(row)

    payload = {
        'schema_version': 1,
        'source': 'data_process_general_e09_record_refline',
        'source_run_dir': os.path.relpath(root, output.parent),
        'requirements': {
            'min_duration_s': float(min_duration_s),
            'min_length_m': float(min_length_m),
            'one_shot_max_length_m': float(one_shot_max_length_m),
            'rolling_min_length_m': float(rolling_min_length_m),
        },
        'trajectory_count': len(trajectories),
        'rejected_count': len(rejected),
        'trajectories': trajectories,
        'rejected': rejected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-duration-s', type=float, default=10.0)
    parser.add_argument('--min-length-m', type=float, default=80.0)
    parser.add_argument('--one-shot-max-length-m', type=float, default=280.0)
    parser.add_argument('--rolling-min-length-m', type=float, default=200.0)
    args = parser.parse_args()
    payload = build_manifest(
        args.input_run_dir, args.output,
        min_duration_s=args.min_duration_s,
        min_length_m=args.min_length_m,
        one_shot_max_length_m=args.one_shot_max_length_m,
        rolling_min_length_m=args.rolling_min_length_m,
    )
    print(f"accepted={payload['trajectory_count']} rejected={payload['rejected_count']}")


if __name__ == '__main__':
    main()
