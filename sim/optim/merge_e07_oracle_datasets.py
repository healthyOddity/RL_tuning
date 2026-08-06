"""Merge corrected E07 oracle rows into the historical full dataset.

Constant-speed rows from schema v2 are numerically equivalent under the new
per-step speed reference. Variable-speed rows are replaced by schema v3 rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from collections import Counter

import numpy as np
import yaml


ROW_EXPORTS = (
    'trajectory_types', 'trajectory_families', 'trajectory_splits',
    'speed_kph', 'dc_soft_losses', 'oracle_soft_losses', 'dc_hard_losses',
    'oracle_hard_losses', 'dc_losses', 'oracle_scalar_losses',
    'improvement_pct', 'oracle_status', 'oracle_best_epoch',
    'oracle_best_hard_epoch', 'oracle_best_soft_epoch',
)


def _scalar(data, key, default=''):
    if key not in data.files:
        return default
    return np.asarray(data[key]).item()


def _atomic_savez(path: str, arrays: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(path), suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def merge_datasets(base_path: str, replacement_path: str,
                   output_dir: str) -> dict:
    base = np.load(base_path, allow_pickle=True)
    replacement = np.load(replacement_path, allow_pickle=True)
    base_keys = [str(k) for k in base['trajectory_keys']]
    replacement_keys = [str(k) for k in replacement['trajectory_keys']]
    base_index = {key: i for i, key in enumerate(base_keys)}
    replacement_index = {key: i for i, key in enumerate(replacement_keys)}
    missing = sorted(set(replacement_keys) - set(base_keys))
    if missing:
        raise ValueError(f'replacement keys absent from base: {missing}')
    if len(replacement_index) != len(replacement_keys):
        raise ValueError('duplicate trajectory keys in replacement dataset')

    arrays = {key: np.array(base[key], copy=True) for key in base.files}
    n_base = len(base_keys)
    n_replacement = len(replacement_keys)
    replaced_fields = []
    for key in base.files:
        if key not in replacement.files:
            continue
        base_value = np.asarray(base[key])
        replacement_value = np.asarray(replacement[key])
        if (base_value.ndim == 0 or replacement_value.ndim == 0
                or base_value.shape[0] != n_base
                or replacement_value.shape[0] != n_replacement
                or base_value.shape[1:] != replacement_value.shape[1:]):
            continue
        for trajectory_key, replacement_i in replacement_index.items():
            arrays[key][base_index[trajectory_key]] = replacement_value[replacement_i]
        replaced_fields.append(key)

    base_fingerprint = str(_scalar(base, 'run_fingerprint'))
    replacement_fingerprint = str(_scalar(replacement, 'run_fingerprint'))
    merged_fingerprint = hashlib.sha256(
        f'{base_fingerprint}|{replacement_fingerprint}|{replacement_keys}'.encode()
    ).hexdigest()
    arrays['schema_version'] = np.array(3, dtype=np.int32)
    arrays['selection_metric'] = np.array('hard_closed_loop_loss')
    arrays['loss_reference_mode'] = np.array('per_step_trajectory_v')
    arrays['run_fingerprint'] = np.array(merged_fingerprint)
    arrays['source_base_dataset'] = np.array(os.path.abspath(base_path))
    arrays['source_replacement_dataset'] = np.array(
        os.path.abspath(replacement_path))
    arrays['replaced_trajectory_keys'] = np.array(replacement_keys)

    output_npz = os.path.join(output_dir, 'oracle_dataset.npz')
    _atomic_savez(output_npz, arrays)

    output_csv = os.path.join(output_dir, 'oracle_dataset.csv')
    param_names = [str(v) for v in arrays['param_names']]
    rows = []
    for i, key in enumerate(base_keys):
        row = {
            'trajectory_key': key,
            'loss_reference_mode': 'per_step_trajectory_v',
            'source_schema': 'v3_replacement' if key in replacement_index
            else 'v2_constant_speed_equivalent',
        }
        for field in ROW_EXPORTS:
            if field in arrays:
                value = arrays[field][i]
                row[field] = value.item() if isinstance(value, np.generic) else value
        for j, param_name in enumerate(param_names):
            row[f'baseline_param_{j}_{param_name}'] = float(
                arrays['baseline_params'][i, j])
            row[f'oracle_param_{j}_{param_name}'] = float(
                arrays['oracle_params'][i, j])
            row[f'delta_param_{j}_{param_name}'] = float(
                arrays['delta_params'][i, j])
        rows.append(row)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dc_losses = arrays['dc_hard_losses'].astype(np.float64)
    oracle_losses = arrays['oracle_hard_losses'].astype(np.float64)
    improvement = ((dc_losses - oracle_losses)
                   / np.maximum(dc_losses, 1e-8) * 100.0)
    statuses = [str(v) for v in arrays['oracle_status']]
    summary = {
        'schema_version': 3,
        'loss_reference_mode': 'per_step_trajectory_v',
        'trajectory_count': n_base,
        'constant_speed_reused_count': n_base - n_replacement,
        'variable_speed_replaced_count': n_replacement,
        'base_dataset': os.path.abspath(base_path),
        'replacement_dataset': os.path.abspath(replacement_path),
        'merged_fingerprint': merged_fingerprint,
        'replaced_fields': replaced_fields,
        'dc_hard_avg_loss': float(np.mean(dc_losses)),
        'oracle_hard_avg_loss': float(np.mean(oracle_losses)),
        'aggregate_improvement_pct': float(
            (np.mean(dc_losses) - np.mean(oracle_losses))
            / max(float(np.mean(dc_losses)), 1e-8) * 100.0),
        'mean_per_trajectory_improvement_pct': float(np.mean(improvement)),
        'oracle_win_count': int(np.sum(oracle_losses < dc_losses)),
        'oracle_status_counts': dict(Counter(statuses)),
        'output_npz': os.path.abspath(output_npz),
        'output_csv': os.path.abspath(output_csv),
    }
    summary_path = os.path.join(output_dir, 'summary.yaml')
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    return {**summary, 'summary_path': os.path.abspath(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Replace variable-speed rows in an E07 oracle dataset')
    parser.add_argument('--base-dataset', required=True)
    parser.add_argument('--replacement-dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    summary = merge_datasets(
        args.base_dataset, args.replacement_dataset, args.output_dir)
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))


if __name__ == '__main__':
    main()
