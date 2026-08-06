"""Train and closed-loop evaluate an E07 RandomForest scene adapter."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, load_config
from model.trajectory import expand_trajectories
from optim.e07_dc_param_oracle_common import (
    DEFAULT_DC_CONFIG,
    PARAM_VECTOR_DIM,
    hard_loss_for_params,
    materialize_smoke_trajectories,
)
from optim.e07_trajectory_manifest import load_e07_manifest, materialize_e07_specs


def materialize_standard_trajectories():
    return [(key, gen()) for key, _label, gen in expand_trajectories(None)]


def _items_by_key(trajectory_items):
    return {key: traj for key, traj in trajectory_items}


def _write_csv(path: str, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_dataset(dataset_path: str):
    data = np.load(dataset_path, allow_pickle=True)
    keys = [str(k) for k in data['trajectory_keys']]
    trajectory_manifest = ''
    if 'trajectory_manifest' in data.files:
        trajectory_manifest = str(np.asarray(data['trajectory_manifest']).item())
    dc_loss_key = (
        'dc_hard_losses'
        if 'dc_hard_losses' in data.files
        else 'dc_losses'
    )
    oracle_loss_key = (
        'oracle_hard_losses'
        if 'oracle_hard_losses' in data.files
        else 'oracle_scalar_losses'
    )
    return {
        'keys': keys,
        'features': data['features'].astype(np.float32),
        'baseline_params': data['baseline_params'].astype(np.float32),
        'oracle_params': data['oracle_params'].astype(np.float32),
        'delta_params': data['delta_params'].astype(np.float32),
        'dc_losses': data[dc_loss_key].astype(np.float32),
        'oracle_scalar_losses': data[oracle_loss_key].astype(np.float32),
        'loss_source': {
            'dc': dc_loss_key,
            'oracle': oracle_loss_key,
        },
        'trajectory_manifest': trajectory_manifest,
    }


def _project_param_vectors(vectors: np.ndarray) -> np.ndarray:
    projected = vectors.copy()
    if projected.shape[1] != PARAM_VECTOR_DIM:
        raise ValueError(
            f'expected {PARAM_VECTOR_DIM} predicted params, got {projected.shape[1]}')
    # Current DC parameter order:
    # T2/T3/T4/T6 table y values occupy [0:28], lon gains [28:34],
    # switch_speed [34].
    projected[:, 0:34] = np.maximum(projected[:, 0:34], 0.0)
    projected[:, 34] = np.clip(projected[:, 34], 0.5, 10.0)
    return projected.astype(np.float32)


def _resolve_manifest_path(dataset_path: str, manifest_path: str | None) -> str | None:
    if not manifest_path:
        return None
    if os.path.exists(manifest_path):
        return manifest_path
    dataset_dir_candidate = os.path.join(os.path.dirname(dataset_path), manifest_path)
    if os.path.exists(dataset_dir_candidate):
        return dataset_dir_candidate
    return manifest_path


def train_random_forest_adapter(
    dataset_path: str,
    cfg: dict,
    trajectory_items=None,
    trajectory_manifest: str | None = None,
    output_dir: str | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
    test_size: float = 0.25,
):
    dataset = _load_dataset(dataset_path)
    n_samples = len(dataset['keys'])
    if n_samples < 2:
        raise ValueError('RandomForest smoke needs at least two samples')
    if trajectory_items is None:
        manifest_path = _resolve_manifest_path(
            dataset_path, trajectory_manifest or dataset['trajectory_manifest'])
        if manifest_path:
            trajectory_items = materialize_e07_specs(load_e07_manifest(manifest_path))
        else:
            trajectory_items = materialize_standard_trajectories()
    traj_by_key = _items_by_key(trajectory_items)

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('results', 'e07_adapter', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(
        n_estimators=int(n_estimators), random_state=int(random_state))
    model.fit(dataset['features'][train_idx], dataset['delta_params'][train_idx])

    pred_delta = model.predict(dataset['features'][test_idx]).astype(np.float32)
    pred_params = _project_param_vectors(
        dataset['baseline_params'][test_idx] + pred_delta)
    param_mse = float(mean_squared_error(
        dataset['oracle_params'][test_idx], pred_params))
    delta_mse = float(mean_squared_error(
        dataset['delta_params'][test_idx], pred_delta))

    rows = []
    adapter_losses = []
    for out_i, sample_i in enumerate(test_idx):
        key = dataset['keys'][sample_i]
        if key not in traj_by_key:
            raise ValueError(f"trajectory key {key} not available for eval")
        adapter_loss = hard_loss_for_params(
            traj_by_key[key], cfg, pred_params[out_i])
        adapter_losses.append(adapter_loss)
        row = {
            'trajectory_key': key,
            'split': 'test',
            'dc_loss': float(dataset['dc_losses'][sample_i]),
            'oracle_loss': float(dataset['oracle_scalar_losses'][sample_i]),
            'adapter_loss': float(adapter_loss),
            'adapter_vs_dc_pct': float(
                (adapter_loss - dataset['dc_losses'][sample_i])
                / max(float(dataset['dc_losses'][sample_i]), 1e-6) * 100.0),
            'adapter_vs_oracle_pct': float(
                (adapter_loss - dataset['oracle_scalar_losses'][sample_i])
                / max(float(dataset['oracle_scalar_losses'][sample_i]), 1e-6)
                * 100.0),
        }
        for j in range(PARAM_VECTOR_DIM):
            row[f'pred_param_{j}'] = float(pred_params[out_i, j])
            row[f'pred_delta_param_{j}'] = float(
                pred_params[out_i, j] - dataset['baseline_params'][sample_i, j])
        rows.append(row)

    model_path = os.path.join(output_dir, 'adapter_random_forest.joblib')
    csv_path = os.path.join(output_dir, 'adapter_eval_results.csv')
    summary_path = os.path.join(output_dir, 'summary.yaml')
    joblib.dump(model, model_path)
    _write_csv(csv_path, rows)

    adapter_losses = np.array(adapter_losses, dtype=np.float32)
    summary = {
        'dataset_path': dataset_path,
        'trajectory_manifest': trajectory_manifest or dataset['trajectory_manifest'],
        'sample_count': int(n_samples),
        'train_count': int(len(train_idx)),
        'test_count': int(len(test_idx)),
        'n_estimators': int(n_estimators),
        'random_state': int(random_state),
        'selection_metric': 'hard_closed_loop_loss',
        'dataset_loss_source': dataset['loss_source'],
        'param_dim': PARAM_VECTOR_DIM,
        'param_mse': param_mse,
        'delta_mse': delta_mse,
        'dc_avg_loss': float(np.mean(dataset['dc_losses'][test_idx])),
        'oracle_avg_loss': float(
            np.mean(dataset['oracle_scalar_losses'][test_idx])),
        'adapter_avg_loss': float(np.mean(adapter_losses)),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    return {
        'output_dir': output_dir,
        'model_path': model_path,
        'csv_path': csv_path,
        'summary_path': summary_path,
        'summary': summary,
    }


def _load_cli_cfg(config_path: str | None, plant: str):
    cfg = load_config(config_path)
    apply_plant_override(cfg, plant)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='Train E07 RandomForest adapter from oracle dataset')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--plant', default='truck_trailer',
                        choices=['truck_trailer'])
    parser.add_argument('--config', default=DEFAULT_DC_CONFIG)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--trajectory-manifest', default=None,
                        help='Optional E07 manifest; defaults to path stored in dataset')
    parser.add_argument('--smoke-fixtures', action='store_true',
                        help='Use two short synthetic trajectories for CLI smoke')
    args = parser.parse_args()

    cfg = _load_cli_cfg(args.config, args.plant)
    items = materialize_smoke_trajectories() if args.smoke_fixtures else None
    result = train_random_forest_adapter(
        dataset_path=args.dataset,
        cfg=cfg,
        trajectory_items=items,
        trajectory_manifest=args.trajectory_manifest,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    print(f"adapter model: {result['model_path']}")
    print(f"eval csv: {result['csv_path']}")
    print(f"summary: {result['summary_path']}")


if __name__ == '__main__':
    main()
