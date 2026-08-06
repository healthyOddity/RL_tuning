"""Benchmark batched RL action evaluation throughput and fidelity."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from datetime import datetime

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, load_config
from model.trajectory import expand_trajectories
from optim.batched_rl_eval import evaluate_batched_actions, evaluate_scalar_actions
from optim.train_batch import RL_ACTION_BOUNDS


def materialize_standard_trajectories(trajectory_types=None, max_trajectories=None):
    rows = []
    excluded = []
    for key, _label, generator in expand_trajectories(trajectory_types):
        if 'park_route' in key:
            excluded.append(key)
            continue
        rows.append((key, generator()))
    if max_trajectories is not None:
        rows = rows[:max_trajectories]
    return rows, excluded


def make_actions(mode, count, seed):
    if mode == 'zero':
        return np.zeros((count, 11), dtype=np.float32)
    if mode == 'random':
        rng = np.random.default_rng(seed)
        bounds = RL_ACTION_BOUNDS.numpy()
        return rng.uniform(-bounds, bounds, size=(count, 11)).astype(np.float32)
    raise ValueError(f"unsupported action mode: {mode}")


def chunk_indices(count, batch_size):
    for start in range(0, count, batch_size):
        yield start, min(start + batch_size, count)


def run_scalar_reference(rows, actions, cfg, baseline_losses, norm_floor):
    keys = [key for key, _traj in rows]
    trajectories = [traj for _key, traj in rows]
    started = time.perf_counter()
    result = evaluate_scalar_actions(
        trajectories, keys, actions, cfg, baseline_losses, norm_floor)
    elapsed = time.perf_counter() - started
    return result, elapsed


def run_batched_once(rows, actions, cfg, baseline_losses, norm_floor, batch_size):
    keys_all = []
    raw_losses = []
    rewards = []
    started = time.perf_counter()
    for start, end in chunk_indices(len(rows), batch_size):
        chunk = rows[start:end]
        keys = [key for key, _traj in chunk]
        trajectories = [traj for _key, traj in chunk]
        result = evaluate_batched_actions(
            trajectories, keys, actions[start:end], cfg,
            baseline_losses, norm_floor)
        keys_all.extend(result['trajectory_keys'])
        raw_losses.extend(result['raw_losses'].tolist())
        rewards.extend(result['rewards'].tolist())
    elapsed = time.perf_counter() - started
    return {
        'trajectory_keys': keys_all,
        'raw_losses': np.asarray(raw_losses, dtype=np.float32),
        'rewards': np.asarray(rewards, dtype=np.float32),
    }, elapsed


def summarize_runs(runs, scalar_throughput):
    grouped = {}
    for run in runs:
        grouped.setdefault((run['action_mode'], run['batch_size']), []).append(run)

    summary = []
    for (action_mode, batch_size), group in grouped.items():
        ok = [run for run in group if run['status'] == 'ok']
        throughputs = [run['transitions_per_second'] for run in ok]
        rel_errors = [run['max_loss_rel_error'] for run in ok]
        elapsed = [run['elapsed_seconds'] for run in ok]
        median_throughput = statistics.median(throughputs) if throughputs else None
        median_elapsed = statistics.median(elapsed) if elapsed else None
        long_tail = False
        if median_elapsed and elapsed:
            long_tail = max(elapsed) > 2.0 * median_elapsed
        summary.append({
            'action_mode': action_mode,
            'batch_size': batch_size,
            'successful_repeats': len(ok),
            'total_repeats': len(group),
            'median_transitions_per_second': median_throughput,
            'speedup_vs_scalar': (
                median_throughput / scalar_throughput
                if median_throughput and scalar_throughput else None),
            'max_loss_rel_error': max(rel_errors) if rel_errors else None,
            'long_tail_risk': long_tail,
        })
    summary.sort(key=lambda row: (row['action_mode'], row['batch_size']))
    return summary


def write_results(payload, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, 'e05b_benchmark.yaml')
    csv_path = os.path.join(output_dir, 'e05b_benchmark.csv')
    with open(yaml_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    fields = [
        'action_mode', 'batch_size', 'successful_repeats', 'total_repeats',
        'median_transitions_per_second', 'speedup_vs_scalar',
        'max_loss_rel_error', 'long_tail_risk',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload['summary'])
    return yaml_path, csv_path


def run_benchmark(args):
    cfg = load_config(args.config)
    apply_plant_override(cfg, args.plant)
    if cfg['vehicle'].get('model_type') != 'truck_trailer':
        raise ValueError('E05B benchmark currently supports truck_trailer only')
    if args.disable_mlp:
        cfg['truck_trailer_vehicle'] = dict(cfg['truck_trailer_vehicle'])
        cfg['truck_trailer_vehicle']['checkpoint_path'] = ''

    rows, excluded = materialize_standard_trajectories(
        args.trajectories, args.max_trajectories)
    if not rows:
        raise ValueError('no trajectories selected')

    baseline_started = time.perf_counter()
    zero_actions = make_actions('zero', len(rows), args.seed)
    zero_baselines = {key: 1.0 for key, _traj in rows}
    zero_reference = evaluate_scalar_actions(
        [traj for _key, traj in rows],
        [key for key, _traj in rows],
        zero_actions,
        cfg,
        zero_baselines,
        norm_floor=1.0)
    baseline_losses = {
        key: max(float(loss), 1e-6)
        for key, loss in zip([key for key, _traj in rows], zero_reference['raw_losses'])
    }
    sorted_baselines = sorted(baseline_losses.values())
    norm_floor = sorted_baselines[len(sorted_baselines) // 2] ** 0.5
    baseline_seconds = time.perf_counter() - baseline_started

    all_runs = []
    scalar_refs = {}
    scalar_throughputs = []
    for action_mode in args.actions:
        actions = make_actions(action_mode, len(rows), args.seed)
        scalar_result, scalar_elapsed = run_scalar_reference(
            rows, actions, cfg, baseline_losses, norm_floor)
        scalar_throughput = len(rows) / scalar_elapsed
        scalar_throughputs.append(scalar_throughput)
        scalar_refs[action_mode] = {
            'elapsed_seconds': scalar_elapsed,
            'transitions': len(rows),
            'transitions_per_second': scalar_throughput,
        }

        for batch_size in args.batch_sizes:
            for repeat in range(1, args.repeats + 1):
                print(
                    f'[E05B] action={action_mode} batch={batch_size} '
                    f'repeat {repeat}/{args.repeats}',
                    flush=True)
                try:
                    batched_result, elapsed = run_batched_once(
                        rows, actions, cfg, baseline_losses,
                        norm_floor, batch_size)
                    denom = np.maximum(np.abs(scalar_result['raw_losses']), 1e-6)
                    rel_error = np.abs(
                        batched_result['raw_losses'] - scalar_result['raw_losses']) / denom
                    all_runs.append({
                        'action_mode': action_mode,
                        'batch_size': batch_size,
                        'repeat': repeat,
                        'status': 'ok',
                        'elapsed_seconds': elapsed,
                        'transitions': len(rows),
                        'transitions_per_second': len(rows) / elapsed,
                        'max_loss_rel_error': float(np.max(rel_error)),
                        'mean_loss_rel_error': float(np.mean(rel_error)),
                        'error': None,
                    })
                except Exception as exc:
                    all_runs.append({
                        'action_mode': action_mode,
                        'batch_size': batch_size,
                        'repeat': repeat,
                        'status': 'error',
                        'elapsed_seconds': None,
                        'transitions': 0,
                        'transitions_per_second': None,
                        'max_loss_rel_error': None,
                        'mean_loss_rel_error': None,
                        'error': f'{type(exc).__name__}: {exc}',
                    })

    scalar_throughput = min(scalar_throughputs) if scalar_throughputs else None
    payload = {
        'metadata': {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'plant': args.plant,
            'config': args.config,
            'disable_mlp': args.disable_mlp,
            'seed': args.seed,
            'trajectory_count': len(rows),
            'trajectory_keys': [key for key, _traj in rows],
            'excluded_routes': excluded,
            'batch_sizes': args.batch_sizes,
            'actions': args.actions,
            'repeats': args.repeats,
            'platform': platform.platform(),
            'python': sys.version.split()[0],
        },
        'baseline_precompute_seconds': baseline_seconds,
        'scalar_reference': scalar_refs,
        'runs': all_runs,
        'summary': summarize_runs(all_runs, scalar_throughput),
    }
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(__file__), '..', 'results', 'rl_bench',
            f'e05b_batched_{stamp}')
    yaml_path, csv_path = write_results(payload, output_dir)
    print(f'[E05B] YAML: {yaml_path}')
    print(f'[E05B] CSV: {csv_path}')
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description='Benchmark batched truck_trailer RL action evaluation')
    parser.add_argument('--plant', default='truck_trailer')
    parser.add_argument('--config', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trajectories', nargs='+', default=None)
    parser.add_argument('--max-trajectories', type=int, default=None)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--actions', nargs='+', default=['zero', 'random'])
    parser.add_argument('--disable-mlp', action='store_true')
    parser.add_argument('--output-dir', default=None)
    return parser


if __name__ == '__main__':
    run_benchmark(build_parser().parse_args())
