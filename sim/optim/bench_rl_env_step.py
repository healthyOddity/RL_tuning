"""Benchmark steady-state RLTuningEnv step throughput."""

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from functools import partial

import numpy as np
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optim.rl_env import RLTuningEnv


VALID_MODES = {'single', 'dummy', 'subproc'}


def parse_mode(value):
    try:
        mode, count_text = value.split(':', 1)
        n_envs = int(count_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid mode {value!r}; expected MODE:N") from exc
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if n_envs <= 0:
        raise ValueError('n_envs must be positive')
    if mode == 'single' and n_envs != 1:
        raise ValueError('single mode requires n_envs=1')
    return mode, n_envs


def configure_torch_threads(num_threads):
    if num_threads is None:
        return
    import torch

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def create_env(plant, config_path, seed, trajectories, baseline_stats,
               torch_threads=None):
    configure_torch_threads(torch_threads)
    env = RLTuningEnv(
        plant=plant,
        config_path=config_path,
        seed=seed,
        trajectory_types=trajectories,
        baseline_stats=baseline_stats,
    )
    env.reset(seed=seed)
    return env


def _build_runner(mode, n_envs, plant, config_path, seed, trajectories,
                  baseline_stats, subproc_torch_threads):
    torch_threads = subproc_torch_threads if mode == 'subproc' else None
    factories = [
        partial(
            create_env, plant, config_path, seed + rank, trajectories,
            baseline_stats, torch_threads)
        for rank in range(n_envs)
    ]
    if mode == 'single':
        return factories[0](), False
    if mode == 'dummy':
        return DummyVecEnv(factories), True
    return SubprocVecEnv(factories, start_method='spawn'), True


def benchmark_once(mode, n_envs, steps_per_env, plant, config_path, seed,
                   trajectories, baseline_stats, repeat,
                   subproc_torch_threads=None):
    runner = None
    started = time.perf_counter()
    try:
        runner, is_vector = _build_runner(
            mode, n_envs, plant, config_path, seed, trajectories,
            baseline_stats, subproc_torch_threads)
        runner.reset()
        startup_seconds = time.perf_counter() - started

        if is_vector:
            action = np.zeros((n_envs, 11), dtype=np.float32)
            runner.step(action)
        else:
            action = np.zeros(11, dtype=np.float32)
            runner.step(action)
            runner.reset()

        timed_started = time.perf_counter()
        for _ in range(steps_per_env):
            runner.step(action)
            if not is_vector:
                runner.reset()
        elapsed_seconds = time.perf_counter() - timed_started
        transitions = n_envs * steps_per_env
        return {
            'mode': mode,
            'n_envs': n_envs,
            'repeat': repeat,
            'torch_threads_per_worker': (
                subproc_torch_threads if mode == 'subproc' else None),
            'status': 'ok',
            'startup_seconds': startup_seconds,
            'elapsed_seconds': elapsed_seconds,
            'transitions': transitions,
            'transitions_per_second': transitions / elapsed_seconds,
            'seconds_per_transition': elapsed_seconds / transitions,
            'error': None,
        }
    except Exception as exc:
        return {
            'mode': mode,
            'n_envs': n_envs,
            'repeat': repeat,
            'torch_threads_per_worker': (
                subproc_torch_threads if mode == 'subproc' else None),
            'status': 'error',
            'startup_seconds': time.perf_counter() - started,
            'elapsed_seconds': None,
            'transitions': 0,
            'transitions_per_second': None,
            'seconds_per_transition': None,
            'error': f'{type(exc).__name__}: {exc}',
        }
    finally:
        if runner is not None:
            runner.close()


def summarize_runs(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault((run['mode'], run['n_envs']), []).append(run)

    summary = []
    for (mode, n_envs), mode_runs in grouped.items():
        successful = [run for run in mode_runs if run['status'] == 'ok']
        throughputs = [run['transitions_per_second'] for run in successful]
        summary.append({
            'mode': mode,
            'n_envs': n_envs,
            'successful_repeats': len(successful),
            'total_repeats': len(mode_runs),
            'median_transitions_per_second': (
                statistics.median(throughputs) if throughputs else None),
        })

    summary.sort(key=lambda row: (row['n_envs'], row['mode']))
    single = next(
        (row['median_transitions_per_second'] for row in summary
         if row['mode'] == 'single' and row['n_envs'] == 1), None)
    for row in summary:
        throughput = row['median_transitions_per_second']
        if single and throughput is not None:
            row['speedup_vs_single'] = throughput / single
            row['parallel_efficiency'] = (
                row['speedup_vs_single'] / row['n_envs'])
        else:
            row['speedup_vs_single'] = None
            row['parallel_efficiency'] = None
    return summary


def write_results(payload, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, 'e05_benchmark.yaml')
    csv_path = os.path.join(output_dir, 'e05_benchmark.csv')
    with open(yaml_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    fields = [
        'mode', 'n_envs', 'successful_repeats', 'total_repeats',
        'median_transitions_per_second', 'speedup_vs_single',
        'parallel_efficiency',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload['summary'])
    return yaml_path, csv_path


def run_benchmark(args):
    baseline_started = time.perf_counter()
    baseline_env = RLTuningEnv(
        plant=args.plant,
        config_path=args.config,
        seed=args.seed,
        trajectory_types=args.trajectories,
    )
    baseline_stats = baseline_env.export_baseline_stats()
    baseline_env.close()
    baseline_seconds = time.perf_counter() - baseline_started

    runs = []
    for mode_value in args.modes:
        mode, n_envs = parse_mode(mode_value)
        for repeat in range(1, args.repeats + 1):
            print(f'[E05] {mode}:{n_envs} repeat {repeat}/{args.repeats}')
            run = benchmark_once(
                mode, n_envs, args.steps_per_env, args.plant, args.config,
                args.seed, args.trajectories, baseline_stats, repeat,
                args.subproc_torch_threads)
            runs.append(run)
            print(
                f"  status={run['status']} "
                f"throughput={run['transitions_per_second']}")

    payload = {
        'metadata': {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'plant': args.plant,
            'config': args.config,
            'seed': args.seed,
            'trajectories': args.trajectories,
            'steps_per_env': args.steps_per_env,
            'repeats': args.repeats,
            'modes': args.modes,
            'logical_processors': os.cpu_count(),
            'platform': platform.platform(),
            'python': sys.version.split()[0],
            'subproc_torch_threads': args.subproc_torch_threads,
        },
        'baseline_precompute_seconds': baseline_seconds,
        'runs': runs,
        'summary': summarize_runs(runs),
    }
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(__file__), '..', 'results', 'rl_bench', stamp)
    yaml_path, csv_path = write_results(payload, output_dir)
    print(f'[E05] YAML: {yaml_path}')
    print(f'[E05] CSV: {csv_path}')
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description='Benchmark RLTuningEnv step throughput')
    parser.add_argument('--plant', default='truck_trailer')
    parser.add_argument('--config', default=None)
    parser.add_argument('--steps-per-env', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subproc-torch-threads', type=int, default=1)
    parser.add_argument('--trajectories', nargs='+', default=None)
    parser.add_argument(
        '--modes', nargs='+',
        default=['single:1', 'dummy:4', 'subproc:2', 'subproc:4',
                 'subproc:8', 'subproc:12'])
    parser.add_argument('--output-dir', default=None)
    return parser


if __name__ == '__main__':
    run_benchmark(build_parser().parse_args())
