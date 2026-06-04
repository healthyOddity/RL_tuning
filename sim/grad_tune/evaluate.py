from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, load_config
from grad_tune.plotting import plot_record_refline_comparison
from grad_tune.real_loss import GradTuneLossWeights
from grad_tune.record_adapter import load_csv_sample
from grad_tune.refline_builder import build_trajectory_points
from sim_loop import run_simulation


def _sim_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _series(history, key):
    return [float(h[key]) for h in history]


def _rmse(values):
    if not values:
        return 0.0
    return (sum(v * v for v in values) / len(values)) ** 0.5


def _rate_rmse(values):
    if len(values) < 2:
        return 0.0
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    return _rmse(diffs)


def compute_eval_metrics(history, sample, weights: GradTuneLossWeights,
                         trajectory=None) -> dict:
    n = min(len(history), sample.n_steps)
    lat = _series(history[:n], 'lateral_error')
    head = _series(history[:n], 'heading_error')
    ref_v = sample.ref_v if trajectory is None else [p.v for p in trajectory[:n]]
    speed = [float(history[i]['v']) - float(ref_v[i]) for i in range(n)]
    steer = _series(history[:n], 'steer')
    acc = _series(history[:n], 'acc')
    lat_mse = _rmse(lat) ** 2
    head_mse = _rmse(head) ** 2
    speed_mse = _rmse(speed) ** 2
    steer_rate_mse = _rate_rmse(steer) ** 2
    acc_rate_mse = _rate_rmse(acc) ** 2
    sim_loss = (
        weights.w_lat * lat_mse
        + weights.w_head * head_mse
        + weights.w_speed * speed_mse
        + weights.w_steer_rate * steer_rate_mse
        + weights.w_acc_rate * acc_rate_mse
    )
    return {
        'sim_loss': float(sim_loss),
        'lat_rmse': _rmse(lat),
        'head_rmse': _rmse(head),
        'speed_rmse': _rmse(speed),
        'steer_rate_rmse': _rate_rmse(steer),
        'acc_rate_rmse': _rate_rmse(acc),
        'n_steps': n,
    }


def _delta_pct(before, after):
    return 0.0 if abs(before) < 1e-12 else (after - before) / before * 100.0


def run_evaluation(args) -> dict:
    cfg_base = load_config(args.baseline_config)
    cfg_tuned = load_config(args.tuned_config)
    apply_plant_override(cfg_base, args.plant)
    apply_plant_override(cfg_tuned, args.plant)

    sample = load_csv_sample(
        args.input_csv, cfg_base, min_duration_s=args.min_duration,
        max_duration_s=args.window_duration,
        preprocess_mode=args.preprocess_mode,
        min_motion_speed=args.min_motion_speed,
        zero_threshold=args.zero_threshold,
        vy_jump_threshold=args.vy_jump_threshold,
        jump_dilate_frames=args.jump_dilate_frames)
    trajectory = build_trajectory_points(sample, source=args.refline_source)

    hist_base = run_simulation(
        trajectory, init_speed=sample.init_v, init_x=sample.init_x,
        init_y=sample.init_y, init_yaw=sample.init_yaw, cfg=cfg_base,
        differentiable=False)
    hist_tuned = run_simulation(
        trajectory, init_speed=sample.init_v, init_x=sample.init_x,
        init_y=sample.init_y, init_yaw=sample.init_yaw, cfg=cfg_tuned,
        differentiable=False)

    weights = GradTuneLossWeights(
        w_lat=args.w_lat, w_head=args.w_head, w_speed=args.w_speed,
        w_steer_rate=args.w_steer_rate, w_acc_rate=args.w_acc_rate)
    base_metrics = compute_eval_metrics(hist_base, sample, weights, trajectory)
    tuned_metrics = compute_eval_metrics(hist_tuned, sample, weights, trajectory)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}
    if not args.no_plots:
        plots = plot_record_refline_comparison(
            hist_base, hist_tuned, trajectory, str(output_dir), sample=sample,
            refline_source=args.refline_source)

    report = {
        'input_csv': str(args.input_csv),
        'baseline_config': str(args.baseline_config),
        'tuned_config': str(args.tuned_config),
        'plant': args.plant,
        'refline_source': args.refline_source,
        'window': {
            'start_s': sample.window_start,
            'end_s': sample.window_end,
            'n_steps': sample.n_steps,
        },
        'preprocess': sample.preprocess_report,
        'baseline': base_metrics,
        'tuned': tuned_metrics,
        'delta': {
            'loss_pct': round(_delta_pct(
                base_metrics['sim_loss'], tuned_metrics['sim_loss']), 6),
            'lat_rmse_pct': round(_delta_pct(
                base_metrics['lat_rmse'], tuned_metrics['lat_rmse']), 6),
            'head_rmse_pct': round(_delta_pct(
                base_metrics['head_rmse'], tuned_metrics['head_rmse']), 6),
            'speed_rmse_pct': round(_delta_pct(
                base_metrics['speed_rmse'], tuned_metrics['speed_rmse']), 6),
            'steer_rate_rmse_pct': round(_delta_pct(
                base_metrics['steer_rate_rmse'],
                tuned_metrics['steer_rate_rmse']), 6),
            'acc_rate_rmse_pct': round(_delta_pct(
                base_metrics['acc_rate_rmse'],
                tuned_metrics['acc_rate_rmse']), 6),
        },
        'plots': plots,
    }
    report_path = output_dir / 'grad_tune_eval.yaml'
    with open(report_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    report['report_yaml'] = str(report_path)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate baseline vs tuned params on the same record refline')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--baseline-config', default=str(_sim_dir() / 'configs' / 'default.yaml'))
    parser.add_argument('--tuned-config', required=True)
    parser.add_argument('--plant', default='truck_trailer',
                        choices=['kinematic', 'dynamic', 'hybrid_dynamic',
                                 'hybrid_v2', 'truck_trailer',
                                 'truck_trailer_dynamics'])
    parser.add_argument('--refline-source', default='record',
                        choices=['record', 'csv'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--window-duration', type=float, default=6.0)
    parser.add_argument('--min-duration', type=float, default=2.0)
    parser.add_argument('--preprocess-mode', default='combined',
                        choices=['combined', 'controller', 'none'])
    parser.add_argument('--min-motion-speed', type=float, default=0.05)
    parser.add_argument('--zero-threshold', type=float, default=1e-7)
    parser.add_argument('--vy-jump-threshold', type=float, default=2.0)
    parser.add_argument('--jump-dilate-frames', type=int, default=3)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--w-lat', type=float, default=10.0)
    parser.add_argument('--w-head', type=float, default=8.0)
    parser.add_argument('--w-speed', type=float, default=3.0)
    parser.add_argument('--w-steer-rate', type=float, default=0.05)
    parser.add_argument('--w-acc-rate', type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        report = run_evaluation(parse_args(argv))
    except Exception as exc:
        print(f"grad_tune evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"eval_report: {report['report_yaml']}")
    print(f"baseline_sim_loss: {report['baseline']['sim_loss']:.6f}")
    print(f"tuned_sim_loss: {report['tuned']['sim_loss']:.6f}")
    print(f"loss_delta_pct: {report['delta']['loss_pct']:+.3f}%")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
