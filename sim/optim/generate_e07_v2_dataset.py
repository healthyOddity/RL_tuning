"""Generate the gated E07 v2 trajectory dataset.

The script produces:
- ``e07_v2_train_manifest.yaml``: accepted trajectories for downstream E07
- ``e07_v2_rejected_manifest.yaml``: candidates excluded by gates
- ``e07_v2_gate_summary.yaml``: per-trajectory gate metrics and reasons
- ``e07_v2_dc_trajectory_preview.png`` when simulation gate is enabled
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, apply_runtime_overrides, load_config
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from optim.e07_manifest_gate import gate_specs_by_geometry, write_gate_outputs
from optim.e07_trajectory_manifest import (
    build_e07_v2_candidate_specs,
    materialize_e07_specs,
)
from optim.post_training import _calc_metrics
from optim.train import tracking_loss
from sim_loop import run_simulation


def _history_loss(history, ref_speed=None):
    tensor_history = [
        {k: torch.tensor(v, dtype=torch.float32) for k, v in h.items()}
        for h in history
    ]
    return tracking_loss(
        tensor_history, ref_speed=ref_speed,
        w_lat=10.0, w_head=8.0, w_speed=3.0,
        w_steer_rate=0.05, w_acc_rate=0.01).item()


def _sim_reasons(loss: float, metrics: dict, max_loss: float,
                 max_lat_rmse: float, max_head_rmse: float,
                 max_lat_max: float) -> list[str]:
    reasons = []
    if not np.isfinite(loss):
        reasons.append('sim_nonfinite_loss')
    if loss > max_loss:
        reasons.append('sim_loss')
    if metrics['lat_rmse'] > max_lat_rmse:
        reasons.append('sim_lat_rmse')
    if metrics['head_rmse'] > max_head_rmse:
        reasons.append('sim_head_rmse')
    if metrics['lat_max'] > max_lat_max:
        reasons.append('sim_lat_max')
    return reasons


def run_dc_sim_gate(specs: list[dict], cfg: dict, max_loss: float = 80.0,
                    max_lat_rmse: float = 1.5,
                    max_head_rmse: float = 0.5,
                    max_lat_max: float = 5.0):
    accepted = []
    rejected = []
    results = []
    histories = []

    for spec in specs:
        key, traj = materialize_e07_specs([spec])[0]
        lat_ctrl = LatControllerTruck(cfg, differentiable=False)
        lon_ctrl = LonController(cfg, differentiable=False)
        lat_ctrl.reset_state()
        lon_ctrl.reset_state()
        traj_speed = traj[0].v
        history = run_simulation(
            traj, init_speed=traj_speed,
            init_x=traj[0].x, init_y=traj[0].y, init_yaw=traj[0].theta,
            cfg=cfg, lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
            differentiable=False, tbptt_k=0)
        loss = _history_loss(history, ref_speed=None)
        metrics = _calc_metrics(history)
        reasons = _sim_reasons(
            loss, metrics, max_loss=max_loss,
            max_lat_rmse=max_lat_rmse, max_head_rmse=max_head_rmse,
            max_lat_max=max_lat_max)
        is_core = spec.get('gate_role') == 'core'
        status = 'accepted' if not reasons else 'rejected'
        if is_core and reasons:
            status = 'accepted_core_override'
        results.append({
            'key': key,
            'trajectory_type': spec['type'],
            'gate_role': spec.get('gate_role'),
            'gate_status': status,
            'reasons': reasons,
            'sim_loss': float(loss),
            'sim_metrics': metrics,
        })
        histories.append((key, traj, history, metrics))
        if reasons and not is_core:
            rejected.append(spec)
        else:
            accepted.append(spec)
    return accepted, rejected, results, histories


def _select_preview_histories(histories, max_plots: int):
    if max_plots is None or len(histories) <= max_plots:
        return list(histories)

    selected = []
    selected_keys = set()
    priority_prefixes = [
        'uturn_',
        'smooth_high_curv_',
        'intersection_turn_',
        'straight_accel_decel_',
        'stop_go_',
        'lane_change_',
        'double_lc_',
        'clothoid_left_',
        'clothoid_right_',
        's_curve_',
        'combined_decel_',
        'lc_accel_',
        'clothoid_decel_',
    ]

    for prefix in priority_prefixes:
        for item in histories:
            key = item[0]
            if key.startswith(prefix) and key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)
                break
        if len(selected) >= max_plots:
            return selected

    for item in histories:
        key = item[0]
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= max_plots:
            break
    return selected


def plot_dc_trajectory_preview(histories, output_dir: str,
                               max_plots: int = 24) -> str | None:
    if not histories:
        return None
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = _select_preview_histories(histories, max_plots)
    ncols = 3
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (key, traj, history, metrics) in zip(axes, rows):
        ax.plot([p.x for p in traj], [p.y for p in traj], 'k--',
                linewidth=1.0, label='ref')
        ax.plot([h['x'] for h in history], [h['y'] for h in history],
                'b-', linewidth=1.0, label='dc')
        ax.set_title(f"{key}\nlat={metrics['lat_rmse']:.3f}m")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    for ax in axes[len(rows):]:
        ax.axis('off')
    axes[0].legend(loc='best')
    fig.tight_layout()
    path = os.path.join(output_dir, 'e07_v2_dc_trajectory_preview.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='Generate gated E07 v2 dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir',
                        default=os.path.join('results', 'e07_trajectories',
                                             'e07_v2_gate'))
    parser.add_argument('--max-scenarios', type=int, default=None)
    parser.add_argument('--skip-sim-gate', action='store_true')
    parser.add_argument('--plant', default='truck_trailer')
    parser.add_argument('--config', default=None)
    parser.add_argument('--disable-mlp', action='store_true', default=True)
    parser.add_argument('--enable-mlp', action='store_false', dest='disable_mlp')
    parser.add_argument('--max-loss', type=float, default=80.0)
    parser.add_argument('--max-lat-rmse', type=float, default=1.5)
    parser.add_argument('--max-head-rmse', type=float, default=0.5)
    parser.add_argument('--max-lat-max', type=float, default=5.0)
    parser.add_argument('--plot-max-scenarios', type=int, default=24)
    args = parser.parse_args()

    specs = build_e07_v2_candidate_specs(seed=args.seed)
    if args.max_scenarios is not None:
        specs = specs[:args.max_scenarios]

    geom_accepted, geom_rejected, geom_results = gate_specs_by_geometry(specs)
    accepted = list(geom_accepted)
    rejected = list(geom_rejected)
    results = list(geom_results)
    preview_path = None

    sim_summary = {'enabled': False}
    if not args.skip_sim_gate:
        cfg = load_config(args.config)
        if args.plant:
            apply_plant_override(cfg, args.plant)
        apply_runtime_overrides(cfg, disable_mlp=args.disable_mlp)
        sim_accepted, sim_rejected, sim_results, histories = run_dc_sim_gate(
            geom_accepted, cfg, max_loss=args.max_loss,
            max_lat_rmse=args.max_lat_rmse,
            max_head_rmse=args.max_head_rmse,
            max_lat_max=args.max_lat_max)
        accepted = sim_accepted
        rejected = geom_rejected + sim_rejected
        results = geom_results + sim_results
        preview_path = plot_dc_trajectory_preview(
            histories, args.output_dir, max_plots=args.plot_max_scenarios)
        sim_summary = {
            'enabled': True,
            'plant': args.plant,
            'disable_mlp': bool(args.disable_mlp),
            'preview_plot': preview_path,
            'thresholds': {
                'max_loss': args.max_loss,
                'max_lat_rmse': args.max_lat_rmse,
                'max_head_rmse': args.max_head_rmse,
                'max_lat_max': args.max_lat_max,
            },
        }

    paths = write_gate_outputs(
        accepted=accepted,
        rejected=rejected,
        results=results,
        output_dir=args.output_dir,
        preset='e07_v2_geometry_gate',
        seed=args.seed,
        extra_summary={
            'candidate_count': len(specs),
            'geometry_gate': {
                'accepted_count': len(geom_accepted),
                'rejected_count': len(geom_rejected),
            },
            'sim_gate': sim_summary,
        },
    )

    print(yaml.safe_dump({
        'candidate_count': len(specs),
        'accepted_count': len(accepted),
        'rejected_count': len(rejected),
        **paths,
        'preview_plot': preview_path,
    }, sort_keys=False, allow_unicode=True))


if __name__ == '__main__':
    main()
