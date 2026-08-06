"""Rolling-preview RL evaluation.

The standard RL evaluator applies one action to a whole trajectory.  This
script evaluates a long trajectory by predicting a new RL action for fixed
preview windows and applying the scheduled actions inside one continuous
closed-loop simulation.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC

from config import apply_plant_override, load_config
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from optim.rl_env import RLTuningEnv, extract_geometric_features
from optim.rl_eval_sources import (
    apply_window_policy,
    build_manifest_scenarios,
    build_real_csv_manifest_scenarios,
    build_real_csv_scenario,
)
from optim.train import tracking_loss
from sim_loop import run_simulation


def _baseline_params(env) -> np.ndarray:
    return np.array([
        1.0, 1.0, 1.0, 1.0,
        env._baseline_station_kp,
        env._baseline_station_ki,
        env._baseline_low_speed_kp,
        env._baseline_low_speed_ki,
        env._baseline_high_speed_kp,
        env._baseline_high_speed_ki,
        env._baseline_switch_speed,
    ], dtype=np.float32)


def _trajectory_distances(trajectory) -> np.ndarray:
    distances = [0.0]
    for previous, current in zip(trajectory, trajectory[1:]):
        distances.append(distances[-1] + float(np.hypot(
            current.x - previous.x, current.y - previous.y)))
    return np.asarray(distances, dtype=float)


def build_preview_windows(trajectory, horizon_sec: float | None = None,
                          stride_sec: float | None = None,
                          horizon_m: float | None = None,
                          stride_m: float | None = None,
                          tail_policy: str = 'hold_last') -> list[dict]:
    if len(trajectory) < 2:
        raise ValueError('trajectory must contain at least 2 points')
    if tail_policy not in {'hold_last', 'dc_fallback'}:
        raise ValueError(f'unknown tail_policy: {tail_policy}')

    distance_mode = horizon_m is not None or stride_m is not None
    time_mode = horizon_sec is not None or stride_sec is not None
    if distance_mode == time_mode:
        raise ValueError('choose exactly one of distance or time horizon mode')

    windows = []
    tail_start = None
    distances = _trajectory_distances(trajectory)
    if distance_mode:
        if horizon_m is None or horizon_m <= 0.0:
            raise ValueError('horizon_m must be positive')
        if stride_m is None or stride_m <= 0.0:
            raise ValueError('stride_m must be positive')
        start_targets = np.arange(0.0, distances[-1] + 1e-9, stride_m)
        starts = np.unique(np.searchsorted(
            distances, start_targets, side='left')).tolist()
        for start in starts:
            remaining_m = distances[-1] - distances[start]
            if remaining_m + 1e-9 < horizon_m:
                tail_start = int(start)
                break
            end = int(np.searchsorted(
                distances, distances[start] + horizon_m, side='left')) + 1
            end = min(end, len(trajectory))
            if end - start < 2:
                tail_start = int(start)
                break
            segment = trajectory[start:end]
            windows.append({
                'start_step': int(start),
                'end_step': int(end),
                't_start': float(segment[0].t),
                't_end': float(segment[-1].t),
                'window_length_m': float(distances[end - 1] - distances[start]),
                'features': extract_geometric_features(segment),
            })
    else:
        if horizon_sec is None or horizon_sec <= 0.0:
            raise ValueError('horizon_sec must be positive')
        if stride_sec is None or stride_sec <= 0.0:
            raise ValueError('stride_sec must be positive')
        dt = max(trajectory[1].t - trajectory[0].t, 1e-6)
        horizon_steps = max(2, int(round(horizon_sec / dt)))
        stride_steps = max(1, int(round(stride_sec / dt)))
        for start in range(0, len(trajectory), stride_steps):
            end = start + horizon_steps
            if end > len(trajectory):
                tail_start = start
                break
            segment = trajectory[start:end]
            windows.append({
                'start_step': start,
                'end_step': end,
                't_start': float(segment[0].t),
                't_end': float(segment[-1].t),
                'window_length_m': float(
                    distances[end - 1] - distances[start]),
                'features': extract_geometric_features(segment),
            })

    if tail_policy == 'dc_fallback' and tail_start is not None:
        segment = trajectory[tail_start:]
        windows.append({
            'start_step': int(tail_start),
            'end_step': len(trajectory),
            't_start': float(segment[0].t),
            't_end': float(segment[-1].t),
            'window_length_m': float(distances[-1] - distances[tail_start]),
            'features': None,
            'fallback_reason': 'short_tail',
        })
    return windows


def predict_rolling_schedule(model, env, trajectory,
                             horizon_sec: float | None = None,
                             stride_sec: float | None = None,
                             horizon_m: float | None = None,
                             stride_m: float | None = None,
                             tail_policy: str = 'hold_last',
                             ood_fallback: bool = False,
                             ema_alpha: float | None = None,
                             action_rate_limit_frac: float | None = None,
                             ) -> list[dict]:
    if ema_alpha is not None and not 0.0 < ema_alpha <= 1.0:
        raise ValueError('ema_alpha must be in (0, 1]')
    if action_rate_limit_frac is not None and action_rate_limit_frac <= 0.0:
        raise ValueError('action_rate_limit_frac must be positive')

    schedule = []
    previous_action = np.zeros_like(env.ACTION_BOUNDS, dtype=np.float32)
    windows = build_preview_windows(
        trajectory, horizon_sec=horizon_sec, stride_sec=stride_sec,
        horizon_m=horizon_m, stride_m=stride_m, tail_policy=tail_policy)
    for window in windows:
        fallback_reason = window.get('fallback_reason')
        if fallback_reason:
            action = np.zeros_like(env.ACTION_BOUNDS, dtype=np.float32)
            schedule.append({
                **window,
                'action': action,
                'is_ood': False,
                'used_policy': False,
                'action_saturation_ratio': 0.0,
            })
            previous_action = action
            continue

        obs = np.concatenate([window['features'], _baseline_params(env)])
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = np.asarray(raw_action, dtype=np.float32)
        is_ood = bool(env.is_ood(window['features']))
        if ood_fallback and is_ood:
            action = np.zeros_like(env.ACTION_BOUNDS, dtype=np.float32)
            fallback_reason = 'ood'
            used_policy = False
        else:
            action = np.clip(raw_action, -env.ACTION_BOUNDS, env.ACTION_BOUNDS)
            if ema_alpha is not None:
                action = ema_alpha * action + (1.0 - ema_alpha) * previous_action
            if action_rate_limit_frac is not None:
                max_delta = action_rate_limit_frac * env.ACTION_BOUNDS
                action = previous_action + np.clip(
                    action - previous_action, -max_delta, max_delta)
            action = np.clip(action, -env.ACTION_BOUNDS, env.ACTION_BOUNDS)
            used_policy = True
        schedule.append({
            **window,
            'action': np.asarray(action, dtype=np.float32),
            'is_ood': is_ood,
            'used_policy': used_policy,
            'fallback_reason': fallback_reason,
            'action_saturation_ratio': float(np.mean(
                np.abs(raw_action) >= env.ACTION_BOUNDS - 1e-6)),
        })
        previous_action = np.asarray(action, dtype=np.float32)
    return schedule


def make_step_callback(env, schedule: list[dict]):
    next_idx = {'value': 0}

    def callback(step, t, lat_ctrl, lon_ctrl, cfg):
        idx = next_idx['value']
        if idx < len(schedule) and step >= schedule[idx]['start_step']:
            env._apply_action(schedule[idx]['action'])
            next_idx['value'] = idx + 1

    return callback


def _tracking_loss(history, ref_speed: float | None = None) -> float:
    tensor_history = [
        {k: torch.tensor(v, dtype=torch.float32) for k, v in h.items()}
        for h in history
    ]
    return tracking_loss(
        tensor_history, ref_speed=ref_speed,
        w_lat=10.0, w_head=8.0, w_speed=3.0,
        w_steer_rate=0.05, w_acc_rate=0.01).item()


def _to_plain(value):
    if isinstance(value, np.ndarray):
        return [_to_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _save_rolling_results(results, output_dir, run_spec=None):
    os.makedirs(output_dir, exist_ok=True)
    rolling_losses = [r['rolling_rl_loss'] for r in results]
    dc_losses = [r['dc_loss'] for r in results]
    payload = {
        'summary': {
            'loss_reference_mode': 'per_step_trajectory_v',
            'trajectory_count': len(results),
            'rolling_rl_avg_loss': float(np.mean(rolling_losses)),
            'dc_avg_loss': float(np.mean(dc_losses)),
            'win_count': int(sum(
                1 for r in results if r['rolling_rl_loss'] < r['dc_loss'])),
        },
        'run_spec': _to_plain(run_spec or {}),
        'results': [_to_plain(r) for r in results],
    }
    path = os.path.join(output_dir, 'rl_rolling_eval_results.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return path


def evaluate_rolling_model(model_path: str, plant: str, dc_config_path: str,
                           trajectory_manifest: str | None = None,
                           real_csv: str | None = None,
                           real_csv_manifest: str | None = None,
                           real_role: str | None = None,
                           output_dir: str | None = None,
                           horizon_sec: float | None = None,
                           stride_sec: float | None = None,
                           horizon_m: float | None = 120.0,
                           stride_m: float | None = 20.0,
                           tail_policy: str = 'dc_fallback',
                           ood_fallback: bool = False,
                           ema_alpha: float | None = None,
                           action_rate_limit_frac: float | None = None,
                           max_scenarios: int | None = None,
                           eval_window_policy: str = 'all') -> list[dict]:
    sources = [trajectory_manifest, real_csv, real_csv_manifest]
    if sum(value is not None for value in sources) > 1:
        raise ValueError('choose at most one external trajectory source')
    if not any(sources):
        raise ValueError('rolling evaluation requires an external trajectory source')

    model = SAC.load(model_path)
    env = RLTuningEnv(plant=plant, config_path=dc_config_path,
                      compute_baseline_losses=False)
    dc_cfg = load_config(dc_config_path)
    if plant:
        apply_plant_override(dc_cfg, plant)
    dc_lat = LatControllerTruck(dc_cfg, differentiable=False)
    dc_lon = LonController(dc_cfg, differentiable=False)

    if trajectory_manifest:
        scenarios = build_manifest_scenarios(
            trajectory_manifest, max_scenarios=max_scenarios)
    elif real_csv:
        scenarios = [build_real_csv_scenario(real_csv, max_steps=max_scenarios)]
    else:
        scenarios = build_real_csv_manifest_scenarios(
            real_csv_manifest, role=real_role,
            max_scenarios=max_scenarios)
    scenarios = apply_window_policy(scenarios, eval_window_policy)

    results = []
    for scenario in scenarios:
        traj = scenario.make_trajectory()
        ref_speed = traj[0].v
        schedule = predict_rolling_schedule(
            model, env, traj, horizon_sec=horizon_sec,
            stride_sec=stride_sec, horizon_m=horizon_m,
            stride_m=stride_m, tail_policy=tail_policy,
            ood_fallback=ood_fallback, ema_alpha=ema_alpha,
            action_rate_limit_frac=action_rate_limit_frac)

        env._restore_baseline_params()
        env.lat_ctrl.reset_state()
        env.lon_ctrl.reset_state()
        rolling_history = run_simulation(
            traj, init_speed=ref_speed,
            init_x=traj[0].x, init_y=traj[0].y, init_yaw=traj[0].theta,
            cfg=env.cfg, lat_ctrl=env.lat_ctrl, lon_ctrl=env.lon_ctrl,
            differentiable=False, tbptt_k=0,
            step_callback=make_step_callback(env, schedule))
        rolling_loss = _tracking_loss(rolling_history, ref_speed=None)
        rolling_legacy_loss = _tracking_loss(rolling_history, ref_speed)
        env._restore_baseline_params()

        dc_lat.reset_state()
        dc_lon.reset_state()
        dc_history = run_simulation(
            traj, init_speed=ref_speed,
            init_x=traj[0].x, init_y=traj[0].y, init_yaw=traj[0].theta,
            cfg=dc_cfg, lat_ctrl=dc_lat, lon_ctrl=dc_lon,
            differentiable=False, tbptt_k=0)
        dc_loss = _tracking_loss(dc_history, ref_speed=None)
        dc_legacy_loss = _tracking_loss(dc_history, ref_speed)

        policy_windows = [item for item in schedule if item['used_policy']]
        fallback_windows = [
            item for item in schedule if item.get('fallback_reason')]

        result = {
            'key': scenario.key,
            'rolling_rl_loss': rolling_loss,
            'dc_loss': dc_loss,
            'rolling_rl_legacy_scalar_loss': rolling_legacy_loss,
            'dc_legacy_scalar_loss': dc_legacy_loss,
            'loss_reference_mode': 'per_step_trajectory_v',
            'delta_pct': (rolling_loss - dc_loss) / max(dc_loss, 1e-8) * 100,
            'source_metadata': scenario.metadata,
            'window_count': len(schedule),
            'policy_window_count': len(policy_windows),
            'ood_window_count': int(sum(item['is_ood'] for item in schedule)),
            'fallback_count': len(fallback_windows),
            'mean_action_saturation_ratio': float(np.mean([
                item['action_saturation_ratio'] for item in policy_windows
            ])) if policy_windows else 0.0,
            'schedule': [{
                'start_step': item['start_step'],
                'end_step': item['end_step'],
                't_start': item['t_start'],
                't_end': item['t_end'],
                'window_length_m': item['window_length_m'],
                'is_ood': item['is_ood'],
                'used_policy': item['used_policy'],
                'fallback_reason': item.get('fallback_reason'),
                'action_saturation_ratio': item['action_saturation_ratio'],
                'action': item['action'],
            } for item in schedule],
        }
        results.append(result)
        print(f"  {scenario.key:30s}  rolling_RL={rolling_loss:.4f}  "
              f"DC={dc_loss:.4f}  windows={len(policy_windows)}  "
              f"fallbacks={len(fallback_windows)}")

    if output_dir:
        run_spec = {
            'model_path': os.path.abspath(model_path),
            'dc_config_path': os.path.abspath(dc_config_path),
            'plant': plant,
            'trajectory_manifest': (
                os.path.abspath(trajectory_manifest)
                if trajectory_manifest else None),
            'real_csv': os.path.abspath(real_csv) if real_csv else None,
            'real_csv_manifest': (
                os.path.abspath(real_csv_manifest)
                if real_csv_manifest else None),
            'real_role': real_role,
            'horizon_sec': horizon_sec,
            'stride_sec': stride_sec,
            'horizon_m': horizon_m,
            'stride_m': stride_m,
            'tail_policy': tail_policy,
            'ood_fallback': bool(ood_fallback),
            'ema_alpha': ema_alpha,
            'action_rate_limit_frac': action_rate_limit_frac,
            'max_scenarios': max_scenarios,
            'eval_window_policy': eval_window_policy,
        }
        yaml_path = _save_rolling_results(
            results, output_dir, run_spec=run_spec)
        print(f"  structured results: {yaml_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rolling-preview RL eval')
    parser.add_argument('--rl-model', type=str, required=True)
    parser.add_argument('--dc-config', type=str, required=True)
    parser.add_argument('--plant', type=str, default='hybrid_v2')
    parser.add_argument('--trajectory-manifest', type=str, default=None)
    parser.add_argument('--real-csv', type=str, default=None)
    parser.add_argument('--real-csv-manifest', type=str, default=None)
    parser.add_argument('--real-role', choices=['one_shot', 'rolling'],
                        default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--horizon-m', type=float, default=None)
    parser.add_argument('--stride-m', type=float, default=None)
    parser.add_argument('--horizon-sec', type=float, default=None,
                        help='Legacy time horizon; prefer --horizon-m')
    parser.add_argument('--stride-sec', type=float, default=None)
    parser.add_argument('--tail-policy', choices=['hold_last', 'dc_fallback'],
                        default='dc_fallback')
    parser.add_argument('--ood-fallback', action='store_true')
    parser.add_argument('--ema-alpha', type=float, default=None)
    parser.add_argument('--action-rate-limit-frac', type=float, default=None)
    parser.add_argument('--max-scenarios', type=int, default=None)
    parser.add_argument('--eval-window-policy', type=str, default='all',
                        choices=['all', 'id_only', 'ood_only'])
    args = parser.parse_args()
    sources = (args.trajectory_manifest, args.real_csv,
               args.real_csv_manifest)
    if sum(value is not None for value in sources) > 1:
        parser.error('choose at most one external trajectory source')
    if not any(sources):
        parser.error('rolling evaluation requires an external trajectory source')
    distance_mode = args.horizon_m is not None or args.stride_m is not None
    time_mode = args.horizon_sec is not None or args.stride_sec is not None
    if distance_mode and time_mode:
        parser.error('choose either distance or time horizon arguments')
    if not distance_mode and not time_mode:
        args.horizon_m = 120.0
        args.stride_m = 20.0
    elif distance_mode and (args.horizon_m is None or args.stride_m is None):
        parser.error('--horizon-m and --stride-m must be provided together')
    elif time_mode and (args.horizon_sec is None or args.stride_sec is None):
        parser.error('--horizon-sec and --stride-sec must be provided together')

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.rl_model),
                                  'rolling_evaluation')
    evaluate_rolling_model(
        model_path=args.rl_model,
        plant=args.plant,
        dc_config_path=args.dc_config,
        trajectory_manifest=args.trajectory_manifest,
        real_csv=args.real_csv,
        real_csv_manifest=args.real_csv_manifest,
        real_role=args.real_role,
        output_dir=output_dir,
        horizon_sec=args.horizon_sec,
        stride_sec=args.stride_sec,
        horizon_m=args.horizon_m,
        stride_m=args.stride_m,
        tail_policy=args.tail_policy,
        ood_fallback=args.ood_fallback,
        ema_alpha=args.ema_alpha,
        action_rate_limit_frac=args.action_rate_limit_frac,
        max_scenarios=args.max_scenarios,
        eval_window_policy=args.eval_window_policy,
    )
