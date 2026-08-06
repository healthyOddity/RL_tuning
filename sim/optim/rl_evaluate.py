"""RL vs DC baseline 对比评估。

在 48 条标准轨迹上对比 RL 推理参数 vs DC baseline 参数的跟踪性能。
生成对比图表：分轨迹 loss 柱状图、参数差异热力图、总体统计摘要。

用法:
    python optim/rl_evaluate.py --rl-model results/rl/hybrid_v2/xxx/sac_model_final.zip \
                                --dc-config configs/tuned/xxx.yaml --plant hybrid_v2
"""
import argparse
import os
import sys
import torch
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC

from config import load_config, apply_plant_override, apply_runtime_overrides
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from sim_loop import run_simulation
from optim.rl_env import RLTuningEnv, extract_geometric_features
from optim.train import tracking_loss
from optim.post_training import _plot_comparison_grid, _calc_metrics
from optim.rl_eval_sources import (
    apply_window_policy,
    build_manifest_scenarios,
    build_real_csv_manifest_scenarios,
    build_real_csv_scenario,
    build_standard_scenarios,
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _build_rl_eval_scenarios(trajectory_types=None, include_park_route=False):
    return [
        (scenario.key, scenario.label, scenario.make_trajectory)
        for scenario in build_standard_scenarios(
            trajectory_types, include_park_route=include_park_route)
    ]


def _select_eval_scenarios(trajectory_types=None, include_park_route=False,
                           trajectory_manifest=None, real_csv=None,
                           real_csv_manifest=None, real_role=None,
                           max_scenarios=None, eval_window_policy='all'):
    sources = [trajectory_manifest, real_csv, real_csv_manifest]
    if sum(value is not None for value in sources) > 1:
        raise ValueError('choose at most one external trajectory source')
    if trajectory_manifest:
        scenarios = build_manifest_scenarios(
            trajectory_manifest, max_scenarios=max_scenarios)
    elif real_csv:
        scenarios = [build_real_csv_scenario(real_csv, max_steps=max_scenarios)]
    elif real_csv_manifest:
        scenarios = build_real_csv_manifest_scenarios(
            real_csv_manifest, role=real_role,
            max_scenarios=max_scenarios)
    else:
        scenarios = build_standard_scenarios(
            trajectory_types, include_park_route=include_park_route)
        if max_scenarios is not None:
            scenarios = scenarios[:max_scenarios]
    return apply_window_policy(scenarios, eval_window_policy)


def _apply_eval_runtime_overrides(cfg, disable_mlp=False):
    if disable_mlp:
        apply_runtime_overrides(cfg, disable_mlp=True)


def evaluate_rl_model(model_path, plant, dc_config_path, output_dir=None,
                      trajectory_types=None, include_park_route=False,
                      trajectory_manifest=None, real_csv=None,
                      real_csv_manifest=None, real_role=None,
                      max_scenarios=None, eval_window_policy='all',
                      disable_mlp=False):
    model = SAC.load(model_path)
    env = RLTuningEnv(plant=plant, config_path=dc_config_path,
                       compute_baseline_losses=False)
    _apply_eval_runtime_overrides(env.cfg, disable_mlp=disable_mlp)

    dc_cfg = load_config(dc_config_path)
    if plant:
        apply_plant_override(dc_cfg, plant)
    _apply_eval_runtime_overrides(dc_cfg, disable_mlp=disable_mlp)
    dc_lat = LatControllerTruck(dc_cfg, differentiable=False)
    dc_lon = LonController(dc_cfg, differentiable=False)

    results = []
    all_base = []
    all_tuned = []
    eval_scenarios = _select_eval_scenarios(
        trajectory_types=trajectory_types,
        include_park_route=include_park_route,
        trajectory_manifest=trajectory_manifest,
        real_csv=real_csv,
        real_csv_manifest=real_csv_manifest,
        real_role=real_role,
        max_scenarios=max_scenarios,
        eval_window_policy=eval_window_policy,
    )
    for scenario in eval_scenarios:
        key = scenario.key
        label = scenario.label
        traj = scenario.make_trajectory()
        traj_speed = traj[0].v
        features = extract_geometric_features(traj)

        obs_21d = np.concatenate([features, np.array([
            1.0, 1.0, 1.0, 1.0,
            env._baseline_station_kp, env._baseline_station_ki,
            env._baseline_low_speed_kp, env._baseline_low_speed_ki,
            env._baseline_high_speed_kp, env._baseline_high_speed_ki,
            env._baseline_switch_speed,
        ], dtype=np.float32)])
        rl_action, _ = model.predict(obs_21d, deterministic=True)
        rl_action = np.clip(rl_action, -env.ACTION_BOUNDS, env.ACTION_BOUNDS)
        action_saturation_ratio = float(np.mean(np.isclose(
            np.abs(rl_action), env.ACTION_BOUNDS, rtol=0.0, atol=1e-6)))

        env._apply_action(rl_action)
        env.lat_ctrl.reset_state()
        env.lon_ctrl.reset_state()
        rl_history = run_simulation(
            traj, init_speed=traj_speed,
            init_x=traj[0].x, init_y=traj[0].y, init_yaw=traj[0].theta,
            cfg=env.cfg, lat_ctrl=env.lat_ctrl, lon_ctrl=env.lon_ctrl,
            differentiable=False, tbptt_k=0)
        rl_tensor_history = [
            {k: torch.tensor(v, dtype=torch.float32) for k, v in h.items()}
            for h in rl_history
        ]
        rl_loss = tracking_loss(
            rl_tensor_history, ref_speed=None,
            w_lat=10.0, w_head=8.0, w_speed=3.0,
            w_steer_rate=0.05, w_acc_rate=0.01).item()
        rl_legacy_loss = tracking_loss(
            rl_tensor_history, ref_speed=traj_speed,
            w_lat=10.0, w_head=8.0, w_speed=3.0,
            w_steer_rate=0.05, w_acc_rate=0.01).item()
        env._restore_baseline_params()

        dc_lat.reset_state()
        dc_lon.reset_state()
        dc_history = run_simulation(
            traj, init_speed=traj_speed,
            init_x=traj[0].x, init_y=traj[0].y, init_yaw=traj[0].theta,
            cfg=dc_cfg, lat_ctrl=dc_lat, lon_ctrl=dc_lon,
            differentiable=False, tbptt_k=0)
        dc_tensor_history = [
            {k: torch.tensor(v, dtype=torch.float32) for k, v in h.items()}
            for h in dc_history
        ]
        dc_loss = tracking_loss(
            dc_tensor_history, ref_speed=None,
            w_lat=10.0, w_head=8.0, w_speed=3.0,
            w_steer_rate=0.05, w_acc_rate=0.01).item()
        dc_legacy_loss = tracking_loss(
            dc_tensor_history, ref_speed=traj_speed,
            w_lat=10.0, w_head=8.0, w_speed=3.0,
            w_steer_rate=0.05, w_acc_rate=0.01).item()

        dc_metrics = _calc_metrics(dc_history)
        rl_metrics = _calc_metrics(rl_history)
        all_base.append((key, label, traj, dc_history, dc_metrics, traj_speed))
        all_tuned.append((key, label, traj, rl_history, rl_metrics, traj_speed))

        is_ood = env.is_ood(features)
        ood_label = ' [!OOD]' if is_ood else ''

        results.append({
            'key': key,
            'rl_loss': rl_loss,
            'dc_loss': dc_loss,
            'rl_legacy_scalar_loss': rl_legacy_loss,
            'dc_legacy_scalar_loss': dc_legacy_loss,
            'loss_reference_mode': 'per_step_trajectory_v',
            'delta_pct': (rl_loss - dc_loss) / max(dc_loss, 1e-8) * 100,
            'is_ood': is_ood,
            'source_metadata': scenario.metadata,
            'rl_action': rl_action.copy(),
            'action_saturation_ratio': action_saturation_ratio,
            'rl_metrics': rl_metrics,
            'dc_metrics': dc_metrics,
        })
        print(f"  {key:30s}  RL={rl_loss:.4f}  DC={dc_loss:.4f}  "
              f"Δ={rl_loss-dc_loss:+.4f} ({(rl_loss-dc_loss)/max(dc_loss,1e-8)*100:+.1f}%){ood_label}")

    rl_losses = [r['rl_loss'] for r in results]
    dc_losses = [r['dc_loss'] for r in results]
    rl_mean = np.mean(rl_losses)
    dc_mean = np.mean(dc_losses)
    win_count = sum(1 for r in results if r['rl_loss'] < r['dc_loss'])
    ood_count = sum(1 for r in results if r['is_ood'])

    print(f"\n{'='*60}")
    print(f"总体统计:")
    print(f"  RL  avg loss: {rl_mean:.4f}")
    print(f"  DC  avg loss: {dc_mean:.4f}")
    print(f"  Δ: {rl_mean - dc_mean:+.4f} ({(rl_mean-dc_mean)/max(dc_mean,1e-8)*100:+.1f}%)")
    print(f"  RL 优于 DC: {win_count}/{len(results)} 条轨迹")
    if ood_count > 0:
        print(f"  OOD 警告: {ood_count}/{len(results)} 条轨迹")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _plot_comparison(results, rl_mean, dc_mean, win_count, output_dir)
        _plot_comparison_grid(all_base, all_tuned, output_dir,
                              plot_type='trajectory', filename='comparison_trajectory.png')
        _plot_comparison_grid(all_base, all_tuned, output_dir,
                              plot_type='lateral_error', filename='comparison_lateral_error.png')
        _plot_comparison_grid(all_base, all_tuned, output_dir,
                              plot_type='speed_error', filename='comparison_speed_error.png')
        _plot_comparison_grid(all_base, all_tuned, output_dir,
                              plot_type='steer', filename='comparison_steer.png')
        _plot_comparison_grid(all_base, all_tuned, output_dir,
                              plot_type='acc', filename='comparison_acc.png')
        run_spec = {
            'model_path': os.path.abspath(model_path),
            'dc_config_path': os.path.abspath(dc_config_path),
            'plant': plant,
            'disable_mlp': bool(disable_mlp),
            'trajectory_types': trajectory_types,
            'include_park_route': bool(include_park_route),
            'trajectory_manifest': (
                os.path.abspath(trajectory_manifest)
                if trajectory_manifest else None),
            'real_csv': os.path.abspath(real_csv) if real_csv else None,
            'real_csv_manifest': (
                os.path.abspath(real_csv_manifest)
                if real_csv_manifest else None),
            'real_role': real_role,
            'max_scenarios': max_scenarios,
            'eval_window_policy': eval_window_policy,
        }
        yaml_path = _save_eval_results(
            results, output_dir, rl_mean, dc_mean, win_count, ood_count,
            run_spec=run_spec)
        print(f"  结构化结果: {yaml_path}")

    return results


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


def _save_eval_results(results, output_dir, rl_mean, dc_mean, win_count,
                       ood_count, run_spec=None):
    os.makedirs(output_dir, exist_ok=True)
    trajectory_count = len(results)
    payload = {
        'summary': {
            'loss_reference_mode': 'per_step_trajectory_v',
            'trajectory_count': trajectory_count,
            'rl_avg_loss': _to_plain(rl_mean),
            'dc_avg_loss': _to_plain(dc_mean),
            'delta': _to_plain(rl_mean - dc_mean),
            'delta_pct': _to_plain((rl_mean - dc_mean) / max(dc_mean, 1e-8) * 100),
            'win_count': int(win_count),
            'ood_count': int(ood_count),
            'mean_action_saturation_ratio': float(np.mean([
                r.get('action_saturation_ratio', 0.0) for r in results
            ])) if results else 0.0,
        },
        'run_spec': _to_plain(run_spec or {}),
        'results': [_to_plain(r) for r in results],
    }
    path = os.path.join(output_dir, 'rl_eval_results.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return path


def _plot_comparison(results, rl_mean, dc_mean, win_count, output_dir):
    keys = [r['key'] for r in results]
    rl_vals = [r['rl_loss'] for r in results]
    dc_vals = [r['dc_loss'] for r in results]
    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    bars1 = ax.bar(x - width/2, rl_vals, width, label='RL (SAC)', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, dc_vals, width, label='DC (BPTT+Adam)', color='#1f77b4', alpha=0.8)
    ax.set_xlabel('轨迹')
    ax.set_ylabel('Tracking Loss')
    ax.set_title(f'RL vs DC 分轨迹 Loss 对比  (RL avg={rl_mean:.4f}, DC avg={dc_mean:.4f}, RL胜{win_count}/{len(results)})')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'rl_vs_dc_comparison.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL vs DC 对比评估')
    parser.add_argument('--rl-model', type=str, required=True,
                        help='SAC 模型路径 (.zip)')
    parser.add_argument('--dc-config', type=str, required=True,
                        help='DC baseline 配置路径 (YAML)')
    parser.add_argument('--plant', type=str, default='hybrid_v2',
                        help='被控对象类型')
    parser.add_argument('--trajectories', nargs='+', default=None,
                        help='轨迹类型名，默认全量 48 条标准轨迹')
    parser.add_argument('--include-park-route', action='store_true',
                        help='追加 DC V1 验证中的 park_route，评估 48+1 场景')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--disable-mlp', action='store_true',
                        help='Disable truck_trailer MLP residual; use base dynamics only')
    parser.add_argument('--trajectory-manifest', type=str, default=None,
                        help='E07-style trajectory manifest for OOD evaluation')
    parser.add_argument('--real-csv', type=str, default=None,
                        help='Real record CSV used as a record-refline scenario')
    parser.add_argument('--real-csv-manifest', type=str, default=None,
                        help='Quality-gated E09 real record manifest')
    parser.add_argument('--real-role', choices=['one_shot', 'rolling'],
                        default=None)
    parser.add_argument('--max-scenarios', type=int, default=None,
                        help='Limit scenarios for smoke tests')
    parser.add_argument('--eval-window-policy', type=str, default='all',
                        choices=['all', 'id_only', 'ood_only'],
                        help='Filter evaluation scenarios by ID/OOD source')
    args = parser.parse_args()
    if sum(value is not None for value in (
            args.trajectory_manifest, args.real_csv,
            args.real_csv_manifest)) > 1:
        parser.error('choose at most one external trajectory source')

    output_dir = args.output
    if output_dir is None:
        rl_dir = os.path.dirname(args.rl_model)
        output_dir = os.path.join(rl_dir, 'evaluation')

    evaluate_rl_model(
        model_path=args.rl_model,
        plant=args.plant,
        dc_config_path=args.dc_config,
        output_dir=output_dir,
        trajectory_types=args.trajectories,
        include_park_route=args.include_park_route,
        trajectory_manifest=args.trajectory_manifest,
        real_csv=args.real_csv,
        real_csv_manifest=args.real_csv_manifest,
        real_role=args.real_role,
        max_scenarios=args.max_scenarios,
        eval_window_policy=args.eval_window_policy,
        disable_mlp=args.disable_mlp,
    )
