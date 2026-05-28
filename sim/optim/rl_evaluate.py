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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC

from config import load_config, apply_plant_override
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from model.trajectory import (expand_trajectories, TRAJECTORY_TYPES,
                              SPEED_BANDS_KPH)
from sim_loop import run_simulation
from optim.rl_env import RLTuningEnv, extract_geometric_features
from optim.train import tracking_loss
from optim.post_training import _plot_comparison_grid, _calc_metrics

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_rl_model(model_path, plant, dc_config_path, output_dir=None):
    model = SAC.load(model_path)
    env = RLTuningEnv(plant=plant, config_path=dc_config_path,
                       compute_baseline_losses=False)

    dc_cfg = load_config(dc_config_path)
    if plant:
        apply_plant_override(dc_cfg, plant)
    dc_lat = LatControllerTruck(dc_cfg, differentiable=False)
    dc_lon = LonController(dc_cfg, differentiable=False)

    results = []
    all_base = []
    all_tuned = []
    for key in env._traj_keys_list:
        traj = env._traj_cache[key]
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
            dc_tensor_history, ref_speed=traj_speed,
            w_lat=10.0, w_head=8.0, w_speed=3.0,
            w_steer_rate=0.05, w_acc_rate=0.01).item()

        dc_metrics = _calc_metrics(dc_history)
        rl_metrics = _calc_metrics(rl_history)
        all_base.append((key, key, traj, dc_history, dc_metrics, traj_speed))
        all_tuned.append((key, key, traj, rl_history, rl_metrics, traj_speed))

        is_ood = env.is_ood(features)
        ood_label = ' [!OOD]' if is_ood else ''

        results.append({
            'key': key,
            'rl_loss': rl_loss,
            'dc_loss': dc_loss,
            'delta_pct': (rl_loss - dc_loss) / max(dc_loss, 1e-8) * 100,
            'is_ood': is_ood,
            'rl_action': rl_action.copy(),
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

    return results


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
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    args = parser.parse_args()

    output_dir = args.output
    if output_dir is None:
        rl_dir = os.path.dirname(args.rl_model)
        output_dir = os.path.join(rl_dir, 'evaluation')

    evaluate_rl_model(
        model_path=args.rl_model,
        plant=args.plant,
        dc_config_path=args.dc_config,
        output_dir=output_dir,
    )
