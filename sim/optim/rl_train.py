"""SAC 强化学习参数整定训练。

基于 stable-baselines3 SAC 算法，在 Contextual Bandit 环境中训练 Agent
学习 轨迹几何特征 → 控制器参数 的映射。

用法:
    python optim/rl_train.py --plant hybrid_v2 --config configs/tuned/xxx.yaml --total-timesteps 50000
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList)

from config import load_config, save_tuned_config, apply_plant_override
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from optim.rl_env import RLTuningEnv


def load_baseline_params(cfg: dict):
    lc = LatControllerTruck(cfg, differentiable=False)
    ln = LonController(cfg, differentiable=False)
    return {
        'T2_y': lc.T2_y.data.clone(),
        'T3_y': lc.T3_y.data.clone(),
        'T4_y': lc.T4_y.data.clone(),
        'T6_y': lc.T6_y.data.clone(),
        'station_kp': ln.station_kp.item(),
        'station_ki': ln.station_ki.item(),
        'low_speed_kp': ln.low_speed_kp.item(),
        'low_speed_ki': ln.low_speed_ki.item(),
        'high_speed_kp': ln.high_speed_kp.item(),
        'high_speed_ki': ln.high_speed_ki.item(),
        'switch_speed': ln.switch_speed.item(),
    }


def export_tuned_yaml(env: RLTuningEnv, action: "np.ndarray", output_dir: str) -> str:
    """将 RL 最优参数导出为 YAML（与 DC tuned 格式一致）。"""
    import numpy as np
    import torch

    action = np.clip(action, -env.ACTION_BOUNDS, env.ACTION_BOUNDS)

    from optim.train import DiffControllerParams
    params = DiffControllerParams(cfg=env.cfg)

    params.lat_ctrl.T2_y.data = env._baseline_T2_y * (1.0 + float(action[0]))
    params.lat_ctrl.T3_y.data = env._baseline_T3_y * (1.0 + float(action[1]))
    params.lat_ctrl.T4_y.data = env._baseline_T4_y * (1.0 + float(action[2]))
    params.lat_ctrl.T6_y.data = env._baseline_T6_y * (1.0 + float(action[3]))

    params.lon_ctrl.station_kp.data = torch.tensor(
        max(0.0, env._baseline_station_kp + float(action[4])))
    params.lon_ctrl.station_ki.data = torch.tensor(
        max(0.0, env._baseline_station_ki + float(action[5])))
    params.lon_ctrl.low_speed_kp.data = torch.tensor(
        max(0.0, env._baseline_low_speed_kp + float(action[6])))
    params.lon_ctrl.low_speed_ki.data = torch.tensor(
        max(0.0, env._baseline_low_speed_ki + float(action[7])))
    params.lon_ctrl.high_speed_kp.data = torch.tensor(
        max(0.0, env._baseline_high_speed_kp + float(action[8])))
    params.lon_ctrl.high_speed_ki.data = torch.tensor(
        max(0.0, env._baseline_high_speed_ki + float(action[9])))
    sw = max(0.5, min(10.0, env._baseline_switch_speed + float(action[10])))
    params.lon_ctrl.switch_speed.data = torch.tensor(sw)

    cfg_out = params.to_config_dict()
    return save_tuned_config(cfg_out, output_dir=output_dir)


def train_rl(plant='hybrid_v2', config_path=None, total_timesteps=50000,
             lr=3e-4, buffer_size=100000, batch_size=256, seed=42,
             trajectories=None, verbose=True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.path.dirname(__file__), '..',
                              'results', 'rl', plant, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    env = RLTuningEnv(plant=plant, config_path=config_path, seed=seed,
                      trajectory_types=trajectories)
    eval_env = RLTuningEnv(plant=plant, config_path=config_path, seed=seed + 1000,
                           trajectory_types=trajectories)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=output_dir,
        name_prefix='sac_model')

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=output_dir,
        log_path=output_dir, eval_freq=5000,
        deterministic=True, render=False)

    model = SAC(
        "MlpPolicy", env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        verbose=1 if verbose else 0,
        seed=seed,
        tensorboard_log=output_dir,
    )

    import time as _time
    t_start = _time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_callback, eval_callback]),
        log_interval=100,
    )

    elapsed = _time.time() - t_start
    print(f"\n训练耗时: {elapsed/60:.1f} 分钟 ({elapsed:.0f} 秒)")

    model_path = os.path.join(output_dir, 'sac_model_final')
    model.save(model_path)

    best_obs = env._build_obs()
    best_action, _ = model.predict(best_obs, deterministic=True)
    yaml_path = export_tuned_yaml(env, best_action, output_dir)

    if verbose:
        print(f"\n训练完成!")
        print(f"  SAC 模型: {model_path}")
        print(f"  最优参数 YAML: {yaml_path}")
        print(f"  输出目录: {output_dir}")

    return {
        'model': model,
        'model_path': model_path,
        'yaml_path': yaml_path,
        'output_dir': output_dir,
        'elapsed_seconds': elapsed,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC RL 参数整定训练')
    parser.add_argument('--plant', type=str, default='hybrid_v2',
                        choices=['kinematic', 'dynamic', 'hybrid_dynamic', 'hybrid_v2', 'truck_trailer'],
                        help='被控对象类型')
    parser.add_argument('--config', type=str, default=None,
                        help='DC baseline 配置路径（warm-start）')
    parser.add_argument('--total-timesteps', type=int, default=50000,
                        help='SAC 总训练步数')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='SAC 学习率')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='经验回放缓冲区大小')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='SAC batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--trajectories', nargs='+', default=None,
                        help='轨迹类型名，默认全量 48 条')
    args = parser.parse_args()

    train_rl(
        plant=args.plant,
        config_path=args.config,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        seed=args.seed,
        trajectories=args.trajectories,
    )
