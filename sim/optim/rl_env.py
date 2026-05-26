"""Contextual Bandit RL 环境 — 将 sim_loop 封装为 Gymnasium Env。

每个 episode = 一条随机采样的轨迹。Agent 看到轨迹几何特征后一次性输出整组参数，
运行 non-differentiable 仿真，reward 为负 tracking_loss（与 DC 公式一致）。

用法:
    from optim.rl_env import RLTuningEnv
    env = RLTuningEnv(plant='hybrid_v2', config_path='configs/tuned/xxx.yaml')
"""
import math
import sys
import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config, apply_plant_override
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from model.trajectory import (expand_trajectories, TRAJECTORY_TYPES,
                              SPEED_BANDS_KPH)
from sim_loop import run_simulation


def extract_geometric_features(trajectory) -> np.ndarray:
    """从参考轨迹提取 10 维连续几何特征。

    Args:
        trajectory: 参考轨迹点列表 (TrajectoryPoint: x, y, theta, kappa, v, t, s, a)

    Returns:
        np.ndarray shape (10,) float32
    """
    curvatures = np.array([p.kappa for p in trajectory], dtype=np.float64)
    speeds = np.array([p.v for p in trajectory], dtype=np.float64)

    abs_curv = np.abs(curvatures)

    max_curvature = float(np.max(abs_curv))
    mean_abs_curvature = float(np.mean(abs_curv))
    curvature_std = float(np.std(curvatures))

    ds = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i].x - trajectory[i - 1].x
        dy = trajectory[i].y - trajectory[i - 1].y
        ds += math.sqrt(dx * dx + dy * dy)
    curvature_integral = float(np.sum(abs_curv)) * max(ds / max(len(curvatures), 1), 0.01)
    trajectory_length = ds

    lat_accels = np.abs(curvatures * speeds * speeds)
    max_lat_accel = float(np.max(lat_accels)) if len(lat_accels) > 0 else 0.0

    speed_mean = float(np.mean(speeds))
    speed_std = float(np.std(speeds))

    curvature_peaks = _count_peaks(abs_curv)
    curvature_sign_changes = _count_sign_changes(curvatures)

    return np.array([
        max_curvature,
        mean_abs_curvature,
        curvature_std,
        curvature_integral,
        max_lat_accel,
        speed_mean,
        speed_std,
        trajectory_length,
        float(curvature_peaks),
        float(curvature_sign_changes),
    ], dtype=np.float32)


def _count_peaks(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    peaks = 0
    threshold = np.mean(values) * 0.3 if np.mean(values) > 1e-9 else 0.001
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1] and values[i] > threshold:
            peaks += 1
    return max(peaks, 1) if np.max(values) > 1e-9 else 0


def _count_sign_changes(values: np.ndarray) -> int:
    if len(values) < 2:
        return 0
    changes = 0
    for i in range(1, len(values)):
        if values[i] * values[i - 1] < 0:
            changes += 1
    return changes


class RLTuningEnv(gym.Env):
    """Contextual Bandit 参数整定环境。

    观测空间 (21D):
        [max_curvature, mean_abs_curvature, curvature_std, curvature_integral,
         max_lat_accel, speed_mean, speed_std, trajectory_length,
         curvature_peaks, curvature_sign_changes]        ← 10 维几何特征
        [T2_mult, T3_mult, T4_mult, T6_mult,             ← 4 查找表缩放因子 (初值=1.0)
         station_kp, station_ki, low_speed_kp, low_speed_ki,
         high_speed_kp, high_speed_ki, switch_speed]     ← 7 PID 值 (初值=baseline)

    动作空间 (11D):
        [ΔT2_mult, ΔT3_mult, ΔT4_mult, ΔT6_mult,         ← 缩放因子增量
         Δstation_kp, Δstation_ki, Δlow_speed_kp, Δlow_speed_ki,
         Δhigh_speed_kp, Δhigh_speed_ki, Δswitch_speed]  ← PID 增量

    最终参数: T_y = baseline_T_y * (1 + mult), PID = baseline_PID + delta
    """

    ACTION_BOUNDS = np.array([
        0.3, 0.3, 0.3, 0.3,          # table mults
        0.1, 0.02, 0.1, 0.02, 0.1, 0.02, 0.5,  # PID deltas
    ], dtype=np.float32)

    def __init__(self, plant='hybrid_v2', config_path=None, seed=None,
                 trajectory_types=None):
        super().__init__()

        self.seed_val = seed

        cfg = load_config(config_path)
        if plant:
            apply_plant_override(cfg, plant)
        self.cfg = cfg
        self.plant = plant

        self.lat_ctrl = LatControllerTruck(cfg, differentiable=False)
        self.lon_ctrl = LonController(cfg, differentiable=False)

        self._capture_baseline_params()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.ACTION_BOUNDS, high=self.ACTION_BOUNDS, dtype=np.float32)

        self._setup_trajectories(trajectory_types)
        self._precompute_features()

        self.episode_count = 0
        self._current_features = np.zeros(10, dtype=np.float32)

    def _capture_baseline_params(self):
        self._baseline_T2_y = self.lat_ctrl.T2_y.data.clone()
        self._baseline_T3_y = self.lat_ctrl.T3_y.data.clone()
        self._baseline_T4_y = self.lat_ctrl.T4_y.data.clone()
        self._baseline_T6_y = self.lat_ctrl.T6_y.data.clone()
        self._baseline_station_kp = self.lon_ctrl.station_kp.item()
        self._baseline_station_ki = self.lon_ctrl.station_ki.item()
        self._baseline_low_speed_kp = self.lon_ctrl.low_speed_kp.item()
        self._baseline_low_speed_ki = self.lon_ctrl.low_speed_ki.item()
        self._baseline_high_speed_kp = self.lon_ctrl.high_speed_kp.item()
        self._baseline_high_speed_ki = self.lon_ctrl.high_speed_ki.item()
        self._baseline_switch_speed = self.lon_ctrl.switch_speed.item()

    def _setup_trajectories(self, trajectory_types):
        expanded = expand_trajectories(trajectory_types)
        self._traj_keys = [key for key, _label, _gen in expanded]
        self._traj_generators = {key: gen for key, _label, gen in expanded}
        self._traj_cache = {key: gen() for key, gen in self._traj_generators.items()}
        self._traj_keys_list = list(self._traj_cache.keys())

    def _precompute_features(self):
        self._feature_means = None
        self._feature_cov_inv = None
        self._feature_threshold = None

        all_features = []
        for key in self._traj_keys_list:
            traj = self._traj_cache[key]
            feats = extract_geometric_features(traj)
            all_features.append(feats)
        all_features = np.stack(all_features, axis=0)

        self._feature_means = np.mean(all_features, axis=0)
        cov = np.cov(all_features, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            self._feature_cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self._feature_cov_inv = np.linalg.pinv(cov)
        self._feature_threshold = _chi2_threshold(10, 0.01)

    def is_ood(self, features: np.ndarray) -> bool:
        if self._feature_cov_inv is None:
            return False
        diff = features - self._feature_means
        d2 = float(diff.T @ self._feature_cov_inv @ diff)
        return d2 > self._feature_threshold

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = self.np_random.integers(0, len(self._traj_keys_list))
        self._current_key = self._traj_keys_list[idx]
        self._current_trajectory = self._traj_cache[self._current_key]
        self._current_features = extract_geometric_features(self._current_trajectory)

        self._restore_baseline_params()
        self.lat_ctrl.reset_state()
        self.lon_ctrl.reset_state()

        obs = self._build_obs()
        return obs, {'trajectory_key': self._current_key}

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.ACTION_BOUNDS, self.ACTION_BOUNDS)
        self._apply_action(action)

        traj = self._current_trajectory
        traj_speed = traj[0].v

        history = run_simulation(
            traj, init_speed=traj_speed,
            init_x=traj[0].x, init_y=traj[0].y, init_yaw=traj[0].theta,
            cfg=self.cfg,
            lat_ctrl=self.lat_ctrl, lon_ctrl=self.lon_ctrl,
            differentiable=False, tbptt_k=0)

        reward = self._compute_reward(history, traj_speed)
        self._restore_baseline_params()

        self.episode_count += 1
        info = {
            'reward': reward,
            'trajectory_key': self._current_key,
            'episode': self.episode_count,
        }
        obs = self._build_obs()
        return obs, reward, True, False, info

    def _apply_action(self, action: np.ndarray):
        self.lat_ctrl.T2_y.data = self._baseline_T2_y * (1.0 + float(action[0]))
        self.lat_ctrl.T3_y.data = self._baseline_T3_y * (1.0 + float(action[1]))
        self.lat_ctrl.T4_y.data = self._baseline_T4_y * (1.0 + float(action[2]))
        self.lat_ctrl.T6_y.data = self._baseline_T6_y * (1.0 + float(action[3]))

        self.lon_ctrl.station_kp.data = torch.tensor(
            max(0.0, self._baseline_station_kp + float(action[4])))
        self.lon_ctrl.station_ki.data = torch.tensor(
            max(0.0, self._baseline_station_ki + float(action[5])))
        self.lon_ctrl.low_speed_kp.data = torch.tensor(
            max(0.0, self._baseline_low_speed_kp + float(action[6])))
        self.lon_ctrl.low_speed_ki.data = torch.tensor(
            max(0.0, self._baseline_low_speed_ki + float(action[7])))
        self.lon_ctrl.high_speed_kp.data = torch.tensor(
            max(0.0, self._baseline_high_speed_kp + float(action[8])))
        self.lon_ctrl.high_speed_ki.data = torch.tensor(
            max(0.0, self._baseline_high_speed_ki + float(action[9])))

        sw = self._baseline_switch_speed + float(action[10])
        sw = max(0.5, min(10.0, sw))
        self.lon_ctrl.switch_speed.data = torch.tensor(sw)

    def _restore_baseline_params(self):
        self.lat_ctrl.T2_y.data = self._baseline_T2_y.clone()
        self.lat_ctrl.T3_y.data = self._baseline_T3_y.clone()
        self.lat_ctrl.T4_y.data = self._baseline_T4_y.clone()
        self.lat_ctrl.T6_y.data = self._baseline_T6_y.clone()
        self.lon_ctrl.station_kp.data = torch.tensor(self._baseline_station_kp)
        self.lon_ctrl.station_ki.data = torch.tensor(self._baseline_station_ki)
        self.lon_ctrl.low_speed_kp.data = torch.tensor(self._baseline_low_speed_kp)
        self.lon_ctrl.low_speed_ki.data = torch.tensor(self._baseline_low_speed_ki)
        self.lon_ctrl.high_speed_kp.data = torch.tensor(self._baseline_high_speed_kp)
        self.lon_ctrl.high_speed_ki.data = torch.tensor(self._baseline_high_speed_ki)
        self.lon_ctrl.switch_speed.data = torch.tensor(self._baseline_switch_speed)

    def _build_obs(self) -> np.ndarray:
        params = np.array([
            1.0, 1.0, 1.0, 1.0,
            self._baseline_station_kp,
            self._baseline_station_ki,
            self._baseline_low_speed_kp,
            self._baseline_low_speed_ki,
            self._baseline_high_speed_kp,
            self._baseline_high_speed_ki,
            self._baseline_switch_speed,
        ], dtype=np.float32)
        return np.concatenate([self._current_features, params])

    def _compute_reward(self, history, ref_speed) -> float:
        from optim.train import tracking_loss
        tensor_history = [
            {k: torch.tensor(v, dtype=torch.float32) for k, v in h.items()}
            for h in history
        ]
        loss = tracking_loss(
            tensor_history, ref_speed=ref_speed,
            w_lat=10.0, w_head=8.0, w_speed=3.0,
            w_steer_rate=0.05, w_acc_rate=0.01,
            return_details=False)
        return -float(loss.item())

    def get_ood_stats(self) -> dict:
        return {
            'feature_means': self._feature_means,
            'feature_cov_inv': self._feature_cov_inv,
            'threshold': self._feature_threshold,
        }


def _chi2_threshold(df: int, p: float) -> float:
    """卡方分布阈值（近似）。df 自由度，p 显著性水平。"""
    if p == 0.01:
        table = {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086,
                 6: 16.812, 7: 18.475, 8: 20.090, 9: 21.666, 10: 23.209,
                 15: 30.578, 20: 37.566}
        return table.get(df, df * 1.6 + 7.0)
    return df * 1.5 + 5.0
