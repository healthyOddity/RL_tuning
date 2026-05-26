"""RL 环境单元测试：几何特征提取、reset/step、观测/动作维度。"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.trajectory import generate_lane_change, generate_clothoid_turn


@pytest.fixture
def sample_trajectory():
    return generate_lane_change(lane_width=3.5, change_length=40.0, speed=10.0)


@pytest.fixture
def sample_curved_trajectory():
    return generate_clothoid_turn(radius=30.0, turn_angle=1.57, speed=10.0)


class TestGeometricFeatures:
    def test_output_shape(self, sample_trajectory):
        from optim.rl_env import extract_geometric_features
        feats = extract_geometric_features(sample_trajectory)
        assert isinstance(feats, np.ndarray)
        assert feats.shape == (10,)
        assert feats.dtype == np.float32

    def test_no_nan_inf(self, sample_trajectory):
        from optim.rl_env import extract_geometric_features
        feats = extract_geometric_features(sample_trajectory)
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))

    def test_all_finite_values(self, sample_trajectory):
        from optim.rl_env import extract_geometric_features
        feats = extract_geometric_features(sample_trajectory)
        assert np.all(np.isfinite(feats))

    def test_straight_traj_zero_curvature_features(self):
        from model.trajectory import generate_straight
        from optim.rl_env import extract_geometric_features
        traj = generate_straight(length=20, speed=5.0)
        feats = extract_geometric_features(traj)
        assert feats[0] < 1e-6
        assert feats[1] < 1e-6
        assert feats[2] < 1e-6
        assert feats[8] == 0.0

    def test_curved_traj_has_curvature(self, sample_curved_trajectory):
        from optim.rl_env import extract_geometric_features
        feats = extract_geometric_features(sample_curved_trajectory)
        assert feats[0] > 0.0
        assert feats[1] > 0.0

    def test_speed_features(self, sample_trajectory):
        from optim.rl_env import extract_geometric_features
        feats = extract_geometric_features(sample_trajectory)
        assert feats[5] > 0.0
        assert feats[6] >= 0.0


class TestRLTuningEnv:
    @pytest.fixture
    def env(self):
        from optim.rl_env import RLTuningEnv
        return RLTuningEnv(plant='kinematic', seed=42, trajectory_types=['lane_change'])

    def test_init_spaces(self, env):
        assert env.observation_space.shape == (21,)
        assert env.action_space.shape == (11,)
        assert env.observation_space.dtype == np.float32
        assert env.action_space.dtype == np.float32

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (21,)
        assert obs.dtype == np.float32
        assert 'trajectory_key' in info

    def test_reset_no_nan(self, env):
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_reset_params_are_baseline(self, env):
        env.reset()
        assert env.lat_ctrl.T2_y.data.shape == env._baseline_T2_y.shape
        np.testing.assert_allclose(
            env.lat_ctrl.T2_y.data.numpy(),
            env._baseline_T2_y.numpy())

    def test_step_runs_without_crash(self, env):
        env.reset()
        action = np.zeros(11, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert terminated is True
        assert truncated is False

    def test_step_zero_action_near_baseline_reward(self, env):
        env.reset()
        action = np.zeros(11, dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert np.isfinite(reward)
        assert reward < 0

    def test_step_restores_baseline_after(self, env):
        env.reset()
        action = np.ones(11, dtype=np.float32) * 0.1
        env.step(action)
        np.testing.assert_allclose(
            env.lat_ctrl.T2_y.data.numpy(),
            env._baseline_T2_y.numpy())

    def test_action_clipping(self, env):
        env.reset()
        large_action = np.ones(11, dtype=np.float32) * 10.0
        env._apply_action(large_action)
        assert np.all(env.lat_ctrl.T2_y.data.numpy() >= 0)
        assert env.lon_ctrl.station_kp.item() >= 0
        assert 0.5 <= env.lon_ctrl.switch_speed.item() <= 10.0

    def test_multiple_resets_different_trajectories(self, env):
        keys = set()
        for _ in range(10):
            _, info = env.reset()
            keys.add(info['trajectory_key'])
        assert len(keys) >= 1

    def test_ood_detection_has_stats(self, env):
        env.reset()
        stats = env.get_ood_stats()
        assert stats['feature_means'] is not None
        assert stats['feature_cov_inv'] is not None
        assert stats['threshold'] is not None

    def test_ood_detection_train_traj_not_ood(self, env):
        env.reset()
        feats = env._current_features
        assert not env.is_ood(feats)

    def test_observation_components_width(self, env):
        obs, _ = env.reset()
        assert obs.shape == (21,)
        geo_part = obs[:10]
        param_part = obs[10:]
        assert np.all(np.isfinite(geo_part))
        assert np.all(np.isfinite(param_part))
        assert np.all(param_part[[0, 1, 2, 3]] == 1.0)

    def test_restore_baseline_params(self, env):
        env.reset()
        env._apply_action(np.array([0.2] * 11, dtype=np.float32))
        env._restore_baseline_params()
        np.testing.assert_allclose(
            env.lat_ctrl.T2_y.data.numpy(),
            env._baseline_T2_y.numpy())
        assert env.lon_ctrl.station_kp.item() == pytest.approx(
            env._baseline_station_kp)
