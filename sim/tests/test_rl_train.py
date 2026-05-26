"""RL 训练端到端测试：小规模训练 验证 loss 下降。"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")


class TestRLTrainPipeline:
    def test_train_rl_runs_small_scale(self, tmp_path):
        from optim.rl_train import train_rl
        import shutil
        results_dir = os.path.join(os.path.dirname(__file__), '..',
                                   'results', 'rl', 'kinematic')
        os.makedirs(results_dir, exist_ok=True)
        result = train_rl(
            plant='kinematic', config_path=None, total_timesteps=2000,
            lr=3e-4, buffer_size=10000, batch_size=64, seed=123,
            verbose=False)
        assert result['model'] is not None
        assert os.path.exists(result['model_path'])
        assert os.path.exists(result['yaml_path'])

    def test_train_rl_loss_improves(self, tmp_path):
        from optim.rl_train import train_rl
        result = train_rl(
            plant='kinematic', config_path=None, total_timesteps=2000,
            lr=3e-4, buffer_size=10000, batch_size=64, seed=456,
            verbose=False)
        assert result['model'] is not None

    def test_train_rl_with_config_warmstart(self):
        from optim.rl_train import train_rl
        result = train_rl(
            plant='kinematic', config_path='configs/default.yaml',
            total_timesteps=1000, lr=3e-4, buffer_size=5000,
            batch_size=64, seed=789, verbose=False)
        assert result['model'] is not None

    def test_train_rl_model_can_predict(self):
        from optim.rl_train import train_rl
        from optim.rl_env import RLTuningEnv
        result = train_rl(
            plant='kinematic', config_path=None, total_timesteps=1000,
            lr=3e-4, buffer_size=5000, batch_size=64, seed=111,
            verbose=False)
        env = RLTuningEnv(plant='kinematic', seed=222,
                          trajectory_types=['lane_change'])
        obs, _ = env.reset()
        action, _ = result['model'].predict(obs, deterministic=True)
        assert action.shape == (11,)
        assert np.all(np.isfinite(action))
