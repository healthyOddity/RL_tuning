"""RL 评估端到端测试：评估脚本完整性和 OOD 检测。"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_plot_functions_importable():
    from optim.post_training import _plot_comparison_grid, _calc_metrics
    assert callable(_plot_comparison_grid)
    assert callable(_calc_metrics)


pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")


@pytest.fixture
def trained_model_path(tmp_path):
    from optim.rl_train import train_rl
    result = train_rl(
        plant='kinematic', config_path=None, total_timesteps=2000,
        lr=3e-4, buffer_size=10000, batch_size=64, seed=333,
        verbose=False)
    return result['model_path']


class TestRLEvaluate:
    def test_evaluate_runs_without_crash(self, trained_model_path):
        from optim.rl_evaluate import evaluate_rl_model
        results = evaluate_rl_model(
            model_path=trained_model_path,
            plant='kinematic',
            dc_config_path='configs/default.yaml',
            output_dir=None)
        assert len(results) > 0
        for r in results:
            assert 'rl_loss' in r
            assert 'dc_loss' in r
            assert 'is_ood' in r
            assert np.isfinite(r['rl_loss'])
            assert np.isfinite(r['dc_loss'])

    def test_evaluate_all_trajectory_keys_present(self, trained_model_path):
        from optim.rl_evaluate import evaluate_rl_model
        from optim.rl_env import RLTuningEnv
        env = RLTuningEnv(plant='kinematic', seed=444,
                          trajectory_types=['lane_change'])
        results = evaluate_rl_model(
            model_path=trained_model_path,
            plant='kinematic',
            dc_config_path='configs/default.yaml',
            output_dir=None)
        result_keys = {r['key'] for r in results}
        for key in env._traj_keys_list:
            assert key in result_keys

    def test_ood_not_flagged_for_train_trajectories(self, trained_model_path):
        from optim.rl_evaluate import evaluate_rl_model
        results = evaluate_rl_model(
            model_path=trained_model_path,
            plant='kinematic',
            dc_config_path='configs/default.yaml',
            output_dir=None)
        ood_count = sum(1 for r in results if r['is_ood'])
        assert ood_count == 0

    def test_evaluate_outputs_finite_values(self, trained_model_path):
        from optim.rl_evaluate import evaluate_rl_model
        results = evaluate_rl_model(
            model_path=trained_model_path,
            plant='kinematic',
            dc_config_path='configs/default.yaml',
            output_dir=None)
        for r in results:
            assert np.isfinite(r['rl_loss'])
            assert np.isfinite(r['dc_loss'])
            assert r['rl_loss'] > 0
            assert r['dc_loss'] > 0
