"""RL 评估端到端测试：评估脚本完整性和 OOD 检测。"""
import pytest
import numpy as np
import yaml
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_plot_functions_importable():
    from optim.post_training import _plot_comparison_grid, _calc_metrics
    assert callable(_plot_comparison_grid)
    assert callable(_calc_metrics)


def test_save_eval_results_writes_plain_yaml(tmp_path):
    from optim.rl_evaluate import _save_eval_results

    results = [{
        'key': 'lane_change_5kph',
        'rl_loss': np.float32(1.25),
        'dc_loss': np.float64(2.5),
        'delta_pct': np.float64(-50.0),
        'is_ood': np.bool_(False),
        'rl_action': np.array([0.1, -0.2], dtype=np.float32),
    }]

    output_path = _save_eval_results(
        results,
        output_dir=str(tmp_path),
        rl_mean=np.float64(1.25),
        dc_mean=np.float64(2.5),
        win_count=1,
        ood_count=0,
        run_spec={'model_path': 'model.zip', 'plant': 'truck_trailer'},
    )

    assert os.path.basename(output_path) == 'rl_eval_results.yaml'
    with open(output_path, 'r', encoding='utf-8') as f:
        saved = yaml.safe_load(f)

    assert saved['summary']['trajectory_count'] == 1
    assert saved['summary']['rl_avg_loss'] == pytest.approx(1.25)
    assert saved['summary']['dc_avg_loss'] == pytest.approx(2.5)
    assert saved['summary']['win_count'] == 1
    assert saved['summary']['mean_action_saturation_ratio'] == 0.0
    assert saved['run_spec']['model_path'] == 'model.zip'
    assert saved['run_spec']['plant'] == 'truck_trailer'
    assert saved['results'][0]['key'] == 'lane_change_5kph'
    assert saved['results'][0]['rl_action'] == pytest.approx([0.1, -0.2])


def test_cli_rejects_multiple_external_sources(tmp_path):
    import subprocess

    manifest_path = tmp_path / 'manifest.yaml'
    csv_path = tmp_path / 'record.csv'
    manifest_path.write_text('specs: []\n', encoding='utf-8')
    csv_path.write_text('timestamp,position_enu.x,position_enu.y,heading\n',
                        encoding='utf-8')

    script = os.path.join(os.path.dirname(__file__), '..', 'optim',
                          'rl_evaluate.py')
    proc = subprocess.run([
        sys.executable,
        script,
        '--rl-model', 'missing.zip',
        '--dc-config', 'configs/default.yaml',
        '--trajectory-manifest', str(manifest_path),
        '--real-csv', str(csv_path),
    ], cwd=os.path.join(os.path.dirname(__file__), '..'),
       text=True, capture_output=True)

    assert proc.returncode != 0
    assert 'choose at most one' in proc.stderr


def test_disable_mlp_runtime_override_clears_truck_checkpoint():
    from config import load_config
    from optim.rl_evaluate import _apply_eval_runtime_overrides

    cfg = load_config()
    cfg['truck_trailer_vehicle']['checkpoint_path'] = 'some_model.pth'

    _apply_eval_runtime_overrides(cfg, disable_mlp=True)

    assert cfg['truck_trailer_vehicle']['checkpoint_path'] == ''


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
