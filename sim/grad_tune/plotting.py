from __future__ import annotations

import os

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_record_refline_comparison(baseline_history, tuned_history, trajectory,
                                   output_dir: str, sample=None,
                                   refline_source: str = 'record'):
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}

    outputs['trajectory'] = _plot_trajectory(
        baseline_history, tuned_history, trajectory,
        os.path.join(output_dir, 'comparison_trajectory.png'), sample,
        refline_source=refline_source)
    outputs['record_trajectory_included'] = sample is not None
    outputs['lateral_error'] = _plot_series(
        baseline_history, tuned_history, 'lateral_error', '横向误差 (m)',
        os.path.join(output_dir, 'comparison_lateral_error.png'))
    outputs['speed_error'] = _plot_speed_error(
        baseline_history, tuned_history, trajectory,
        os.path.join(output_dir, 'comparison_speed_error.png'))
    outputs['steer'] = _plot_series(
        baseline_history, tuned_history, 'steer', '方向盘转角 (deg)',
        os.path.join(output_dir, 'comparison_steer.png'))
    outputs['acc'] = _plot_series(
        baseline_history, tuned_history, 'acc', '加速度指令 (m/s^2)',
        os.path.join(output_dir, 'comparison_acc.png'))
    return outputs


def _plot_trajectory(base, tuned, trajectory, path, sample=None,
                     refline_source: str = 'record'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([p.x for p in trajectory], [p.y for p in trajectory],
            'k--', label=f'refline({refline_source})', linewidth=1.2)
    if sample is not None:
        ax.plot(sample.x, sample.y, color='0.35', linestyle='-',
                label='record', linewidth=1.4, alpha=0.9)
    ax.plot([h['x'] for h in base], [h['y'] for h in base],
            'b-', label='baseline', alpha=0.8)
    ax.plot([h['x'] for h in tuned], [h['y'] for h in tuned],
            'r-', label='tuned', alpha=0.8)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_series(base, tuned, key, ylabel, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([h['t'] for h in base], [h[key] for h in base],
            'b-', label='baseline', alpha=0.8)
    ax.plot([h['t'] for h in tuned], [h[key] for h in tuned],
            'r-', label='tuned', alpha=0.8)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_speed_error(base, tuned, trajectory, path):
    ref_v = [p.v for p in trajectory]
    n_b = min(len(base), len(ref_v))
    n_t = min(len(tuned), len(ref_v))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([base[i]['t'] for i in range(n_b)],
            [base[i]['v'] - ref_v[i] for i in range(n_b)],
            'b-', label='baseline', alpha=0.8)
    ax.plot([tuned[i]['t'] for i in range(n_t)],
            [tuned[i]['v'] - ref_v[i] for i in range(n_t)],
            'r-', label='tuned', alpha=0.8)
    ax.axhline(y=0.0, color='k', linewidth=0.6, alpha=0.4)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('速度误差 (m/s)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
