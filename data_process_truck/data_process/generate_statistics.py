import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def collect_stats(train_dir: Path):
    subdirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if not subdirs:
        logger.warning('No subdirectories found in %s', train_dir)
        return []

    results = []
    for subdir in subdirs:
        csv_paths = sorted(subdir.glob('*_train.csv'))
        if not csv_paths:
            continue

        category = subdir.name
        total_duration = 0.0
        total_frames = 0
        vx_diff_high = 0
        vy_high = 0
        yawrate_high = 0
        steer_high = 0

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            if 'Time_s' in df.columns:
                total_duration += df['Time_s'].max() - df['Time_s'].min()
            total_frames += len(df)

            if 'Vx_mps' in df.columns:
                diff_vals = df['Vx_mps'].diff().abs().fillna(0)
                vx_diff_high += int((diff_vals >= 0.02).sum())

            if 'Vy_mps' in df.columns:
                vy_high += int((df['Vy_mps'].abs() >= 0.5).sum())

            if 'Yawrate_degps' in df.columns:
                yawrate_high += int((df['Yawrate_degps'].abs() >= 15).sum())

            if 'Steer_deg_cmd' in df.columns:
                steer_high += int((df['Steer_deg_cmd'].abs() >= 30).sum())

        results.append({
            'category': category,
            'total_duration': total_duration,
            'total_frames': total_frames,
            'vx_diff_ratio': vx_diff_high / total_frames if total_frames > 0 else 0,
            'vy_ratio': vy_high / total_frames if total_frames > 0 else 0,
            'yawrate_ratio': yawrate_high / total_frames if total_frames > 0 else 0,
            'steer_ratio': steer_high / total_frames if total_frames > 0 else 0,
        })

    return results


def generate_txt(stats, step_times, output_path):
    lines = []
    lines.append('=' * 60)
    lines.append('Pipeline Statistics Report')
    lines.append('=' * 60)
    lines.append('')

    if any(t >= 0 for t in step_times):
        total = sum(t for t in step_times if t >= 0)
        lines.append('--- Runtime ---')
        lines.append(f'  总耗时: {total:.1f}s')
        labels = ['Step1 (extract)', 'Step2 (classify)', 'Step3 (csvdata)']
        for label, t in zip(labels, step_times):
            if t >= 0:
                pct = t / total * 100 if total > 0 else 0
                lines.append(f'  {label}: {t:.1f}s ({pct:.1f}%)')
        lines.append('')

    total_duration = sum(s['total_duration'] for s in stats)
    total_frames = sum(s['total_frames'] for s in stats)
    lines.append(f'--- 数据总量 ---')
    lines.append(f'  类别数: {len(stats)}')
    lines.append(f'  总时长: {total_duration:.1f}s')
    lines.append(f'  总帧数: {total_frames}')
    lines.append('')

    lines.append(f'--- 各类别时长 ---')
    lines.append(f'  {"类别":<20s} {"时长(s)":<12s} {"帧数":<8s}')
    lines.append(f'  {"-"*40}')
    for s in sorted(stats, key=lambda x: x['category']):
        lines.append(f'  {s["category"]:<20s} {s["total_duration"]:<12.1f} {s["total_frames"]:<8d}')
    lines.append('')

    lines.append(f'--- 动态指标较大值占比 ---')
    header = f'  {"类别":<20s} {"Vx_diff":<10s} {"Vy":<10s} {"Yawrate":<10s} {"Steer":<10s}'
    lines.append(header)
    lines.append(f'  {"-"*60}')
    for s in sorted(stats, key=lambda x: x['category']):
        lines.append(
            f'  {s["category"]:<20s} {s["vx_diff_ratio"]:<10.1%} {s["vy_ratio"]:<10.1%} '
            f'{s["yawrate_ratio"]:<10.1%} {s["steer_ratio"]:<10.1%}'
        )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info('Report saved to %s', output_path)


def generate_png(stats, output_path):
    if not stats:
        logger.warning('No stats to plot, skipping PNG')
        return

    sorted_stats = sorted(stats, key=lambda x: x['category'])
    categories = [s['category'] for s in sorted_stats]
    x = np.arange(len(categories))
    bar_width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    durations = [s['total_duration'] for s in sorted_stats]
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    ax1.bar(x, durations, width=bar_width, color=colors)
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Total Duration (s)')
    ax1.set_title('Total Duration per Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=30, ha='right')
    for i, v in enumerate(durations):
        ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    indicators = ['vx_diff_ratio', 'vy_ratio', 'yawrate_ratio', 'steer_ratio']
    ind_labels = ['Vx_diff', 'Vy', 'Yawrate', 'Steer']
    n_indicators = len(indicators)
    group_width = bar_width * n_indicators
    x_group = x - group_width / 2 + bar_width / 2

    for i, (key, label) in enumerate(zip(indicators, ind_labels)):
        values = [s[key] * 100 for s in sorted_stats]
        pos = x_group + i * bar_width
        bars = ax2.bar(pos, values, bar_width, label=label)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

    ax2.set_xlabel('Category')
    ax2.set_ylabel('High-value Ratio (%)')
    ax2.set_title('Dynamic Indicator High-value Ratio per Category')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=30, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Chart saved to %s', output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate statistics report for pipeline output')
    parser.add_argument('train_dir', type=Path, help='2_train/ directory to analyze')
    parser.add_argument('--step-times', nargs=3, type=float, default=[-1, -1, -1],
                        help='Step execution times in seconds (Step1 Step2 Step3)')
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = args.train_dir

    if not train_dir.is_dir():
        logger.error('Train directory not found: %s', train_dir)
        return 1

    stats = collect_stats(train_dir)
    if not stats:
        logger.warning('No train data found in %s', train_dir)
        return 0

    generate_txt(stats, args.step_times, train_dir / 'statistics_report.txt')
    generate_png(stats, train_dir / 'statistics_report.png')
    return 0


if __name__ == '__main__':
    exit(main())
