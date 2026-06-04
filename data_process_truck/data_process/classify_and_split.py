#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from classify_scenario_from_trajectory import classify_trajectory_pipeline
from csvdata_new_truck import find_combined_segments, zero_small_values

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

SCENE_MAP = {
    'constant_curve': 'turn',
    'lane_change': 'lane_change',
    'straight_hold': 'straight',
    'follow_stop_go': 'straight',
}




def load_signal_config(config_dir):
    path = config_dir / 'signal_config.yaml'
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_thresholds(config_dir):
    path = config_dir / 'threshold_config.yaml'
    with path.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    scenario = raw.get('scenario', {})
    return {
        'curve_kappa': scenario.get('curve_kappa_threshold', 0.003),
        'curve_kappa_point': scenario.get('curve_kappa_point_threshold', 0.005),
        'curve_sustained_ratio': scenario.get('curve_sustained_ratio', 0.3),
        'stop_speed': scenario.get('stop_speed_threshold', 0.3),
        'lane_change_lat_offset': scenario.get('lane_change_lat_offset_threshold', 0.6),
        'lane_change_lat_std': 0.12,
        'low_speed_ratio': 0.5,
        'stop_count': 1,
        'speed_changes': 3,
    }


def map_columns(df, signal_config):
    mapped = pd.DataFrame(index=df.index)
    for alias, original in signal_config.items():
        if original in df.columns:
            mapped[alias] = pd.to_numeric(df[original], errors='coerce').fillna(0.0)
        else:
            mapped[alias] = 0.0
    return mapped





def split_by_labels(labels, timestamps, min_duration=2.0):
    n = len(labels)
    if n == 0:
        return []

    segments = []
    start = 0
    for i in range(1, n):
        if labels[i] != labels[start]:
            duration = timestamps[i - 1] - timestamps[start]
            if duration >= min_duration:
                segments.append((start, i, labels[start]))
            start = i

    duration = timestamps[-1] - timestamps[start]
    if duration >= min_duration:
        segments.append((start, n, labels[start]))

    return segments


def speed_bin(mapped_df, speed_low, speed_mid):
    if 'ego_v' not in mapped_df.columns or mapped_df.empty:
        return None
    median_v = mapped_df['ego_v'].median()
    if median_v < speed_low:
        return 'low'
    elif median_v < speed_mid:
        return 'mid'
    else:
        return 'high'


def process_file(csv_path, output_dir, signal_config, thresholds, smooth_window,
                  speed_low, speed_mid, min_segment_duration=2.0,
                  enable_plot=True):
    df = pd.read_csv(csv_path)

    filter_df = pd.DataFrame(index=df.index)
    if 'controller_enable' in df.columns:
        filter_df['controller_enable'] = df['controller_enable']
    speed_kmh = df['VehicleInfoBDData.BDCFF10D0_VCU_VehicleSpeed'].values if 'VehicleInfoBDData.BDCFF10D0_VCU_VehicleSpeed' in df.columns else np.zeros(len(df))
    filter_df['Vx_mps'] = zero_small_values(speed_kmh / 3.6, 1e-7)
    vy_raw = df['vy'].values if 'vy' in df.columns else np.zeros(len(df))
    filter_df['Vy_mps'] = zero_small_values(vy_raw, 1e-7)
    filter_df['Time_s'] = df['timestamp']

    valid_segments = find_combined_segments(filter_df)
    if not valid_segments:
        logger.warning(f"跳过 {csv_path.name}：无有效控制段")
        return None

    stem = csv_path.stem.replace('_interpolated', '')
    scene_counters = {}
    stats = Counter()

    for vs_start, vs_end in valid_segments:
        sub_df = df.iloc[vs_start:vs_end + 1].reset_index(drop=True)
        sub_mapped = map_columns(sub_df, signal_config)

        if sub_mapped.empty or 'ego_v' not in sub_mapped.columns:
            continue

        labels = classify_trajectory_pipeline(sub_mapped, thresholds, smooth_window)

        scene_segments = split_by_labels(labels, sub_mapped['timestamp'].values, min_segment_duration)
        if not scene_segments:
            continue

        for seg_start, seg_end, label in scene_segments:
            segment_mapped = sub_mapped.iloc[seg_start:seg_end].copy()
            scene_label = SCENE_MAP[label]
            spd = speed_bin(segment_mapped, speed_low, speed_mid)
            if spd is None:
                continue

            scene_counters[scene_label] = scene_counters.get(scene_label, 0) + 1
            seq = scene_counters[scene_label]

            out_name = f'{stem}_{scene_label}{seq}.csv'
            dest_dir = output_dir / f'{scene_label}_{spd}'
            dest_dir.mkdir(parents=True, exist_ok=True)
            sub_df.iloc[seg_start:seg_end].to_csv(dest_dir / out_name, index=False)

            if enable_plot:
                png_name = f'{stem}_{scene_label}{seq}_trajectory.png'
                plot_trajectory(
                    segment_mapped, labels[seg_start:seg_end], dest_dir / png_name,
                    f'{stem}_{scene_label}{seq}  [{scene_label}_{spd}]'
                )

            scene_speed_key = f'{scene_label}_{spd}'
            stats[scene_speed_key] += 1

    if not stats:
        logger.warning(f"跳过 {csv_path.name}：拆分后无有效场景段")
        return None

    return stats


def plot_trajectory(mapped_df, labels, output_path, scene_name):
    scenario_colors = {
        'lane_change': 'blue',
        'constant_curve': 'red',
        'follow_stop_go': 'green',
        'straight_hold': 'gray',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(scene_name, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for scenario, color in scenario_colors.items():
        mask = np.array(labels) == scenario
        if np.any(mask):
            ax.scatter(mapped_df.loc[mask, 'x'], mapped_df.loc[mask, 'y'],
                       c=color, label=scenario, s=1, alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory with Scenario Classification')
    ax.legend(markerscale=8, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[0, 1]
    ax.plot(mapped_df['timestamp'], mapped_df['ref_kappa'], 'b-', linewidth=0.5, label='Curvature')
    ax.axhline(y=0.003, color='r', linestyle='--', label='Threshold (0.003)')
    ax.axhline(y=-0.003, color='purple', linestyle='--', label='Threshold (-0.003)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Curvature (1/m)')
    ax.set_title('Trajectory Curvature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(mapped_df['timestamp'], mapped_df['ego_v'], 'g-', linewidth=0.5, label='Ego Speed')
    if 'ref_v' in mapped_df.columns:
        ax.plot(mapped_df['timestamp'], mapped_df['ref_v'], 'r--', linewidth=0.5, label='Ref Speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    scenario_ids = {s: i for i, s in enumerate(scenario_colors.keys())}
    ts = mapped_df['timestamp']
    scenario_values = [scenario_ids[s] for s in labels]
    ax.fill_between(ts, np.zeros(len(labels)), scenario_values, step='mid', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scenario')
    ax.set_yticks(list(scenario_ids.values()))
    ax.set_yticklabels(list(scenario_ids.keys()))
    ax.set_title('Scenario Classification Timeline')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='按场景和速度对 interpolated CSV 文件进行分类并分发'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='包含 *interpolated.csv 文件的输入目录'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='输出根目录（场景子目录将创建于此）'
    )
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=5,
        help='标签平滑窗口大小（默认 5）'
    )
    parser.add_argument(
        '--speed-low',
        type=float,
        default=8.33,
        help='低速档上限，m/s（默认 8.33 ≈ 30 km/h）'
    )
    parser.add_argument(
        '--speed-mid',
        type=float,
        default=16.67,
        help='中速档上限，m/s（默认 16.67 ≈ 60 km/h）'
    )
    parser.add_argument(
        '--min-segment-duration',
        type=float,
        default=2.0,
        help='场景段最小时长（秒），短于此的段被忽略（默认 2.0）'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='不生成轨迹可视化图'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_dir = SCRIPT_DIR.parent / 'config'
    signal_config = load_signal_config(config_dir)
    thresholds = load_thresholds(config_dir)

    input_dir = args.input_dir
    if not input_dir.is_dir():
        logger.error(f"输入目录不存在：{input_dir}")
        return 1

    csv_files = sorted(input_dir.glob('*interpolated.csv'))
    if not csv_files:
        logger.error(f"在 {input_dir} 中未找到 *interpolated.csv 文件")
        return 1

    logger.info(f"找到 {len(csv_files)} 个 interpolated CSV 文件")

    stats = Counter()
    skipped = 0
    total = 0

    for csv_path in csv_files:
        result = process_file(
            csv_path, args.output_dir, signal_config, thresholds,
            args.smooth_window, args.speed_low, args.speed_mid,
            min_segment_duration=args.min_segment_duration,
            enable_plot=not args.no_plot
        )
        if result is None:
            skipped += 1
        else:
            stats.update(result)
        total += 1

    logger.info(f"处理完成：总计 {total} 个文件，成功 {total - skipped} 个，跳过 {skipped} 个")
    if stats:
        logger.info("各场景+速度组合文件数：")
        for combo, count in sorted(stats.items()):
            logger.info(f"  {combo}: {count}")
    return 0


if __name__ == '__main__':
    exit(main())
