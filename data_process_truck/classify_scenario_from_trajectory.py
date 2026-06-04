#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于轨迹自动识别场景类型工具

功能：
- 从轨迹（x, y 坐标）自动识别场景类型
- 支持四种场景：lane_change（变道）、constant_curve（弯道）、
                follow_stop_go（跟车起停）、straight_hold（直道）
- 输出带场景标记的 CSV 文件和可视化图

使用方法：
python classify_scenario_from_trajectory.py --input your_log.csv --output output_dir
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import signal
from scipy.spatial.distance import cdist

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Task 1: 轨迹特征提取模块
# ============================================================================

def compute_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算轨迹曲率（基于微分几何公式）
    
    公式：κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    
    参数:
        x: X 坐标序列
        y: Y 坐标序列
    
    返回:
        曲率序列（与输入长度相同）
    """
    n = len(x)
    curvature = np.zeros(n)
    
    # 使用中心差分计算一阶和二阶导数
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # 计算曲率
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2) ** 1.5
    
    # 避免除零
    mask = denominator > 1e-6
    curvature[mask] = numerator[mask] / denominator[mask]
    
    return curvature


def compute_lateral_offset(x: np.ndarray, y: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    计算横向偏移（通过局部拟合中心线）
    
    方法：
    1. 使用滑动窗口拟合局部轨迹的中心线
    2. 计算每个点到中心线的垂直距离
    
    参数:
        x: X 坐标序列
        y: Y 坐标序列
        window_size: 滑动窗口大小
    
    返回:
        横向偏移序列
    """
    n = len(x)
    lateral_offset = np.zeros(n)
    
    half_window = window_size // 2
    
    for i in range(n):
        # 确定窗口范围
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # 提取窗口内的点
        x_window = x[start:end]
        y_window = y[start:end]
        
        # 如果窗口内点太少，跳过
        if len(x_window) < 3:
            continue
        
        # 拟合直线（使用 PCA 或线性回归）
        # 这里使用简单的线性拟合
        if np.std(x_window) > np.std(y_window):
            # X 变化更大，拟合 y = f(x)
            try:
                coeffs = np.polyfit(x_window, y_window, 1)
                line_y = coeffs[0] * x[i] + coeffs[1]
                lateral_offset[i] = y[i] - line_y
            except:
                lateral_offset[i] = 0
        else:
            # Y 变化更大，拟合 x = f(y)
            try:
                coeffs = np.polyfit(y_window, x_window, 1)
                line_x = coeffs[0] * y[i] + coeffs[1]
                lateral_offset[i] = x[i] - line_x
            except:
                lateral_offset[i] = 0
    
    return lateral_offset


def compute_speed_features(ego_v: np.ndarray, timestamp: np.ndarray) -> dict[str, Any]:
    """
    计算速度特征（用于跟车场景识别）
    
    参数:
        ego_v: 车速序列
        timestamp: 时间戳序列
    
    返回:
        速度特征字典
    """
    dt = np.gradient(timestamp)
    dt[dt == 0] = 1e-3  # 避免除零
    
    # 加速度
    acc = np.gradient(ego_v) / dt
    
    # 速度变化次数（施加死区过滤噪声）
    dv = np.diff(ego_v)
    dv_sign = np.sign(dv)
    dv_sign[np.abs(dv) < 0.02] = 0
    speed_changes = np.sum(np.abs(np.diff(dv_sign)) == 2)
    
    # 加减速切换次数（施加死区过滤噪声）
    acc_sign = np.sign(acc)
    acc_sign[np.abs(acc) < 0.1] = 0
    acc_switches = np.sum(np.abs(np.diff(acc_sign)) == 2)
    
    # 停车检测（速度<0.3m/s）
    stop_mask = ego_v < 0.3
    stop_count = 0
    if len(stop_mask) > 0:
        # 统计停车事件次数
        stop_diff = np.diff(stop_mask.astype(int))
        stop_count = np.sum(stop_diff == 1)
    
    # 低速段占比
    low_speed_ratio = np.mean(ego_v < 2.0)
    
    return {
        'acc_rms': np.sqrt(np.mean(acc**2)),
        'speed_changes': speed_changes,
        'acc_switches': acc_switches,
        'stop_count': stop_count,
        'low_speed_ratio': low_speed_ratio,
    }


def segment_trajectory(df: pd.DataFrame, window_size: int = 100, step: int = 50) -> list[dict[str, Any]]:
    """
    将轨迹分段（滑动窗口）
    
    参数:
        df: 包含轨迹数据的 DataFrame
        window_size: 窗口大小（点数）
        step: 步长
    
    返回:
        分段列表，每段包含索引范围和特征
    """
    n = len(df)
    segments = []
    
    for start in range(0, n - window_size, step):
        end = start + window_size
        segment_df = df.iloc[start:end]
        
        # 计算特征
        x = segment_df['x'].values
        y = segment_df['y'].values
        ego_v = segment_df['ego_v'].values if 'ego_v' in df.columns else np.zeros(len(x))
        timestamp = segment_df['timestamp'].values
        
        curvature = compute_curvature(x, y)
        lateral_offset = compute_lateral_offset(x, y)
        speed_features = compute_speed_features(ego_v, timestamp)

        ref_kappa_in_seg = segment_df['ref_kappa'].values if 'ref_kappa' in segment_df.columns else None
        if ref_kappa_in_seg is not None:
            ref_kappa_mean = float(np.mean(np.abs(ref_kappa_in_seg)))
            ref_kappa_raw = ref_kappa_in_seg.copy()
        else:
            ref_kappa_mean = 0.0
            ref_kappa_raw = None
        
        segments.append({
            'start': start,
            'end': end,
            'curvature_mean': np.mean(curvature),
            'curvature_max': np.max(curvature),
            'curvature_std': np.std(curvature),
            'curvature_raw': curvature,
            'ref_kappa_mean': ref_kappa_mean,
            'ref_kappa_raw': ref_kappa_raw,
            'lateral_offset_mean': np.mean(lateral_offset),
            'lateral_offset_std': np.std(lateral_offset),
            'lateral_offset_peak': np.max(np.abs(lateral_offset)),
            'speed_features': speed_features,
            'x': x,
            'y': y,
            'ego_v': ego_v
        })
    
    return segments


# ============================================================================
# Task 2: 场景识别规则引擎
# ============================================================================

def classify_lane_change(segment: dict[str, Any], thresholds: dict[str, float]) -> bool:
    """
    变道场景识别规则
    
    规则：
    1. 横向偏移呈现单峰或双峰变化
    2. 曲率接近 0（直道变道）
    """
    # 横向偏移峰值足够大
    if segment['lateral_offset_peak'] < thresholds.get('lane_change_lat_offset', 0.6):
        return False
    
    # 曲率较小（排除弯道）
    kappa = segment.get('ref_kappa_mean', segment.get('curvature_mean', 0.0))
    if kappa > thresholds.get('curve_kappa', 0.003):
        return False
    
    # 横向偏移标准差大（说明有变化）
    if segment['lateral_offset_std'] < thresholds.get('lane_change_lat_std', 0.2):
        return False
    
    return True


def classify_constant_curve(segment: dict[str, Any], thresholds: dict[str, float]) -> bool:
    """
    弯道场景识别规则

    规则：
    1. ref_kappa 平均曲率大于阈值（表示道路本身有弯度）
    2. ref_kappa 持续大于阈值的点达到一定占比
    """
    kappa_thresh = thresholds.get('curve_kappa', 0.003)
    min_ratio = thresholds.get('curve_sustained_ratio', 0.3)

    # Primary: 使用 ref_kappa（参考路径曲率，反映实际道路弯度）
    ref_kappa_mean = segment.get('ref_kappa_mean')
    if ref_kappa_mean is not None:
        if ref_kappa_mean >= kappa_thresh:
            return True
        ref_kappa = segment.get('ref_kappa_raw')
        if ref_kappa is not None and len(ref_kappa) > 0:
            if np.mean(np.abs(ref_kappa) >= kappa_thresh) >= min_ratio:
                return True
        return False

    # Fallback: 当 ref_kappa 不可用时，使用梯度法曲率
    if segment['curvature_mean'] >= kappa_thresh:
        return True
    curvature = segment.get('curvature_raw')
    if curvature is not None and len(curvature) > 0:
        if np.mean(np.abs(curvature) >= kappa_thresh) >= min_ratio:
            return True

    return False


def classify_follow_stop_go(segment: dict[str, Any], thresholds: dict[str, float]) -> bool:
    """
    跟车起停场景识别规则
    
    规则：
    1. 速度频繁变化
    2. 有停车过程
    """
    speed_feat = segment['speed_features']
    
    # 低速段占比高
    if speed_feat['low_speed_ratio'] >= thresholds.get('low_speed_ratio', 0.5):
        # 有停车
        if speed_feat['stop_count'] >= thresholds.get('stop_count', 1):
            return True
        
        # 或者速度变化频繁
        if speed_feat['speed_changes'] >= thresholds.get('speed_changes', 3):
            return True
    
    return False


def classify_scenario_direct(df: pd.DataFrame, idx: int, thresholds: dict[str, float]) -> str:
    """
    直接基于数据点的场景分类器（逐点分类）
    
    参数:
        df: 原始 DataFrame
        idx: 当前点的索引
        thresholds: 阈值字典
    
    返回:
        场景类型
    """
    # 优先使用 ref_kappa（如果有）—— 使用局部窗口持续性判断
    if 'ref_kappa' in df.columns:
        window = 10
        start = max(0, idx - window)
        end = min(len(df), idx + window)
        local_kappa = np.abs(df.loc[start:end, 'ref_kappa'].values)
        if np.mean(local_kappa >= thresholds.get('curve_kappa_point', 0.005)) >= 0.3:
            return 'constant_curve'
    
    # 检查速度特征（用于跟车场景）
    if 'ego_v' in df.columns:
        v = df.loc[idx, 'ego_v']
        
        # 低速或停车检测
        if v < thresholds.get('stop_speed', 0.5):
            return 'follow_stop_go'
        
        # 检查速度变化（用于跟车）
        window = 20
        start = max(0, idx - window)
        end = min(len(df), idx + window)
        speed_window = df.loc[start:end, 'ego_v'].values
        
        if len(speed_window) > 10:
            speed_std = np.std(speed_window)
            speed_changes = np.sum(np.abs(np.diff(np.sign(np.diff(speed_window)))) > 0)
            
            # 速度变化频繁且低速段占比高（需有真实低速/停车证据）
            if speed_changes >= 3 and np.mean(speed_window < 5.0) > 0.3 and np.min(speed_window) < 0.5:
                return 'follow_stop_go'
    
    # 检查横向偏移（用于变道）
    if 'lateral_error' in df.columns:
        window = 30
        start = max(0, idx - window)
        end = min(len(df), idx + window)
        lat_err = abs(df.loc[idx, 'lateral_error'])

        if lat_err >= thresholds.get('lane_change_lat_offset', 0.6):
            if 'ref_kappa' in df.columns:
                local_kappa = np.mean(np.abs(df.loc[start:end, 'ref_kappa'].values))
                if local_kappa < thresholds.get('curve_kappa', 0.003):
                    return 'lane_change'
            else:
                return 'lane_change'

        # 检查横向偏移变化
        lat_window = df.loc[start:end, 'lateral_error'].values

        if len(lat_window) > 10:
            lat_std = np.std(lat_window)
            lat_peak = np.max(np.abs(lat_window))

            if lat_std >= thresholds.get('lane_change_lat_std', 0.12) and lat_peak >= 0.6:
                if 'ref_kappa' in df.columns:
                    local_kappa = np.mean(np.abs(df.loc[start:end, 'ref_kappa'].values))
                    if local_kappa < thresholds.get('curve_kappa', 0.003):
                        return 'lane_change'
                else:
                    return 'lane_change'

    # 默认直道
    return 'straight_hold'


def smooth_scenario_labels(labels: list[str], window_size: int = 5) -> list[str]:
    """
    平滑场景标签（避免频繁切换）
    
    使用多数投票平滑
    """
    n = len(labels)
    smoothed = labels.copy()
    half_window = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window_labels = labels[start:end]
        
        # 多数投票
        from collections import Counter
        counter = Counter(window_labels)
        smoothed[i] = counter.most_common(1)[0][0]
    
    return smoothed


def classify_trajectory_pipeline(df: pd.DataFrame, thresholds: dict[str, float], smooth_window: int = 10) -> list[str]:
    """
    组合管道：逐点分类 → 段级修正 → 平滑

    段级修正优先级：
    1. classify_lane_change() 通过 → lane_change
    2. classify_constant_curve() 通过 → constant_curve
    3. classify_follow_stop_go() 通过 → follow_stop_go
    4. 兜底 → 保留逐点分类结果

    参数:
        df: 输入 DataFrame
        thresholds: 阈值字典
        smooth_window: 平滑窗口大小

    返回:
        修正后的标签列表
    """
    # Step 1: 逐点分类
    labels = []
    for i in range(len(df)):
        label = classify_scenario_direct(df, i, thresholds)
        labels.append(label)

    # Step 2: 段级修正
    segments = segment_trajectory(df)
    for segment in segments:
        start, end = segment['start'], segment['end']

        if classify_lane_change(segment, thresholds):
            for j in range(start, end):
                labels[j] = 'lane_change'
            continue

        if classify_constant_curve(segment, thresholds):
            for j in range(start, end):
                labels[j] = 'constant_curve'
            continue

        if classify_follow_stop_go(segment, thresholds):
            for j in range(start, end):
                labels[j] = 'follow_stop_go'
            continue

    # Step 3: 平滑
    labels = smooth_scenario_labels(labels, window_size=smooth_window)

    return labels


# ============================================================================
# Task 3: 输出模块
# ============================================================================

def generate_event_columns(df: pd.DataFrame, scenario_labels: list[str]) -> pd.DataFrame:
    """
    生成事件标记列（6 个 event 字段）
    
    参数:
        df: 原始 DataFrame
        scenario_labels: 每个点对应的场景标签
    
    返回:
        添加了 6 个 event_*列的 DataFrame
    """
    result = df.copy()
    n = len(df)

    # 初始化 6 个事件列为 0（整数）
    result['event_lane_change'] = 0
    result['event_curve_enter'] = 0
    result['event_curve_exit'] = 0
    result['event_follow_start'] = 0
    result['event_stop_start'] = 0
    result['event_stop_end'] = 0

    labels = np.array(scenario_labels)

    # 找到所有场景段边界
    boundaries = [0]
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            boundaries.append(i)
    boundaries.append(n)

    # 对每个场景段，整段设置对应的 event
    for idx in range(len(boundaries) - 1):
        seg_start = boundaries[idx]
        seg_end = boundaries[idx + 1]
        scenario = labels[seg_start]

        if scenario == 'lane_change':
            result.loc[seg_start:seg_end - 1, 'event_lane_change'] = 1
        elif scenario == 'follow_stop_go':
            result.loc[seg_start:seg_end - 1, 'event_follow_start'] = 1
        elif scenario == 'constant_curve':
            result.loc[seg_start:seg_end - 1, 'event_curve_enter'] = 1

        # 弯道出口：从弯道切换到其他场景的边界行
        if seg_start > 0 and labels[seg_start - 1] == 'constant_curve':
            result.loc[seg_start, 'event_curve_exit'] = 1

    # 停车检测：在 follow_stop_go 段内，速度跨 0.3 时标记
    if 'ego_v' in df.columns:
        for i in range(1, n):
            if labels[i] == 'follow_stop_go' and df.loc[i, 'ego_v'] < 0.3:
                if df.loc[i - 1, 'ego_v'] >= 0.3:
                    result.loc[i, 'event_stop_start'] = 1
            if labels[i] != 'follow_stop_go' and labels[i - 1] == 'follow_stop_go':
                if df.loc[i - 1, 'ego_v'] < 0.3 and df.loc[i, 'ego_v'] >= 0.3:
                    result.loc[i, 'event_stop_end'] = 1

    # 转换为整数类型
    for col in ['event_lane_change', 'event_curve_enter', 'event_curve_exit',
                'event_follow_start', 'event_stop_start', 'event_stop_end']:
        result[col] = result[col].astype(int)

    return result


def plot_trajectory_with_scenario(
    df: pd.DataFrame,
    scenario_labels: list[str],
    output_path: Path
) -> None:
    """
    绘制带场景标记的轨迹图
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 颜色映射
    scenario_colors = {
        'lane_change': 'blue',
        'constant_curve': 'red',
        'follow_stop_go': 'green',
        'straight_hold': 'gray'
    }
    
    # 图 1: 轨迹俯视图
    ax = axes[0, 0]
    for scenario in scenario_colors:
        mask = np.array(scenario_labels) == scenario
        if np.any(mask):
            ax.scatter(
                df.loc[mask, 'x'],
                df.loc[mask, 'y'],
                c=scenario_colors[scenario],
                label=scenario,
                s=1,
                alpha=0.6
            )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory with Scenario Classification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 图 2: 曲率时序图
    ax = axes[0, 1]
    if 'ref_kappa' in df.columns:
        kappa = df['ref_kappa'].values
    else:
        kappa = compute_curvature(df['x'].values, df['y'].values)
    
    ax.plot(df['timestamp'], kappa, 'b-', linewidth=0.5, label='Curvature')
    ax.axhline(y=0.003, color='r', linestyle='--', label='Threshold (0.003)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Curvature (1/m)')
    ax.set_title('Trajectory Curvature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图 3: 速度时序图
    ax = axes[1, 0]
    if 'ego_v' in df.columns:
        ax.plot(df['timestamp'], df['ego_v'], 'g-', linewidth=0.5, label='Ego Speed')
        if 'ref_v' in df.columns:
            ax.plot(df['timestamp'], df['ref_v'], 'r--', linewidth=0.5, label='Ref Speed')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Speed Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Speed data not available', ha='center', va='center')
        ax.set_title('Speed Profile (Not Available)')
    
    # 图 4: 场景时序图
    ax = axes[1, 1]
    scenario_ids = {s: i for i, s in enumerate(scenario_colors.keys())}
    scenario_values = [scenario_ids[s] for s in scenario_labels]
    
    ax.fill_between(df['timestamp'], 0, scenario_values, step='mid', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scenario')
    ax.set_yticks(list(scenario_ids.values()))
    ax.set_yticklabels(list(scenario_ids.keys()))
    ax.set_title('Scenario Classification Timeline')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"轨迹可视化图已保存：{output_path}")


def generate_scenario_report(
    df: pd.DataFrame,
    scenario_labels: list[str],
    output_path: Path
) -> None:
    """
    生成场景统计报告
    """
    from collections import Counter
    
    # 统计各场景的点数
    counter = Counter(scenario_labels)
    total_points = len(scenario_labels)
    
    # 计算时长（如果有时间戳）
    if 'timestamp' in df.columns:
        dt = df['timestamp'].diff()
        dt_mean = dt.mean() if len(dt) > 0 else 0.1
    else:
        dt_mean = 0.1  # 假设 10Hz
    
    # 生成报告
    report_lines = [
        "=" * 60,
        "场景识别统计报告",
        "=" * 60,
        "",
        f"总数据点数：{total_points}",
        f"平均采样间隔：{dt_mean:.3f} s",
        "",
        "各场景统计:",
        "-" * 60
    ]
    
    for scenario, count in counter.most_common():
        percentage = count / total_points * 100
        duration = count * dt_mean
        report_lines.append(
            f"{scenario:20s}: {count:6d} 点 ({percentage:5.1f}%), "
            f"时长约 {duration:.1f} s"
        )
    
    report_lines.extend([
        "-" * 60,
        "",
        "场景说明:",
        "  lane_change     : 变道场景（横向偏移变化，曲率小）",
        "  constant_curve  : 弯道场景（曲率持续大于阈值）",
        "  follow_stop_go  : 跟车起停场景（速度变化频繁，有停车）",
        "  straight_hold   : 直道场景（曲率接近 0，横向偏移稳定）",
        "",
        "=" * 60
    ])
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"场景统计报告已保存：{output_path}")
    print('\n'.join(report_lines))


# ============================================================================
# Task 4: 主函数和命令行接口
# ============================================================================

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_data(input_path: Path, signal_config_path: Path | None = None) -> pd.DataFrame:
    """
    加载输入 CSV 文件，支持通过信号配置映射列名

    参数:
        input_path: CSV 文件路径
        signal_config_path: 信号配置 YAML 文件（内部名 → CSV列名的映射）
    """
    logger.info(f"加载数据：{input_path}")
    df = pd.read_csv(input_path)

    # 如果提供了信号配置，进行列名映射
    if signal_config_path is not None:
        signal_config = _load_yaml(signal_config_path)
        mapped_df = pd.DataFrame()
        for target, source in signal_config.items():
            if source in df.columns:
                mapped_df[target] = pd.to_numeric(df[source], errors="coerce").fillna(0.0)
        if not mapped_df.empty:
            for col in mapped_df.columns:
                df[col] = mapped_df[col]
            logger.info(f"已通过信号配置映射 {len(mapped_df.columns)} 个字段（保留原始列名）")
        else:
            logger.warning("信号配置映射后无有效列，使用原始数据")

    # 检查必需列
    required_cols = ['timestamp', 'x', 'y']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列：{missing}")

    logger.info(f"数据加载完成，共 {len(df)} 行，{len(df.columns)} 列")
    return df


def run_classification(
    input_path: Path,
    output_dir: Path,
    thresholds: dict[str, float] | None = None,
    signal_config_path: Path | None = None
) -> None:
    """
    执行场景识别主流程（逐点分类版本）
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    df = load_data(input_path, signal_config_path)
    
    # 默认阈值
    if thresholds is None:
        thresholds = {
            'curve_kappa': 0.003,
            'curve_kappa_point': 0.005,
            'curve_sustained_ratio': 0.3,
            'lane_change_lat_offset': 0.6,
            'lane_change_lat_std': 0.12,
            'low_speed_ratio': 0.5,
            'stop_count': 1,
            'speed_changes': 3,
            'stop_speed': 0.5
        }
    
    logger.info("开始场景识别...")
    
    # 逐点分类（不再使用分段）
    point_labels = []
    for i in range(len(df)):
        label = classify_scenario_direct(df, i, thresholds)
        point_labels.append(label)
    
    # 平滑标签
    logger.info("平滑场景标签...")
    point_labels = smooth_scenario_labels(point_labels, window_size=10)  # 增大平滑窗口
    
    # 生成输出
    logger.info("生成输出文件...")
    
    # 1. 带场景标记的 CSV
    output_csv = output_dir / 'output_classified.csv'
    df_classified = generate_event_columns(df, point_labels)
    df_classified.to_csv(output_csv, index=False)
    logger.info(f"带场景标记的 CSV 已保存：{output_csv}")
    
    # 2. 轨迹可视化图
    output_fig = output_dir / 'trajectory_with_scenario.png'
    plot_trajectory_with_scenario(df, point_labels, output_fig)
    
    # 3. 场景统计报告
    output_report = output_dir / 'scenario_report.txt'
    generate_scenario_report(df, point_labels, output_report)
    
    logger.info("场景识别完成！")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='基于轨迹自动识别场景类型工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python classify_scenario_from_trajectory.py --input your_log.csv --output output_dir
  
  # 调整阈值
  python classify_scenario_from_trajectory.py --input your_log.csv --output output_dir --curve-kappa 0.004
  
  # 使用自定义窗口大小
  python classify_scenario_from_trajectory.py --input your_log.csv --output output_dir --window-size 150
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='输入 CSV 文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='输出目录'
    )
    
    # 阈值参数
    parser.add_argument(
        '--curve-kappa',
        type=float,
        default=0.003,
        help='弯道曲率阈值（默认：0.003 1/m）'
    )
    
    parser.add_argument(
        '--lane-change-offset',
        type=float,
        default=0.6,
        help='变道横向偏移阈值（默认：0.6 m）'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='滑动窗口大小（默认：100 点）'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=50,
        help='滑动窗口步长（默认：50 点）'
    )
    
    parser.add_argument(
        '--signal-config',
        type=Path,
        default=None,
        help='信号配置 YAML 文件（内部名 → CSV列名的映射），'
             '当 CSV 列名与标准名不一致时使用'
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 构建阈值配置
    thresholds = {
        'curve_kappa': args.curve_kappa,
        'lane_change_lat_offset': args.lane_change_offset,
        'lane_change_lat_std': 0.12,
        'low_speed_ratio': 0.5,
        'stop_count': 1,
        'speed_changes': 3
    }
    
    logger.info(f"输入文件：{args.input}")
    logger.info(f"输出目录：{args.output}")
    logger.info(f"弯道曲率阈值：{args.curve_kappa} 1/m")
    logger.info(f"变道横向偏移阈值：{args.lane_change_offset} m")
    
    try:
        run_classification(args.input, args.output, thresholds, args.signal_config)
    except Exception as e:
        logger.error(f"场景识别失败：{e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
