"""
将 extract_topic_truck.py 生成的 CSV 转换为 train.py 所需的格式。

增加了 vy 跳变过滤：
  - 检测 |Vy_mps| > 2 m/s 的定位跳变帧
  - 跳变帧及其前后各 3 帧标记为无效，在分段时自动丢弃
  - 跳变处数据被分割为两个干净 segment
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from convert_timestamp import process_timestamps


def convert_csv(input_csv, output_csv, segment_static_data=True, segment_mode='combined', zero_threshold=1e-7, min_zero_duration=10, min_controller_duration=10, process_timestamp=False):
    """
    将 extract_topic_truck.py 生成的 CSV 转换为 train.py 所需的格式

    参数:
        segment_static_data: 是否按数据进行分段 (默认 True)
        segment_mode: 分段模式 (默认 'combined')
            - 'combined': 组合分段 (controller_enable==1 AND 非静止)
            - 'static': 仅按静止数据分段
            - 'controller': 仅按 controller_enable==1 分段
        zero_threshold: 小量阈值, 小于此值视为 0 (默认 1e-7)
        min_zero_duration: 判定为连续静止的最小行数 (默认 10)
        min_controller_duration: 判定为有效控制的最小行数 (默认 10)
    """
    df = pd.read_csv(input_csv)

    new_df = pd.DataFrame()

    if 'controller_enable' in df.columns:
        new_df['controller_enable'] = df['controller_enable']

    new_df['Time_original'] = df['timestamp_ms'] / 1000.0
    new_df['Time_s'] = df['timestamp']

    new_df['Steer_deg_cmd'] = df['steering_target']
    new_df['Target_Steer_L1_deg_cmd'] = df['steering_target']

    new_df['Torque_L1_Nm_cmd'] = df['torque_fl']
    new_df['Torque_R1_Nm_cmd'] = df['torque_fr']
    new_df['Torque_L2_Nm_cmd'] = df['torque_rl']
    new_df['Torque_R2_Nm_cmd'] = df['torque_rr']

    new_df['Vx_mps'] = df['VehicleInfoBDData.BDCFF10D0_VCU_VehicleSpeed'] / 3.6
    new_df['Vy_mps'] = df['vy']

    new_df['X_m'] = df['position_enu.x']
    new_df['Y_m'] = df['position_enu.y']
    new_df['Yaw_deg'] = df['euler_angles.z']

    new_df['Yawrate_degps'] = np.rad2deg(df['VehicleInfoBDData.BD18F0090B_VDC2_YawRate']) - 0.184365
    new_df['YawRate_diff'] = df['diff_yawrate']

    print(f"Step 1: 小量置零 (阈值={zero_threshold:.2e})")
    new_df['Vx_mps'] = zero_small_values(new_df['Vx_mps'].values, zero_threshold)
    new_df['Vy_mps'] = zero_small_values(new_df['Vy_mps'].values, zero_threshold)
    new_df['Yawrate_degps'] = zero_small_values(new_df['Yawrate_degps'].values, zero_threshold)

    if segment_static_data:
        input_path = Path(input_csv)
        base_output_path = Path(output_csv)

        output_dir = base_output_path.parent
        base_filename = base_output_path.stem

        print(f"Step 2: 查找片段 (分段模式={segment_mode}, 最小行数={max(min_zero_duration, min_controller_duration)})")

        if segment_mode == 'combined':
            segments = find_combined_segments(
                new_df,
                min_zero_duration=min_zero_duration,
                min_controller_duration=min_controller_duration
            )
        elif segment_mode == 'controller':
            segments = find_controller_enable_segments(
                new_df,
                min_duration=min_controller_duration
            )
        else:
            segments = find_dynamic_segments(
                new_df,
                min_zero_duration=min_zero_duration
            )

        if not segments:
            print("警告: 未找到有效的片段！")
            return

        print(f"找到 {len(segments)} 个片段")

        for i, (start_idx, end_idx) in enumerate(segments):
            segment_df = new_df.iloc[start_idx:end_idx+1].copy()
            segment_output_path = output_dir / f"{base_filename}_segment_{i+1:03d}.csv"
            segment_df.to_csv(segment_output_path, index=False)
            print(f" 分段 {i+1}: 行 {start_idx}~{end_idx} (共 {end_idx-start_idx+1} 行) -> {segment_output_path.name}")
    else:
        new_df.to_csv(output_csv, index=False)
        print(f"转换完成! 结果已保存到 {output_csv}")


def zero_small_values(values, threshold):
    result = values.copy()
    result[np.abs(result) < threshold] = 0.0
    return result


def find_controller_enable_segments(df, min_duration=10):
    if 'controller_enable' not in df.columns:
        print("警告: 数据中不存在 'controller_enable' 列，返回全部数据")
        return [(0, len(df) - 1)]
    
    is_enabled = df['controller_enable'] == 1
    enabled_segments = find_continuous_segments(is_enabled, min_length=min_duration)
    
    if not enabled_segments:
        print("警告: 未找到 controller_enable==1 的片段")
        return []
    
    return enabled_segments


def find_combined_segments(df, min_zero_duration=10, min_controller_duration=10):
    """
    找出同时满足以下条件的连续片段:
    1. controller_enable==1 (有效控制信号)
    2. 动态状态 (非静止)
    3. vy 无跳变 (|Vy_mps| <= 2 m/s, 跳变帧及前后各 3 帧排除)
    4. dt 正常 (0.005 <= dt <= 0.039, 异常帧排除并在该处分段)

    参数:
        df: 输入 DataFrame
        min_zero_duration: 判定为连续静止的最小行数
        min_controller_duration: 判定为有效控制的最小行数

    返回:
        [(start1, end1), (start2, end2), ...] 满足条件的片段起止索引
    """
    if 'controller_enable' not in df.columns:
        print("警告: 数据中不存在 'controller_enable' 列，回退到静止分段模式")
        return find_dynamic_segments(df, min_zero_duration=min_zero_duration)
    
    is_static = (
        (df['Vx_mps'] == 0) &
        (df['Vy_mps'] == 0)
    )
    
    is_enabled = df['controller_enable'] == 1

    vy_jump = np.abs(df['Vy_mps'].values) > 2.0 # 跳变阈值为 2 m/s
    n_dilate = 3
    vy_jump_dilated = vy_jump.copy()
    for i in range(len(vy_jump)):
        if vy_jump[i]:
            lo = max(0, i - n_dilate)
            hi = min(len(vy_jump) - 1, i + n_dilate)
            vy_jump_dilated[lo:hi+1] = True

    vy_jump_count = int(np.sum(vy_jump))
    if vy_jump_count > 0:
        print(f"  检测到 {vy_jump_count} 帧 vy 跳变 (|vy|>2 m/s)，已排除跳变帧及前后各 {n_dilate} 帧")

    dt = df['Time_s'].diff()
    dt_bad = (dt > 0.039) | (dt < 0.005)
    dt_bad.iloc[0] = False

    dt_bad_count = int(np.sum(dt_bad))
    if dt_bad_count > 0:
        print(f"  检测到 {dt_bad_count} 帧 dt 异常 (dt<0.005 或 dt>0.039)，已排除并在异常处分段")

    is_valid = is_enabled & ~is_static & ~vy_jump_dilated & ~dt_bad
    
    valid_segments = find_continuous_segments(is_valid, min_length=max(min_zero_duration, min_controller_duration))
    
    if not valid_segments:
        print("警告: 未找到满足条件的片段 (controller_enable==1 AND 动态 AND vy无跳变)")
        return []
    
    return valid_segments


def find_dynamic_segments(df, min_zero_duration=10):
    is_static = (
        (df['Vx_mps'] == 0) &
        (df['Vy_mps'] == 0) 
    )

    static_segments = find_continuous_segments(is_static, min_length=min_zero_duration)

    if not static_segments:
        return [(0, len(df) - 1)]

    dynamic_segments = []
    prev_end = -1

    for static_start, static_end in static_segments:
        if static_start > prev_end + 1:
            dynamic_segments.append((prev_end + 1, static_start - 1))
        prev_end = static_end

    if prev_end < len(df) - 1:
        dynamic_segments.append((prev_end + 1, len(df) - 1))

    dynamic_segments = [
        seg for seg in dynamic_segments
        if (seg[1] - seg[0] + 1) >= min_zero_duration
    ]

    return dynamic_segments


def find_continuous_segments(bool_series, min_length=10):
    segments = []
    in_segment = False
    start_idx = 0

    for i, is_true in enumerate(bool_series):
        if is_true and not in_segment:
            in_segment = True
            start_idx = i
        elif not is_true and in_segment:
            in_segment = False
            segment_length = i - start_idx
            if segment_length >= min_length:
                segments.append((start_idx, i - 1))

    if in_segment:
        segment_length = len(bool_series) - start_idx
        if segment_length >= min_length:
            segments.append((start_idx, len(bool_series) - 1))

    return segments


def segment_only(input_csv, output_dir=None, min_zero_duration=10, process_timestamp=False):
    df = pd.read_csv(input_csv)

    if process_timestamp:
        df = process_timestamps(df)
    input_path = Path(input_csv)

    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    dynamic_segments = find_dynamic_segments(df, min_zero_duration)

    if not dynamic_segments:
        print(f"警告: 文件 {input_path.name} 未找到有效的动态片段！")
        return

    print(f"文件 {input_path.name} 找到 {len(dynamic_segments)} 个动态片段")

    for i, (start_idx, end_idx) in enumerate(dynamic_segments):
        segment_df = df.iloc[start_idx:end_idx+1]
        output_path = output_dir / f"{input_path.stem}_segment_{i+1:03d}.csv"
        segment_df.to_csv(output_path, index=False)
        print(f" 分段 {i+1}: 行 {start_idx}~{end_idx} (共 {end_idx-start_idx+1} 行) -> {output_path.name}")


def convert_multiple_csv(input_files, output_dir=None, segment_static_data=True, segment_mode='combined',
                         zero_threshold=1e-7, min_zero_duration=10, min_controller_duration=10, process_timestamp=False):
    for input_file in input_files:
        input_path = Path(input_file)

        if output_dir:
            base_output_path = Path(output_dir) / f"{input_path.stem}_train.csv"
            output_path = base_output_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            base_output_path = input_path.parent / f"{input_path.stem}_train.csv"

        print(f"\n处理文件 {input_file}")
        try:
            convert_csv(
                str(input_path),
                str(base_output_path),
                segment_static_data=segment_static_data,
                segment_mode=segment_mode,
                zero_threshold=zero_threshold,
                min_zero_duration=min_zero_duration,
                min_controller_duration=min_controller_duration,
                process_timestamp=process_timestamp
            )
        except Exception as e:
            print(f"转换失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='转换 CSV 文件格式 (含 vy 跳变过滤)')
    parser.add_argument('input', nargs='?', help='输入的 CSV 文件路径 (单文件模式)')
    parser.add_argument('output', nargs='?', help='输出的 CSV 文件路径 (单文件模式)')
    parser.add_argument('-m', '--multiple', nargs='+', help='批量处理多个输入文件, 输出到各自目录下')
    parser.add_argument('-o', '--output-dir', help='与 -m 配合使用, 指定统一的输出目录')

    parser.add_argument('--segment-static', action='store_true', default=True, help='按数据进行分段 (默认启用)')
    parser.add_argument('--no-segment', action='store_true', help='关闭分段，直接输出全部数据')
    parser.add_argument('--segment-mode', type=str, default='combined', choices=['static', 'controller', 'combined'],
                        help='分段模式: combined=组合分段(controller_enable==1 AND 非静止, 默认), static=仅按静止分段, controller=仅按controller_enable分段')
    parser.add_argument('--only-segment', action='store_true', help='仅按静止数据进行分段, 不进行数据转换')
    parser.add_argument('--zero-threshold', type=float, default=1e-7, help='小量置零阈值, 默认 1e-7')
    parser.add_argument('--min-zero-duration', type=int, default=10, help='判定为连续静止的最小行数, 默认 10')
    parser.add_argument('--min-controller-duration', type=int, default=10, help='判定为有效控制的最小行数, 默认 10')
    parser.add_argument('--process-timestamp', action='store_true', help='启用时间戳归零处理')

    args = parser.parse_args()

    if args.only_segment and args.no_segment:
        parser.error("--only-segment 和 --no-segment 不能同时使用")
    
    segment_data = not args.no_segment

    if args.multiple:
        if args.only_segment:
            for input_file in args.multiple:
                segment_only(
                    input_file,
                    args.output_dir,
                    args.min_zero_duration,
                    process_timestamp=args.process_timestamp
                )
        else:
            convert_multiple_csv(
                args.multiple,
                args.output_dir,
                segment_static_data=segment_data,
                segment_mode=args.segment_mode,
                zero_threshold=args.zero_threshold,
                min_zero_duration=args.min_zero_duration,
                min_controller_duration=args.min_controller_duration,
                process_timestamp=args.process_timestamp
            )
    elif args.input and args.output:
        if args.only_segment:
            segment_only(
                args.input,
                args.output,
                args.min_zero_duration,
                process_timestamp=args.process_timestamp
            )
        else:
            convert_csv(
                args.input,
                args.output,
                segment_static_data=segment_data,
                segment_mode=args.segment_mode,
                zero_threshold=args.zero_threshold,
                min_zero_duration=args.min_zero_duration,
                min_controller_duration=args.min_controller_duration,
                process_timestamp=args.process_timestamp
            )
    else:
        parser.print_help()
