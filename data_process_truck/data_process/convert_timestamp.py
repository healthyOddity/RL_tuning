import csv
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def convert_timestamp(ts):
    """将Unix时间戳转换为北京时间字符串"""
    utc_time = datetime.fromtimestamp(ts, datetime.timezone.utc)
    # utc_time = datetime.utcfromtimestamp(float(ts))用法已弃用，因此更新
    beijing_time = utc_time + timedelta(hours=8)
    return beijing_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 保留毫秒


def process_timestamps(df):
    """
    处理DataFrame中的时间列（Time_s）
    1. 计算时间基准（第一行时间戳）
    2. 应用时间戳归零（当前时间戳 - 时间基准）
    3. 添加北京时间列
    """
    # 提取时间列
    timestamps = df['Time_s'].values

    # 计算时间基准
    time_base = timestamps[0] if len(timestamps) > 0 else 0.0

    # 创建新DataFrame副本
    processed_df = df.copy()

    # 应用时间戳归零
    processed_df['Time_s'] = timestamps - time_base

    # 添加北京时间列
    processed_df['Beijing_Time'] = [convert_timestamp(ts) for ts in timestamps]

    return processed_df


def main():
    parser = argparse.ArgumentParser(description='将CSV文件中的Unix时间戳转换为北京时间')
    parser.add_argument('input', nargs='?', help='输入CSV文件路径')
    parser.add_argument('output', nargs='?', help='输出CSV文件路径(可选)')

    args = parser.parse_args()

    # 如果没有提供输入路径，提示用户输入
    if not args.input:
        args.input = input("请输入CSV文件路径: ")

    # 设置默认输出路径
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_with_time{input_path.suffix}")

    # 存储原始时间戳和行数据
    original_timestamps = []
    rows_data = []

    # 处理CSV文件
    with open(args.input, 'r') as infile:
        reader = csv.reader(infile)

        # 读取标题行
        headers = next(reader)
        headers.append('Beijing_Time')

        # 处理每一行数据
        for row in reader:
            if len(row) > 0:  # 确保行不为空
                try:
                    timestamp = row[0]  # Time_s是第一列
                    # 保存原始时间戳和行数据
                    original_timestamps.append(float(timestamp))
                    rows_data.append(row)
                except (ValueError, IndexError):
                    # 跳过无效行
                    continue

    # 计算时间基准（第一行时间戳）
    if original_timestamps:
        time_base = original_timestamps[0]
    else:
        time_base = 0.0

    # 写入新文件
    with open(args.output, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)

        # 处理每一行数据，应用时间戳归零
        for i, row in enumerate(rows_data):
            try:
                # 修改时间戳：从0开始，保持增量不变
                row[0] = str(original_timestamps[i] - time_base)

                # 使用原始时间戳转换北京时间
                beijing_time = convert_timestamp(original_timestamps[i])
                row.append(beijing_time)
                writer.writerow(row)
            except (ValueError, IndexError):
                # 跳过无效行
                continue

    print(f"转换完成！新文件已保存至: {args.output}")


if __name__ == "__main__":
    main()