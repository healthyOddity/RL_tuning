# Record 数据预处理工具使用说明

## 工具概述

本目录包含六个用于 Apollo Cyber Record 数据预处理的脚本工具：

1. **extract_topic_truck_trq.py** - Record 文件读取和数据提取（主脚本）
2. **csvdata_new_truck.py** - CSV 格式转换和动态片段分割
3. **classify_and_split.py** - 场景分类与文件拆分（按场景 + 速度档位归类）
4. **pipeline.py** - 全自动化数据预处理流水线（一键执行全流程）
5. **convert_timestamp.py** - 时间戳格式转换工具
6. **generate_statistics.py** - 训练数据统计报告生成（自动调用）

---

## 1. extract_topic_truck_trq.py - Record 数据提取

### 功能说明
从 Apollo Cyber Record 文件中提取指定的 topic 数据，进行线性插值、低通滤波、坐标转换等处理，导出为 CSV 格式。

注意：本脚本调用了csvdata_new_truck.py脚本，用于将提取出的csv文件转化成模型训练所需的格式。

### 核心处理流程
1. **Record 读取**：使用 `cyber_record.record.Record` 读取 .record 文件
2. **Topic 过滤**：只处理 TARGET_FIELDS 中定义的目标 topic
3. **时间戳对齐**：以 `/rina/control_debug` 的时间戳为基准
4. **线性插值**：对不同频率的 topic 数据进行插值对齐
5. **角度处理**：特殊处理 360 度边界跳变（如 euler_angles.z）
6. **速度计算**：基于 ENU 坐标计算横向速度 vy
7. **扭矩计算**：根据档位传动比计算轮端扭矩，并应用低通滤波
8. **角速度计算**：计算 yaw 角速度 diff_yawrate

### 提取的 Topic 和字段

```python
TARGET_FIELDS = {
    "/rina/localization/global_pose": [
        "position_enu.x", "position_enu.y", 
        "euler_angles.z", "heading"
    ],
    "/rina/s2s/VehicleInfoToADU": [
        "VehicleInfoBData.BDCFF10D0_VCU_VehicleSpeed",
        "VehicleInfoBData.BD18F009B_VDC2_YawRate",
        "VehicleInfoADData.AD18F009B_SteeringWheelAngle",
        "VehicleInfoADData.AD18FFE213_EHPS_SteerWheelAng",
        "VehicleInfoADData.AD18F101D0_TCU_CurrentGear"
    ],
    "/rina/control_debug": [
        "debug.simple_lon_debug.station_error",
        "debug.simple_lon_debug.speed_error",
        "debug.simple_lat_debug.lateral_error",
        "debug.simple_lat_debug.heading_error",
        "steering_target",
        "target_torque",
        "path_remain"
    ],
    "/rina/perception/planner": [
        "ref_kappa", "ref_dkappa", "ref_s",
        "ref_theta", "ref_x", "ref_y",
        "ref_v", "ref_a"
    ],
    "/rina/decctrloutput/AlgSysStateInfo": [
        "AlgSysState.TopStateData_Struct.ADS_State"
    ]
}
```

### 输出 CSV 列说明

| 列名 | 说明 | 单位 |
|------|------|------|
| timestamp_ms | 毫秒级时间戳 | ms |
| timestamp | 秒级时间戳（从 0 开始） | s |
| controller_enable | 控制器使能信号（ADS_State==4 时为 1） | - |
| position_enu.x/y | ENU 坐标系位置 | m |
| euler_angles.z | 航向角（yaw） | deg |
| VehicleInfoBData.BDCFF10D0_VCU_VehicleSpeed | 车速 | km/h |
| VehicleInfoADData.AD18FFE213_EHPS_SteerWheelAng | 方向盘转角 | deg |
| steering_target | 目标方向盘转角 | deg |
| target_torque | 目标扭矩 | Nm |
| torque_wheel | 轮端扭矩（档位传动比计算后） | Nm |
| torque_filtered | 滤波后的轮端扭矩 | Nm |
| torque_fl/fr/rl/rr | 四轮扭矩分配 | Nm |
| vy | 横向速度（车辆坐标系） | m/s |
| vx_veh | 纵向速度（车辆坐标系） | m/s |
| diff_yawrate | 角速度 | deg/s |
| ref_* | 规划参考线数据 | - |

### 命令行参数

```bash
python extract_topic_truck_trq.py input_path [--output-dir OUTPUT_DIR] 
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| input_path | str | 是 | 输入文件路径 **或** 目录路径 |
| --output-dir | str | 否 | 输出目录路径（默认：当前目录） |

### ⚠️ 重要注意事项

#### 参数互斥说明
- **无互斥参数** - 该脚本只有两个简单参数
- `input_path` 可以是：
  - **单个 .record 文件**：输出到 `--output-dir` 指定目录
  - **目录路径**：递归处理目录下所有 `.record` 和 `.record.*` 文件，输出到各自所在目录

#### input/output 含义
- **input_path**：
  - 单文件模式：`C:\data\test.record`
  - 批量模式：`C:\data\record_folder\`（递归处理子目录）
- **output-dir**：
  - 单文件模式：指定输出 CSV 的目录
  - 批量模式：**被忽略**，输出到各 record 文件所在目录

### 使用示例

#### 示例 1：处理单个 record 文件
```bash
# 激活环境
conda activate pypose

# 处理单个文件，输出到当前目录
python extract_topic_truck_trq.py C:\data\test.record

# 处理单个文件，输出到指定目录
python extract_topic_truck_trq.py C:\data\test.record --output-dir C:\output\csv_files
```

#### 示例 2：批量处理目录下所有 record 文件
```bash
# 递归处理目录下所有 .record 文件（包括子目录）
python extract_topic_truck_trq.py C:\data\record_folder\
```

#### 示例 3：处理带序号的 record 文件
```bash
# 自动识别 .record.000, .record.001 等分片文件
python extract_topic_truck_trq.py C:\data\multi_segment.record
```

---

## 2. csvdata_new_truck.py - CSV 格式转换

### 功能说明
将 `extract_topic_truck_trq.py` 生成的 CSV 转换为训练所需的格式，支持：
- 列重命名和重组
- 小量值置零（阈值过滤）
- **组合分段（默认）**：同时基于 controller_enable==1 和静止/运动状态分段
- 时间戳处理（可选）

### 核心处理流程
1. **列映射**：将原始列名映射为训练脚本友好的名称
2. **小量置零**：将绝对值小于阈值的 vx, vy, yawrate 置为 0
3. **分段处理**（默认启用）：
   - **combined 模式（默认）**：筛选 controller_enable==1 且非静止的数据
   - **static 模式**：仅按静止状态分段（兼容旧逻辑）
   - **controller 模式**：仅按 controller_enable==1 分段
4. **片段输出**：每个有效片段保存为独立的 CSV 文件

### 输出 CSV 列说明

| 原始列名 | 新列名 | 说明 |
|---------|--------|------|
| timestamp_ms | Time_original | 原始毫秒时间戳 |
| timestamp | Time_s | 秒级时间戳（从 0 开始） |
| steering_target | Steer_deg_cmd / Target_Steer_L1_deg_cmd | 方向盘转角 |
| torque_fl/fr/rl/rr | Torque_L1/R1/L2/R2_Nm_cmd | 四轮扭矩 |
| VehicleSpeed | Vx_mps | 纵向速度（km/h → m/s） |
| vy | Vy_mps | 横向速度 |
| position_enu.x/y | X_m / Y_m | ENU 位置 |
| euler_angles.z | Yaw_deg | 航向角 |
| YawRate | Yawrate_degps | 偏航角速度（弧度→角度） |
| diff_yawrate | YawRate_diff | 角速度（微分计算） |

### 命令行参数

```bash
# 单文件模式
python csvdata_new_truck.py input_csv output_csv [选项]

# 批量模式
python csvdata_new_truck.py -m file1.csv file2.csv ... [-o OUTPUT_DIR] [选项]
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| input | str | 条件必填 | 输入 CSV 文件路径（单文件模式） |
| output | str | 条件必填 | 输出 CSV 文件路径（单文件模式） |
| -m, --multiple | list | 条件必填 | 批量处理多个文件（与 input/output 互斥） |
| -o, --output-dir | str | 否 | 批量模式的统一输出目录 |
| --no-segment | flag | 否 | 关闭分段，直接输出全部数据 |
| --segment-mode | str | 否 | 分段模式：combined(默认), static, controller |
| --only-segment | flag | 否 | 仅分段，不进行数据转换（与 --no-segment 互斥） |
| --zero-threshold | float | 否 | 小量置零阈值（默认：1e-7） |
| --min-zero-duration | int | 否 | 判定为连续静止的最小行数（默认：10） |
| --min-controller-duration | int | 否 | 判定为有效控制的最小行数（默认：10） |
| --process-timestamp | flag | 否 | 启用时间戳归零处理 |

### 分段模式详解

| 模式 | 筛选条件 | 适用场景 |
|------|---------|---------|
| **combined** (默认) | controller_enable==1 AND (vx≠0 OR vy≠0) | 训练数据（推荐） |
| static | vx≠0 OR vy≠0 | 兼容旧逻辑 |
| controller | controller_enable==1 | 保留静止控制数据 |

### ⚠️ 重要注意事项

#### 参数互斥说明

**互斥组 1：输入模式**
- ❌ **不能同时使用** `input/output` 和 `-m/--multiple`
- ✅ **正确用法**：
  ```bash
  # 单文件模式
  python csvdata_new_truck.py input.csv output.csv
  
  # 批量模式
  python csvdata_new_truck.py -m file1.csv file2.csv file3.csv
  ```

**互斥组 2：分段模式**
- ❌ **不能同时使用** `--no-segment` 和 `--only-segment`
- ✅ **正确用法**：
  ```bash
  # 转换 + 分段（默认行为）
  python csvdata_new_truck.py input.csv output.csv
  
  # 转换 + 不分段
  python csvdata_new_truck.py input.csv output.csv --no-segment
  
  # 仅分段（不转换列名）
  python csvdata_new_truck.py input.csv output.csv --only-segment
  ```

#### input/output 含义差异

**单文件模式**：
- `input`：**必须是文件路径**（如 `data.csv`）
- `output`：**必须是文件路径**（如 `output_train.csv`）

**批量模式**：
- `-m`：文件列表（如 `file1.csv file2.csv`）
- `-o`：**目录路径**（可选，默认输出到各输入文件的同级目录）

### 使用示例

#### 示例 1：单文件转换（默认组合分段）
```bash
# 默认行为：使用 combined 模式分段（controller_enable==1 AND 非静止）
python csvdata_new_truck.py input.csv output.csv

# 转换 + 小量置零（自定义阈值）
python csvdata_new_truck.py input.csv output.csv --zero-threshold 1e-5
```

#### 示例 2：单文件转换 + 不分段
```bash
# 转换但不分段，输出全部数据
python csvdata_new_truck.py input.csv output.csv --no-segment
```

#### 示例 3：使用不同分段模式
```bash
# 使用 static 模式（兼容旧逻辑）
python csvdata_new_truck.py input.csv output.csv --segment-mode static

# 使用 controller 模式（保留静止控制数据）
python csvdata_new_truck.py input.csv output.csv --segment-mode controller

# 自定义最小有效行数
python csvdata_new_truck.py input.csv output.csv --min-controller-duration 20
```

#### 示例 4：批量转换（默认组合分段）
```bash
# 批量处理多个 CSV，默认使用 combined 模式分段
python csvdata_new_truck.py -m file1.csv file2.csv file3.csv
```

#### 示例 5：批量转换（统一输出目录）
```bash
# 批量处理，统一输出到指定目录
python csvdata_new_truck.py -m file1.csv file2.csv -o C:\output\converted\
```

#### 示例 6：仅分段（不转换列名）
```bash
# 只对 CSV 进行动态片段分割，不改变列名
python csvdata_new_truck.py input.csv output.csv --only-segment

# 批量仅分段
python csvdata_new_truck.py -m file1.csv file2.csv --only-segment
```

---

## 3. convert_timestamp.py - 时间戳转换

### 功能说明
将 CSV 文件中的 Unix 时间戳（秒）转换为：
1. **归零时间戳**：从 0 开始的相对时间（Time_s 列）
2. **北京时间**：可读的日期时间字符串（Beijing_Time 列）

### 核心处理流程
1. 读取 CSV 第一列（Time_s）
2. 提取第一行时间戳作为基准
3. 所有时间戳减去基准（归零处理）
4. 将原始 Unix 时间戳转换为北京时间（UTC+8）
5. 添加 Beijing_Time 列到 CSV

### 输出格式

| 原始列 | 处理后 | 说明 |
|--------|--------|------|
| Time_s | Time_s | 归零后的时间（秒，从 0 开始） |
| - | Beijing_Time | 北京时间字符串（YYYY-MM-DD HH:MM:SS.mmm） |

### 命令行参数

```bash
python convert_timestamp.py [input_csv] [output_csv]
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| input | str | 否 | 输入 CSV 文件路径（不提供则提示输入） |
| output | str | 否 | 输出 CSV 文件路径（不提供则自动生成） |

### 默认输出路径
如果未提供 `output` 参数，自动生成：
```
{输入目录}/{输入文件名}_with_time.{扩展名}
```

### 使用示例

#### 示例 1：交互式（不提供参数）
```bash
python convert_timestamp.py
# 提示：请输入 CSV 文件路径：C:\data\test.csv
# 自动输出到：C:\data\test_with_time.csv
```

#### 示例 2：指定输入文件
```bash
python convert_timestamp.py C:\data\test.csv
# 自动输出到：C:\data\test_with_time.csv
```

#### 示例 3：指定输入和输出
```bash
python convert_timestamp.py C:\data\test.csv C:\output\converted.csv
```

---

## 4. pipeline.py — 全自动化数据预处理流水线

### 概述

`pipeline.py` 串联四个预处理脚本，实现一键全流程：

```
Record 文件 → [Step 1] 插值 CSV → [Step 2] 场景分类拆分 → [Step 3] 训练格式转换 → [统计] 生成报告
```

### 依赖脚本

| 步骤 | 脚本 | 功能 |
|------|------|------|
| Step 1 | `extract_topic_truck_trq.py` | `.record` → `*_interpolated.csv` |
| Step 2 | `classify_and_split.py` | 预过滤有效控制段 → 按场景标签边界拆分到 `1_scenes/{scene}_{speed}/` |
| Step 3 | `csvdata_new_truck.py -m --no-segment` | 列映射 + 小量置零，输出训练格式 CSV 到 `2_train/{scene}_{speed}/` |
| 统计 | `generate_statistics.py` | 自动生成训练数据统计报告（Step 3 完成后自动调用） |

### Step 2 新行为说明

- **预过滤**：调用 `find_combined_segments()`（从 `csvdata_new_truck.py` 导入），先排除 `controller_enable=0` 或车辆静止的无效段落，仅对有效控制段进行场景分类。
- **场景段拆分**：不再为整个文件分配一个主导场景标签，而是按场景标签边界将轨迹拆分为多个场景段。一条轨迹可能产出多个场景 CSV（如 `stem_turn1.csv`、`stem_lane_change1.csv`、`stem_straight1.csv`）。
- **场景 CSV 保留原始列名**（来自 `extract_topic_truck_trq.py` 的输出格式），供下游兼容使用。

### Step 3 新行为说明

- 使用 `--no-segment` 模式调用 `csvdata_new_truck.py`，仅进行**列映射 + 小量置零**，不再重复分段（分段已在 Step 2 完成）。
- 输出到独立目录 `2_train/`，与 `1_scenes/` 完全分离。

### 统计报告生成

- Step 3 完成后 **自动调用** `generate_statistics.py`，生成统计报告。
- 输出文件：
  - `2_train/statistics_report.txt` — 文本格式摘要
  - `2_train/statistics_report.png` — 可视化图表
- 报告内容：
  - 各步骤运行耗时
  - 每个场景 / 速度档位的总时长
  - 每个类别的动态指标占比（Vx_accel、Vy、Yawrate、Steer angle）
- 可通过 `--no-stats` 跳过统计生成
- 也可独立运行：`python generate_statistics.py 2_train/`

### 目录约定

```
{output_root}/
  0_interpolated/          ← Step 1 输出：原始插值 CSV
  1_scenes/                ← Step 2 输出：场景 CSV（原始列名，仅有效控制段）
    turn_low/              ← 转弯 + 低速（Vx < 30km/h）
      stem_turn1.csv
      stem_turn1_trajectory.png
    turn_mid/              ← 转弯 + 中速（30 ≤ Vx < 60km/h）
    turn_high/             ← 转弯 + 高速（Vx ≥ 60km/h）
    lane_change_low/
    lane_change_mid/
    lane_change_high/
    straight_low/
    straight_mid/
    straight_high/
  2_train/                 ← Step 3 输出：训练格式 CSV（单独目录）
    turn_low/
      stem_turn1_train.csv
    turn_mid/              ← 与 1_scenes/ 子目录结构相同
    ...
    statistics_report.txt  ← 自动生成的统计报告（文本）
    statistics_report.png  ← 自动生成的统计图表
```

### 命令行参数

```bash
python pipeline.py <record_dir> [options]

位置参数：
  record_dir            Record 文件所在目录（递归搜索 *.record 及 *.record.*）

可选参数：
  --output-root DIR       输出根目录（默认：record_dir 的上级目录）
  --skip-step1            跳过 Step 1（提取插值 CSV）
  --skip-step2            跳过 Step 2（场景分类拆分）
  --skip-step3            跳过 Step 3（生成训练片段）
  --force-step1           强制重跑 Step 1，即使产出已存在
  --only-step1            仅执行 Step 1
  --only-step2            仅执行 Step 2
  --only-step3            仅执行 Step 3
  --no-stats              跳过统计报告生成（统计仅在完整流水线模式下自动生成）
  --smooth-window N       场景标签平滑窗口大小（默认 5，透传给 classify_and_split.py）
  --speed-low N.N         低速档位上限 m/s（默认 8.33，≈30km/h）
  --speed-mid N.N         中速档位上限 m/s（默认 16.67，≈60km/h）
  --min-segment-duration N.N  最小场景段时长（秒，默认 2.0）
```

### 使用示例

#### 示例 1：全流程一键执行
```bash
python pipeline.py C:\data\records\ --output-root C:\data\output\
```
- 产出：`0_interpolated/` + `1_scenes/` + `2_train/`（含 `statistics_report.txt` + `statistics_report.png`）

#### 示例 2：增量处理（只跑新增步骤）
```bash
# 已完成 Step 1，继续 Step 2+3
python pipeline.py C:\data\records\ --output-root C:\data\output\ --skip-step1
```

#### 示例 3：跳过统计生成
```bash
python pipeline.py C:\data\records\ --output-root C:\data\output\ --no-stats
```

#### 示例 4：只跑场景分类（不转换训练格式）
```bash
python pipeline.py C:\data\records\ --output-root C:\data\output\ --skip-step3
```

#### 示例 5：仅跑 Step 3 转换已有场景数据
```bash
python pipeline.py </path/to/anywhere> --output-root C:\data\output\ --only-step3
```
> 注意：`--only-step3` 模式下不会自动生成统计报告，如需统计请单独运行 `generate_statistics.py`。

#### 示例 6：仅跑 Step 2 重新分类
```bash
python pipeline.py </path/to/anywhere> --output-root C:\data\output\ --only-step2
```

#### 示例 7：调整场景分类参数
```bash
python pipeline.py C:\data\records\ --smooth-window 7 --speed-low 6.0 --speed-mid 14.0 --min-segment-duration 3.0
```

---

## 5. classify_and_split.py — 场景分类与文件拆分

`classify_and_split.py` 由 `pipeline.py` 的 Step 2 自动调用，也可单独使用。

### 功能说明

- 读取 `*_interpolated.csv` 文件
- **预过滤**：调用 `find_combined_segments()`（从 `csvdata_new_truck.py` 导入），排除 `controller_enable=0` 或车辆静止的无效段落，仅对有效控制段进行后续分类
- 调用 `classify_scenario_direct()` 逐行进行场景标记（依赖 `../classify_scenario_from_trajectory.py`）
- 滑动窗口平滑标签（默认 window=5）
- **场景段拆分**：按场景标签边界将轨迹拆分为多个连续段，一条轨迹可产出多个场景 CSV（如 `stem_turn1.csv`、`stem_lane_change1.csv`、`stem_straight1.csv`）
- 基于 `ego_v` 中位数判定速度档位
- 场景映射：`constant_curve`→`turn`，`lane_change`→`lane_change`，其他→`straight`
- 过滤掉时长不足 `--min-segment-duration` 秒的场景段
- 将场景 CSV 保存到 `{output_dir}/{scene}_{speed}/` 子目录，**保留原始列名**（来自 `extract_topic_truck_trq.py` 输出格式）

### 命令行参数

```bash
python classify_and_split.py <input_dir> <output_dir> [options]

位置参数：
  input_dir             包含 *_interpolated.csv 的目录
  output_dir            输出根目录（将在其下创建 {scene}_{speed}/ 子目录）

可选参数：
  --smooth-window N         场景标签平滑窗口大小（默认 5）
  --speed-low N.N           低速档位上限 m/s（默认 8.33，≈30km/h）
  --speed-mid N.N           中速档位上限 m/s（默认 16.67，≈60km/h）
  --min-segment-duration N.N 最小场景段时长（秒，默认 2.0）
```

### 使用示例

#### 示例 1：基本用法
```bash
python classify_and_split.py C:\data\0_interpolated\ C:\data\1_scenes\
```

#### 示例 2：调整阈值
```bash
python classify_and_split.py C:\data\0_interpolated\ C:\data\1_scenes\ --smooth-window 7 --speed-low 6.0 --speed-mid 14.0
```

#### 示例 3：设置最小场景段时长
```bash
python classify_and_split.py C:\data\0_interpolated\ C:\data\1_scenes\ --min-segment-duration 3.0
```

### 依赖

- 父目录中的 `classify_scenario_from_trajectory.py`（提供 `classify_scenario_direct` 和 `smooth_scenario_labels`）
- 同目录中的 `csvdata_new_truck.py`（提供 `find_combined_segments` 函数）
- `../config/signal_config.yaml`（列名映射）
- `../config/threshold_config.yaml`（场景阈值）

---

## 6. 配置文件参考

场景分类和信号映射由 `../config/` 目录下的 YAML 配置文件管理。

### signal_config.yaml — 信号列名映射

定义脚本如何从 CSV 列名映射到统一的内部信号名称：

```yaml
# 自车实际状态
x: position_enu.x           # 自车实际 X 坐标
y: position_enu.y           # 自车实际 Y 坐标
yaw: euler_angles.z         # 自车实际航向角

# 横向控制相关
lateral_error: debug.simple_lat_debug.lateral_error  # 横向跟踪误差 e_y
heading_error: debug.simple_lat_debug.heading_error  # 航向跟踪误差 e_psi

# 参考轨迹
ref_kappa: ref_kappa        # 参考轨迹曲率
ref_x: ref_x                # 参考轨迹 X 坐标
ref_y: ref_y                # 参考轨迹 Y 坐标
```

完整清单含 **40+ 信号映射**，涵盖：
- **实际状态**：位置、航向、速度、加速度
- **参考轨迹**：曲率、位置、速度、加速度
- **控制状态**：使能信号、误差、前馈/反馈
- **事件标记**：变道、弯道、跟车、停车等

### threshold_config.yaml — 场景阈值

定义场景分类所需的阈值参数：

```yaml
lateral:                     # 横向性能阈值
  ey_rms_soft: 0.12
  overshoot_high: 0.25

longitudinal:                # 纵向性能阈值
  jerk_rms_high: 1.80
  stop_over_threshold: 0.35

scenario:                    # 场景分类阈值
  min_duration: 3.0
  curve_kappa_threshold: 0.003          # 弯道判定最小曲率
  curve_kappa_point_threshold: 0.005    # 逐点弯道判定曲率
  curve_sustained_ratio: 0.3            # 弯道持续占比
  lane_change_lat_offset_threshold: 0.6 # 变道横向偏移阈值
  stop_speed_threshold: 0.3             # 停车速度阈值（m/s）
```

---

## 7. 场景分类逻辑说明

场景分类由 `classify_scenario_from_trajectory.py` 实现，分为三个层级。

此外，在分类之前 `classify_and_split.py` 会先调用 `find_combined_segments()` 进行**有效控制段预过滤**：排除 `controller_enable=0` 或车辆静止（vx→0 且 vy→0）的段落，仅对 controller 使能且车辆在运动的有效段落进行后续场景分类。

### 7.1 逐点分类（Point-wise Classification）

`classify_scenario_direct()` 对每个数据点独立判断：

1. **弯道检测** — 局部窗口内 `ref_kappa` 均值 ≥ `curve_kappa_point`（0.005）且占比 ≥ 30%，标记为 `constant_curve`
2. **停车/低速检测** — `ego_v < stop_speed`（0.5 m/s），标记为 `follow_stop_go`
3. **速度变化检测** — 窗口内速度变化频繁（speed_changes ≥ 3）且存在真实低速（min_speed < 0.5），标记为 `follow_stop_go`
4. **变道检测** — 基于 `lateral_error` 偏移量判定：当 `lateral_error` 偏移 ≥ `lane_change_lat_offset`（0.6）且曲率小于弯道阈值时，标记为 `lane_change`
5. **默认** — 以上均不满足时标记为 `straight_hold`

### 7.2 段级修正（Segment-level Correction）

`classify_trajectory_pipeline()` 对连续轨迹段进行整体修正，优先级从高到低：

**标签覆盖优先级：**
```
lane_change > constant_curve > follow_stop_go > 原有逐点结果
```

| 优先级 | 分类函数 | 判断依据 |
|--------|---------|---------|
| 1 (最高) | `classify_lane_change()` | 曲率较小 + 横向偏移标准差大 |
| 2 | `classify_constant_curve()` | ref_kappa 均值 ≥ curve_kappa 阈值，或持续占比 ≥ sustained_ratio |
| 3 | `classify_follow_stop_go()` | 低速段占比高 + 有停车或速度频繁变化 |
| 4 (兜底) | — | 保留逐点分类结果 |

### 7.3 标签平滑（Label Smoothing）

`classify_scenario_from_trajectory.py` 第 3.3 节使用了 `smooth_scenario_labels()`：

- **多数投票**：以 `--smooth-window`（默认 5）为窗口，取窗口内频次最高的标签
- **作用**：消除标签的毛刺抖动，保证场景切换的稳定性

### 7.4 场景段拆分与速度档位映射

`classify_and_split.py` 将平滑后的标签按边界切分为多个场景段，然后将细粒度标签映射为最终场景名，并按速度中位数分档：

```
场景段拆分：
  按标签边界将轨迹切分为多个连续段（一个轨迹可能产出多个场景 CSV）

场景映射：
  constant_curve → turn
  lane_change    → lane_change
  其他（straight_hold / follow_stop_go） → straight

速度档位（基于 ego_v 中位数）：
  Vx < speed_low (8.33 m/s ≈ 30 km/h)  → {scene}_low
  speed_low ≤ Vx < speed_mid (16.67 m/s ≈ 60 km/h) → {scene}_mid
  Vx ≥ speed_mid                        → {scene}_high

最小段长过滤：
  丢弃时长 < --min-segment-duration（默认 2.0 秒）的场景段
```

最终目录结构：
```
1_scenes/
  turn_low/           ← 转弯 + 低速
    stem_turn1.csv
    stem_turn1_trajectory.png
  turn_mid/           ← 转弯 + 中速
  lane_change_low/    ← 变道 + 低速
  lane_change_mid/    ← 变道 + 中速
  straight_low/       ← 直行 + 低速
  straight_mid/       ← 直行 + 中速
  ...
```

### 7.5 事件标记（Event Columns）

`generate_event_columns()` 根据场景标签生成 6 个事件标记列：

| 事件列 | 触发条件 |
|--------|---------|
| event_lane_change | 该行标签为 lane_change |
| event_curve_enter | 该行标签为 constant_curve |
| event_curve_exit | 从 constant_curve 切换到其他场景的边界行 |
| event_follow_start | 该行标签为 follow_stop_go |
| event_stop_start | follow_stop_go 段内，速度跨过 0.3 m/s 下降沿 |
| event_stop_end | follow_stop_go 段结束后，速度跨过 0.3 m/s 上升沿 |

---

## 完整工作流示例

### 场景 1：全自动化流水线（推荐）
```bash
# 一键执行全流程：records → 场景分类 → 训练数据 → 统计报告
# 输出目录结构：
#   output/
#     0_interpolated/      ← Step 1：插值 CSV
#     1_scenes/            ← Step 2：场景 CSV（原始列名，按标签边界拆分）
#       turn_low/          ← 转弯 + 低速（<30km/h）
#         stem_turn1.csv
#         stem_turn1_trajectory.png
#       turn_mid/          ← 转弯 + 中速（30-60km/h）
#       turn_high/         ← 转弯 + 高速（≥60km/h）
#       lane_change_low/   ← 变道 + 低速
#       lane_change_mid/   ← 变道 + 中速
#       lane_change_high/  ← 变道 + 高速
#       straight_low/      ← 直行 + 低速
#       straight_mid/      ← 直行 + 中速
#       straight_high/     ← 直行 + 高速
#     2_train/             ← Step 3：训练格式 CSV + 统计报告
#       turn_low/
#         stem_turn1_train.csv
#       ...
#       statistics_report.txt
#       statistics_report.png

python pipeline.py C:\data\records\ --output-root C:\data\output\
```

### 场景 2：跳过统计报告
```bash
# 全流程但不生成统计
python pipeline.py C:\data\records\ --output-root C:\data\output\ --no-stats
```

### 场景 3：跳过已完成的步骤（增量处理）
```bash
# 已完成 Step 1（0_interpolated/ 已有 CSV），只跑 Step 2+3
python pipeline.py C:\data\records\ --output-root C:\data\output\ --skip-step1
```

### 场景 4：仅做场景分类（不转换训练格式）
```bash
# 只跑 Step 1 + Step 2，不生成 2_train/
python pipeline.py C:\data\records\ --output-root C:\data\output\ --skip-step3
```

### 场景 5：单个 record 文件处理（手动分步）
```bash
# 1. 提取 topic 数据（自动转换为训练格式，使用 combined 模式分段）
python extract_topic_truck_trq.py C:\data\test.record --output-dir C:\output\

# 输出：
# - C:\output\test_interpolated.csv（原始插值数据）
# - C:\output\test_train_segment_001.csv（第一个有效片段）
# - C:\output\test_train_segment_002.csv（第二个有效片段）
# ...
```

### 场景 6：批量处理多个 record 文件（手动分步）
```bash
# 1. 批量提取（递归处理子目录）
python extract_topic_truck_trq.py C:\data\record_folder\

# 2. 批量转换（统一输出目录，默认 combined 模式）
python csvdata_new_truck.py -m C:\data\record_folder\*_interpolated.csv -o C:\output\
```

### 场景 7：关闭分段处理
```bash
# 转换但不分段，输出完整数据
python csvdata_new_truck.py input.csv output.csv --no-segment
```

### 场景 8：仅处理时间戳
```bash
# 已有 CSV，只需添加北京时间列
python convert_timestamp.py C:\data\existing.csv
# 输出：C:\data\existing_with_time.csv
```

### 场景 9：仅生成统计报告（已有训练数据）
```bash
# 对已有的 2_train/ 目录独立生成统计报告
python generate_statistics.py C:\data\output\2_train\
# 产出：C:\data\output\2_train\statistics_report.txt + statistics_report.png
```

---

## 常见问题 FAQ

### Q1: ModuleNotFoundError: No module named 'cyber_record'
**A**: 需要在 pypose 环境中安装依赖：
```bash
conda activate pypose
pip install cyber_record
```

### Q2: ModuleNotFoundError: No module named 'convert_timestamp'
**A**: 这是本地文件导入问题，不是包安装问题。确保 `convert_timestamp.py` 与 `csvdata_new_truck.py` 在同一目录。

### Q3: 如何调整静止检测的灵敏度？
**A**: 使用 `--zero-threshold` 和 `--min-zero-duration` 参数：
```bash
# 更宽松（更容易判定为静止）
python csvdata_new_truck.py input.csv output.csv --zero-threshold 1e-5 --min-zero-duration 5

# 更严格（更难判定为静止）
python csvdata_new_truck.py input.csv output.csv --zero-threshold 1e-8 --min-zero-duration 20
```

### Q4: 输出的 CSV 文件在哪里？
**A**: 
- 单文件模式：`--output-dir` 指定的目录（默认当前目录）
- 批量模式：各 record/csv 文件所在的同级目录

### Q5: 如何处理分片的 record 文件（.record.000, .record.001）？
**A**: `extract_topic_truck_trq.py` 自动识别 `.record` 和 `.record.*` 格式：
```bash
python extract_topic_truck_trq.py C:\data\multi.record
# 自动处理 multi.Record.000, multi.Record.001, ...
```

### Q6: 统计报告没有生成？
**A**: 统计报告仅在**完整流水线模式**（非 `--only-stepN`）下自动生成。如需对已有训练数据单独生成统计：
```bash
python generate_statistics.py 2_train/
```

---

## 脚本改进建议

### 建议 1：统一配置文件
**问题**：TARGET_FIELDS 硬编码在脚本中，修改不便
**建议**：提取到 YAML/JSON 配置文件
```yaml
# config.yaml
topics:
  /rina/localization/global_pose:
    - position_enu.x
    - position_enu.y
  /rina/control_debug:
    - debug.simple_lat_debug.heading_error
```

### 建议 2：增加进度条
**问题**：处理大文件时无法感知进度
**建议**：使用 `tqdm` 库显示进度条
```python
from tqdm import tqdm
for topic, message, timestamp_ns in tqdm(record.read_messages(), desc="Processing"):
    ...
```

### 建议 3：增加日志模块
**问题**：使用 print 输出，无法控制日志级别
**建议**：使用 Python logging 模块
```python
import logging
logging.info("处理完成")
logging.warning("缺少某字段")
logging.error("文件不存在")
```

### 建议 4：增加数据验证
**问题**：无法检测异常数据（如 NaN、inf）
**建议**：在输出前进行数据质量检查
```python
# 检查 NaN/inf
if np.isnan(value) or np.isinf(value):
    logging.warning(f"检测到异常值：{field_path}={value}")
```

### 建议 5：增加可视化功能
**问题**：无法快速预览数据质量
**建议**：添加 `--plot` 选项，生成数据曲线图
```bash
python csvdata_new_truck.py input.csv output.csv --plot
# 生成 steering_angle.png, speed.png 等
```

### 建议 6：参数验证增强
**问题**：参数错误时提示不够友好
**建议**：在 argparse 后增加自定义验证
```python
if args.segment_static and args.only_segment:
    parser.error("--segment-static 和 --only-segment 不能同时使用")
```

### 建议 7：增加缓存机制
**问题**：重复处理相同 record 文件效率低
**建议**：基于文件 hash 的缓存
```python
import hashlib
file_hash = hashlib.md5(open(record_file, 'rb').read()).hexdigest()
if os.path.exists(f"cache/{file_hash}.csv"):
    # 使用缓存
```

---

## 依赖包清单

```bash
# 核心依赖
pip install cyber_record
pip install record_msg  # 可选，用于简化操作

# 数据处理
pip install numpy
pip install scipy
pip install pandas

# 工具库
pip install tqdm  # 进度条（建议）
pip install matplotlib  # 可视化（建议）
```

---

## 版本信息

- **脚本版本**: 2.0
- **最后更新**: 2026-05-15
- **Python 版本**: 3.11+
- **测试环境**: Windows 10/11, conda pypose 环境

---

## 8. BLF CAN 数据接入（新增）

### 8.1 audit_can_signals.py - DBC/BLF 信号审计

用于在正式转换前回答三个问题：

- DBC 中定义了哪些 message/signal
- BLF 中实际出现了哪些 CAN ID，频率和时间范围如何
- 当前训练链路需要的字段能否由 DBC+BLF 覆盖

输出文件：

| 文件 | 说明 |
|------|------|
| dbc_signal_inventory.csv | DBC 中所有 message/signal 清单 |
| blf_message_coverage.csv | BLF 中实际出现的 CAN ID、帧数、频率、解码状态 |
| target_signal_coverage.csv | 当前训练目标字段覆盖情况 |
| missing_signal_report.md | 缺失信号报告，按 Blocking/Degraded/Optional 分组 |
| decoded_sample.csv | 少量解码样例 |

使用示例：

```bash
python audit_can_signals.py --dbc vehicle.dbc --blf data.blf --output-dir audit_out
```

只审计 DBC：

```bash
python audit_can_signals.py --dbc vehicle.dbc --output-dir audit_out
```

### 8.2 BLF 转训练 CSV

当前 BLF 链路分两步执行，先把 `.blf + .dbc` 解码为按时间戳对齐的 raw CSV，再把 raw CSV 转成现有车辆模型训练格式。

Step 1：解码 BLF 到 raw CSV：

```bash
python extract_blf_raw_signals.py --dbc vehicle_dbc_dir --blf data.blf --output out/raw_aligned.csv --long-output out/raw_long.csv --sample-period-ms 10 --max-gap-ms 50
```

Step 2：转成 `*_interpolated.csv` 和训练 CSV：

```bash
python convert_blf_raw_to_train.py --raw out/raw_aligned.csv --interpolated-output out/data_blf_interpolated.csv --train-output out/data_blf_train.csv --sample-period-ms 20
```

该转换保持 50Hz 输出，`VCU_VehicleCtrlMod==3` 视为 `controller_enable=1`；`total_motor_request_torque_nm` 会按 `current_gear` 对应传动比换算为轮边扭矩后再按后轮平分。

如果 BLF 中包含 `INS_A` 定位报文，默认会提取 `INS570D_INS_Latitude`、`INS570D_INS_Longitude`、`INS570D_INS_HeadingAngle`、`INS570D_INS_PitchAngle`、`INS570D_INS_RollAngle`、`INS570D_INS_VBx`、`INS570D_INS_VBy`。经纬度按 WGS84 椭球转换为 ENU 坐标，原点在天津 `(117.064613, 39.068178)` 与北京 `(116.714101539, 40.181167491)` 两个候选中用 haversine 距离自动选择最近点；`HeadingAngle` 作为 `Yaw_deg`，`VBy` 作为 `Vy_mps`，`Pitch/Roll/VBx/VBy` 会作为补充列追加到未分段训练 CSV 末尾。

### 8.3 映射配置

默认映射配置位于：

```text
config/blf_signal_mapping.yaml
```

需要先根据真实 DBC 审计结果填写各字段的 `message_name` 和 `signal_name`。普通底盘 CAN 通常只能覆盖车速、横摆角速度、方向盘角、档位等信号；定位、ADU 控制目标、ADS 状态、规划参考线和 `control_debug` 信号需要确认 BLF 中是否存在对应报文。

### 8.4 依赖说明

BLF 链路需要额外依赖：

```bash
pip install cantools python-can
```

注意：不要在未确认环境要求前直接安装依赖。若运行脚本时报缺少 `cantools` 或 `python-can`，请先与环境维护者确认后再安装。
