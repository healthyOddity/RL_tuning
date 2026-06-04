# grad_tune 使用说明

## 2026-06-04 补充：record refline 与梯度报告

当前 `grad_tune` 默认使用：

```text
--plant truck_trailer
--refline-source record
```

`--plant` 仍然可以在命令行直接配置；本轮实车微调流程只验证
`truck_trailer`。`--refline-source record` 会用实车 `x/y/yaw/ego_v`
构造仿真 refline：`s` 由实车轨迹弧长累计得到，`kappa` 由实车 yaw
对弧长差分得到，`a` 由实车速度对时间差分得到。保留
`--refline-source csv` 仅用于排查 CSV 原始 `ref_x/ref_y/ref_s/ref_v/ref_a`
与实车轨迹的差异。

每次运行会额外输出：

```text
results/grad_tune/<run_id>/gradient_report.yaml
```

`gradient_report.yaml` 主要用于人工检查，说明文字以中文为主；YAML key
保留英文以方便程序读取。报告记录反传 loss 表达式、speed proxy 表达式、
学习率、梯度裁剪比例、每个开放参数的原始梯度、裁剪后梯度、原始更新量、
限幅更新量、物理投影差值、更新前值和更新后值。

`grad_tune` 用于从实车 record/CSV 中自动提取有效片段，基于实车误差构造
proxy-gradient loss，跑一次可微仿真并输出一版微调后的 DC 控制参数。

当前验收路径是：

```text
record 或 *_interpolated.csv
  -> 默认预处理，筛选有效运动片段并保留全部原始列
  -> 提取 refline 和实车 error
  -> truck_trailer/其他 plant 仿真
  -> 用实车 error 加权仿真误差反传
  -> 保存 tuned YAML
  -> 可选 baseline vs tuned 评估和可视化
```

## 运行环境

从 `train_file/differentiable-control/sim` 目录运行：

```powershell
conda activate pypose
```

不要直接用系统 Python；当前依赖在 `pypose` 环境中。

## 输入数据

推荐输入是 `extract_topic_truck_trq.py` 生成的原始插值 CSV：

```text
--input-csv path\to\*_interpolated.csv
```

也可以预留 record 输入：

```text
--record-dir path\to\record_dir
```

`--record-dir` 会调用：

```text
data_process_truck/data_process/extract_topic_truck_trq.py
```

如果一个 record 目录产生多个 CSV，需要改用 `--input-csv` 明确指定其中一段。

## 必要信号

CSV 至少需要包含以下信息，支持原始列名或内部别名：

- 时间与使能：`timestamp`, `controller_enable`
- 实车状态：`position_enu.x`, `position_enu.y`, `euler_angles.z`
- 实车速度：`debug.simple_lon_debug.current_speed`
- 实车误差：`debug.simple_lat_debug.lateral_error`,
  `debug.simple_lat_debug.heading_error`,
  `debug.simple_lon_debug.speed_error`
- 参考线：`ref_x`, `ref_y`, `ref_theta`, `ref_kappa`,
  `ref_s`, `ref_v`, `ref_a`

说明：

- `heading_error` 按弧度处理。
- `euler_angles.z` 如果数值看起来是角度，会自动转成弧度。
- 如果存在 `vy` 或 `Vy_mps`，预处理会用它做横向速度跳变过滤。

## 默认预处理

主流程默认执行预处理，不需要额外开关。预处理只筛选有效行，保留原始 CSV 的
全部列，不会像模型训练 CSV 转换那样裁掉 refline/error/debug 信号。

默认模式是：

```text
--preprocess-mode combined
```

筛选条件：

- `controller_enable == 1`
- 车辆或参考线处于运动状态：
  `max(abs(ego_v), abs(ref_v)) > --min-motion-speed`
- 采样间隔正常：`0.005 <= dt <= 0.039`
- 如果有 `vy`/`Vy_mps`，排除 `abs(vy) > --vy-jump-threshold` 的跳变点及前后若干帧
- 选择最长的连续有效片段

常用预处理参数：

```text
--preprocess-mode combined|controller|none
--min-motion-speed 0.05
--zero-threshold 1e-7
--vy-jump-threshold 2.0
--jump-dilate-frames 3
--min-duration 2.0
```

模式含义：

- `combined`：默认，控制使能 + 非静止 + dt 正常 + 可选 vy 跳变过滤。
- `controller`：只要求控制使能和 dt 正常，保留静止段。
- `none`：不做预处理，仅用于排查问题。

报告中的 `preprocess` 字段会记录实际选中的行号和 warning，例如：

```yaml
preprocess:
  mode: combined
  selected_start_index: 308
  selected_end_index: 1930
  selected_rows: 1623
  total_rows: 2808
  warnings: []
```

## 一步微调

示例：

```powershell
python -m grad_tune.one_step_tune `
  --input-csv C:\path\to\20260417_154100.00000_interpolated.csv `
  --config configs\default.yaml `
  --plant truck_trailer `
  --refline-source record `
  --param-mode all `
  --output-dir results\grad_tune `
  --tuned-dir configs\tuned `
  --window-duration 6.0 `
  --eval-after
```

`--param-mode` 可选：

- `all`：横向和纵向控制参数都参与一次梯度更新。
- `lat_offset`：只更新横向控制参数。
- `lon_speed`：只更新纵向控制参数。

更新会做梯度裁剪、单参数变化比例限制和物理范围投影；plant 和 MLP 权重不会被更新。

## Loss 逻辑

调参阶段报告用实车 error 计算真实 loss：

```text
w_lat   * lateral_error_record^2
w_head  * heading_error_record^2
w_speed * speed_error_record^2
```

反向传播时使用 proxy-gradient 形式：

```text
2*w_lat   * lateral_error_record.detach() * lateral_error_sim
2*w_head  * heading_error_record.detach() * heading_error_sim
2*w_speed * speed_error_record.detach()   * speed_error_sim
```

平顺性项不使用 record 的 steer/acc rate，而是使用仿真输出：

```text
w_steer_rate * steer_rate_sim^2
w_acc_rate   * acc_rate_sim^2
```

这样可以让梯度来自当前 DC controller + plant，同时 loss 方向由实车 error 指定。

## 输出文件

一步微调会输出：

```text
results/grad_tune/<run_id>/tuned_*.yaml
results/grad_tune/<run_id>/grad_tune_report.yaml
results/grad_tune/<run_id>/grad_tune_report.txt
results/grad_tune/<run_id>/gradient_report.yaml
configs/tuned/tuned_*.yaml
```

使用 `--eval-after` 时还会输出：

```text
results/grad_tune/<run_id>/evaluation/grad_tune_eval.yaml
results/grad_tune/<run_id>/evaluation/comparison_trajectory.png
results/grad_tune/<run_id>/evaluation/comparison_lateral_error.png
results/grad_tune/<run_id>/evaluation/comparison_speed_error.png
results/grad_tune/<run_id>/evaluation/comparison_steer.png
results/grad_tune/<run_id>/evaluation/comparison_acc.png
```

轨迹图图例：

- `refline`：CSV 中的参考线
- `record`：实车实际轨迹
- `baseline`：微调前参数仿真轨迹
- `tuned`：微调后参数仿真轨迹

## 单独评估

已有 tuned YAML 时，可以单独评估：

```powershell
python -m grad_tune.evaluate `
  --input-csv C:\path\to\20260417_154100.00000_interpolated.csv `
  --baseline-config configs\default.yaml `
  --tuned-config configs\tuned\tuned_xxx.yaml `
  --plant truck_trailer `
  --output-dir results\grad_tune_eval
```

`evaluate` 也默认执行同一套预处理，确保评估窗口和调参窗口一致。

## 注意事项

- 默认会跳过静止起步段，避免 truck_trailer 低速/零速附近的非期望微小位移污染评估。
- 如果需要复现实验中的原始全窗口行为，可加 `--preprocess-mode none`。
- 如果 config 中 `truck_trailer_vehicle.checkpoint_path` 存在，会直接使用该 checkpoint；
  如果路径不存在，会尝试回退到
  `configs/checkpoints/best_truck_trailer_error_model.pth`。
- 输出的 tuned YAML 只是离线候选参数，不会自动部署到实车。
