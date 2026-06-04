# Grad Tune Codex Session Handoff

日期：2026-06-04

本文档总结本次 Codex session 中关于 `grad_tune` 实车数据梯度微调方案的讨论、实现、排查结论和当前可用命令，方便新开 session 后快速接续。

## 1. 总目标

目标是在 `train_file/differentiable-control` 的 `dev/grad_tune` 分支中实现一条自动化微调链路：

```text
输入一段实测 record 或已提取的 *_interpolated.csv
  -> 预处理，筛选有效实车片段
  -> 提取 refline、实车 lateral/heading/speed error
  -> 用实车 refline 初始化 DC 仿真
  -> 通过 truck_trailer plant/MLP + controller 可微链路获得参数梯度
  -> 用实车 error 定义 proxy-gradient loss
  -> 做一次 bounded gradient step
  -> 输出 tuned YAML 和完整 result
```

核心思想不是“让 tuned 参数在代理仿真里一定更好”，而是：

```text
实车 error 表示真实系统偏差；
代理模型提供 参数 -> 仿真误差 的局部可微敏感度；
用 real_error * d(sim_error)/d(param) 做一阶参数微调。
```

因此 `evaluation` 里的仿真 loss 只能作为 sanity check 和可视化参考，不能作为实车微调是否必然有效的唯一标准。

## 2. 当前主要代码

实现集中在：

```text
sim/grad_tune/
```

关键文件：

```text
sim/grad_tune/one_step_tune.py      # 一步微调主流程
sim/grad_tune/evaluate.py           # baseline vs tuned 评估
sim/grad_tune/record_adapter.py     # CSV/record 读取，sample 构造
sim/grad_tune/preprocess.py         # 默认预处理，有效片段筛选
sim/grad_tune/refline_builder.py    # 从实车 refline 构造 TrajectoryPoint
sim/grad_tune/real_loss.py          # 实车 error proxy-gradient loss
sim/grad_tune/param_policy.py       # 参数冻结、裁剪、步长限制、物理投影
sim/grad_tune/plotting.py           # 轨迹和误差可视化
sim/grad_tune/README.md             # 中文使用说明
```

相关测试：

```text
sim/tests/test_grad_tune_refline.py
sim/tests/test_grad_tune_loss.py
sim/tests/test_grad_tune_param_policy.py
sim/tests/test_grad_tune_smoke.py
sim/tests/test_grad_tune_evaluate.py
```

## 3. 输入数据与 record 支持

当前首选输入是 `extract_topic_truck_trq.py` 生成的原始插值 CSV：

```text
--input-csv path/to/*_interpolated.csv
```

也预留了 record 输入：

```text
--record-dir path/to/record_dir
```

`record_dir_to_csv()` 会调用：

```text
data_process_truck/data_process/extract_topic_truck_trq.py
```

如果 record 目录产生多个 CSV，需要改用 `--input-csv` 明确指定某一段。

注意：不要直接调用 `csvdata_new_truck.py convert_csv()` 作为 grad_tune 输入，因为该脚本会转换成模型训练格式并裁掉 refline/debug/error 等 grad_tune 必要信号。我们只迁移了它的“有效片段筛选思想”，在 `grad_tune/preprocess.py` 中保留原始全列。

## 4. 默认预处理

早期发现原始 CSV 前面存在静止起步段，truck_trailer 在零速/低速附近会有非期望微小位移，导致轨迹图看起来像“起步反向走一段”。因此主流程默认先做预处理。

默认模式：

```text
--preprocess-mode combined
```

筛选条件：

- `controller_enable == 1`
- `max(abs(ego_v), abs(ref_v)) > --min-motion-speed`
- `0.005 <= dt <= 0.039`
- 如果有 `vy` 或 `Vy_mps`，排除 `abs(vy) > --vy-jump-threshold` 的跳变点及前后若干帧
- 选择最长连续有效片段

常用参数：

```text
--preprocess-mode combined|controller|none
--min-motion-speed 0.05
--zero-threshold 1e-7
--vy-jump-threshold 2.0
--jump-dilate-frames 3
--min-duration 2.0
```

在验证 CSV 上，默认预处理选中的窗口是：

```text
selected_start_index: 308
selected_end_index: 1930
window: 6.159963623046875s -> 12.159963623046874s
```

## 5. Loss 设计和关键修正

### 5.1 横向与航向

实车字段：

```text
debug.simple_lat_debug.lateral_error
debug.simple_lat_debug.heading_error
```

已确认：

- `heading_error` 单位是弧度。
- lateral/heading error 的符号与 DC 仿真公式基本一致。

proxy-gradient 形式：

```text
2 * w_lat  * lateral_error_record.detach() * lateral_error_sim
2 * w_head * heading_error_record.detach() * heading_error_sim
```

### 5.2 平顺性

不使用 record 的 steer rate / acc rate 作为 backward 平顺性项。

原因：微调目标是通过当前 controller + plant 的可微链路更新参数，平顺性项应约束 tuned 后仿真输出本身，而不是拟合实车历史控制动作。

当前使用：

```text
w_steer_rate * steer_rate_sim^2
w_acc_rate   * acc_rate_sim^2
```

### 5.3 纵向 speed error 最新修正

最初实现使用：

```text
speed_error_sim = v_sim - ref_v
```

后来确认实车 `debug.simple_lon_debug.speed_error` 的定义是：

```text
speed_error_record = ref_v - s_dot_matched

s_dot = v * cos(theta_vehicle - theta_ref) / (1 - k_ref * d)
```

其中：

- `d` 是 Frenet 横向偏差。
- `k_ref` 是匹配点参考曲率。
- `s_dot_matched` 是车辆在 Frenet 坐标系下沿参考轨迹的纵向速度分量。

因此已将 sim 侧 speed proxy 改为同口径：

```text
speed_error_sim = ref_v - s_dot_sim
s_dot_sim = v_sim * cos(heading_error_sim) / (1 - ref_kappa * lateral_error_sim)
```

代码位于：

```text
sim/grad_tune/real_loss.py
```

报告中会记录：

```yaml
speed_error_mode: frenet_ref_v_minus_s_dot
speed_denom_min: ...
speed_denom_max: ...
```

当前实现对分母做了保护：

```text
denom_safe = clamp(1 - ref_kappa * lateral_error_sim, min=0.2, max=5.0)
```

这是本轮最后一个关键修正。修正前，仿真 eval loss 在默认学习率下变大；修正后，在同一 CSV 上 eval loss 下降。

## 6. Evaluation 的定位

`evaluate.py` 会用同一段 refline 分别跑 baseline 和 tuned 参数，并输出：

```text
grad_tune_eval.yaml
comparison_trajectory.png
comparison_lateral_error.png
comparison_speed_error.png
comparison_steer.png
comparison_acc.png
```

轨迹图图例：

```text
refline  # CSV 参考线
record   # 实车实际轨迹
baseline # 微调前参数仿真轨迹
tuned    # 微调后参数仿真轨迹
```

重要判断：

- evaluation 是仿真条件下的 sanity check。
- 不能把 “tuned sim_loss 必须下降” 作为实车微调硬门禁。
- 但如果修正 error 口径后 sim evaluation 也改善，说明当前代理梯度和这个仿真指标方向至少没有明显冲突。

## 7. checkpoint 与 MLP

本轮确认：

```text
truck_trailer_vehicle.checkpoint_path
```

会被 `vehicle_factory.py` 解析并加载。

用过的 checkpoint：

```text
configs/checkpoints/best_truck_trailer_error_model_0525.pth
configs/checkpoints/best_truck_trailer_error_model_train_loss_0525.pth
```

结论：

- 普通 `best_truck_trailer_error_model_0525.pth` 在本段数据上表现更好。
- `train_loss_0525` 在本段数据上速度可能更接近，但横向偏差明显更大，整体不如普通 0525。
- 用户确认当前 MLP 结构与云端一致。
- 当前 truck_trailer MLP 使用 `LeakyReLU(negative_slope=0.02)`。

## 8. 已跑通的关键结果

### 8.1 预处理后但 speed error 未修正的结果

路径：

```text
sim/results/grad_tune_preprocessed/20260417_154100_00000_interpolated_20260603_191142/
```

当时 evaluation：

```text
baseline_sim_loss: 1.646704
tuned_sim_loss:    1.754377
loss_delta_pct:   +6.538677%
```

该结果促发了 speed error 口径排查。

### 8.2 Frenet speed error 修正后的最终结果

输入 CSV：

```text
C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\data_process\data\data_0512\data_0512\0_interpolated\20260417_154100.00000_interpolated.csv
```

输出 result：

```text
sim/results/grad_tune_frenet_speed/20260417_154100_00000_interpolated_20260604_110107/
```

完整报告：

```text
sim/results/grad_tune_frenet_speed/20260417_154100_00000_interpolated_20260604_110107/grad_tune_report.yaml
```

评估报告：

```text
sim/results/grad_tune_frenet_speed/20260417_154100_00000_interpolated_20260604_110107/evaluation/grad_tune_eval.yaml
```

标准 tuned 参数：

```text
sim/configs/tuned/tuned_948456e_20260604_110107.yaml
```

evaluation 指标：

```text
baseline_sim_loss: 1.646704
tuned_sim_loss:    1.543899
loss_delta_pct:   -6.243122%

lat_rmse_pct:     -0.682492%
head_rmse_pct:    -0.883895%
speed_rmse_pct:   -3.384606%
```

报告确认：

```text
speed_error_mode: frenet_ref_v_minus_s_dot
speed_denom_min: 1.0
speed_denom_max: 1.0003196001052856
```

### 8.3 测试

完整 grad_tune 测试命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CONDA_REPORT_ERRORS='false'
conda run -n pypose python -m pytest `
  sim\tests\test_grad_tune_refline.py `
  sim\tests\test_grad_tune_loss.py `
  sim\tests\test_grad_tune_param_policy.py `
  sim\tests\test_grad_tune_smoke.py `
  sim\tests\test_grad_tune_evaluate.py -q
```

最近结果：

```text
16 passed, 1 warning
```

warning 是 `.pytest_cache` 写入权限问题，不影响测试。

## 9. 当前推荐运行命令

从：

```text
train_file/differentiable-control/sim
```

运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CONDA_REPORT_ERRORS='false'
conda run -n pypose python -m grad_tune.one_step_tune `
  --input-csv C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\data_process\data\data_0512\data_0512\0_interpolated\20260417_154100.00000_interpolated.csv `
  --config C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim\configs\default.yaml `
  --plant truck_trailer `
  --param-mode all `
  --output-dir C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim\results\grad_tune_frenet_speed `
  --tuned-dir C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim\configs\tuned `
  --window-duration 6.0 `
  --eval-after
```

## 10. Git add 建议

不要盲目 `git add train_file/differentiable-control/data_process_truck` 整个目录。该目录中有很多既有脚本和可能无关文件。

如果要提交本轮 grad_tune 相关实现，建议至少包含：

```text
sim/grad_tune/
sim/tests/test_grad_tune_refline.py
sim/tests/test_grad_tune_loss.py
sim/tests/test_grad_tune_param_policy.py
sim/tests/test_grad_tune_smoke.py
sim/tests/test_grad_tune_evaluate.py
sim/grad_tune/README.md
docs/grad_tune/specs/spec.md
docs/grad_tune/specs/checklist.md
docs/grad_tune/specs/tasks.md
docs/grad_tune/session_summary_20260603.md
docs/grad_tune/session_handoff_20260604.md
```

如果 record -> CSV 能力需要在干净 checkout 中可用，还需要选择性加入：

```text
data_process_truck/data_process/extract_topic_truck_trq.py
```

以及它实际 import 依赖的必要文件。不要直接加入整个 `data_process_truck` 目录。

## 11. 后续建议

短期：

1. 用更多实车 CSV 跑 `grad_tune_frenet_speed` 流程，验证 Frenet speed 口径修正是否稳定改善。
2. 对 tuned 参数做实车 replay 或实车小范围试验，验证真实 error 是否下降。
3. 如果某些场景 evaluation 仍上升，不要直接判定失败，先检查 real error 与 sim error 口径是否一致。

中期：

1. 支持 multi-record batch gradient，降低单段数据偶然性。
2. 在报告中增加 record real loss before/after 的实车复跑对比入口。
3. 如果确认实车 speed_error 还包含滤波、站位误差或限幅，需要进一步把 sim 侧 speed proxy 对齐到同口径。

原则：

```text
实车微调阶段最重要的是 error 定义一致 + 代理模型局部敏感度方向可信；
仿真 evaluation 是辅助诊断，不是唯一验收指标。
```

## 12. 2026-06-04 追加：record refline 与梯度报告

### 12.1 本轮目标

根据计划文件：

```text
docs/plans/2026-06-04-grad-tune-observability-record-refline-plan.md
```

完成两项改进：

1. 在每次 `grad_tune` 结果目录中记录更完整的梯度与参数更新信息，包括反传表达式、学习率、梯度裁剪、每个参数的更新前值、原始梯度、裁剪后梯度、原始更新量、限幅更新量、物理投影差值和更新后值。
2. 新增 `--refline-source record`，默认使用实车 `x/y/yaw/ego_v` 构造仿真 refline，避免 CSV 中 `ref_x/ref_y/ref_s` 与实车轨迹在纵向 station 上不一致导致仿真 refline 不连续或明显偏离实车轨迹。

### 12.2 已完成

代码改动集中在：

```text
sim/grad_tune/data_schema.py
sim/grad_tune/record_adapter.py
sim/grad_tune/refline_builder.py
sim/grad_tune/observability.py
sim/grad_tune/real_loss.py
sim/grad_tune/param_policy.py
sim/grad_tune/one_step_tune.py
sim/grad_tune/evaluate.py
sim/grad_tune/plotting.py
sim/grad_tune/README.md
```

测试改动集中在：

```text
sim/tests/test_grad_tune_refline.py
sim/tests/test_grad_tune_loss.py
sim/tests/test_grad_tune_param_policy.py
sim/tests/test_grad_tune_smoke.py
sim/tests/test_grad_tune_evaluate.py
```

已实现能力：

- `GradTuneSample` 现在保留实车 `yaw` 和 `ego_v` 数组。
- `build_trajectory_points(sample, source='record')` 默认用实车轨迹构造 `TrajectoryPoint`：
  - `x/y` 来自实车位置；
  - `theta` 来自实车 yaw；
  - `s` 由实车轨迹弧长累计；
  - `kappa` 由 yaw 对弧长差分；
  - `v` 来自实车速度；
  - `a` 由实车速度对时间差分。
- `build_trajectory_points(sample, source='csv')` 保留原 CSV `ref_x/ref_y/ref_s/ref_v/ref_a` 行为，作为排查入口。
- `one_step_tune.py` 和 `evaluate.py` 新增：

```text
--refline-source record|csv
```

默认值为 `record`。

- `one_step_tune.py` 现在会输出：

```text
gradient_report.yaml
```

该报告以中文说明为主，保留英文 YAML key 方便程序读取。

- `compute_backward_loss()` 支持传入当前仿真 trajectory。使用 `record` refline 时，speed proxy 中的 `ref_v/ref_kappa` 来自仿真 trajectory，并在报告中记录：

```yaml
speed_refline_source: trajectory
```

### 12.3 关键决策

1. 本轮不再比较其它 plant，仍然保留现有 `--plant` 参数入口，但实际验证只跑 `truck_trailer`。
2. 默认 refline 从 CSV refline 改为 record refline，原因是本轮数据中发现 record 与 CSV refline 在纵向 station 上存在明显差异；用实车轨迹构造 refline 能保证仿真 refline 与实车轨迹连续且空间上接近。
3. `gradient_report.yaml` 面向人工检查，说明文字、公式解释和更新规则说明使用中文；字段名仍用英文，便于后续脚本读取。
4. 保留 `--refline-source csv`，用于复现旧流程和排查 CSV 原始 refline。
5. 当前 `real_loss` 仍按 CSV 中记录的实车 lateral/heading/speed error 计算；record refline 主要改变仿真所用轨迹和 speed proxy 的参考速度/曲率。后续若要让实车 error 也完全以 record refline 重算，需要另起任务明确口径。

### 12.4 本轮测试

focused grad_tune 测试命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CONDA_REPORT_ERRORS='false'
conda run -n pypose python -m pytest `
  sim\tests\test_grad_tune_refline.py `
  sim\tests\test_grad_tune_loss.py `
  sim\tests\test_grad_tune_param_policy.py `
  sim\tests\test_grad_tune_smoke.py `
  sim\tests\test_grad_tune_evaluate.py -q
```

结果：

```text
21 passed, 1 warning
```

warning 仍是 `.pytest_cache` 写入权限问题，不影响测试。

### 12.5 新 CSV 实跑结果

输入 CSV：

```text
C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\data_process\data\data_0512\data_0512\0_interpolated\20260413_185635_interpolated.csv
```

运行命令从 `train_file/differentiable-control/sim` 执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CONDA_REPORT_ERRORS='false'
conda run -n pypose python -m grad_tune.one_step_tune `
  --input-csv C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\data_process\data\data_0512\data_0512\0_interpolated\20260413_185635_interpolated.csv `
  --config C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim\configs\default.yaml `
  --plant truck_trailer `
  --refline-source record `
  --param-mode all `
  --output-dir C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim\results\grad_tune_record_refline `
  --tuned-dir C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim\configs\tuned `
  --window-duration 6.0 `
  --eval-after
```

结果目录：

```text
sim/results/grad_tune_record_refline/20260413_185635_interpolated_20260604_155718/
```

关键文件：

```text
sim/results/grad_tune_record_refline/20260413_185635_interpolated_20260604_155718/tuned_948456e_20260604_155718.yaml
sim/results/grad_tune_record_refline/20260413_185635_interpolated_20260604_155718/grad_tune_report.yaml
sim/results/grad_tune_record_refline/20260413_185635_interpolated_20260604_155718/gradient_report.yaml
sim/results/grad_tune_record_refline/20260413_185635_interpolated_20260604_155718/evaluation/grad_tune_eval.yaml
sim/configs/tuned/tuned_948456e_20260604_155718.yaml
```

报告确认：

```yaml
plant: truck_trailer
refline_source: record
speed_refline_source: trajectory
```

评价指标：

```text
baseline_sim_loss: 5.015672842552176
tuned_sim_loss:    4.961389013018595
loss_delta_pct:   -1.082284%

lat_rmse_pct:          +2.749330%
head_rmse_pct:         +0.521837%
speed_rmse_pct:        -0.323760%
steer_rate_rmse_pct:   -3.263810%
acc_rate_rmse_pct:     -0.545064%
```

解释：record refline 模式下总体仿真 evaluation loss 下降约 1.08%，主要来自 speed 与平顺性项改善；lateral/head RMSE 略有上升。evaluation 仍然只作为 sanity check 和诊断指标，不能单独作为实车微调是否有效的最终依据。

### 12.6 未完成与后续建议

未完成：

- 尚未把实车 lateral/heading/speed error 按 record refline 重新计算；当前仍使用 CSV 中已有 debug error。
- 尚未做 multi-record batch gradient。
- 尚未做实车 replay 或实车小范围验证。

后续建议：

1. 明确实车 debug error 的参考点语义。如果要彻底切换到 record refline，需要重新定义并计算 real lateral/heading/speed error。
2. 用更多 CSV 跑 `--refline-source record`，确认 gradient 方向和 evaluation 指标是否稳定。
3. 如果 `truck_trailer` MLP 仍出现明显纵向响应偏差，优先排查 MLP checkpoint 与当前数据分布、扭矩输入口径和车辆参数是否一致。
