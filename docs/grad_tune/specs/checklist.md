# 代理模型梯度微调验收 Checklist

## 输入与预处理

- [ ] 抽取后 CSV 能被入口识别。
- [ ] `--record-dir` 入口已预留；若 record 解析依赖不可用，给出明确错误并提示先运行 `extract_topic_truck_trq.py` 生成 CSV。
- [ ] 不执行任何 `pip install`、`conda install` 或其他安装操作。
- [ ] 必需字段存在：`timestamp`、`controller_enable`、`x`、`y`、`yaw`、`ego_v`、`lateral_error`、`heading_error`、`debug.simple_lon_debug.speed_error`、`ref_x`、`ref_y`、`ref_yaw/ref_theta`、`ref_kappa`、`ref_s`、`ref_v`、`ref_a`。
- [ ] `heading_error` 按弧度处理。
- [ ] `speed_error` 直接使用 `debug.simple_lon_debug.speed_error`。
- [ ] 有效窗口满足 `controller_enable == 1`。
- [ ] 若存在 `event_takeover`，有效窗口内没有接管。
- [ ] 时间戳单调递增。
- [ ] dt 异常帧被剔除或触发明确错误。
- [ ] 有效窗口时长不低于配置阈值。
- [ ] CSV 信号重采样到 `simulation.dt` 对齐时间轴。

## Refline 与仿真

- [ ] `ref_x/ref_y/ref_yaw/ref_kappa/ref_s/ref_v/ref_a` 被转换为 `TrajectoryPoint`。
- [ ] `ref_yaw` 缺失时能使用 `ref_theta`。
- [ ] 仿真初始状态来自有效窗口首帧。
- [ ] 使用 `plant=truck_trailer`。
- [ ] `run_simulation(..., differentiable=True)` 能产生 history。
- [ ] history 长度与对齐后的实车 error 长度一致，或有明确裁剪策略。

## Loss

- [ ] 报告用 `J_real_report` 使用实车三项 tracking error 平方计算。
- [ ] backward tracking 项使用：

```text
2*w_lat   * stopgrad(lateral_error_record) * lateral_error_sim
2*w_head  * stopgrad(heading_error_record) * heading_error_sim
2*w_speed * stopgrad(speed_error_record)   * speed_error_sim
```

- [ ] `speed_error_sim = v_sim - ref_v_aligned`。
- [ ] 平顺性项只使用仿真输出：

```text
w_steer_rate * mean(diff(steer_sim)^2)
w_acc_rate   * mean(diff(acc_sim)^2)
```

- [ ] 不使用 record 的 `steer_rate`、`acc_rate` 参与 backward。
- [ ] `loss_backward.backward()` 后开放参数存在非空梯度。

## 参数策略

- [ ] 支持 `lat_offset`、`lon_speed`、`all` 三种参数开放模式。
- [ ] `lat_offset`/`lon_speed` 模式下，非开放参数 `requires_grad=False`。
- [ ] `lat_offset`/`lon_speed` 模式下，开放参数 `requires_grad=True`。
- [ ] `all` 模式下，现有 DC 控制器可训练参数 `requires_grad=True`，plant/MLP 参数不参与更新。
- [ ] 梯度 NaN/Inf 被检测并拒绝输出候选参数。
- [ ] 梯度范数被记录。
- [ ] 梯度裁剪是否触发被记录。
- [ ] 单参数 delta 不超过配置上限，默认 `3% ~ 5%`。
- [ ] 参数物理边界投影生效。
- [ ] `lat_offset`/`lon_speed` 模式下，未开放参数输出前后完全一致。
- [ ] `all` 模式下，所有发生变化的参数均列入报告。

## 输出

- [ ] 生成 `tuned_<run_id>.yaml`。
- [ ] 生成 `grad_tune_report.yaml`。
- [ ] 生成 `grad_tune_report.txt`。
- [ ] 报告包含输入路径、配置路径、plant、窗口起止时间、样本数。
- [ ] 报告包含参数开放模式、开放参数名、参数更新前后值和 delta。
- [ ] 报告包含 `J_real_report`、`loss_backward`、各项 RMSE、梯度范数。
- [ ] 输出 YAML 可被 `load_config()` 加载。
- [ ] 最终实现说明列出新增文件、修改文件和建议 `git add` 路径。

## 测试

- [ ] `test_grad_tune_refline.py` 覆盖 refline 构造。
- [ ] `test_grad_tune_loss.py` 覆盖 backward loss 的梯度存在性。
- [ ] `test_grad_tune_param_policy.py` 覆盖冻结、裁剪、投影。
- [ ] `test_grad_tune_smoke.py` 使用小型 CSV fixture 跑通端到端。
- [ ] 测试命令在 `conda activate pypose` 环境下通过。

## 拒绝输出候选参数的条件

- [ ] 缺少必需字段。
- [ ] 无有效控制窗口。
- [ ] 有效窗口过短。
- [ ] 时间轴无法对齐。
- [ ] refline 点数不足。
- [ ] 仿真 history 为空。
- [ ] `loss_backward` 非有限值。
- [ ] 当前开放模式下参与更新的参数梯度全为空或全为非有限值。
- [ ] 参数更新后违反硬边界且无法投影回合法范围。
