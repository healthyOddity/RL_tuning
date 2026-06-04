# 代理模型梯度微调实施 Tasks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现输入已抽取 CSV 后，自动输出一版代理模型梯度 based 候选参数，并预留 record 目录输入接口。

**Architecture:** 新增 `sim/grad_tune` 包，复用现有 `DiffControllerParams`、`run_simulation`、`load_config`、`save_tuned_config`。数据适配、refline 构造、loss、参数策略和 CLI 编排分文件实现。

**Tech Stack:** Python、PyTorch、PyYAML、pandas、pytest、现有 DC `sim` 包。

---

## 文件结构

新增：

```text
sim/grad_tune/__init__.py
sim/grad_tune/data_schema.py
sim/grad_tune/record_adapter.py
sim/grad_tune/refline_builder.py
sim/grad_tune/real_loss.py
sim/grad_tune/param_policy.py
sim/grad_tune/one_step_tune.py
sim/tests/test_grad_tune_refline.py
sim/tests/test_grad_tune_loss.py
sim/tests/test_grad_tune_param_policy.py
sim/tests/test_grad_tune_smoke.py
```

复用：

```text
sim/optim/train.py
sim/sim_loop.py
sim/config.py
data_process_truck/data_process/extract_topic_truck_trq.py
data_process_truck/data_process/pipeline.py
data_process_truck/config/signal_config.yaml
```

## Task 1: 定义数据结构与字段校验

**Files:**

- Create: `sim/grad_tune/__init__.py`
- Create: `sim/grad_tune/data_schema.py`
- Test: `sim/tests/test_grad_tune_refline.py`

- [ ] 创建 `GradTuneSample` 数据结构，包含对齐后的时间轴、实车 error、refline 字段和初始状态。
- [ ] 定义必需字段列表，包含 `debug.simple_lon_debug.speed_error`。
- [ ] 实现 `validate_columns(df)`：缺字段时报出明确字段名。
- [ ] 实现 `select_yaw_column(df)`：优先 `ref_yaw`，否则 `ref_theta`。
- [ ] 写测试覆盖缺字段、`ref_yaw/ref_theta` fallback、`heading_error` 不做单位转换。
- [ ] 运行：`python -m pytest sim/tests/test_grad_tune_refline.py -q`。

## Task 2: 实现 CSV 到单工况样本的适配

**Files:**

- Create: `sim/grad_tune/record_adapter.py`
- Test: `sim/tests/test_grad_tune_smoke.py`

- [ ] 实现 `load_csv_sample(input_csv, cfg)`。
- [ ] 按 `controller_enable == 1` 截取连续有效窗口。
- [ ] 若存在 `event_takeover`，剔除接管窗口。
- [ ] 检查时间戳单调和 dt 范围。
- [ ] 重采样到 `cfg["simulation"]["dt"]`。
- [ ] 将 `debug.simple_lon_debug.speed_error` 映射为 `speed_error_record`。
- [ ] 初始状态取有效窗口首帧的 `x/y/yaw/ego_v`。
- [ ] 写小型 CSV fixture smoke test，覆盖完整加载流程。

## Task 3: 构造实车 Refline

**Files:**

- Create: `sim/grad_tune/refline_builder.py`
- Test: `sim/tests/test_grad_tune_refline.py`

- [ ] 实现 `build_trajectory_points(sample)`。
- [ ] 每一帧生成 `TrajectoryPoint(x, y, theta, kappa, v, a, s, t)`。
- [ ] 确保 `theta` 使用 rad。
- [ ] 对 refline 点数不足 2 的情况抛明确错误。
- [ ] 测试生成点数量、首末点值、`ref_theta` fallback。

## Task 4: 实现实车 Error 加权的 Backward Loss

**Files:**

- Create: `sim/grad_tune/real_loss.py`
- Test: `sim/tests/test_grad_tune_loss.py`

- [ ] 实现 `compute_real_report_loss(sample, weights)`。
- [ ] 实现 `compute_backward_loss(history, sample, weights)`。
- [ ] backward tracking 项使用：

```text
2*w_lat   * stopgrad(lateral_error_record) * lateral_error_sim
2*w_head  * stopgrad(heading_error_record) * heading_error_sim
2*w_speed * stopgrad(speed_error_record)   * speed_error_sim
```

- [ ] `speed_error_sim` 使用 `history["v"] - sample.ref_v`。
- [ ] 平顺性项只使用 `history["steer"]` 和 `history["acc"]`。
- [ ] 不读取、不使用 record 的 `steer_rate` 或 `acc_rate`。
- [ ] 测试 `.backward()` 后一个 dummy 参数能收到非空梯度。

## Task 5: 实现参数开放模式、冻结、裁剪和投影

**Files:**

- Create: `sim/grad_tune/param_policy.py`
- Test: `sim/tests/test_grad_tune_param_policy.py`

- [ ] 定义参数开放模式配置：`lat_offset`、`lon_speed`、`all`。
- [ ] 实现 `freeze_except(params, param_mode)`。
- [ ] 实现 `collect_open_parameters(params, param_mode)`。
- [ ] `all` 模式开放现有 DC 控制器可训练参数，但不开放 plant/MLP 参数。
- [ ] 实现梯度 NaN/Inf 检测。
- [ ] 实现全局梯度范数裁剪。
- [ ] 实现单参数 delta 百分比限制，默认 `0.05`。
- [ ] 复用现有物理边界：纵向增益非负、`switch_speed` 范围、横向时间参数非负。
- [ ] 测试 `lat_offset`/`lon_speed` 模式下非开放参数不更新，开放参数更新受限。
- [ ] 测试 `all` 模式下控制器参数可更新，且每个参数 delta 受限。

## Task 6: 实现 One-Step Tune CLI

**Files:**

- Create: `sim/grad_tune/one_step_tune.py`
- Test: `sim/tests/test_grad_tune_smoke.py`

- [ ] 实现 CLI 参数：

```text
--input-csv
--record-dir
--config
--plant
--param-mode
--output-dir
--max-delta-ratio
--w-lat
--w-head
--w-speed
--w-steer-rate
--w-acc-rate
```

- [ ] `--input-csv` 第一版必须支持。
- [ ] `--record-dir` 作为预留接口；依赖可用时先调用 `data_process_truck/data_process/extract_topic_truck_trq.py` 产出 CSV，再进入 `--input-csv` 同一流程。
- [ ] 当前环境无 record 或 record 解析依赖不可用时，`--record-dir` 报明确错误，并提示先用 `extract_topic_truck_trq.py` 生成 CSV。
- [ ] 加载配置并强制 `plant=truck_trailer` 或 CLI 指定 plant。
- [ ] 创建 `DiffControllerParams(cfg)`。
- [ ] 根据 `--param-mode` 冻结非开放参数；`all` 模式保留控制器参数可训练。
- [ ] 构造 refline 并调用 `run_simulation(..., differentiable=True)`。
- [ ] 计算 `J_real_report` 和 `loss_backward`。
- [ ] 执行一次 backward 和一次参数更新。
- [ ] 保存候选 YAML 与报告。

## Task 7: 报告输出

**Files:**

- Modify: `sim/grad_tune/one_step_tune.py`
- Test: `sim/tests/test_grad_tune_smoke.py`

- [ ] 输出目录格式：

```text
sim/results/grad_tune/<run_id>/
```

- [ ] 保存：

```text
tuned_<run_id>.yaml
grad_tune_report.yaml
grad_tune_report.txt
```

- [ ] 报告包含输入路径、配置路径、plant、窗口起止时间、样本数。
- [ ] 报告包含参数开放模式、开放参数名、参数前后值、delta、delta 百分比。
- [ ] 报告包含 `J_real_report`、`loss_backward`、RMSE、梯度范数、裁剪/投影状态。
- [ ] 报告包含本次实现新增、修改脚本清单，按可 `git add` 的相对路径列出。
- [ ] 测试输出文件存在且 YAML 可加载。

## Task 8: 端到端验证

**Files:**

- Test: `sim/tests/test_grad_tune_smoke.py`

- [ ] 准备一个最小 CSV fixture，包含必需字段和 2 秒以上数据。
- [ ] 运行：

```text
python -m pytest sim/tests/test_grad_tune_refline.py sim/tests/test_grad_tune_loss.py sim/tests/test_grad_tune_param_policy.py sim/tests/test_grad_tune_smoke.py -q
```

- [ ] 手动运行 CLI：

```text
python -m sim.grad_tune.one_step_tune \
  --input-csv <fixture_csv> \
  --config sim/configs/default.yaml \
  --plant truck_trailer \
  --param-mode all \
  --output-dir sim/results/grad_tune
```

- [ ] 验证输出 YAML 能被 `load_config()` 读取。
- [ ] 用 `--param-mode lat_offset` 验证未开放参数没有变化。
- [ ] 用 `--param-mode all` 验证发生变化的参数均满足 delta 限制。
- [ ] 验证 report 明确写出未使用 record 平顺性数据参与 backward。

## Task 9: 文档回链

**Files:**

- Modify: `docs/grad_tune/specs/spec.md`
- Modify: `docs/grad_tune/specs/checklist.md`
- Modify: `docs/grad_tune/specs/tasks.md`

- [ ] 若实现中参数块映射与 spec 不一致，更新 spec。
- [ ] 若验收命令变化，更新 checklist。
- [ ] 若新增必要文件，更新 tasks 文件结构。
- [ ] 不更新与本方案无关的文档。

## Task 10: 最终交付清单

**Files:**

- Modify: final response only

- [ ] 最终回复列出新增文件。
- [ ] 最终回复列出修改文件。
- [ ] 最终回复给出建议 `git add` 路径。
- [ ] 清单只包含本次实现相关文件，不包含仓库中已有的其他未跟踪文件。
