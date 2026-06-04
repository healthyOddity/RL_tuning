# 代理模型梯度 Based 实车 Record 微调方案 Spec

## 1. 背景与目标

现有 DC 框架已经具备可微闭环仿真能力：控制器参数 `theta` 通过 PyTorch `nn.Parameter` 暴露，`plant=truck_trailer` 可在闭环中产生 `lateral_error`、`heading_error`、速度、转向和加速度输出，并通过 autograd 反传到控制器参数。

本方案目标是在现场微调阶段输入一段实车数据，自动完成：

1. 从已抽取 CSV 中读取实车 refline、实车 tracking error 和初始状态；预留 record 目录输入接口。
2. 用实车 refline 驱动 `truck_trailer` plant 闭环仿真。
3. 用实车 tracking error 决定误差下降方向，用代理模型提供 error 对参数的梯度。
4. 执行一次小步、受限的梯度下降，输出一版候选参数 YAML。

第一版以 `extract_topic_truck_trq.py` 已抽取出的原始 CSV 作为主输入，因为当前项目环境没有可用 record。`--record-dir` 入口只做接口预留或在依赖可用时调用现有 data_process 流程。

验收标准是：**输入一段满足字段要求的抽取后 CSV，输出一版可回灌的候选参数配置，并生成本轮微调报告。**

## 2. Why：为什么不能直接替换 Loss

真实目标可以写成：

```text
J_real(theta) = sum_t [
  w_lat   * e_lat_real(t, theta)^2
+ w_head  * e_head_real(t, theta)^2
+ w_speed * e_speed_real(t, theta)^2
]
```

理想梯度为：

```text
dJ_real/dtheta =
sum_t 2*w_lat   * e_lat_real(t, theta)   * d e_lat_real(t, theta)   / dtheta
+     2*w_head  * e_head_real(t, theta)  * d e_head_real(t, theta)  / dtheta
+     2*w_speed * e_speed_real(t, theta) * d e_speed_real(t, theta) / dtheta
```

但 record 已经采集完成，程序里读到的是固定数组：

```text
e_lat_record(t), e_head_record(t), e_speed_record(t)
```

如果直接写：

```text
loss = sum_t w * e_record(t)^2
```

则 `loss` 与当前仿真中的 `theta` 没有可微计算链：

```text
theta -> controller -> plant -> sim_error
record_error -> loss
```

因此：

```text
d loss / d theta = 0
```

本方案采用一阶近似：

```text
d e_real(t, theta) / dtheta ~= d e_sim(t, theta) / dtheta
```

于是使用如下 backward surrogate：

```text
loss_backward =
sum_t [
  2*w_lat   * e_lat_record(t).detach()   * e_lat_sim(t)
+ 2*w_head  * e_head_record(t).detach()  * e_head_sim(t)
+ 2*w_speed * e_speed_record(t).detach() * e_speed_sim(t)
]
```

其梯度为：

```text
d loss_backward / dtheta =
sum_t [
  2*w_lat   * e_lat_record(t)   * d e_lat_sim(t)   / dtheta
+ 2*w_head  * e_head_record(t)  * d e_head_sim(t)  / dtheta
+ 2*w_speed * e_speed_record(t) * d e_speed_sim(t) / dtheta
]
```

这保留了实车误差方向，同时利用代理模型提供可微梯度通路。

## 3. 作用域

### 3.1 In Scope

- 单段或单工况抽取后 CSV 输入。
- `--record-dir` 输入接口预留；依赖可用时可先调用 `data_process_truck/data_process/extract_topic_truck_trq.py` 生成 CSV，再进入同一后处理流程。
- CSV 预处理、有效窗口截取、refline 构造。
- 使用 `truck_trailer` plant 闭环仿真。
- 一次梯度下降生成候选参数，不做自动多轮迭代。
- 支持横向优先、纵向优先两类参数块，以及 `all` 全量小步更新模式。
- 使用实车 `lateral_error`、`heading_error`、`speed_error` 三项参与 tracking error 梯度。
- 使用仿真输出的 `steer`、`acc` 做平顺性正则。
- 输出候选参数 YAML 和微调报告。

### 3.2 Out of Scope

- 不做参数自动下发实车。
- 不做 A/B 复测自动判定。
- 不做多工况联合优化。
- 不使用 record 的 `steer_rate`、`acc_rate` 参与 backward。
- 不在本阶段校准 plant 或训练新的 MLP residual。
- 不自动安装依赖；任何安装操作必须先询问作者。

## 4. 输入字段要求

### 4.1 必需字段

从 `extract_topic_truck_trq.py` 抽取后的 CSV 至少需要：

```text
timestamp
controller_enable
x
y
yaw
ego_v
lateral_error
heading_error
debug.simple_lon_debug.speed_error
ref_x
ref_y
ref_yaw 或 ref_theta
ref_kappa
ref_s
ref_v
ref_a
```

已确认约束：

- `debug.simple_lon_debug.speed_error` 可直接作为实车 `speed_error`。
- `heading_error` 单位为弧度。

### 4.2 可选字段

```text
event_takeover
acc_cmd
steer_out
yaw_rate
ego_a
station_error
preview_speed_error
```

这些字段可用于数据质量检查和报告，但第一版不要求参与 backward。

## 5. 数据处理规则

### 5.1 有效窗口

预处理必须截取连续有效窗口：

- `controller_enable == 1`
- 无人工接管，若 `event_takeover` 存在则要求为 0
- 时间戳单调递增
- dt 在合理范围内，建议 `0.005 <= dt <= 0.039`
- 窗口时长满足最低要求，建议不少于 2 秒

### 5.2 时间对齐

仿真步长使用配置中的 `simulation.dt`，默认 0.02 s。record 信号需要重采样到仿真时间轴：

```text
t_sim = 0, dt, 2*dt, ..., T
```

要求对齐后的数组长度与仿真 history 长度一致，或按二者较短长度裁剪。

### 5.3 Refline 构造

从 record 的 `ref_x/ref_y/ref_yaw/ref_kappa/ref_s/ref_v/ref_a` 构造 `TrajectoryPoint` 列表。字段单位要求：

- `ref_x/ref_y`: m
- `ref_yaw/ref_theta`: rad
- `ref_kappa`: 1/m
- `ref_s`: m
- `ref_v`: m/s
- `ref_a`: m/s^2

若 `ref_yaw` 缺失但 `ref_theta` 存在，使用 `ref_theta`。

## 6. Loss 设计

### 6.1 实车 Tracking Error 项

第一版使用三项实车误差：

```text
e_lat_record   = lateral_error
e_head_record  = heading_error
e_speed_record = debug.simple_lon_debug.speed_error
```

仿真对应项：

```text
e_lat_sim   = history["lateral_error"]
e_head_sim  = history["heading_error"]
e_speed_sim = history["v"] - ref_v_aligned
```

backward surrogate：

```text
loss_tracking_backward =
mean(
  2*w_lat   * stopgrad(e_lat_record)   * e_lat_sim
+ 2*w_head  * stopgrad(e_head_record)  * e_head_sim
+ 2*w_speed * stopgrad(e_speed_record) * e_speed_sim
)
```

### 6.2 平顺性项

平顺性只使用仿真输出：

```text
loss_smooth =
  w_steer_rate * mean(diff(steer_sim)^2)
+ w_acc_rate   * mean(diff(acc_sim)^2)
```

不使用 record 的 `steer_rate`、`acc_rate` 参与 backward。原因是 record 平顺性信号可能混入定位噪声、执行器反馈、CAN 延迟、外界扰动或接管边界；第一版只将平顺性作为候选参数在代理闭环中的正则约束。

### 6.3 总 Backward Loss

```text
loss_backward = loss_tracking_backward + loss_smooth
```

### 6.4 报告用 Real Loss

报告中仍需要计算实车 tracking loss 标量：

```text
J_real_report =
mean(
  w_lat   * e_lat_record^2
+ w_head  * e_head_record^2
+ w_speed * e_speed_record^2
)
```

该值只用于记录和验收，不直接 `.backward()`。

## 7. 参数更新策略

### 7.1 参数开放模式

第一版支持三种开放模式：

- `lat_offset`: 横向直道偏移/回正类工况。
- `lon_speed`: 纵向定速或速度阶跃类工况。
- `all`: 开放现有 DC 框架中已暴露的全部可训练控制器参数。

推荐默认先用 `lat_offset` 或 `lon_speed`，因为单工况下分块更新更便于归因和排查。但 `all` 模式在工程上可以接受，前提是严格执行单参数 delta 百分比限制、全局梯度裁剪、物理边界投影和报告审查。

`all` 模式的适用条件：

- 本轮目标是离线生成候选参数，不自动下发实车。
- 只执行一次 backward 和一次小步更新。
- 每个参数更新幅度默认不超过当前值的 `3% ~ 5%`。
- 报告必须列出全部发生变化的参数。

参数开放模式到具体 `nn.Parameter` 的映射应配置化，避免硬编码散落在训练逻辑中。

### 7.2 冻结规则

`lat_offset` 和 `lon_speed` 模式下，除开放参数块外，其他所有参数必须设置：

```python
param.requires_grad_(False)
```

开放参数块内参数保留 `requires_grad=True`。

`all` 模式下，保留所有现有 DC 可训练控制器参数的 `requires_grad=True`，但仍必须排除 plant/MLP 参数和任何非控制器参数。

### 7.3 梯度与步长限制

候选更新为：

```text
theta_candidate = project(theta_current - alpha * P * clipped_grad)
```

必须包含：

- 梯度 NaN/Inf 清理。
- 全局梯度范数裁剪。
- 单参数最大变化限制，默认不超过当前值的 `3% ~ 5%`。
- 参数物理边界投影，例如增益非负、时间参数非负、`switch_speed` 范围限制。

### 7.4 一次更新

第一版只执行一次 backward 和一次参数更新，不做多 epoch 训练。

## 8. 输出

每次运行必须输出：

```text
sim/results/grad_tune/<run_id>/
  tuned_<run_id>.yaml
  grad_tune_report.yaml
  grad_tune_report.txt
```

报告至少包含：

- 输入 CSV 路径；若由 record 入口触发，则同时记录 record 路径和中间 CSV 路径。
- 使用配置路径。
- plant 类型。
- 有效窗口起止时间和样本数。
- 开放参数块和开放参数名。
- `J_real_report`。
- `loss_backward` 数值。
- 各项 error RMSE。
- 梯度范数。
- 每个开放参数的更新前、更新后、delta、delta 百分比。
- 是否触发梯度裁剪或参数投影。
- 输出参数文件路径。
- 新增、修改的脚本清单，按 `git add` 可直接使用的相对路径列出。

## 9. CLI 设计

建议入口：

```text
python -m sim.grad_tune.one_step_tune \
  --input-csv <interpolated_or_segment_csv> \
  --config sim/configs/default.yaml \
  --plant truck_trailer \
  --param-mode all \
  --output-dir sim/results/grad_tune
```

预留 record 输入接口：

```text
python -m sim.grad_tune.one_step_tune \
  --record-dir <record_dir> \
  --config sim/configs/default.yaml \
  --plant truck_trailer \
  --param-mode lat_offset \
  --output-dir sim/results/grad_tune
```

第一版必须支持 `--input-csv`。`--record-dir` 在当前无 record 环境中可以只做接口预留；若 `cyber_record` 等 record 解析依赖不可用，应报明确错误，并提示先运行 `extract_topic_truck_trq.py` 生成 CSV。

## 10. 验收标准

功能验收：

1. 输入满足字段要求的抽取后 CSV 后，程序能自动生成候选参数 YAML。
2. 输出报告包含本轮窗口、loss、梯度、参数 delta 和输出路径。
3. `heading_error` 按 rad 处理。
4. `speed_error` 使用 `debug.simple_lon_debug.speed_error`。
5. backward tracking 项使用实车 `lateral_error/heading_error/speed_error` 作为 stopgrad 权重。
6. 平顺性项只使用仿真 `steer/acc`，不使用 record 的平顺性数据。
7. `lat_offset`/`lon_speed` 模式下非开放参数没有更新；`all` 模式下只允许控制器参数更新。
8. 每个发生更新的参数幅度不超过配置限制。
9. 输出 YAML 可被现有 `load_config()` 加载。
10. 无任何自动安装依赖行为。
11. 报告或最终回复列出本次实现新增、修改的脚本路径，便于作者执行 `git add`。

质量验收：

1. 对缺字段、窗口过短、dt 异常、无有效控制段给出明确错误。
2. 梯度为 NaN/Inf 时拒绝输出候选参数。
3. 梯度范数过小或过大时报告 warning。
4. 单元测试覆盖 refline 构造、loss backward、参数冻结、参数限幅。
5. 至少提供一个基于小型 CSV fixture 的端到端 smoke test。

## 11. 推荐实现文件

```text
sim/grad_tune/
  __init__.py
  data_schema.py       # 标准列名、字段校验、对齐后数据结构
  record_adapter.py    # record 或 CSV 到单工况数据包
  refline_builder.py   # refline columns 到 TrajectoryPoint
  real_loss.py         # J_real_report 和 loss_backward
  param_policy.py      # 参数块、冻结、裁剪、投影
  one_step_tune.py     # CLI 入口和流程编排

sim/tests/
  test_grad_tune_refline.py
  test_grad_tune_loss.py
  test_grad_tune_param_policy.py
  test_grad_tune_smoke.py
```

该拆分保持第一版范围清晰：数据适配、loss、参数策略和 CLI 编排各自独立，便于测试和后续扩展多工况。

## 12. 实现交付说明要求

AI 执行实现后，最终回复必须列出：

```text
新增文件:
- path/to/new_file.py

修改文件:
- path/to/modified_file.py

建议 git add:
- path/to/new_file.py
- path/to/modified_file.py
```

该清单必须只包含本次实现相关文件，不包含未触碰的既有未跟踪文件。
