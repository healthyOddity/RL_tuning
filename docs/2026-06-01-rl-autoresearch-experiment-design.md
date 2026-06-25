# RL 参数自动整定实验设计与执行计划

日期：2026-06-01

一句话状态判断：当前项目已经可以直接执行 `Default vs DC`、`Pure RL`、`DC+RL` 训练与 `DC vs DC+RL` 复现实验，但完整四臂汇总、单轨迹 BPTT、并行 RL、Gradient-Informed SAC、系统化 holdout 和 safe scheduling 还需要补充少量到中等规模代码。

本文基于 `docs/2026-06-01-rl-experiment-roadmap-discussion.md` 和当前代码状态，将后续研究拆成可逐步执行、可审核、可填写结果的实验协议。本文采用有人在环的 autoresearch 流程：每个实验先确认 protocol，再执行命令或补代码，跑完后只把经过 sanity check 的结果写入结论。

## 2026-06-11 更新：当前 truck_trailer 证据状态与执行顺序

当前 truck_trailer 主线已经从历史 `hybrid_v2` 探索切换为以下组合：

```text
DC baseline:
sim/results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml

RL run:
sim/results/rl/truck_trailer/20260609_175233/
```

对 `20260609_175233` 的当前状态检查：

| 文件/记录 | 状态 |
|---|---|
| `sac_model_final.zip` | 存在 |
| `best_model.zip` | 存在 |
| `sac_model_17000_steps.zip` | 存在 |
| `sac_model_final_replay_buffer.pkl` | 存在 |
| `evaluations.npz` | 存在，评估 timesteps 为 `[12000, 17000]` |
| 模型内部 `num_timesteps` | `best_model.zip`、`sac_model_17000_steps.zip`、`sac_model_final.zip` 均为 `17000` |
| `evaluation/` | 已生成，包含 `result.txt`、`rl_vs_dc_comparison.png` 和 comparison 系列 png |
| `evaluation_with_park_route/` | 已生成，包含 `rl_eval_results.yaml`、`rl_vs_dc_comparison.png` 和 comparison 系列 png |

对 `20260608_203406_mlp0525` 的 DC/BPTT 结果检查：

| 指标 | Default | DC tuned | 改善 |
|---|---:|---:|---:|
| training loss | 3.9215 | 1.0190 | -74.02% |
| avg lat RMSE, 49 场景 | 1.0539 m | 0.3124 m | -70.36% |
| avg head RMSE, 49 场景 | 0.0350 rad | 0.0235 rad | -32.82% |
| lat RMSE 改善场景数 | - | 45/49 | - |
| head RMSE 改善场景数 | - | 44/49 | - |

当前阶段性结论：

1. **E02 Default vs DC 已完成**：`20260608_203406_mlp0525` 可作为 truck_trailer 主线的 DC baseline 结果。
2. **E01 DC+RL 正式评估已完成**：`20260609_175233/evaluation/result.txt` 显示 DC+RL 在标准 48 条轨迹上 avg loss 从 4.0761 降到 2.9773，平均改善 27.0%，胜出 39/48。
3. **E01P park_route 诊断已完成**：`evaluation_with_park_route/rl_eval_results.yaml` 显示 49 场景中 OOD 数量为 1，`park_route` 被标记为 OOD，且 DC+RL 在该场景上严重退化。
4. 在 Pure RL 对照和四臂汇总完成前，不应直接进入 rolling 部署或论文最终结论。

### E01 当前执行入口：truck_trailer DC+RL 正式评估（48 条标准轨迹）

目的：回答当前 17000-step SAC policy 是否在 truck_trailer 标准 48 条轨迹上优于 DC tuned baseline。该入口对应正文第 4 节 E01，不再把历史 `hybrid_v2/20260526_195622` 作为当前主线 E01。

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_evaluate.py `
  --rl-model results/rl/truck_trailer/20260609_175233/sac_model_final.zip `
  --dc-config results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml `
  --plant truck_trailer `
  --output results/rl/truck_trailer/20260609_175233/evaluation
```

建议同时把终端输出保存为日志，因为当前 `rl_evaluate.py` 主要保存图片，不保存完整 per-scenario 数值表：

```powershell
New-Item -ItemType Directory -Force results/rl/truck_trailer/20260609_175233/evaluation
# PowerShell 中可把 E01-TT 命令末尾追加：
2>&1 | Tee-Object results/rl/truck_trailer/20260609_175233/evaluation/log.txt
```

成功标准：

| 检查项 | 标准 |
|---|---|
| 输出目录 | `evaluation/` 生成 comparison 系列 png |
| 轨迹数 | 48 |
| 关键指标 | 记录 RL avg loss、DC avg loss、胜出轨迹数、OOD 数 |
| 结论口径 | 只评价 48 条标准轨迹，不包含 park_route |

### E01P：truck_trailer + park_route 泛化诊断

目的：把 `park_route` 作为强 OOD / 综合路线压力测试，诊断 one-shot 全局参数 agent 的边界。它不应与 48 条标准轨迹混在一起作为同一个结论。

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_evaluate.py `
  --rl-model results/rl/truck_trailer/20260609_175233/sac_model_final.zip `
  --dc-config results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml `
  --plant truck_trailer `
  --include-park-route `
  --output results/rl/truck_trailer/20260609_175233/evaluation_with_park_route
```

解释原则：

- 若 48 条标准轨迹效果好，而 `park_route` 差，优先解释为 one-shot trajectory-level agent 对复合路线的能力边界。
- 若 48 条标准轨迹也不稳定，先检查 RL action、reward、训练步数、baseline loss 归一化和评估脚本，而不是推进 rolling scheduling。
- `park_route` 后续更适合进入 rolling preview 参数调度实验：未来 5s 局部轨迹特征 -> 周期调用 agent -> 参数平滑/限速/OOD 回退。

E01P 已在 2026-06-15 使用同一 DC baseline 和 `20260609_175233/sac_model_final.zip` 完成。结构化结果保存在：

```text
sim/results/rl/truck_trailer/20260609_175233/evaluation_with_park_route/rl_eval_results.yaml
```

| 指标 | 数值 |
|---|---:|
| 场景数 | 49 |
| RL avg loss | 157467.5696 |
| DC avg loss | 4.5817 |
| RL 胜出轨迹数 | 39/49 |
| OOD 数量 | 1/49 |
| `park_route` RL loss | 7715768.0000 |
| `park_route` DC loss | 28.8503 |
| `park_route` RL lat RMSE | 877.9075 m |
| `park_route` DC lat RMSE | 0.5638 m |

E01P 的结论是：标准 48 条轨迹上的 DC+RL 整体收益不能外推到 `park_route`。当前 SAC policy 在 `park_route` 上输出接近动作边界的参数调整，导致 tracking loss 和横向误差极端放大；这更像 one-shot trajectory-level policy 对强 OOD/复合路线的能力边界，而不是标准轨迹结论被推翻。因此后续不应把 `park_route` 并入 48 条均值，也不应在没有 OOD fallback、参数限速和平滑机制前推进部署 claim。

### E01A：评估后必须补的诊断能力

无论 E01-TT 结果好坏，都建议在下一轮代码改进中优先补：

1. `rl_evaluate.py` 保存 per-scenario 结果表，例如 `rl_eval_results.yaml` 或 `.csv`。
2. 每条轨迹记录 `rl_loss`、`dc_loss`、`delta_pct`、`lat_rmse`、`head_rmse`、`is_ood`、`ood_distance`、`rl_action`。
3. 增加 `--ood-policy none|scale|fallback_dc`，先只用于评估，不改变训练。
4. 增加 action 分布检查：是否贴边、是否某些维度长期饱和。

这一步比继续盲目加 timesteps 更重要，因为当前 RL 是否有效要先靠逐轨迹诊断判断。

## 0. 当前证据与代码状态

### 0.1 已有关键实验

已有历史探索实验目录：

```text
sim/results/rl/hybrid_v2/20260526_195622/
```

该实验已完成 `DC tuned baseline vs DC+RL`：

| 指标 | 数值 |
|---|---:|
| DC avg loss | 59.1975 |
| DC+RL avg loss | 40.8389 |
| 平均改善 | -31.0% |
| DC+RL 胜出轨迹数 | 48/48 |

结果记录位置：

```text
sim/results/rl/hybrid_v2/20260526_195622/evaluation/log.txt
```

当前 truck_trailer 主线已有结果：

| 实验 | 目录 | 状态 | 关键结论 |
|---|---|---|---|
| E02: Default vs DC | `sim/results/training/truck_trailer/20260608_203406_mlp0525/` | 已完成 | training loss -74.02%；49 场景 avg lat RMSE -70.36%；avg head RMSE -32.82% |
| E01: DC vs DC+RL | `sim/results/rl/truck_trailer/20260609_175233/` | 已完成标准 48 条评估 | RL avg loss 2.9773，DC avg loss 4.0761，平均改善 -27.0%，RL 胜出 39/48 |
| E01P: park_route 诊断 | 同上 | 已完成 | `park_route` 被判定为 OOD；RL loss 7715768.0000 vs DC loss 28.8503，说明 one-shot policy 在强 OOD/复合路线下失效 |

### 0.2 当前代码支持矩阵

| 能力 | 当前状态 | 证据/入口 | 结论 |
|---|---|---|---|
| BPTT/DC 全局整定 | 已支持 | `sim/optim/train.py` | 可直接执行 |
| truck_trailer 批量 BPTT | 已支持 | `sim/optim/train_batch.py` | 仅 truck_trailer/hybrid_dynamic 批量路径 |
| Default vs tuned 验证 | 已支持 | `sim/optim/post_training.py` | hybrid_v2 可走 scalar 验证 |
| SAC 训练 | 已支持 | `sim/optim/rl_train.py` | 可直接执行 |
| DC warm-start RL | 已支持 | `rl_train.py --config configs/tuned/xxx.yaml` | 可直接执行 |
| Pure RL around default/random | 部分支持 | `rl_train.py --config configs/default.yaml` 或 random yaml | 可直接执行，但需定义“纯 RL”口径 |
| DC vs DC+RL 评估 | 已支持 | `sim/optim/rl_evaluate.py` | 可直接执行 |
| Default/DC/PureRL/DC+RL 四臂统一汇总 | 部分支持 | 需合并多个日志 | 建议补汇总脚本 |
| 单轨迹 BPTT vs 全局 BPTT | 部分支持 | `train.py --trajectories <type>` 只能到轨迹类型，不到单速度轨迹 | 需要补单轨迹/速度选择脚本 |
| RL 并行仿真 | 未实现 | `rl_train.py` 当前单环境 | 需要补 benchmark 与 `--n-envs` |
| Gradient-Informed SAC | 未实现 | `RLTuningEnv` observation 仍是 10D 几何 + 11D 参数 | 需要新设计 |
| systematic holdout | 部分支持 | 轨迹类型 holdout 可用 `--trajectories`；速度 holdout 不可直接指定 | 需要补 speed/plant split 支持 |
| safe scheduling / 参数表调度 | 未实现 | 当前是连续动作 SAC | 需要新 baseline |

## 1. 执行原则

### 1.1 人在环检查点

每个实验至少经过四个检查点：

| 检查点 | 触发时机 | 需要人工判断 |
|---|---|---|
| G0: protocol 审核 | 跑实验前 | 对比组、指标、预算是否合理 |
| G1: 长任务确认 | 训练预计超过 30 分钟、占 GPU/CPU、或新增依赖前 | 是否允许执行 |
| G2: sanity check | 跑完后 | 日志是否完整、是否 NaN、是否复现实验口径 |
| G3: 结论审核 | 写入结论前 | 是否能支撑论文 claim |

### 1.2 统一指标

所有控制效果优先报告以下指标：

| 指标 | 含义 | 来源 |
|---|---|---|
| avg tracking loss | 综合跟踪 loss | `tracking_loss` |
| per-trajectory win count | 逐轨迹胜出数 | A/B 对比日志 |
| lat RMSE | 横向误差均方根 | `_calc_metrics` |
| head RMSE | 航向误差均方根 | `_calc_metrics` |
| speed RMSE | 速度误差均方根 | 可从 history 计算 |
| wall time | 训练/评估耗时 | 脚本输出 |
| seed mean/std | 多 seed 稳定性 | 汇总脚本 |

### 1.3 推荐工作目录

后续命令默认在以下目录执行：

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
```

除非明确说明，不安装新依赖；如果需要安装，必须先单独确认。

## 2. 实验总览

| 编号 | 实验 | 目的 | 当前代码是否可直接做 | 优先级 |
|---|---|---|---|---|
| E00 | 运行环境与复现性 smoke test | 确认入口可运行 | 是 | P0 |
| E01 | 复现已有 DC vs DC+RL | 固化已有结果 | 是 | P0 |
| E02 | Default vs DC | 证明 BPTT/DC 相比原始参数有效 | 是 | P0 |
| E03 | Pure RL around default/random | 证明无 DC warm-start 的 RL 能力 | 是 | P0 |
| E04 | 四臂消融汇总 | Default/DC/PureRL/DC+RL 统一结论 | 部分，需要汇总脚本 | P0 |
| E05 | 并行/批量 RL 仿真 benchmark | 降低后续实验成本 | 已完成，未通过接入门槛 | Closed |
| E06 | `--n-envs`/batched 并行训练 | 加速 SAC 训练 | Skip，当前不改 `rl_train.py` | Skip |
| E07 | 单轨迹 BPTT vs 全局 BPTT | 验证“全局 BPTT 是多场景折中” | 部分，需要补单轨迹选择 | P1 |
| E08 | 多 seed 稳定性 | 检查偶然性 | 是，但耗时高 | P1 |
| E09 | holdout 泛化 | 验证场景自适应不是记忆训练集 | 部分，需要补 speed split | P1 |
| E10 | Gradient-Informed SAC | 对齐调研定义的完整方法 | 否，需要新实现 | P2 |
| E11 | safe scheduling / 参数表调度 | 对齐部署安全路线 | 否，需要新 baseline | P2 |
| E12 | RL policy 应用验证 | 把 policy 当作参数调度器评估 | 部分，需要真实/回放数据接口 | P2 |

建议先完成当前 truck_trailer 主线的 E03/E04，再决定是否投入 E05-E12。理由：E01、E01P 和 E02 已完成，当前最缺的是 Pure RL 对照和四臂汇总；如果四臂消融不成立，后面的算法对比、rolling 部署和 Gradient-Informed SAC 都缺少主 claim 支撑。

## 3. E00：运行环境与复现性 Smoke Test

### 3.1 目的

确认当前环境能 import 关键模块、配置可加载、已有模型和 tuned yaml 路径存在。

### 3.2 执行命令

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
@'
import os
from config import load_config, apply_plant_override
from optim.rl_env import RLTuningEnv

cfg = load_config("configs/tuned/tuned_2cdcb49_20260526_193652.yaml")
apply_plant_override(cfg, "hybrid_v2")
print("config loaded:", cfg["vehicle"]["model_type"])

model_path = "results/rl/hybrid_v2/20260526_195622/sac_model_final.zip"
print("model exists:", os.path.exists(model_path), model_path)

env = RLTuningEnv(
    plant="hybrid_v2",
    config_path="configs/tuned/tuned_2cdcb49_20260526_193652.yaml",
    compute_baseline_losses=False,
)
obs, info = env.reset(seed=42)
print("obs shape:", obs.shape)
print("trajectory:", info["trajectory_key"])
'@ | python -
```

### 3.3 成功标准

| 检查项 | 预期 |
|---|---|
| config loaded | 正常打印 |
| model exists | `True` |
| obs shape | `(21,)` |
| trajectory | 任一标准轨迹 key |

### 3.4 结果填写

| 字段 | 结果 |
|---|---|
| 执行日期 |  |
| Python 环境 |  |
| 是否通过 |  |
| 报错/异常 |  |
| 备注 |  |

## 4. E01：truck_trailer DC vs DC+RL 正式评估

### 4.1 目的

确认当前 truck_trailer 主线的 17000-step SAC policy 是否能在标准 48 条轨迹上优于 DC tuned baseline。该实验是四臂消融中的 **B vs D**，也是后续算法对比、rolling 部署和论文结论的前置门槛。

### 4.2 实验组

| 组别 | 配置/模型 |
|---|---|
| DC | `results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml` |
| DC+RL | `results/rl/truck_trailer/20260609_175233/sac_model_final.zip` |

### 4.3 执行命令

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_evaluate.py `
  --rl-model results/rl/truck_trailer/20260609_175233/sac_model_final.zip `
  --dc-config results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml `
  --plant truck_trailer `
  --output results/rl/truck_trailer/20260609_175233/evaluation
```

### 4.4 成功标准

| 指标 | 预期 |
|---|---|
| 轨迹数 | 48 |
| DC+RL 胜出数 | 明显多于 DC 胜出数，最好接近 48/48 |
| 平均改善 | DC+RL avg loss 低于 DC avg loss |
| 输出图 | comparison 系列 png 生成 |
| 日志 | 保存终端输出或后续生成 per-scenario 结果表 |

### 4.5 结果填写

| 指标 | DC | DC+RL | 改善 |
|---|---:|---:|---:|
| avg loss | 4.0761 | 2.9773 | -27.0% |
| 胜出轨迹数 | 9/48 | 39/48 | +30 条 |
| OOD 数量 | 未记录 | 未记录 | 标准 48 条不含 `park_route` |

结论填写：

```text
E01 标准 48 条 truck_trailer 轨迹评估已完成。`20260609_175233` 的 17000-step SAC policy 在 DC tuned baseline 基础上继续降低平均 tracking loss：DC avg loss 为 4.0761，DC+RL avg loss 为 2.9773，平均改善 27.0%，并在 39/48 条轨迹上优于 DC。因此当前证据支持“DC+RL 在标准 in-distribution 轨迹上整体优于 DC”的阶段性结论。需要保留的限制是：9 条轨迹仍然退化，当前 `result.txt` 不是结构化 per-scenario 表，也未记录 action 分布、OOD 标记和 `park_route` 诊断，所以该结果还不能直接外推到强 OOD、rolling 调度或部署安全 claim。
```

异常记录：

```text
`evaluation/result.txt` 中逐轨迹文本存在少量重复行，但末尾总体统计完整，本文仅引用总体统计和可直接核验的胜出轨迹数。后续应优先让 `rl_evaluate.py` 输出结构化 CSV/YAML，避免人工从终端文本整理结果。
```

## 5. E02：Default vs DC

### 5.1 目的

证明 BPTT/DC tuned 参数相对 `default.yaml` 的收益。这是四臂消融中的 A vs B。

### 5.2 当前可行性

已完成。`sim/results/training/truck_trailer/20260608_203406_mlp0525/` 是当前 truck_trailer 主线的 DC/BPTT 结果目录，包含训练曲线、49 场景验证图、`experiment_log.yaml` 和 tuned YAML。

### 5.3 执行命令

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/train_batch.py `
  --epochs 50 `
  --plant truck_trailer
```

### 5.4 成功标准

| 指标 | 预期 |
|---|---|
| 场景数 | 49，包含 48 条标准轨迹 + `park_route` |
| DC 平均 lat/head RMSE | 优于 default |
| 退化场景 | 需要列出，不能只报均值 |

### 5.5 结果填写

| 指标 | Default | DC | 改善 |
|---|---:|---:|---:|
| training loss | 3.9215 | 1.0190 | -74.02% |
| avg lat RMSE, 49 场景 | 1.0539 m | 0.3124 m | -70.36% |
| avg head RMSE, 49 场景 | 0.0350 rad | 0.0235 rad | -32.82% |
| lat improved scenarios | - | 45/49 | - |
| head improved scenarios | - | 44/49 | - |

代表性退化场景：

| 场景 | Default | DC | 退化比例 | 备注 |
|---|---:|---:|---:|---|
| `s_curve_25kph` lat RMSE | 0.2419 | 0.2662 | +10.04% | head RMSE 同时退化 +90.73% |
| `clothoid_decel_18kph` lat RMSE | 0.1845 | 0.2134 | +15.68% | head RMSE 仍改善 |
| `lc_accel_18kph` lat RMSE | 0.1898 | 0.2562 | +34.98% | head RMSE 同时退化 +73.31% |
| `park_route` lat RMSE | 0.2695 | 0.5416 | +101.00% | 综合园区路线，强 OOD/复合路线压力测试 |
| `clothoid_decel_35kph` head RMSE | 0.0509 | 0.0570 | +11.89% | lat RMSE 仍改善 -16.6% |

结论填写：

```text
E02 已完成。DC/BPTT 对 truck_trailer 的整体收益明确：训练 loss 降低 74.02%，49 场景平均 lat RMSE 降低 70.36%，平均 head RMSE 降低 32.82%。但 DC tuned baseline 不是全场景无退化，`park_route`、`lc_accel_18kph`、`s_curve_25kph` 等场景存在局部退化。因此 E02 支持“DC 相比 default 有显著总体收益”，同时也给后续 RL 场景自适应和 rolling 调度留下动机。
```

## 6. E03：Pure RL around Default/Random

### 6.1 目的

验证不使用 DC warm-start 时，SAC 是否仍能学到有效的参数调度策略。这是四臂消融中的 A vs C，并为 C vs D 提供基础。

### 6.2 “纯 RL”口径

当前环境动作是相对 baseline 的增量，因此这里的 Pure RL 不是完全黑箱地从任意参数空间搜索，而是以下两种可执行定义：

| 口径 | baseline config | 说明 | 推荐性 |
|---|---|---|---|
| Pure RL-default | `configs/default.yaml` | 围绕原始参数学增量 | 推荐先做 |
| Pure RL-random | `configs/random_seed123.yaml` 等 | 围绕随机初值学增量 | 作为鲁棒性补充 |

### 6.3 执行命令：Pure RL-default

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
python optim/rl_train.py `
  --plant truck_trailer `
  --config configs/default.yaml `
  --total-timesteps 17000 `
  --seed 42
```

训练完成后记录输出目录，例如：

```text
results/rl/truck_trailer/20260615_155054/
```

然后评估 Pure RL 相对 default：

```powershell
python optim/rl_evaluate.py `
  --rl-model results/rl/truck_trailer/20260615_155054/sac_model_final.zip `
  --dc-config configs/default.yaml `
  --plant truck_trailer `
  --output results/rl/truck_trailer/20260615_155054/evaluation
```

### 6.4 执行命令：Pure RL-random，可选

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
python optim/rl_train.py `
  --plant hybrid_v2 `
  --config configs/random_seed123.yaml `
  --total-timesteps 20000 `
  --seed 42
```

### 6.5 成功标准

| 指标 | 预期 |
|---|---|
| 训练完成 | 生成 `sac_model_final.zip` |
| Pure RL vs default | 平均 loss 低于 default |
| 与 DC+RL 比较 | 多数情况下可能弱于 DC+RL；若强于 DC+RL，需要重点分析 |

### 6.6 结果填写

| 实验 | seed | baseline config | run id | baseline avg loss | RL avg loss | 改善 | 胜出轨迹数 |
|---|---:|---|---|---:|---:|---:|---:|
| Pure RL-default | 42 | `configs/default.yaml` | `20260615_155054` | 32.1460 | 12.9343 | -59.8% | 48/48 |
| Pure RL-random | 42 | `configs/random_seed123.yaml` |  |  |  |  |  |

注：上表只统计标准 48 条轨迹，明确排除 `park_route`。含 `park_route` 的 `rl_eval_results.yaml` 总表中 49 条均值会被千万级 OOD loss 主导，不能作为四臂消融的 avg loss。`park_route` 只进入 OOD/部署安全诊断。

结论填写：

```text
E03 Pure RL-default 已完成。结果来自 `results/rl/truck_trailer/20260615_155054/evaluation/rl_eval_results.yaml`，并按 `is_ood=false` 过滤出标准 48 条轨迹后重新计算：Default avg loss 为 32.1460，Pure RL avg loss 为 12.9343，Pure RL 胜出 48/48，平均 loss 降低 59.8%。这说明不使用 DC warm-start 时，SAC 仍能围绕 default 学到有效参数调度；但 Pure RL 的 12.9343 仍明显弱于 DC 的 4.0761 和 DC+RL 的 2.9773。因此单 seed 证据支持“DC warm-start 是当前最优路线的重要前置”，但最终稳定性还需要 E08 多 seed 验证。
```

## 7. E04：四臂消融汇总

### 7.1 目的

统一回答四个系统版本的关系：

| 版本 | 名称 | 定义 |
|---|---|---|
| A | Default | `configs/default.yaml` |
| B | DC | `results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml` |
| C | Pure RL | `results/rl/truck_trailer/20260615_155054/sac_model_final.zip`，baseline 为 `configs/default.yaml` |
| D | DC+RL | `results/rl/truck_trailer/20260609_175233/sac_model_final.zip`，baseline 为 B |

### 7.2 当前可行性

当前四臂主实验的单 seed 证据已经基本完成。A vs B 由 E02 支撑，A vs C 由 E03 支撑，B vs D 由 E01 支撑；A/B/C/D 逐轨迹统一表仍建议补一个小脚本从结构化 YAML/日志中生成，避免后续多 seed 时手工整理出错。

### 7.3 建议补充脚本

新增脚本建议：

```text
sim/optim/rl_collect_ablation.py
```

功能：

1. 读取 E02 的 `experiment_log.yaml` 或 validation 输出；
2. 读取 E03 的 `evaluation_vs_default/log.txt`；
3. 读取 E01 的 `evaluation_rerun_*/log.txt`；
4. 汇总每条轨迹的 Default/DC/PureRL/DC+RL loss；
5. 输出 `ablation_summary.csv` 和 `ablation_summary.md`。

实现难度：低到中等。主要工作是统一不同日志格式。若不想先写脚本，也可以手工填表，但不利于多 seed。

### 7.4 汇总表模板

| 轨迹 | Default loss | DC loss | Pure RL loss | DC+RL loss | best |
|---|---:|---:|---:|---:|---|
| lane_change_5kph |  |  |  |  |  |
| lane_change_18kph |  |  |  |  |  |
| ... |  |  |  |  |  |

总体统计：

| 对比 | 平均改善 | 胜出轨迹数 | 结论 |
|---|---:|---:|---|
| A vs B | -87.3% | 待逐轨迹表固化 | DC/BPTT 是主收益来源 |
| A vs C | -59.8% | 48/48 | Pure RL 有效，但弱于 DC |
| A vs D | -90.7% | 待逐轨迹表固化 | DC+RL 是当前最强单 seed 方案 |
| B vs D | -27.0% | 39/48 | RL 在 DC warm-start 基础上继续提供增益 |
| C vs D | DC+RL 比 Pure RL 低 77.0% | 待逐轨迹表固化 | DC warm-start 明显优于从 default 直接 RL |
| B vs C | Pure RL 比 DC 高 217.3% | 待逐轨迹表固化 | Pure RL 尚不能替代 DC/BPTT |

结论填写：

```text
四臂主线的单 seed 结论已经基本闭环：Default 最弱，Pure RL 明显改善 Default 但仍弱于 DC，DC+RL 在标准 48 条 in-distribution 轨迹上最强。当前最稳妥的论文表述是“可微整定提供主要全局收益，RL 在 DC warm-start 上提供场景自适应增益”；不应表述为“纯 RL 可替代 DC”，也不应把 `park_route` 纳入四臂 avg loss。`park_route` 是 OOD/部署安全诊断，应单列。
```

## 8. E05：RL 环境 step 吞吐 benchmark

### 8.1 目的

先测 `RLTuningEnv.step()` 的真实耗时，确认并行化收益空间。该实验不训练完整 SAC，只做环境吞吐测试。

### 8.2 当前可行性

当前没有 benchmark 脚本，需要新增最小脚本。E05 只新增独立 benchmark，不改 `rl_train.py`，目的是先回答“并行环境是否值得接入训练”。

### 8.3 建议补充脚本

新增：

```text
sim/tests/bench_rl_env_step.py
```

最小功能：

| 参数 | 含义 |
|---|---|
| `--plant` | plant 类型 |
| `--config` | baseline config |
| `--steps` | 每个环境 step 数 |
| `--n-envs` | 环境数量 |
| `--vec` | `none` / `dummy` / `subproc` |

建议实现方式：

1. `none`：直接创建单个 `RLTuningEnv`，循环 `env.step(env.action_space.sample())`。
2. `dummy`：用 `stable_baselines3.common.vec_env.DummyVecEnv` 包装多个 env factory，验证向量接口与 reset/step 返回值。
3. `subproc`：用 `SubprocVecEnv` 包装多个 env factory；Windows 下必须把脚本入口放在 `if __name__ == "__main__":` 下面，env factory 必须是可 pickle 的顶层函数或闭包中只捕获简单参数。
4. 每个 env 使用不同 seed，例如 `seed + rank`。
5. 记录 wall time、总 step 数、step/s、单步平均耗时，并把结果保存为 `results/rl_bench/bench_rl_env_step_<timestamp>.csv` 或直接打印成表。
6. benchmark 不做 SAC 学习、不保存模型、不改训练逻辑。

测试矩阵：

| 模式 | n_envs | 目的 |
|---|---:|---|
| single | 1 | 当前基线 |
| DummyVecEnv | 4 | 检查向量接口开销 |
| SubprocVecEnv | 4 | 检查 Windows 多进程可用性 |
| SubprocVecEnv | 8 | 检查加速收益 |
| SubprocVecEnv | 12 | 检查上限和内存压力 |

### 8.4 成功标准

| 指标 | 成功标准 |
|---|---|
| `SubprocVecEnv(n=4)` | 可以正常启动和结束 |
| `n_envs=8` 加速比 | 至少 3x，才值得接入训练 |
| CPU/内存 | 无明显打满导致卡死 |
| baseline loss 预计算 | 不应重复到不可接受 |

判断规则：

- 如果 `SubprocVecEnv(n=4)` 都无法稳定启动或退出，先不要做 E06。
- 如果 `n_envs=4/8` 相对 single 没有明显加速，说明当前瓶颈可能不是可并行的 step，或者进程开销抵消收益，先不要改训练主流程。
- 如果 `n_envs=8` 有至少 3x 加速，才值得接入 `rl_train.py` 并用于多 seed / holdout。

### 8.5 结果填写

正式结果（truck_trailer，标准 48 条轨迹，10 steps/env，3 repeats，取中位数）：

| 模式 | n_envs | worker torch threads | median wall time(s) | transitions/s | 加速比 | 并行效率 | 稳定性 |
|---|---:|---:|---:|---:|---:|---:|---|
| single | 1 | 默认 | 37.45 | 0.2670 | 1.00x | 100.0% | 3/3 正常 |
| dummy | 4 | 默认 | 209.36 | 0.1911 | 0.68x | 17.1% | 3/3 正常，但串行更慢 |
| subproc | 2 | 默认 | 58.35 | 0.3428 | 1.22x | 61.2% | 3/3 正常 |
| subproc | 4 | 1 | 94.08 | 0.4252 | 1.59x | 39.8% | 3/3 正常 |
| subproc | 8 | 1 | 154.05 | 0.5193 | 1.95x | 24.3% | 3/3 正常 |
| subproc | 12 | 1 | 204.38 | 0.5871 | 2.20x | 18.3% | 2 次正常，1 次 1150.02s 长尾 |

补充信息：24 个逻辑处理器；PyTorch 默认每进程为 16 个 intra-op / 24 个 inter-op 线程，因此正式复测将 Subproc worker 限制为 1 个线程。baseline loss 只在主进程预计算一次，耗时约 302.54s，再通过内存统计注入各 worker，避免按环境数重复计算。

结论填写：

```text
E05 已完成。Windows SubprocVecEnv 可以正常启动、运行和退出，baseline 统计内存注入也避免了各 worker 重复预计算；但当前 PC 上环境吞吐扩展性不足。稳定配置中 subproc8 仅达到 1.95x，中位吞吐为 0.5193 transitions/s，低于进入 E06 所要求的 3x；subproc12 虽达到 2.20x 中位加速，但并行效率仅 18.3%，并出现一次 1150.02s 严重长尾。因此 E05 未通过 E06 接入门槛，当前不应修改 rl_train.py 接入 --n-envs。若后续重启 E06，应先定位进程调度、Torch/BLAS 线程和仿真内部共享资源导致的扩展性瓶颈。
```

### 8.6 E05B batched 仿真正式结果

正式结果目录：`sim/results/rl_bench/e05b_batched_formal_20260623`。

E05B 的单进程 batched evaluator 在 48 条标准轨迹上通过了部分吞吐测试，但未通过 fidelity gate。`random` action 下 batch 8 的中位吞吐达到 0.2784 transitions/s，相对 scalar random reference 约 4.82x；但同一配置的 `max_loss_rel_error` 为 13.39%，`mean_loss_rel_error` 为 1.43%。`zero` action 的最大误差更高，batch 4/16 为 31.60%，batch 8 为 29.63%。由于当前 RL tuning 在部分轨迹上的收益也可能只有约 5%，这种单轨迹 loss 误差足以改变 reward、改善/退化符号和后续策略学习方向。

结论：E05B 不应接入普通 SAC 训练主线。当前可信基准仍是 scalar `RLTuningEnv`/`run_simulation` 路径；batched 路径只能保留为实验性吞吐工具或后续 per-trajectory 对齐调试对象。若未来重启 batched RL，必须先导出逐轨迹 scalar/batched loss、检查改善符号一致性，并把 `max_loss_rel_error` 降到足以区分 5% 级 RL 收益的范围内。

## 9. E06：接入 `--n-envs` 并行 SAC 训练

### 9.1 目的

将 E05 中验证有效的并行环境接入 `rl_train.py`，降低多 seed、holdout、算法对比的成本。

### 9.2 当前可行性

Skip。E05 的 `SubprocVecEnv` 路线没有达到 3x 稳定加速门槛；E05B 的 batched evaluator 虽有局部吞吐收益，但正式 48 轨迹 fidelity 未通过。当前不修改 `rl_train.py` 接入 `--n-envs` 或 batched SAC。后续多 seed、holdout 和算法对比应按 scalar 路径规划少量关键实验；只有在 batched 路径完成逐轨迹等价性修复后，才重新评估 E06。

### 9.3 建议实现范围

只做最小改造：

| 改动 | 说明 |
|---|---|
| `rl_train.py --n-envs` | 默认 1，保持旧行为 |
| `--vec-env` | `dummy` 或 `subproc` |
| env factory | 每个 env 使用不同 seed |
| baseline loss cache | 至少避免明显重复预计算，或先记录重复成本 |
| EvalCallback | eval env 仍用单环境，降低复杂度 |

建议源码改法：

1. 在 `optim/rl_train.py` CLI 增加 `--n-envs`，默认 `1`；增加 `--vec-env choices=["dummy","subproc"]`，默认 `dummy` 或仅当 `--n-envs > 1` 时生效。
2. 抽出 `make_env(rank)` 工厂，内部创建 `RLTuningEnv(plant=args.plant, config_path=args.config, ...)`，并在 reset 或 env 初始化后使用 `seed + rank`。
3. `n_envs == 1` 时保持当前单环境路径，保证旧行为完全不变。
4. `n_envs > 1` 时：
   - `dummy` 使用 `DummyVecEnv([make_env(i) ...])`；
   - `subproc` 使用 `SubprocVecEnv([make_env(i) ...], start_method="spawn")` 或按 SB3/Windows 推荐方式处理；
   - 训练 env 使用 vec env，eval env 仍使用单个 `RLTuningEnv`。
5. 保存路径需要包含 `n_envs` 和 `vec_env` 元信息，避免覆盖单环境结果。
6. 先不要做 baseline loss cache 的复杂共享；如果 E05 显示预计算成本明显，可以后续单独加缓存。第一版只要求正确、可复现、旧路径不变。

暂时不做：

| 暂不做 | 原因 |
|---|---|
| 多算法 SAC/TD3/PPO 同时接入 | 会扩大变量 |
| Gradient-Informed SAC 同时接入 | 会混淆加速收益 |
| 自动选择 n_envs | 先人工指定 |

### 9.4 验证命令

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
python optim/rl_train.py `
  --plant hybrid_v2 `
  --config configs/tuned/tuned_2cdcb49_20260526_193652.yaml `
  --total-timesteps 2000 `
  --seed 42 `
  --n-envs 4 `
  --vec-env subproc
```

### 9.5 结果填写

| n_envs | vec | timesteps | wall time | 相比单环境 | final eval reward/loss | 是否稳定 |
|---:|---|---:|---:|---:|---:|---|
| 1 | none | 2000 |  | 1.0x |  |  |
| 4 | subproc | 2000 |  |  |  |  |
| 8 | subproc | 2000 |  |  |  |  |

结论填写：

```text

```

## 10. E07：单轨迹 BPTT vs 全局 BPTT

### 10.1 目的

验证全局 DC tuned 参数是否是多场景折中。如果单轨迹最优参数分散明显，RL 场景自适应的动机更强。

### 10.2 当前可行性

当前 `train.py --trajectories <type>` 会展开该类型的所有速度段，不能直接指定单个 `lane_change_35kph`。要严格做 48 个 single-trajectory optimum，需要补脚本或扩展 CLI。

### 10.3 建议补充脚本

新增：

```text
sim/optim/train_single_trajectory_grid.py
```

功能：

1. 遍历 `TRAJECTORY_TYPES × SPEED_BANDS_KPH`；
2. 每次只生成一条轨迹；
3. 调用 `train()` 或拆出支持单条 trajectory object 的入口；
4. 保存每条轨迹的 tuned yaml、final loss、参数差异；
5. 汇总为 `single_vs_global_bptt.csv`。

### 10.4 近似替代实验

如果暂时不补脚本，可先做 8 个轨迹类型级别的 BPTT：

```powershell
python optim/train.py --plant hybrid_v2 --trajectories lane_change --epochs 100
python optim/train.py --plant hybrid_v2 --trajectories double_lc --epochs 100
python optim/train.py --plant hybrid_v2 --trajectories clothoid_left --epochs 100
python optim/train.py --plant hybrid_v2 --trajectories clothoid_right --epochs 100
python optim/train.py --plant hybrid_v2 --trajectories s_curve --epochs 100
python optim/train.py --plant hybrid_v2 --trajectories combined_decel --epochs 100
python optim/train.py --plant hybrid_v2 --trajectories clothoid_decel --epochs 100
python optim/train.py --plant hybrid_v2 --trajectories lc_accel --epochs 100
```

注意：`train.py` 当前训练结束后会自动调用 `post_training`，没有 `--no-post-training` 开关；如果只想保存单类型 BPTT 参数而不自动验证，需要先补一个 `--no-post-training` CLI 开关。

### 10.5 结果填写

| 轨迹/类型 | single/group BPTT loss | global BPTT loss | improvement | 参数差异范数 | 结论 |
|---|---:|---:|---:|---:|---|
| lane_change_5kph |  |  |  |  |  |
| ... |  |  |  |  |  |

参数分散统计：

| 参数 | mean | std | min | max | cv |
|---|---:|---:|---:|---:|---:|
| T2_y scale |  |  |  |  |  |
| T3_y scale |  |  |  |  |  |
| station_kp |  |  |  |  |  |
| switch_speed |  |  |  |  |  |

结论填写：

```text

```

## 11. E08：多 seed 稳定性

### 11.1 目的

证明 DC+RL 的收益不是单一随机种子的偶然结果。

### 11.2 实验设计

| 组别 | seed | config |
|---|---:|---|
| DC+RL | 42 | DC tuned |
| DC+RL | 43 | DC tuned |
| DC+RL | 44 | DC tuned |
| Pure RL-default | 42 | default |
| Pure RL-default | 43 | default |
| Pure RL-default | 44 | default |

### 11.3 执行命令模板

```powershell
python optim/rl_train.py `
  --plant hybrid_v2 `
  --config configs/tuned/tuned_2cdcb49_20260526_193652.yaml `
  --total-timesteps 20000 `
  --seed <SEED>
```

### 11.4 结果填写

| 方法 | seed | avg loss | vs baseline 改善 | 胜出轨迹数 | run id | 备注 |
|---|---:|---:|---:|---:|---|---|
| DC+RL | 42 |  |  |  |  |  |
| DC+RL | 43 |  |  |  |  |  |
| DC+RL | 44 |  |  |  |  |  |
| Pure RL | 42 |  |  |  |  |  |
| Pure RL | 43 |  |  |  |  |  |
| Pure RL | 44 |  |  |  |  |  |

统计结论：

| 方法 | mean avg loss | std | mean improvement | mean win count |
|---|---:|---:|---:|---:|
| DC+RL |  |  |  |  |
| Pure RL |  |  |  |  |

结论填写：

```text

```

## 12. E09：Holdout 泛化实验

### 12.1 目的

区分“训练集内调优”与“场景泛化能力”。

### 12.2 当前可行性

轨迹类型 holdout 可部分执行，因为 `rl_train.py --trajectories` 支持选择轨迹类型；速度 holdout 当前不直接支持，因为 `expand_trajectories` 虽有 `speed_bands` 参数，但 `RLTuningEnv` 和 CLI 没暴露。plant holdout 需要验证 observation 和控制器参数在跨 plant 时是否仍一致。

### 12.3 轨迹类型 holdout

示例：留出 `s_curve`，训练其它 7 类。

```powershell
python optim/rl_train.py `
  --plant hybrid_v2 `
  --config configs/tuned/tuned_2cdcb49_20260526_193652.yaml `
  --total-timesteps 20000 `
  --seed 42 `
  --trajectories lane_change double_lc clothoid_left clothoid_right combined_decel clothoid_decel lc_accel
```

当前 `rl_evaluate.py` 默认评估 env 的全量轨迹。若要只评估留出 `s_curve`，需要给 `rl_evaluate.py` 增加 `--trajectories` 参数。

### 12.4 速度 holdout

需要补代码：

| 文件 | 改动 |
|---|---|
| `rl_env.py` | `RLTuningEnv.__init__` 增加 `speed_bands=None` |
| `rl_train.py` | CLI 增加 `--speed-bands` |
| `rl_evaluate.py` | CLI 增加 `--trajectories` 和 `--speed-bands` |

### 12.5 结果填写

| holdout 类型 | 训练集 | 测试集 | 方法 | seed | avg loss | vs DC/default | 胜出数 |
|---|---|---|---|---:|---:|---:|---:|
| trajectory | exclude s_curve | s_curve | DC+RL | 42 |  |  |  |
| trajectory | exclude clothoid_decel | clothoid_decel | DC+RL | 42 |  |  |  |
| speed | exclude 45/55 | 45/55 | DC+RL | 42 |  |  |  |
| plant | hybrid_v2 | truck_trailer | DC+RL | 42 |  |  |  |

结论填写：

```text

```

## 13. E10：Gradient-Informed SAC

### 13.1 目的

对齐调研主线中的完整方法：`BPTT warm-start + Gradient-Informed SAC`。

### 13.2 当前可行性

当前未实现。需要先定义“梯度信息”如何计算、缓存、归一化并拼入 observation。

### 13.3 最小可行设计

| 模块 | 设计 |
|---|---|
| 梯度来源 | 对每条轨迹在 DC baseline 上跑一次 differentiable simulation，计算 loss 对 11 个可调参数的梯度 |
| observation 扩展 | 现有 21D + `grad_sign_11` + `log_grad_norm_11`，共 43D |
| 缓存 | 每个 `(plant, config, trajectory_key)` 缓存一次，避免 RL step 内重复 BPTT |
| 对照组 | E2: BPTT+SAC 无梯度；E3: BPTT+Gradient-Informed SAC |
| 风险 | 梯度噪声、归一化尺度、计算成本、与 RL reward 泄漏关系 |

### 13.4 需要新增/修改

| 文件 | 改动 |
|---|---|
| `rl_env.py` | 增加 `use_gradient_features=False` |
| `rl_train.py` | CLI 增加 `--gradient-features` |
| 新脚本 | `precompute_bptt_grad_features.py` |
| 测试 | 检查 observation shape、无 NaN、缓存命中 |

### 13.5 结果填写

| 方法 | seed | obs dim | avg loss | vs E2 改善 | 胜出轨迹数 | 训练耗时 |
|---|---:|---:|---:|---:|---:|---:|
| BPTT+SAC | 42 | 21 |  |  |  |  |
| BPTT+Gradient-SAC | 42 | 43 |  |  |  |  |

结论填写：

```text

```

## 14. E11：Safe Scheduling / 参数表调度

### 14.1 目的

建立比连续 SAC 更保守、可解释的部署 baseline：先离线生成多组安全参数，RL 或规则只负责选择/插值。

### 14.2 当前可行性

当前未实现。可以作为部署安全章节或工程落地章节的后续实验。

### 14.3 最小实验设计

| 步骤 | 内容 |
|---|---|
| 1 | 从 Default、DC、单类型 BPTT、随机扰动后验证通过的参数中建立候选参数表 |
| 2 | 对每个候选参数跑全量安全验证，过滤掉失控/大误差参数 |
| 3 | 用规则或分类策略按几何特征选择参数 |
| 4 | 与连续 SAC policy 比较性能与安全性 |

### 14.4 结果填写

| 方法 | avg loss | worst-case loss | 失控数 | 胜出轨迹数 | 可解释性 | 部署风险 |
|---|---:|---:|---:|---:|---|---|
| DC |  |  |  |  | 高 | 低 |
| Continuous SAC |  |  |  |  | 中 | 中 |
| Safe parameter table |  |  |  |  | 高 | 低 |

结论填写：

```text

```

## 15. E12：RL Policy 应用验证

### 15.1 目的

验证 RL 产物不是单个导出的 YAML，而是 `sac_model_final.zip` 作为参数调度器，在新轨迹或真实日志片段上推理参数并评估控车效果。

### 15.2 当前可行性

仿真轨迹上可部分执行；真实日志需要先定义日志解析和特征提取接口。

### 15.3 最小仿真版流程

1. 输入一条未见轨迹；
2. 提取 10D 几何特征；
3. 拼接 11D baseline 参数；
4. SAC policy 输出 11D action；
5. action 解码为控制器参数；
6. run_simulation 评估；
7. 与 Default/DC/SafeTable 对比。

### 15.4 需要补充

| 能力 | 状态 |
|---|---|
| 单轨迹 policy inference 脚本 | 需要新增 |
| action 到 yaml/控制器参数的复用接口 | 可复用 `export_tuned_yaml` 和 `_apply_action`，但需要整理 |
| 参数变化率限制 | 未实现 |
| OOD 回退到 DC | 可基于 `env.is_ood()` 扩展 |
| 真实日志特征提取 | 未实现 |

### 15.5 Rolling Preview 调度建议

建议新增独立脚本，不直接塞进现有 `rl_evaluate.py`：

```text
sim/optim/rl_rolling_preview_evaluate.py
```

原因是 rolling preview 的时间粒度、状态缓存、参数平滑和安全回退都不同于当前 one-shot 评估。保持脚本独立可以不干扰 E01/E03/E04 的静态四臂结果。

最小可行流程：

1. 输入一条长轨迹或真实日志片段，按固定窗口切片，例如 preview horizon 5s，stride 0.5s 或 1.0s。
2. 每个窗口提取与当前 RL observation 一致的几何/速度特征，并拼接当前 baseline 参数。
3. SAC policy 输出 action，解码成候选控制器参数。
4. 对候选参数做安全处理：
   - clamp 到训练时允许的 action/参数边界；
   - 对连续窗口参数做低通滤波或 EMA；
   - 限制每个参数的最大变化率；
   - 对 OOD 窗口 fallback 到 DC 参数；
   - 可选地加入 loss/sanity guard，发现发散则回退。
5. 用平滑后的参数序列跑仿真或回放评估，输出相对 Default/DC/one-shot RL 的 tracking loss、lat/head RMSE、最大误差和触发 fallback 次数。

工作量判断：这不是只加一个脚本那么小。若只做仿真轨迹上的 prototype，大约是中等工作量；如果要接真实日志、实时特征、参数表导出和部署安全诊断，则是中到偏大的工程量。建议先做仿真版独立脚本，验证 rolling 是否能修复 `park_route` 或至少避免极端发散，再决定是否工程化。

### 15.6 结果填写

| 数据来源 | 轨迹/日志 | 方法 | avg loss/metrics | 是否 OOD | 是否触发回退 | 备注 |
|---|---|---|---:|---|---|---|
| sim |  | DC |  |  |  |  |
| sim |  | DC+RL policy |  |  |  |  |
| real log |  | DC |  |  |  |  |
| real log |  | DC+RL policy |  |  |  |  |

结论填写：

```text

```

## 16. 推荐执行顺序

### 第一轮：不改代码，先拿可复现证据

1. E03：Pure RL-default 单 seed，先明确“纯 RL”口径。
2. E04：汇总 Default / DC / Pure RL / DC+RL 四臂核心指标。
3. E00 仅在环境变化、依赖变化或迁移机器后重跑。

已完成项：E01 DC vs DC+RL 已由 `20260609_175233/evaluation/result.txt` 支撑，E01P park_route 诊断已由 `20260609_175233/evaluation_with_park_route/rl_eval_results.yaml` 支撑，E02 Default vs DC 已由 `20260608_203406_mlp0525` 支撑，三者不需要重复跑。

第一轮结束后应能回答：

```text
Default、DC、Pure RL、DC+RL 谁最强？DC warm-start 是否必要？当前 truck_trailer 的 DC+RL 是否真的优于 DC？
```

### 第二轮：scalar 多 seed 稳定性

1. E05/E05B：记录为加速路线未通过接入门槛。
2. Skip E06：不把 `--n-envs` 或 batched reward 接入 `rl_train.py`。
3. 用 scalar 路径规划少量关键 E08 多 seed 实验，优先保证结论可信。

第二轮结束后应能回答：

```text
在不接入并行/批量仿真的前提下，哪些 scalar 多 seed/holdout 实验最值得优先运行？
```

### 第三轮：补论文动机与泛化

1. E07：单轨迹或轨迹类型 BPTT vs 全局 BPTT。
2. E08：多 seed 稳定性。
3. E09：轨迹类型 holdout，随后补速度 holdout。

第三轮结束后应能回答：

```text
全局 DC 是否确实是折中？RL 是否真的具备场景泛化，而不是只记住训练轨迹？
```

### 第四轮：推进调研定义的完整方法

1. E10：Gradient-Informed SAC。
2. E11：safe scheduling baseline。
3. E12：policy 应用验证。

第四轮结束后应能回答：

```text
梯度信息是否给 SAC 带来额外价值？连续 policy 是否能被更安全的参数表调度替代或蒸馏？RL 产物能否作为离线参数调度器使用？
```

## 17. 论文实验报告骨架

### 17.1 可支撑的论文问题

| 论文问题 | 对应实验 |
|---|---|
| 可微整定是否优于 default？ | E02 |
| RL 是否能在 DC 基础上继续提升？ | E01/E04 |
| DC warm-start 对 RL 是否必要？ | E03/E04 |
| RL 收益是否稳定？ | E08 |
| RL 是否泛化？ | E09 |
| 为什么需要场景自适应？ | E07 |
| 梯度信息是否有额外价值？ | E10 |
| 方法是否具备部署安全路径？ | E11/E12 |

### 17.2 实验报告结果区

总体结论：

```text

```

核心表 1：四臂消融

| 方法 | avg loss | 相比 Default | 相比 DC | 胜出轨迹数 | seed 数 |
|---|---:|---:|---:|---:|---:|
| Default | 32.1460 | - | +688.6% | - | 1 |
| DC | 4.0761 | -87.3% | - | 待逐轨迹表固化 | 1 |
| Pure RL | 12.9343 | -59.8% | +217.3% | 48/48 vs Default | 1 |
| DC+RL | 2.9773 | -90.7% | -27.0% | 39/48 vs DC | 1 |

注：Default 和 Pure RL 数值来自 `20260615_155054/evaluation/rl_eval_results.yaml`，按 `is_ood=false` 过滤标准 48 条后计算；DC 和 DC+RL 数值来自 `20260609_175233/evaluation/result.txt` 的标准 48 条评估统计。不要使用含 `park_route` 的 49 条均值作为四臂 avg loss。

核心表 2：泛化

| 训练/测试设置 | DC avg loss | DC+RL avg loss | 改善 | 胜出数 | 结论 |
|---|---:|---:|---:|---:|---|
| in-distribution | 4.0761 | 2.9773 | -27.0% | 39/48 | 标准 48 条轨迹上 DC+RL 整体优于 DC，但仍有 9 条退化 |
| park_route OOD | 28.8503 | 7715768.0000 | +26744067.8% | 0/1 | one-shot DC+RL policy 在强 OOD/复合路线下失效 |
| trajectory holdout |  |  |  |  |  |
| speed holdout |  |  |  |  |  |
| plant holdout |  |  |  |  |  |

核心表 3：加速

| 配置 | wall time | 加速比 | 是否采用 |
|---|---:|---:|---|
| single env |  | 1.0x |  |
| n_envs=4 |  |  |  |
| n_envs=8 |  |  |  |
| n_envs=12 |  |  |  |

核心表 4：扩展方法

| 方法 | avg loss | vs BPTT+SAC | 训练耗时 | 风险/备注 |
|---|---:|---:|---:|---|
| BPTT+SAC |  |  |  |  |
| BPTT+Gradient-SAC |  |  |  |  |
| Safe parameter table |  |  |  |  |

### 17.3 当前不可直接写成结论的 claim

以下 claim 在完成对应实验前不能写进论文结论：

| claim | 缺少证据 |
|---|---|
| “RL 方法具有泛化能力” | 缺 E09 holdout |
| “Gradient-Informed SAC 优于普通 SAC” | 缺 E10 |
| “该方法具备部署安全性” | 缺 E11/E12 |
| “DC warm-start 是跨 seed 稳定必要的” | 缺 E08 多 seed 稳定性；单 seed E03/E04 已支持当前路线优于 Pure RL |
| “并行仿真可显著加速训练” | 当前不支持：E05 未达 3x 稳定加速，E05B 未通过 fidelity gate |

## 18. 下一步执行建议

E03/E04 单 seed 四臂主线已经基本完成。E05/E05B 的结论是并行/批量仿真暂时不适合作为普通 RL tuning 的训练加速主线：SubprocVecEnv 未达到 3x 稳定加速门槛，batched evaluator 在正式 48 轨迹上存在足以影响 5% 级 RL 收益判断的单轨迹 loss 误差。因此当前不做 E06，不把 `--n-envs` 或 batched reward 接入 `rl_train.py`；后续应继续以 scalar `RLTuningEnv`/`run_simulation` 作为可信训练与评估路径。

下一步优先跑少量 scalar E08：DC+RL seed 42/43/44 与 Pure RL-default seed 42/43/44。目标是验证“DC+RL 优于 DC、Pure RL 弱于 DC+RL”不是 seed 42 偶然。`park_route` 不进入四臂 avg loss，应进入单独的部署安全路线：新增独立 `rl_rolling_preview_evaluate.py`，做 preview window 参数调度、参数平滑/限速和 OOD fallback 到 DC。rolling preview 是中等以上工作量，先做仿真 prototype，不要直接承诺部署安全 claim。
