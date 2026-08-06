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

## 2026-06-30 更新：32000/37000/42000 checkpoint 正式评估

为避免只依据 TensorBoard `eval/mean_reward` 选择模型，已对 32000、37000、42000 三个候选 SAC policy 执行同一套 `rl_evaluate.py` 正式评估，并按标准 48 条轨迹统计 `is_ood=false` 的结果。该口径下不混入 `park_route`。

| 候选模型 | 模型路径 | 评估目录 | RL avg loss | DC avg loss | avg loss 改善 | RL 胜出轨迹数 |
|---|---|---|---:|---:|---:|---:|
| 32000 best | `sim/results/rl/truck_trailer/20260623_continue_DCRL_30000/best_model.zip` | `sim/results/rl/truck_trailer/20260623_continue_DCRL_30000/evaluation_32000` | 2.9303 | 4.0761 | -28.11% | 41/48 |
| 37000 best | `sim/results/rl/truck_trailer/20260625_continue_DCRL_42000_from_32000/best_model.zip` | `sim/results/rl/truck_trailer/20260625_continue_DCRL_42000_from_32000/evaluation_37000` | 2.8626 | 4.0761 | -29.77% | 43/48 |
| 42000 final | `sim/results/rl/truck_trailer/20260625_continue_DCRL_42000_from_32000/sac_model_final.zip` | `sim/results/rl/truck_trailer/20260625_continue_DCRL_42000_from_32000/evaluation_42000_final` | 2.8233 | 4.0761 | -30.73% | 45/48 |

正式 48 条评估的结论是：`42000 final` 的 avg loss 最低、胜出轨迹数最多，因此当前应作为标准 48 条口径下的最佳 DC+RL policy。`20260625_continue_DCRL_42000_from_32000/best_model.zip` 的模型内部 `num_timesteps` 为 37000，它是 `EvalCallback` 按 5 个随机 eval episode 的 mean reward 保存的训练期候选最优；该信号可用于筛选 checkpoint，但不应替代固定 48 条轨迹评估。

需要注意，`evaluation_42000_final` 若包含 `park_route`，顶层 49 场景 summary 会被 OOD loss 严重污染；论文和实验表格应使用 `is_ood=false` 的 48 条标准轨迹统计，同时把 `park_route` 单独列为 OOD 诊断。

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
| BPTT oracle 数据集 | 部分支持 | `train_batch.py` 可做多轨迹 BPTT；`expand_trajectories()` 当前固定 8 类 × 6 速度段 | 需要补单轨迹/参数化轨迹生成与 oracle 汇总脚本 |
| supervised scene adapter | 未实现 | 当前只有 RL policy 学习 `features -> action` | 需要新增轻量监督训练与统一评估脚本 |
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
| E05 | 并行 RL 仿真 benchmark | 降低后续实验成本 | 否，需要补脚本 | P0 |
| E06 | `--n-envs` 并行训练 | 加速 SAC 训练 | 否，需要改 `rl_train.py` | P0/P1 |
| E07 | BPTT oracle + supervised scene adapter | 验证“全局 BPTT 是多场景折中”，并建立非 RL 场景自适应 baseline | 部分，需要补 oracle 数据集与 adapter 脚本 | P1 |
| E08 | 多 seed 稳定性 | 检查偶然性 | 是，但耗时高 | P1 |
| E09 | 未训练轨迹部署泛化 | 先验证 policy/adapter 在新增未训练轨迹上的部署表现；必要时再做重训式 holdout | 部分，需要补新增轨迹与评估入口 | P1 |
| E10 | Gradient-Informed SAC | 对齐调研定义的完整方法 | 否，需要新实现 | P2 |
| E11 | safe scheduling / 参数表调度 | 对齐部署安全路线 | 否，需要新 baseline | P2 |
| E12 | RL policy 应用验证 | 把 policy 当作参数调度器评估 | 部分，需要真实/回放数据接口 | P2 |

建议先完成当前继续训练版 DC+RL 的正式评估，并按同一套标准 48 轨迹协议更新 E01/E04；若新模型在 avg loss、win count、worst-case 和 action 饱和检查上都不劣于旧模型，则后续论文主线采用新模型。随后优先补 E07 的 BPTT oracle 数据集与 supervised scene adapter baseline，因为“场景自适应”并非 RL 独有，RL 的论文价值需要通过与非 RL adapter 的公平对比来支撑。E08 多 seed 仍有必要，但可排在后面作为稳定性确认；E09 先做新增未训练轨迹上的部署评估，不把“重训式 holdout”作为当前必做项。

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

## 9. E06：接入 `--n-envs` 并行 SAC 训练

### 9.1 目的

将 E05 中验证有效的并行环境接入 `rl_train.py`，降低多 seed、holdout、算法对比的成本。

### 9.2 当前可行性

暂缓实施。E05 已证明 Windows `spawn` 和 baseline 统计内存注入可用，但 subproc8 稳定加速仅 1.95x，未达到 3x 接入门槛；subproc12 还有严重长尾。当前不修改 `rl_train.py`。若后续重新启动 E06，风险点仍包括环境进程扩展性、Torch/BLAS 线程、EvalCallback 频率、UTD 保持和日志路径。

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

## 10. E07：BPTT oracle 数据集与 supervised scene adapter baseline

### 10.1 目的

验证全局 DC tuned 参数是否是多场景折中，并建立一个非 RL 的场景自适应 baseline。这个实验用于修正论文动机：引入 RL 的理由不能简单写成“为了场景自适应”，因为逐场景 BPTT 生成最优参数标签、再训练 `trajectory features -> 参数` 的 supervised adapter 同样属于场景自适应整定。更准确的假设是：DC 提供稳定 warm-start，RL 通过闭环 reward 探索参数残差，可能优于直接模仿 BPTT oracle 的监督式 adapter。

### 10.2 当前可行性

当前标准轨迹集只有 8 类 × 6 个速度段 = 48 条，足够做四臂消融和 adapter smoke test，但不足以支撑“supervised adapter 具备强泛化能力”的结论。2026-07-03 的 E07 诊断表明，`train_batch.py` / `run_simulation_batch()` 当前不是普通 scalar DC baseline 的一比一复刻，不能作为 oracle 标签生成工具。监督式 adapter 仍需要新增数据集生成、训练和评估脚本，但 oracle 标签必须来自 scalar per-trajectory DC tuning，或来自经过 scalar 复验选优的候选。

当前 `expand_trajectories(type_names, speed_bands)` 的速度段固定为 `[5, 18, 25, 35, 45, 55]`，轨迹几何参数也由 `_SPEED_PARAMS` 固定。若要构造 adapter 数据集，需要新增参数化轨迹采样入口，而不是只复用这 48 条标准轨迹。

### 10.3 E07 batched DC-param oracle 失败记录（2026-07-03）

本轮曾实现 `sim/optim/generate_bptt_dc_param_oracle_dataset.py`，尝试用 batched 仿真加速生成每条轨迹独立的 35D DC 参数 oracle。该实现已移到 `sim/tests/diagnose_e07_batched_dc_param_oracle.py` 作为失败诊断脚本；E07 trajectory manifest、RandomForest adapter schema 与分析脚本保留复用，但 batched oracle 本身降级为失败实验记录，不再作为后续 adapter 标签来源。

失败原因不是“action oracle 与 DC 参数不一致”，因为该版本已经改成 DC-param oracle；真正问题是 batched 仿真没有严格复刻 scalar DC baseline 的闭环流程。代码与小样本诊断发现：

1. 历史 batched oracle 内部优化调用 `run_simulation_batch(... hard_mode=False)`，走 batched smooth 训练路径，而不是 scalar `run_simulation()` 的同一条执行路径；
2. scalar 纵控给 `LonController` 传 `car.speed_kph`，batched 纵控传 `vehicle.speed_signed_kph`，会影响低速、倒车、PID 分支和限幅；
3. scalar 横纵向控制器与 batched 控制器不是同一实现复用，而是两套 smooth/STE 近似实现，闭环状态会放大小差异；
4. scalar 仿真按 `int(traj_duration / dt)` 推进，batched 按 `T_max` 推进并用 `valid_mask` 参与 loss，loss 采样边界也不完全一致。

不优化、只用 baseline 参数时，batched 与 scalar 已经存在明显误差：

| trajectory | scalar loss | batched soft | batched hard | 结论 |
|---|---:|---:|---:|---|
| `double_lc_30kph_extra` | 0.2100 | 0.1974 | 0.1979 | 小误差 |
| `combined_decel_45kph` | 6.9291 | 7.9828 | 6.9784 | soft 路径偏差明显 |
| `s_curve_55kph` | 6.4908 | 6.3895 | 6.3840 | 小误差 |
| `lane_change_18kph` | 0.9453 | 0.3801 | 0.3804 | 严重偏差 |
| `stop_go_25kph_d0.8_stop3` | 34.4503 | 37.1546 | 35.5220 | stop-go 有偏差 |

进一步的 scalar 逐条 DC 对照说明：`double_lc_30kph_extra` 在 scalar DC 下也会因 `lr=0.05` 与 final-only 保存而过冲，但 `combined_decel_45kph`、`s_curve_55kph`、`lane_change_18kph` 的 scalar DC 结果明显优于 batched oracle。因此，E07 后续不再使用 batched 仿真作为 oracle 生成器。batched 相关脚本只保留为失败诊断和后续 fidelity 修复参考；正式 E07 adapter 数据集必须改为 scalar per-trajectory DC oracle。

### 10.4 当前正式执行脚本

第一步先生成 E07 参数化轨迹 manifest：

```text
sim/optim/generate_e07_trajectory_manifest.py
```

功能：

1. 默认 `param228` 保留标准 48，并扩充速度、U 型弯、大曲率圆弧、直线加减速和停车起步；
2. 输出 YAML manifest，作为 oracle 和后续 holdout 分析的统一轨迹入口；
3. 不改变 `expand_trajectories()` 的标准 48 评估入口，避免污染既有 DC/RL baseline。

第二步生成正式 scalar DC-param oracle：

```text
sim/optim/generate_scalar_dc_param_oracle_dataset.py
```

功能：

1. 读取 manifest 或默认标准 48；
2. 每条轨迹单独从 global DC baseline 初始化；
3. 复用 scalar `run_simulation()`、`tracking_loss()` 和 `DiffControllerParams`，优化 35D DC 参数；
4. 保存 best-so-far 参数，而不是 final-only 参数；
5. 默认 reject worse：若 best scalar loss 未优于 global DC，则回退 baseline 参数并标记状态；
6. 输出 `oracle_dataset.csv`、`oracle_dataset.npz`、`summary.yaml` 和逐 epoch trace。

第三步训练并闭环评估 supervised adapter：

```text
sim/optim/train_e07_random_forest_adapter.py
```

当前第一版 adapter 使用 RandomForest，训练 `features[45D] -> delta_params[35D]`。评估必须把预测参数放回 scalar `run_simulation()`，比较 `global DC / scalar oracle / adapter` 的 closed-loop tracking loss，而不是只看参数 MSE。后续候选方法按复杂度从低到高：

| 方法 | 作用 | 建议 |
|---|---|---|
| kNN / RBF interpolation | 小数据下的局部插值 baseline | 适合作为最简单 adapter |
| Ridge / ElasticNet | 检查线性映射是否已足够 | 必做强 baseline |
| RandomForest / ExtraTrees | 小样本、非线性、无需深度学习 | 可作为主监督 baseline |
| 小 MLP | 与 RL policy 形式接近 | 只有样本数达到数百条后再做 |
| mixture-of-experts / 参数表调度 | 解释性和部署安全性更好 | 可并入 E11 |

`evaluate_scene_adapter.py` 必须用闭环仿真 loss 评估，而不是只看参数 MSE。统一比较：

```text
Default / global DC / BPTT oracle / supervised adapter / Pure RL / DC+RL
```

### 10.5 数据集规模判断

| 数据规模 | 作用 | 结论口径 |
|---:|---|---|
| 48 条 | smoke test；验证 oracle 生成、adapter 训练、闭环评估链路 | 不能支撑强泛化，只能作为方法可行性 |
| 100-200 条 | 轻量 supervised baseline；可做简单 train/val/test split | 可支撑“非 RL adapter 可学习一部分场景规律” |
| 300-500 条 | 小 MLP / RandomForest 更可靠；可做 trajectory/speed/geometry holdout | 建议作为硕士论文中较稳妥的 adapter 数据规模 |
| 1000+ 条 | 更像完整数据驱动泛化实验 | 成本较高，不作为当前闭环必需项 |

推荐当前目标是 **300 条左右**，来源优先用参数化仿真轨迹生成。设计原则：

1. 覆盖速度：低速泊车/园区 3-15 kph，中速 18-35 kph，高速 45-60 kph；
2. 覆盖曲率：直线、单换道、双换道、左/右 clothoid、S 弯、组合弯、变速弯；
3. 覆盖曲率变化率：平缓、正常、急变三档；
4. 覆盖速度变化：恒速、加速、减速、加减速组合；
5. 保留独立 OOD：`park_route`、极小半径、复合长路线、实车 refline 片段不进入 adapter 训练集。

若使用公开数据集，建议只把它们作为 refline/场景来源，不直接当成控制参数标签来源。公开自动驾驶数据集通常提供轨迹、地图或行为数据，但不会提供 truck_trailer 控制器的 BPTT 最优参数，因此仍需要本仿真器生成 oracle 标签。

### 10.6 48 条标准轨迹上的近似替代实验

如果暂时不扩充数据集，可先做 8 个轨迹类型级别的 BPTT，作为“全局 DC 是折中”的辅助证据：

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

### 10.7 结果填写

| split | 样本数 | 方法 | avg loss | median loss | worst loss | vs global DC | win count |
|---|---:|---|---:|---:|---:|---:|---:|
| train |  | global DC |  |  |  |  |  |
| train |  | BPTT oracle |  |  |  |  |  |
| train |  | supervised adapter |  |  |  |  |  |
| test |  | global DC |  |  |  |  |  |
| test |  | supervised adapter |  |  |  |  |  |
| test |  | DC+RL |  |  |  |  |  |

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

## 12. E09：新增未训练轨迹部署泛化实验

### 12.1 目的

验证已训练 DC+RL policy 和后续 supervised adapter 在未训练仿真 refline、实车 record-refline 和复合长路线上的部署表现。当前阶段不把“重训式 holdout”作为必做项，因为重训成本高，而且论文更需要回答的是：已得到的场景自适应参数调度器能否直接用于未见 refline，以及 one-shot policy 的失败边界能否通过 rolling preview + 安全约束缓解。

### 12.2 当前可行性

当前 E09 拆成四个子实验，避免把 one-shot policy 评估和 rolling 调度系统评估混在同一个结论里：

| 子实验 | 评估方式 | 是否重训 | 实现入口 | 当前价值 |
|---|---|---|---|---|
| E09-A generated OOD one-shot | 整条 E07 manifest refline 只调用一次 agent | 否 | 扩展 `rl_evaluate.py` | 直接测试当前最佳 DC+RL policy 对未训练仿真轨迹的泛化 |
| E09-B real record-refline one-shot | 整条实车 record-refline 只调用一次 agent | 否 | 扩展 `rl_evaluate.py` | 建立实车几何片段上的 one-shot baseline 和失败边界 |
| E09-C generated OOD rolling | 固定 horizon/stride 周期调用 agent | 否 | 新增 `rl_rolling_preview_evaluate.py` | 验证局部预瞄调度能否改善复合/长 refline |
| E09-D real record-refline rolling | 实车 record-refline 上 rolling 调度 | 否 | 新增 `rl_rolling_preview_evaluate.py` | 贴近部署形态，评估 smoothing、rate limit、OOD fallback 是否能避免极端发散 |

E09-A/B 与当前 `rl_evaluate.py` 的机制一致：`整条 refline -> extract_geometric_features -> SAC predict 一次 -> run_simulation 完整轨迹`。因此它们适合通过可选参数接入现有脚本。E09-C/D 的机制不同：它们需要按时间窗口反复提取局部特征、切换参数、平滑参数并记录 fallback，因此应独立成 rolling evaluation 脚本。

### 12.3 E09-A：E07 generated OOD one-shot evaluation

E07 已经生成 `results/e07_trajectories/e07_param228_manifest.yaml`，其中包含标准 48 条、额外速度段、U-turn、高曲率圆弧、直线加减速和 stop-and-go。E09-A 不重新造轨迹生成器，直接复用 `sim/optim/e07_trajectory_manifest.py` 的 manifest 与 materialize 逻辑，筛选未训练/OOD 轨迹做 one-shot 评估。

建议 `rl_evaluate.py` 增加：

```text
--trajectory-manifest results/e07_trajectories/e07_param228_manifest.yaml
--trajectory-types-from-manifest uturn high_curvature_arc stop_and_go standard_speed_extra
--max-trajectories N
```

示例命令：

```powershell
python optim/rl_evaluate.py `
  --plant truck_trailer `
  --dc-config results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml `
  --rl-model results/rl/truck_trailer/20260625_continue_DCRL_42000_from_32000/sac_model_final.zip `
  --trajectory-manifest results/e07_trajectories/e07_param228_manifest.yaml `
  --trajectory-types-from-manifest uturn high_curvature_arc stop_and_go standard_speed_extra `
  --output results/rl_deploy/e09a_generated_ood_oneshot_<date>
```

输出继续使用 `rl_eval_results.yaml`，但必须增加数据来源、manifest key/type、`is_ood`、action 饱和比例、worst-case loss 等字段。E09-A 不启用 fallback，fallback 次数应为 0。

### 12.4 E09-B：实车 record-refline one-shot evaluation

E09-B 使用实车记录轨迹作为仿真 refline，而不是把每一时刻 CSV 中只保留的第一个 `ref_x/ref_y` 拼成 refline。原因是当前数据每一时刻的完整 refline 已丢失，只保留当前匹配/预瞄点，直接连线会混入在线规划或匹配点变化，几何解释不稳定。record-refline 口径与 grad tune evaluation 保持一致。

实车 record-refline 的最小字段建议为：

| 字段 | 用途 |
|---|---|
| `position_enu.x/y` | record-refline 几何路径 |
| `euler_angles.z` 或 `heading` | record-refline 航向，需统一为 rad |
| `timestamp` | 时间轴与窗口切片 |
| 实车速度字段 | refline 速度，必要时平滑 |
| `s` | 由 `x/y` 累积弧长重建 |
| `kappa` | 由 `x/y/theta/s` 平滑后派生 |
| `a` | 由速度对时间差分后平滑 |

建议 `rl_evaluate.py` 增加：

```text
--real-csv <interpolated.csv>
--refline-source record
--window-duration <seconds>
```

示例命令：

```powershell
python optim/rl_evaluate.py `
  --plant truck_trailer `
  --dc-config results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml `
  --rl-model results/rl/truck_trailer/20260625_continue_DCRL_42000_from_32000/sac_model_final.zip `
  --real-csv ..\..\data_process\data\data_0512\data_0512\0_interpolated\20260413_185635_interpolated.csv `
  --refline-source record `
  --window-duration 60.0 `
  --output results/rl_deploy/e09b_real_record_oneshot_<date>
```

E09-B 的结论口径是“实车记录轨迹作为 replay/refline，测试 policy 在真实几何片段上的 one-shot 部署行为”，不是“跟踪原始在线规划 refline”。

### 12.5 E09-C/D：rolling preview evaluation

rolling preview 不应塞进 `rl_evaluate.py`。建议新增：

```text
sim/optim/rl_rolling_preview_evaluate.py
```

最小流程：

1. 输入 E07 manifest 轨迹、`park_route` 或实车 record-refline。
2. 按固定 `horizon_s` 与 `stride_s` 生成局部窗口，例如 horizon 5s、stride 1s。
3. 每个窗口提取与当前 RL observation 一致的 10D 几何/速度特征，并拼接 baseline 参数。
4. SAC policy 输出 action，解码成候选控制器参数。
5. 对连续窗口参数做安全处理：
   - clamp 到训练 action 边界；
   - EMA 平滑；
   - 参数变化率限制；
   - OOD 窗口 fallback 到 DC baseline；
   - 可选发散 guard，发现极端 loss/状态异常时回退。
6. 用参数序列跑完整 refline 仿真，输出 tracking loss、lat/head/speed RMSE、max error、action 饱和比例、OOD 窗口数和 fallback 次数。

示例命令：

```powershell
python optim/rl_rolling_preview_evaluate.py `
  --plant truck_trailer `
  --dc-config results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml `
  --rl-model results/rl/truck_trailer/20260625_continue_DCRL_42000_from_32000/sac_model_final.zip `
  --trajectory-manifest results/e07_trajectories/e07_param228_manifest.yaml `
  --trajectory-types-from-manifest uturn high_curvature_arc stop_and_go `
  --horizon-s 5.0 `
  --stride-s 1.0 `
  --ema-alpha 0.2 `
  --ood-fallback dc `
  --output results/rl_deploy/e09c_generated_ood_rolling_<date>
```

E09-C/D 的结论口径不同于 E09-A/B：它回答的是“局部预瞄调度 + 安全约束是否能把当前 one-shot policy 变成更接近部署形态的参数调度器”，不能直接归因于 SAC policy 本身。

### 12.6 重训式 holdout 的位置

重训式 holdout 不是当前必做项。如果后续要加强论文严谨性，可做两个最小版本：

| 类型 | 训练集 | 测试集 | 是否重训 | 目的 |
|---|---|---|---|---|
| trajectory holdout | exclude `s_curve` | `s_curve` | 是 | 检查是否记忆轨迹类型 |
| speed holdout | exclude 45/55/60 kph | 高速段 | 是 | 检查速度外推 |

速度 holdout 需要补代码：

| 文件 | 改动 |
|---|---|
| `rl_env.py` | `RLTuningEnv.__init__` 增加 `speed_bands=None` |
| `rl_train.py` | CLI 增加 `--speed-bands` |
| `rl_evaluate.py` | CLI 增加 `--trajectories` 和 `--speed-bands` |

### 12.7 结果填写

| 测试集 | 方法 | avg loss | median loss | worst loss | win count vs DC | OOD/窗口数 | fallback 次数 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| generated unseen | DC |  |  |  |  |  |  |  |
| generated unseen | one-shot DC+RL |  |  |  |  |  |  |  |
| generated unseen | rolling DC+RL + fallback |  |  |  |  |  |  |  |
| generated unseen | supervised adapter |  |  |  |  |  |  |  |
| real refline | DC |  |  |  |  |  |  |  |
| real refline | one-shot DC+RL |  |  |  |  |  |  |  |
| real refline | rolling DC+RL + fallback |  |  |  |  |  |  |  |

结论填写：

```text
E09-A/B 回答当前 policy 是否可直接泛化到未见 refline；E09-C/D 回答 rolling preview + safety wrapper 是否能缓解 one-shot 在长路线、复合路线和真实几何片段上的失败边界。所有 E09 结论必须区分 one-shot policy 表现与 rolling 调度系统表现。

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

### 第一轮：冻结 scalar 主线，更新最佳 DC+RL 证据

1. 当前继续训练的 DC+RL 跑完后，按标准 48 条轨迹重新 eval。
2. 若新模型在 avg loss、median loss、win count、worst-case、action 饱和比例上均不劣于 `20260609_175233`，则 E01/E04 采用新模型结论；否则保留旧模型作为论文主线。
3. `park_route` 仍作为强 OOD/部署安全诊断，不能并入标准 48 条 avg loss。
4. E00 仅在环境变化、依赖变化或迁移机器后重跑。

已完成项：E02 Default vs DC 已由 `20260608_203406_mlp0525` 支撑；E03/E04 单 seed 四臂主线已经基本完成；E05/E05B 加速路线未通过门槛，E06 不再作为当前主线。

第一轮结束后应能回答：

```text
当前最佳 DC+RL 是否稳定优于 DC？是否应替换旧版 17000-step 模型作为论文主结果？
```

### 第二轮：补 E07，建立非 RL 场景自适应 baseline

1. 先用 `e07_trajectory_manifest.py` 生成并门控 E07 v2 参数化轨迹，剔除 base DC 本身不可控或几何不合理的 refline。
2. 使用 `generate_scalar_dc_param_oracle_dataset.py` 逐条从当前 global DC baseline 初始化，保存 best-so-far 参数，并对未改善样本执行 reject-worse 回退。
3. 使用 `train_scene_adapter.py` 与 `evaluate_scene_adapter.py` 训练并评估 kNN/Ridge/RandomForest/小 MLP 等非 RL adapter。
4. 对比 `global DC / scalar DC oracle / supervised adapter / DC+RL`，用 scalar 闭环 tracking loss 而不是参数 MSE 下结论。

第二轮结束后应能回答：

```text
场景自适应是否必须依赖 RL？监督式 adapter 与 DC+RL 的差距有多大？RL 的额外价值是否真实存在？
```

### 第三轮：做 E09 部署泛化，不优先重训 holdout

1. 先生成新增未训练仿真轨迹，不重训，直接部署当前最佳 DC+RL policy 和 supervised adapter。
2. 接入实车 refline 片段，构造 `x/y/theta/t/s/v/a/kappa` 参考线，做回放/仿真部署。
3. 对 one-shot policy 与 rolling preview policy 都加同一套 clamp、EMA 平滑、参数变化率限制和 OOD fallback。
4. 只有当论文需要更强统计泛化证据时，再做重训式 trajectory/speed holdout。

第三轮结束后应能回答：

```text
当前 policy/adapter 能否部署到未训练轨迹和实车 refline？失败边界在哪里？fallback 是否能避免极端发散？
```

### 第四轮：补多 seed 稳定性与可选扩展

1. E08 多 seed 后置执行：DC+RL seed 42/43/44 和 Pure RL-default seed 42/43/44；该实验不需要改代码，但耗时高。
2. 若 supervised adapter 接近 DC+RL，可优先发展 E11 参数表/mixture-of-experts 调度，强化解释性和部署安全性。
3. 若 DC+RL 明显优于 supervised adapter，再考虑 E10 Gradient-Informed SAC，用 BPTT 梯度特征解释 RL 的额外价值。

第四轮结束后应能回答：

```text
DC+RL 的优势是否跨 seed 稳定？连续 policy 是否需要被更安全的参数表或 mixture-of-experts 蒸馏？梯度信息是否值得接入 SAC？
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
| “并行仿真可显著加速训练” | 缺 E05/E06 benchmark |

## 18. 下一步执行建议

当前不再把 E05/E06 并行加速作为后续主线。E05 multiprocess 和 E05B batched 仿真都没有同时满足速度与 fidelity 门槛，E06 跳过；E07 batched DC-param oracle 也因未能一比一复刻 scalar DC baseline 而降级为失败记录。后续 RL 训练与 E07 oracle 标签生成都保持 scalar 路线，接受训练成本，必要时用多进程并行调度多条独立 scalar 轨迹。

下一步优先级调整为：

1. **更新最佳 DC+RL 结果**：继续训练完成后，按标准 48 条轨迹 eval；若新模型在 avg loss、win count、worst-case 和 action 饱和检查上均优于或不劣于旧模型，则用新模型替换 `20260609_175233` 作为论文主结果。
2. **补 E07 oracle/adaptor baseline**：48 条标准轨迹只够 smoke test；论文级 supervised scene adapter 建议扩展到约 300 条参数化仿真轨迹，使用 scalar per-trajectory DC tuning 生成 `oracle DC 参数` 标签。每条轨迹从当前 global DC baseline 初始化，保存 best-so-far 参数而不是 final epoch；学习率先以 `0.01/0.005` 做小样本网格确认，再训练 kNN/Ridge/RandomForest/小 MLP 等非 RL adapter，并用闭环 loss 与 DC+RL 公平比较。
3. **补 E09 部署泛化**：先不做重训式 holdout；直接把当前最佳 DC+RL policy 和 supervised adapter 部署到新增未训练仿真轨迹、`park_route`、实车 refline 片段，评估 one-shot 与 rolling preview + fallback 的控制效果和失败边界。
4. **后置 E08 多 seed**：多 seed 仍然必要，但不需要改代码且耗时较高，可在主线结论更清晰后执行，用于确认 DC+RL 优势不是 seed 42 偶然。

当前论文口径应改为：DC/BPTT 提供主要全局收益，supervised adapter 作为非 RL 场景自适应 baseline，DC+RL 的价值需要体现在相对 global DC 和 supervised adapter 的闭环 tracking loss 改善上；若未训练轨迹或实车 refline 中出现极端退化，则必须把 OOD fallback、参数平滑和限速作为部署仿真的一部分，而不能直接宣称具备部署安全性。
