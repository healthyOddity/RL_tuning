# RL 参数自动整定实验设计与执行计划

日期：2026-06-01

一句话状态判断：当前项目已经可以直接执行 `Default vs DC`、`Pure RL`、`DC+RL` 训练与 `DC vs DC+RL` 复现实验，但完整四臂汇总、单轨迹 BPTT、并行 RL、Gradient-Informed SAC、系统化 holdout 和 safe scheduling 还需要补充少量到中等规模代码。

本文基于 `docs/2026-06-01-rl-experiment-roadmap-discussion.md` 和当前代码状态，将后续研究拆成可逐步执行、可审核、可填写结果的实验协议。本文采用有人在环的 autoresearch 流程：每个实验先确认 protocol，再执行命令或补代码，跑完后只把经过 sanity check 的结果写入结论。

## 0. 当前证据与代码状态

### 0.1 已有关键实验

已有实验目录：

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
| E05 | 并行 RL 仿真 benchmark | 降低后续实验成本 | 否，需要补脚本 | P0 |
| E06 | `--n-envs` 并行训练 | 加速 SAC 训练 | 否，需要改 `rl_train.py` | P0/P1 |
| E07 | 单轨迹 BPTT vs 全局 BPTT | 验证“全局 BPTT 是多场景折中” | 部分，需要补单轨迹选择 | P1 |
| E08 | 多 seed 稳定性 | 检查偶然性 | 是，但耗时高 | P1 |
| E09 | holdout 泛化 | 验证场景自适应不是记忆训练集 | 部分，需要补 speed split | P1 |
| E10 | Gradient-Informed SAC | 对齐调研定义的完整方法 | 否，需要新实现 | P2 |
| E11 | safe scheduling / 参数表调度 | 对齐部署安全路线 | 否，需要新 baseline | P2 |
| E12 | RL policy 应用验证 | 把 policy 当作参数调度器评估 | 部分，需要真实/回放数据接口 | P2 |

建议先完成 E00-E05，再决定是否投入 E06-E12。理由：如果基础复现、四臂消融和加速收益不成立，后面的复杂研究分支不值得立即展开。

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

## 4. E01：复现已有 DC vs DC+RL

### 4.1 目的

确认 `20260526_195622` 的 DC+RL 结果可以用当前代码重新评估，避免后续分析基于不可复现结果。

### 4.2 实验组

| 组别 | 配置/模型 |
|---|---|
| DC | `configs/tuned/tuned_2cdcb49_20260526_193652.yaml` |
| DC+RL | `results/rl/hybrid_v2/20260526_195622/sac_model_final.zip` |

### 4.3 执行命令

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
python optim/rl_evaluate.py `
  --rl-model results/rl/hybrid_v2/20260526_195622/sac_model_final.zip `
  --dc-config configs/tuned/tuned_2cdcb49_20260526_193652.yaml `
  --plant hybrid_v2 `
  --output results/rl/hybrid_v2/20260526_195622/evaluation_rerun_20260601
```

### 4.4 成功标准

| 指标 | 预期 |
|---|---|
| 轨迹数 | 48 |
| DC+RL 胜出数 | 接近或等于 48/48 |
| 平均改善 | 接近已有 -31.0% |
| 输出图 | comparison 系列 png 生成 |

### 4.5 结果填写

| 指标 | 原始结果 | 复现实验结果 |
|---|---:|---:|
| DC avg loss | 59.1975 |  |
| DC+RL avg loss | 40.8389 |  |
| 平均改善 | -31.0% |  |
| 胜出轨迹数 | 48/48 |  |
| OOD 数量 |  |  |

结论填写：

```text

```

异常记录：

```text

```

## 5. E02：Default vs DC

### 5.1 目的

证明 BPTT/DC tuned 参数相对 `default.yaml` 的收益。这是四臂消融中的 A vs B。

### 5.2 当前可行性

可直接执行。`post_training.py` 支持 `hybrid_v2` 的 scalar per-scenario 验证。

### 5.3 执行命令

```powershell
cd C:\Users\huangjiangyu\Desktop\hirain\L4\train_file\differentiable-control\sim
python optim/post_training.py `
  --config configs/tuned/tuned_2cdcb49_20260526_193652.yaml `
  --plant hybrid_v2 `
  --output-dir results/validation/hybrid_v2/E02_default_vs_dc_20260601
```

### 5.4 成功标准

| 指标 | 预期 |
|---|---|
| 场景数 | 48 或 49，取决于 post_training 是否包含 `park_route` |
| DC 平均 lat/head RMSE | 优于 default |
| 退化场景 | 需要列出，不能只报均值 |

### 5.5 结果填写

| 指标 | Default | DC | 改善 |
|---|---:|---:|---:|
| avg lat RMSE |  |  |  |
| avg head RMSE |  |  |  |
| avg speed RMSE |  |  |  |
| improved scenarios |  |  |  |
| degraded scenarios |  |  |  |

代表性退化场景：

| 场景 | Default | DC | 退化比例 | 备注 |
|---|---:|---:|---:|---|
|  |  |  |  |  |

结论填写：

```text

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
  --plant hybrid_v2 `
  --config configs/default.yaml `
  --total-timesteps 20000 `
  --seed 42
```

训练完成后记录输出目录，例如：

```text
results/rl/hybrid_v2/<PURE_RL_RUN_ID>/
```

然后评估 Pure RL 相对 default：

```powershell
python optim/rl_evaluate.py `
  --rl-model results/rl/hybrid_v2/<PURE_RL_RUN_ID>/sac_model_final.zip `
  --dc-config configs/default.yaml `
  --plant hybrid_v2 `
  --output results/rl/hybrid_v2/<PURE_RL_RUN_ID>/evaluation_vs_default
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
| Pure RL-default | 42 | `configs/default.yaml` |  |  |  |  |  |
| Pure RL-random | 42 | `configs/random_seed123.yaml` |  |  |  |  |  |

结论填写：

```text

```

## 7. E04：四臂消融汇总

### 7.1 目的

统一回答四个系统版本的关系：

| 版本 | 名称 | 定义 |
|---|---|---|
| A | Default | `configs/default.yaml` |
| B | DC | `configs/tuned/tuned_2cdcb49_20260526_193652.yaml` |
| C | Pure RL | `rl_train.py --config configs/default.yaml` 训练出的 policy |
| D | DC+RL | `20260526_195622/sac_model_final.zip` |

### 7.2 当前可行性

部分可直接执行。A vs B、A vs C、B vs D 都有现有入口；但 A/B/C/D 四臂统一表需要把 `post_training.py` 和 `rl_evaluate.py` 的日志合并，建议补一个小脚本。

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
| A vs B |  |  |  |
| A vs C |  |  |  |
| A vs D |  |  |  |
| B vs D |  |  |  |
| C vs D |  |  |  |
| B vs C |  |  |  |

结论填写：

```text

```

## 8. E05：RL 环境 step 吞吐 benchmark

### 8.1 目的

先测 `RLTuningEnv.step()` 的真实耗时，确认并行化收益空间。该实验不训练完整 SAC，只做环境吞吐测试。

### 8.2 当前可行性

当前没有 benchmark 脚本，需要新增最小脚本。建议先不要改 `rl_train.py`。

### 8.3 建议补充脚本

新增：

```text
sim/optim/bench_rl_env_step.py
```

最小功能：

| 参数 | 含义 |
|---|---|
| `--plant` | plant 类型 |
| `--config` | baseline config |
| `--steps` | 每个环境 step 数 |
| `--n-envs` | 环境数量 |
| `--vec` | `none` / `dummy` / `subproc` |

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

### 8.5 结果填写

| 模式 | n_envs | steps/env | wall time(s) | step/s | 加速比 | CPU 占用 | 内存 | 是否通过 |
|---|---:|---:|---:|---:|---:|---|---|---|
| single | 1 | 20 |  |  | 1.0x |  |  |  |
| dummy | 4 | 20 |  |  |  |  |  |  |
| subproc | 4 | 20 |  |  |  |  |  |  |
| subproc | 8 | 20 |  |  |  |  |  |  |
| subproc | 12 | 20 |  |  |  |  |  |  |

结论填写：

```text

```

## 9. E06：接入 `--n-envs` 并行 SAC 训练

### 9.1 目的

将 E05 中验证有效的并行环境接入 `rl_train.py`，降低多 seed、holdout、算法对比的成本。

### 9.2 当前可行性

需要改代码。风险点包括 Windows `spawn`、环境 pickle、baseline loss 重复预计算、EvalCallback 频率和日志路径。

### 9.3 建议实现范围

只做最小改造：

| 改动 | 说明 |
|---|---|
| `rl_train.py --n-envs` | 默认 1，保持旧行为 |
| `--vec-env` | `dummy` 或 `subproc` |
| env factory | 每个 env 使用不同 seed |
| baseline loss cache | 至少避免明显重复预计算，或先记录重复成本 |
| EvalCallback | eval env 仍用单环境，降低复杂度 |

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

### 15.5 结果填写

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

1. E00：smoke test。
2. E01：复现 DC vs DC+RL。
3. E02：Default vs DC。
4. E03：Pure RL-default 单 seed。
5. E04：先手工汇总四臂核心指标。

第一轮结束后应能回答：

```text
Default、DC、Pure RL、DC+RL 谁最强？DC warm-start 是否必要？已有 DC+RL 结论是否可复现？
```

### 第二轮：先解决训练成本

1. E05：写 benchmark 脚本，只测 step 吞吐。
2. 若 E05 证明 `SubprocVecEnv` 有效，再做 E06。
3. 用并行能力重跑 E03/E08，降低多 seed 成本。

第二轮结束后应能回答：

```text
当前 PC 是否支持 RL 并行加速？合理 n_envs 是多少？后续多 seed/holdout 是否可承受？
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
| Default |  |  |  |  |  |
| DC |  |  |  |  |  |
| Pure RL |  |  |  |  |  |
| DC+RL |  |  |  |  |  |

核心表 2：泛化

| 训练/测试设置 | DC avg loss | DC+RL avg loss | 改善 | 胜出数 | 结论 |
|---|---:|---:|---:|---:|---|
| in-distribution |  |  |  |  |  |
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
| “DC warm-start 是必要的” | 缺 E03/E04 C vs D |
| “并行仿真可显著加速训练” | 缺 E05/E06 benchmark |

## 18. 下一步执行建议

建议下一次从 E00-E02 开始，先不改代码。若三项都通过，再做 E03 的 Pure RL-default 单 seed。完成后就能形成第一版四臂消融证据。

如果 E03 训练时间仍然不可接受，应暂停多 seed，先执行 E05/E06 的并行加速路线。
