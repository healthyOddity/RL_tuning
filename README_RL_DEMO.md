# RL 强化学习参数整定模块

基于 SAC 算法的 Contextual Bandit 参数整定模块，作为 Differentiable Control 框架的场景自适应扩展层。

## 整体框架

```
┌──────────────────────────────────────────────────────────────────┐
│                    RL 参数整定框架                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] DC基线建立 (BPTT+Adam)                                      │
│      python optim/train.py --epochs 6 --plant kinematic          │
│      → 输出: configs/tuned/xxx.yaml (baseline_params)            │
│                                ↓                                 │
│  [2] SAC 训练                                                    │
│      python optim/rl_train.py --plant kinematic                  │
│      → 加载 baseline 参数, SAC 学习特征→参数映射                  │
│      → 输出: results/rl/kinematic/{timestamp}/sac_model_final    │
│                                ↓                                 │
│  [3] 评估对比                                                    │
│      python optim/rl_evaluate.py --rl-model xxx --dc-config xxx  │
│      → 48 条轨迹上对比 RL vs DC 性能                              │
│      → 输出: 分轨迹 loss 对比图 + 统计摘要                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 核心架构：Contextual Bandit

每条轨迹视为一个独立的决策问题，Agent 看到轨迹的几何特征后，一次性输出整组参数：

```
每条轨迹 = 1 个 episode
  │
  reset(): 采样轨迹 → extract_geometric_features → 10维特征
  │
  step(action): 解码为参数增量 → run_simulation(non-differentiable) → reward
  │
  Agent 学习: "这条轨迹的几何形状 → 最优参数偏移"
```

**关键设计区别**：RL 不替代梯度下降，而是学习"根据场景选择参数"的条件映射。
BPTT 找到全局最优区域后，SAC 在该区域内学习轨迹个性化的参数偏移。

## 三个核心代码文件

### 1. `sim/optim/rl_env.py` — RL 环境封装

将 `sim_loop` + 控制器封装为标准 Gymnasium `Env` 接口。

| 组件 | 说明 |
|------|------|
| `extract_geometric_features()` | 从轨迹中提取 10 维连续几何特征 |
| `RLTuningEnv.__init__()` | 加载 DC baseline 参数，定义 21 维观测 + 11 维动作 |
| `RLTuningEnv.reset()` | 随机采样一条轨迹，提取特征，构建初始观测 |
| `RLTuningEnv.step()` | 解码动作为参数 → 仿真 → 计算 reward → 返回 |
| `is_ood()` | Mahalanobis 距离检测训练分布外轨迹 |

**21 维观测空间：**

| 分组 | 维度 | 内容 |
|------|------|------|
| 几何特征 | 10 维 | max_curvature, mean_abs_curvature, curvature_std, max_lat_accel, speed_mean, speed_std, trajectory_length 等 |
| 当前参数 | 11 维 | T2/T3/T4/T6 缩放因子 + 7 个 PID 标量值 |

**11 维动作空间：**

| 分组 | 维度 | 范围 |
|------|------|------|
| 查找表缩放因子 | 4 维 (T2/T3/T4/T6) | [-0.3, 0.3] |
| PID Kp 增量 | 3 维 (station/low/high) | [-0.1, 0.1] |
| PID Ki 增量 | 3 维 (station/low/high) | [-0.02, 0.02] |
| switch_speed 增量 | 1 维 | [-0.5, 0.5] |

### 2. `sim/optim/rl_train.py` — SAC 训练

基于 stable-baselines3 SAC 算法训练 Agent。

```bash
python optim/rl_train.py --plant kinematic --total-timesteps 50000 --config configs/tuned/xxx.yaml
```

| CLI 参数 | 默认值 | 说明 |
|----------|--------|------|
| `--plant` | hybrid_v2 | 被控对象 |
| `--config` | None | DC baseline 配置（warm-start） |
| `--total-timesteps` | 50000 | SAC 训练步数 |
| `--lr` | 3e-4 | SAC 学习率 |
| `--buffer-size` | 100000 | 经验回放缓冲区大小 |
| `--batch-size` | 256 | SAC batch size |
| `--seed` | 42 | 随机种子 |

训练完成后自动保存：
- SAC 模型：`results/rl/{plant}/{timestamp}/sac_model_final`
- 最优参数 YAML：`results/rl/{plant}/{timestamp}/sac_model_final`（由 `export_tuned_yaml()` 导出）

### 3. `sim/optim/rl_evaluate.py` — 评估对比

在 48 条标准轨迹上对比 RL 推理参数 vs DC baseline 参数。

```bash
python optim/rl_evaluate.py --rl-model results/rl/kinematic/xxx/sac_model_final \
                            --dc-config configs/default.yaml \
                            --plant kinematic
```

对比方式：对每条轨迹提取几何特征 → SAC 推理输出参数 → 跑仿真 → 算 loss。同时用 DC baseline 参数跑一次仿真，逐轨迹对比。

## 数据流

```python
# 训练时的 episode 流程 (rl_train.py → rl_env.py)
reset():
  traj = sample_trajectory()          ← 从 48 条中随机采样
  features = extract_geometric(traj)  ← 10 维特征
  obs = concat(features, params)      ← 21 维观测
  return obs

step(action):
  T2_new = baseline_T2 * (1 + action[0])   ← 解码动作
  station_kp_new = baseline_kp + action[4]
  # ...安全约束: PID ≥ 0, switch_speed ∈ [0.5, 10]

  history = run_simulation(            ← non-differentiable 仿真
    traj, lat_ctrl=..., lon_ctrl=...,
    differentiable=False)

  reward = -tracking_loss(history)     ← 与 DC 一致的 loss 公式
  restore_baseline()                   ← 恢复 baseline 参数
  return reward

# 推理时的决策流 (rl_evaluate.py)
for traj in all_48_trajectories:
  features = extract_geometric(traj)
  obs = concat(features, baseline_params)  # 21 维
  action = SAC.predict(obs)                # 推理
  apply(action, controller)
  loss = simulate(traj)
```

## 与 DC 框架的关系

| 维度 | DC (BPTT+Adam) | RL (SAC) | 协同方式 |
|------|---------------|----------|---------|
| 优化目标 | 48 条轨迹平均 loss | 逐条轨迹 reward | Phase 1: DC 全局优化 |
| 参数 | 一组成员全局参数 | 每条轨迹一组个性化参数 | Phase 2: SAC 场景精调 |
| 样本效率 | 高 (30min/6epochs) | 中 (~1h) | BPTT 为 SAC 提供初始化 |
| 场景自适应 | 无 | 有 (几何特征→参数映射) | RL 补充 DC 缺失的能力 |
| 梯度利用 | 直接 BPTT | 通过状态空间间接利用 | 未来: 梯度注入观测 |

## 安装

```bash
cd train_file/differentiable-control
pip install -r requirements.txt
# RL 额外依赖
pip install stable-baselines3 gymnasium
```

已包含在 `requirements.txt` 中。

## 快速开始

```bash
cd sim

# 1. 运行 RL 环境测试（不需要 GPU）
python -m pytest tests/test_rl_env.py -v

# 2. 训练 SAC（kinematic 模型，约 1 小时）
python optim/rl_train.py --plant kinematic --total-timesteps 50000

# 3. 评估对比（48 条轨迹）
python optim/rl_evaluate.py --rl-model results/rl/kinematic/xxx/sac_model_final \
                            --dc-config configs/default.yaml \
                            --plant kinematic

# 4. 完整训练+评估（含 BPTT baseline）
python optim/train.py --plant kinematic --epochs 6
python optim/rl_train.py --plant kinematic --config configs/tuned/xxx.yaml
python optim/rl_evaluate.py --rl-model results/rl/kinematic/xxx/sac_model_final \
                            --dc-config configs/tuned/xxx.yaml
```

## 测试

```bash
cd sim
python -m pytest tests/test_rl_env.py -v       # 环境测试（不依赖 stable-baselines3）
python -m pytest tests/test_rl_train.py -v     # 训练端到端测试
python -m pytest tests/test_rl_evaluate.py -v  # 评估端到端测试
```

## 文件结构

```
sim/
├── optim/
│   ├── rl_env.py              # Gymnasium 环境 + 几何特征提取 + OOD 检测
│   ├── rl_train.py            # SAC 训练入口
│   ├── rl_evaluate.py         # RL vs DC 对比评估
│   └── train.py               # DC BPTT 训练（RL 的 Phase 1 基线）
├── tests/
│   ├── test_rl_env.py         # 环境单元测试
│   ├── test_rl_train.py       # 训练端到端测试
│   ├── test_rl_evaluate.py    # 评估端到端测试
│   └── test_rl_integration.py # 旧版 RL Demo（van 分支遗留）
└── results/
    └── rl/                    # RL 训练输出
        └── {plant}/
            └── {timestamp}/
                ├── sac_model_final      # SAC 模型
                ├── sac_model_XXXXX_steps  # Checkpoint
                ├── best_model.zip       # Eval 最优模型
                └── evaluations.npz      # 评估日志
```

## 结果输出

当前在 kinematic 模型上的评估结果（48 条轨迹，SAC vs DC default）：

| 指标 | RL (SAC) | DC (default) | 说明 |
|------|----------|-------------|------|
| 平均 loss | 3.025 | 7.662 | △ = 60.5% |
| 胜场 | 47/48 | 1/48 | SAC 为每条轨迹选参 |

注意：此对比中 DC 为未经 BPTT 优化的默认参数。与 BPTT 调后参数的公平对比正在准备中。

## OOD 检测

RL 环境内置 Mahalanobis 距离检测，评估训练分布外的轨迹时会标注警告：

```
clothoid_decel_55kph    RL=22.48  DC=44.92  Δ=-50.0%  [!OOD]
```

OOD 检测基于 48 条训练轨迹几何特征的均值协方差矩阵，阈值对应 p<0.01。
