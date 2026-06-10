# RL 自动驾驶控制器参数自动整定 — 教学文档

本文记录将强化学习（RL）引入 Differentiable Control (DC) 框架实现自动驾驶控制器参数自动整定的完整学习过程。

---

## 一、问题背景

DC 框架使用 **BPTT + Adam** 进行控制器参数优化，在 48 条轨迹上求平均 loss 后做一次梯度更新。优势是样本效率高（30 分钟 / 6 epochs），局限是输出一组全局最优参数——对所有场景的妥协，缺乏场景自适应能力。

RL (SAC) 作为互补方法，学习"轨迹几何特征 → 控制器参数"的映射，为每条轨迹提供个性化参数。两者是递进关系，非替代关系。

---

## 二、整体框架：Contextual Bandit

每条轨迹视为一个独立的决策问题，Agent 看到轨迹几何特征后一次性输出整组参数：

```
每条轨迹 = 1 个 episode
  │
  reset(): 采样轨迹 → extract_geometric_features → 10维几何特征
  │
  step(action): 解码为参数增量 → run_simulation(non-differentiable) → reward = -tracking_loss
  │
  SAC 学习: "轨迹的几何形状 → 最优参数偏移"
```

### 三阶段 Pipeline

```
[1] BPTT 梯度优化 (DC 基线)
    python optim/train.py --plant kinematic --epochs 6
    → 输出: baseline_params (全局最优邻域)

[2] SAC 场景精调
    python optim/rl_train.py --plant kinematic --config configs/tuned/xxx.yaml
    → 加载 baseline, SAC 学习场景自适应偏移
    → 输出: sac_model

[3] 评估对比
    python optim/rl_evaluate.py --rl-model xxx --dc-config xxx
    → 48 条轨迹上对比 RL vs DC 性能
```

---

## 三、RL 问题建模

### 3.1 状态空间（21 维观测）

| 分组 | 维度 | 内容 |
|------|------|------|
| 几何特征 | 10 | max_curvature, mean_abs_curvature, curvature_std, curvature_integral, max_lat_accel, speed_mean, speed_std, trajectory_length, curvature_peaks, curvature_sign_changes |
| 当前参数 | 11 | T2/T3/T4/T6 缩放因子 + station_kp/ki, low_speed_kp/ki, high_speed_kp/ki, switch_speed |

**设计原则**：全部为连续数值特征，不使用轨迹类型名称。Agent 学到的是"高曲率→长预瞄"，而非"lane_change→参数 X"。

### 3.2 动作空间（11 维参数增量）

| 分组 | 维度 | 范围 |
|------|------|------|
| 查找表缩放因子 (T2/T3/T4/T6) | 4 | [-0.3, 0.3] |
| PID Kp 增量 | 3 | [-0.1, 0.1] |
| PID Ki 增量 | 3 | [-0.02, 0.02] |
| switch_speed 增量 | 1 | [-0.5, 0.5] |

最终参数 = baseline × (1 + scaling) + PID_delta。安全约束：PID 增益 ≥ 0，查找表 y 值 ≥ 0。

### 3.3 奖励函数

```
reward = -tracking_loss
       = -(10×lat_mse + 8×head_mse + 3×speed_mse + 0.05×steer_rate + 0.01×acc_rate)
```

与 DC 训练完全一致，保证 RL 优化方向与梯度下降对齐，使结果可对比。这是稠密奖励，每一步都给信号。

---

## 四、算法：SAC (Soft Actor-Critic)

### 4.1 为什么选 SAC

| 特性 | 对本任务的适配 |
|------|---------------|
| **Off-policy** | 经验回放可复用历史数据，48 条轨迹反复利用 |
| **连续动作空间** | 天然支持 11 维参数增量输出 |
| **Entropy 正则化** | 自动平衡探索与利用，防止早熟收敛 |
| **稳定性好** | 相比 TD3 对超参数更鲁棒 |

### 4.2 网络结构

```
SACPolicy
  ├── actor (Actor)
  │     输入: 21 维观测
  │     └→ Linear(21→256)+ReLU → Linear(256→256)+ReLU
  │        └→ μ: Linear(256→11) + log_σ: Linear(256→11)
  │     输出: 动作均值 + 对数标准差 → tanh 压缩到 [-1,1]
  │
  ├── critic (ContinuousCritic) — 两个独立 Q 网络
  │     Q1: 输入(21+11=32) → Linear(256)+ReLU → Linear(256)+ReLU → Linear(1)
  │     Q2: 同上（独立网络）
  │     训练时取 min(Q1, Q2) 防止 Q 值过估计
  │
  └── critic_target — critic 的软拷贝（τ=0.005 慢速跟随）

总参数量: ~226,000
```

网络简单的原因是：输入是 21 维特征向量（非图像），不需要 CNN 逐层提取特征。RL 领域的共识是小网络 + 好算法 > 大网络 + 数据不足——复杂度在算法逻辑而非网络深度。

### 4.3 训练机制（model.learn 内部的核心 80 行）

**每次 train() 调用的步骤：**

1. 从 ReplayBuffer 随机抽 256 条历史记录 ← **Off-policy 经验回放**
2. 计算 target Q（Bellman 方程）：`target_q = reward + γ × min(critic_target(next_obs, next_action))`
3. 更新 Critic：`critic_loss = MSE(current_q, target_q)` → backward
4. Actor 根据当前状态输出动作
5. Critic 给 Actor 的动作打分：`q_value = min(critic(obs, actor(obs)))`
6. 更新 Actor：`actor_loss = ent_coef × log_prob - q_value` → backward ← **Entropy 正则化**
7. 软更新目标网络：polyak_update(τ=0.005)
8. 自动调节探索温度：`ent_coef = 'auto'` ← 训练初高温度多探索，后期低温度精利用

### 4.4 Actor-Critic 协作：球员与教练

| 角色 | 网络 | 职责 |
|------|------|------|
| Actor（球员） | 策略网络 | 根据观测输出动作 |
| Critic（教练） | 价值网络 | 评估 Actor 的动作值多少钱（Q 值） |

球员每次射门，教练打分。球员往高分方向调整，教练根据实际进球结果修正自己的评分标准。

### 4.5 Entropy 正则化

Actor 的损失函数：`actor_loss = ent_coef × log_prob - Q(s,a)`

- `Q(s,a)` 项：最大化 Critic 评分（往高分走）
- `ent_coef × log_prob` 项：惩罚过于确定的行为（保持随机性探索）

训练初期 `ent_coef` 自动调高 → 多试错。后期自动调低 → 精打细算。

### 4.6 MDP 问题建模

在 SAC 之前，必须将任务形式化为 **Markov 决策过程（MDP）**。MDP 是所有 RL 问题的通用数学语言，包含五要素：

```
MDP = (S, A, P, R, γ)
```

| MDP 要素 | 在我们的设计中的对应 | 代码位置 |
|----------|---------------------|---------|
| **S 状态** | 21 维 = 10 几何特征 + 11 参数快照 | `RLTuningEnv.observation_space` |
| **A 动作** | 11 维参数增量 | `RLTuningEnv.action_space` |
| **P 转移** | `run_simulation(traj, params)` | `RLTuningEnv.step()` |
| **R 奖励** | `-tracking_loss` | `RLTuningEnv._compute_reward()` |
| **γ 折扣** | 0.99（SAC 默认） | `SAC(gamma=0.99)` |

**Markov 性质**要求"未来只取决于现在，不取决于过去"。我们的设计满足此性质：状态包含当前参数快照（Agent 知道"我在哪"）和轨迹几何特征（Agent 知道"前方是什么路"），决策时不需要知道上一条轨迹是什么。

**重要特例：我们的任务是 Contextual Bandit（退化 MDP）**——每条轨迹只有一个决策步：

```
标准 MDP:  s₀ → a₀ → r₀ → s₁ → a₁ → r₁ → s₂ → ... → 终止
Bandit:    观测(轨迹特征) → 动作(一组参数) → 跑完整轨迹 → reward → 终止 (done=True)
```

这意味着 `done` 在第一步后永远为 `True`，Bellman 方程中的未来项永远为零。

### 4.7 Bellman 方程详解

Bellman 方程是 SAC Critic 学习的核心。最简形式：

```
Q(s, a) = r + γ × max[Q(s', a')]
            ↑       ↑        ↑
        立即奖励  折扣因子  未来最优价值
```

**意思**：在状态 `s` 做动作 `a` 的总价值 = 立刻拿到的奖励 + 未来所有奖励的折现总和。

SAC 代码中对应的实现 (`sac.py` 的 `train()`)：

```python
with th.no_grad():
    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
```

**在 Bandit 任务中的退化**：因为 `done` 永远是 `True`，`(1-done) = 0`，因此：

```
target_q = reward = -tracking_loss
```

Bellman 方程退化为直接学习 reward——Critic 学习的就是"这个参数跑这条轨迹，亏损多少"，极其简单直接。

**为什么要折现未来（γ = 0.99）**：如果不是 Bandit 而是 Sequential RL（每 2 秒调一次参），γ 让 Agent 更在意当下、不那幺在意遥远未来：

```
第0步: r₀ = -0.5
第5步: r₅ = -50 × 0.99⁵ = -50 × 0.951 = -47.55
```

### 4.8 Q 值过估计 & 双 Q 机制

**单 Q 网络的问题**：Critic 输出的 Q 值包含估计误差。当 Actor 利用这些信号时，总是往高估的方向走：

```
真实 Q(s,a) = 5.0
Q1 估计 = 7.2  ← 高估
Actor 收到信号："这个动作值 7.2 分！"
Actor 疯狂输出 → reward 只有 4.8 → Critic 继续高估 → 恶性循环 → 训练崩溃
```

**双 Q 的解决**（SAC 和 TD3 的标配）：用一个独立网络 Q2 来约束：

```python
next_q_values = th.min(next_q_values, dim=1, keepdim=True)  # 取两个 Q 的最小值
```

两个独立的 Q 网络各自估计。只有当两个人都高估时才会出问题（概率极低）。Actor 更新同样取 `min(Q1, Q2)`——保守策略，宁可低估也不被高估带偏。这是 TD3 论文 (Fujimoto 2018) 实验验证的必需技术。

### 4.9 Target Network & Polyak 软更新

**为什么不能直接拷贝 critic → critic_target？** 如果每步硬拷贝：

```python
critic_target.load_state_dict(critic.state_dict())  # 危险
```

Critic 往东走 1 步，Target 也往东走 1 步。下一次 Critic 更新时 Target 已更新为新值，Bellman 目标变成**移动靶** → 训练震荡，不收敛。

**Polyak 软更新**（τ=0.005）：

```python
θ_target = 0.005 × θ_critic + 0.995 × θ_target
```

Target 每次只向 Critic 方向移动 0.5%，是一个**慢速移动的平均值**。Target 滞后 Critic 约 200 步，Bellman 目标用的是"过去 200 步的平均 Critic"而非"刚才那一刻的 Critic"，训练因此稳定。

**重要**：Target Network 是**全局的**，不属于任何一条轨迹。每次 `train()` 从 ReplayBuffer 随机抽 256 条来自不同轨迹的记录，更新的是全局 Critic → 全局 Target。

### 4.10 数据采集与训练的交错

SAC 的 `learn()` 不是"采完全部再训"，而是**实时交错**进行：

```python
while num_timesteps < total_timesteps:
    action = actor(obs)                                    # 推理
    new_obs, reward, done, _ = env.step(action)           # 环境反馈
    replay_buffer.add(obs, action, reward, new_obs, done)  # 存回放池

    if num_timesteps > learning_starts:                   # 默认 100 步后开始训
        if num_timesteps % train_freq == 0:              # 默认每 1 步
            train(gradient_steps=gradient_steps)          # 默认做 1 次梯度更新
```

**关键参数**：
- `train_freq`：多少步训一次（默认 1）
- `gradient_steps`：每次训练做几次梯度更新（默认 1）
- 在 Bandit 中，1 timestep = 1 条完整轨迹仿真 + 1 次 train()

**ReplayBuffer vs 轨迹缓存**：
- 轨迹缓存：48 条参考轨迹（几何形状），用于 SAC 运行仿真
- ReplayBuffer：50,000 条训练记录 (obs, action, reward)，用于 SAC 训练时抽数据
- `batch_size=256` 是从 50,000 条里随机抽，256 << 50,000，完全够

### 4.11 如何读 SAC 训练曲线

| 曲线 | 健康状态 | 异常信号 |
|------|---------|---------|
| **actor_loss** | 初期下降，后期稳定在负值 | 持续为正 → Actor 找不到好方向；剧烈震荡 → 学习率太大 |
| **critic_loss** | 初期快速下降，后期平稳 | 持续不降 → Q 网络容量不够或 reward 信号太弱；突然爆炸 → 梯度爆炸 |
| **ent_coef** | 初期高（~0.5+），缓慢下降 | 一直很高 → Agent 还在乱试；降太快 → 探索不足，局部最优 |

我们的任务收敛很快：稠密 reward + Bandit 简化 = Critic 能快速学到价值函数。

### 4.12 超参数对任务的影响

| 参数 | 默认值 | 调大效果 | 调小效果 |
|------|--------|---------|---------|
| **lr** | 3e-4 | 收敛快但不稳定 | 稳定但慢 |
| **gamma** | 0.99 | 对 Bandit 无影响（done 永远是 True） | 无影响 |
| **tau** | 0.005 | Target 紧跟 Critic → 可能震荡 | Target 太滞后 → 收敛慢 |
| **batch_size** | 256 | 梯度估计更准 | 更快但噪声大 |
| **buffer_size** | 100000 | 记住更久的历史 | 只记住最近的 |
| **ent_coef** | auto | 手动设置可能过小或过大 | 自动调节，推荐 |

### 4.13 SAC 策略梯度 vs BPTT 解析梯度

这是整个 DC+RL 方案的数学基石：

```
BPTT 的梯度:
  ∂L/∂θ = ∂L/∂history × ∂history/∂u_T × ∂u_T/∂u_{T-1} × ... × ∂u₁/∂θ
  精确计算，通过全链路可微的反向传播，无估计误差

SAC 的策略梯度:
  ∇J ≈ (1/N) Σ [∇log π(a|s) × (Q(s,a) - baseline)]
  蒙特卡洛估计，方差与 1/N 成正比，需要海量采样才能逼近真值
```

BPTT 利用的是**物理模型结构**（动力学可微，梯度通过仿真器反向传播），SAC 估计的是**统计规律**（从数据中拟合 Q 函数再用策略梯度定理近似）。物理 > 统计，在样本效率上天差地别。NeurIPS 2025 论文从理论上证明了这一点：SAC 需要 O(1/ε³) 的样本复杂度的 BPTT 只需要 O(log(1/ε))。

### 4.14 tanh 压缩动作的数学

SAC 的动作通过 `tanh` 压缩到 [-1, 1]（env 内部再线性映射到实际范围）：

```
a_raw = μ + σ × ε           ← 高斯采样（无界）
a_squashed = tanh(a_raw)    ← 压缩到 [-1, 1]（平滑可微）
```

**为什么不用简单截断**：`clip(a, -1, 1)` 在边界处梯度为零，Actor 无法从"推过头"的错误中学习。`tanh` 处处可微，梯度处处非零。

**梯度消失的补偿**：当 a_raw 的绝对值 > 3 时 tanh 饱和，导数接近 0。SAC 在计算 log_prob 时补偿这一饱和效应：

```python
log_prob = log_prob - (1 - tanh(a)² + ε).log().sum()
#          ↑ 原始高斯概率     ↑ tanh 雅可比行列式修正
```

这让 Actor 即使把动作推得太极端，梯度信号仍能流过去纠正。

---

## 五、Reward 设计深度分析

### 5.1 稠密奖励 vs 稀疏奖励

| | 稠密奖励 | 稀疏奖励 |
|---|---|---|
| 给分时机 | 每步都给 | 仅特殊事件（碰撞） |
| SAC 学习难度 | 容易 | 极难 |
| 你的当前设计 | ✅ tracking_loss 是稠密的 | — |

**稠密奖励是 SAC 快速收敛的关键。** tracking_loss 覆盖了横向跟踪、航向跟踪、速度跟踪、转向平滑、加速平滑五个维度，每一步都给 SAC 梯度信号。

### 5.2 tracking_loss 的权重含义

```
loss = 10×lat_mse  +  8×head_mse  +  3×speed_mse  +  0.05×steer_rate_mse  +  0.01×acc_rate_mse
       ↑               ↑              ↑               ↑                      ↑
    横向最重要      航向次重要      速度再次要      转向平顺关注低         加速平顺关注最低
```

这些权重是从 DC 训练继承的，反映了自动驾驶控制器调参中横向跟踪的优先地位。

---

## 六、BPTT + SAC 的协同论证

### 6.1 理论基础

NeurIPS 2025 论文 (Sharma et al.) 证明了 PID 调参问题满足 **梯度主导性**（Polyak-Łojasiewicz 不等式）：

```
|J(K) - J(K*)| ≤ α × ||∇J(K)||²
```

这意味着每个梯度为零的点都是全局最优。BPTT 利用全链路可微计算精确梯度，能高效收敛到全局最优邻域——这是 SAC 从零探索做不到的。

### 6.2 分工

| 阶段 | 方法 | 做什么 |
|------|------|--------|
| Phase 1 | BPTT + Adam | 梯度下降快速定位全局最优邻域（30 分钟） |
| Phase 2 | SAC | 在邻域内学习场景自适应偏移（~1 小时） |

### 6.3 为什么 SAC 不能替代 BPTT

SAC 的策略梯度是对真实梯度的高方差蒙特卡洛估计，而 BPTT 的梯度是精确计算。在 11 维参数空间中，SAC 从零探索需要的样本复杂度为 O(1/ε³·log(1/ε))，50K 步远远不够。

---

## 七、实验设计

### 7.1 消融实验

| # | 组 | 说明 | 回答的问题 |
|---|-----|------|-----------|
| 1 | Default | 未调参的原始参数 | 底线有多差 |
| 2 | BPTT 单独 | 只做梯度优化 | DC 的上限 |
| 3 | SAC 单独 | 不依赖梯度 | RL 的上限，预计远差于 BPTT |
| 4 | BPTT → SAC 融合 | 梯度优化 + 场景精调 | 1+1 > 2? |
| **5** | **BPTT 全局参数直接评估** | 所有轨迹共用一组 DC 参数 | **场景自适应真实增量** |

第 5 组是核心——它回答"RL 的场景自适应到底有没有带来额外收益"。如果第 4 组和第 5 组差距只有 2-3%，说明 BPTT 已经足够；如果 10-15%+，则场景自适应是真实需求。

### 7.2 算法对比

SAC vs TD3（同是 off-policy，公平比较）vs PPO（on-policy，样效率更低，训练时间更长）。如果资源有限，优先 SAC vs TD3。

### 7.3 评估指标

与 DC 训练完全一致：
- 横向跟踪误差 RMSE
- 航向误差 RMSE
- 速度误差 RMSE
- 转向平滑度
- 加速平滑度

---

## 八、代码架构

### 文件结构

```
sim/optim/
├── rl_env.py          # Gymnasium 环境 + 几何特征 + OOD 检测
├── rl_train.py        # SAC 训练入口
├── rl_evaluate.py     # RL vs DC 对比评估
└── train.py           # DC BPTT 训练 (Phase 1)

sim/tests/
├── test_rl_env.py     # 环境测试（20+ 项）
├── test_rl_train.py   # 训练端到端测试
└── test_rl_evaluate.py # 评估端到端测试
```

### 关键复用

RL 环境直接复用 DC 框架的以下模块（零改写）：
- `run_simulation(differentiable=False)` — 闭环仿真
- `LatControllerTruck(differentiable=False)` — 横向控制器
- `LonController(differentiable=False)` — 纵向控制器
- `tracking_loss()` — 作为 reward 的负值
- `expand_trajectories()` — 轨迹集合生成
- `load_config()` / `save_tuned_config()` — 配置管理

新增代码仅约 570 行（环境 270 + 训练 140 + 评估 160），SAC 算法实现由 stable-baselines3 库完全封装。

---

## 九、关键结论

1. **RL 不替代梯度下降——它们互补。** BPTT 利用问题结构高效收敛，SAC 学习场景自适应偏移。BPTT 给出全局最优，SAC 在其邻域内做个性化精调。

2. **SAC 网络简单是合理的。** 21 维输入不需要 CNN 逐层提取，RL 的样本量撑不起大网络。RL 的复杂度在算法逻辑（双Q、熵正则、经验回放），不在网络深度。

3. **稠密 reward 是 SAC 快速收敛的关键。** tracking_loss 提供了每一步的物理信号，避免了稀疏 reward 的海量探索需求。

4. **梯度主导性理论证明了 BPTT 的收敛优势。** 不是巧合，是问题结构决定的。SAC 的方差估计 + 随机探索永远无法匹敌 BPTT 的精确梯度。

5. **场景自适应是真实价值。** 47/48 条轨迹 SAC 优于 default 参数，证明了"不同轨迹需要不同参数"的假设。但需要在与 BPTT 调后参数的公平对比中验证增量价值。

---

## 十、常用命令

```powershell
cd differentiable-control/sim

# 0. 推荐使用 pypose 环境的 Python，避免 PowerShell 中 conda activate 未生效
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe -c "import stable_baselines3; print(stable_baselines3.__version__)"

# 1. 环境/语法检查
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe -m py_compile optim/rl_train.py optim/rl_evaluate.py optim/rl_env.py

# 2. DC/BPTT 参数整定主流程（truck_trailer）
# train_batch.py 是 truck_trailer 当前主线，输出 tuned_*.yaml 作为 RL warm-start baseline。
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/train_batch.py `
    --plant truck_trailer `
    --epochs 50

# 3. RL/SAC 从 DC tuned baseline 开始训练（truck_trailer）
# --checkpoint-freq 会周期性保存 sac_model_{N}_steps.zip 和对应 replay buffer。
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_train.py `
    --plant truck_trailer `
    --config "results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml" `
    --total-timesteps 20000 `
    --checkpoint-freq 1000

# 4. 指定输出目录训练，便于断点续训写回同一个 run
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_train.py `
    --plant truck_trailer `
    --config "results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml" `
    --total-timesteps 20000 `
    --checkpoint-freq 1000 `
    --output-dir "results/rl/truck_trailer/202606xx_xxxxxx"

# 5. 从 checkpoint / best_model / final_model 断点续训
# --total-timesteps 表示本次额外继续训练多少步，不是累计总步数。
# 新版本 checkpoint 会保存 replay buffer；旧 best_model.zip 没有 replay buffer 时会用空 buffer 继续。
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_train.py `
    --plant truck_trailer `
    --config "results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml" `
    --resume "results/rl/truck_trailer/20260609_175233/best_model.zip" `
    --output-dir "results/rl/truck_trailer/20260609_175233" `
    --total-timesteps 15000 `
    --checkpoint-freq 1000

# 6. 查看模型实际累计 timesteps
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe -c "from stable_baselines3 import SAC; m=SAC.load(r'results/rl/truck_trailer/20260609_175233/best_model.zip'); print(m.num_timesteps)"

# 7. RL vs DC tuned baseline 评估（默认 48 条标准轨迹）
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_evaluate.py `
    --rl-model "results/rl/truck_trailer/202606xx_xxxxxx/sac_model_final.zip" `
    --dc-config "results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml" `
    --plant truck_trailer `
    --output-dir "results/rl/truck_trailer/202606xx_xxxxxx/evaluation"

# 8. 追加 park_route 综合园区路线评估（48 + 1）
# park_route 是强 OOD / 综合路线，用于诊断泛化边界，不建议直接等同于标准训练集表现。
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_evaluate.py `
    --rl-model "results/rl/truck_trailer/202606xx_xxxxxx/sac_model_final.zip" `
    --dc-config "results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml" `
    --plant truck_trailer `
    --include-park-route `
    --output-dir "results/rl/truck_trailer/202606xx_xxxxxx/evaluation_with_park_route"

# 9. 只训练/评估部分轨迹类型，用于 smoke test 或定位问题
C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_train.py `
    --plant truck_trailer `
    --config "results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml" `
    --trajectories lane_change `
    --total-timesteps 1000 `
    --checkpoint-freq 500

C:\Users\huangjiangyu\.conda\envs\pypose\python.exe optim/rl_evaluate.py `
    --rl-model "results/rl/truck_trailer/202606xx_xxxxxx/sac_model_final.zip" `
    --dc-config "results/training/truck_trailer/20260608_203406_mlp0525/tuned_4740dec_20260608_203243.yaml" `
    --plant truck_trailer `
    --trajectories lane_change `
    --output-dir "results/rl/truck_trailer/202606xx_xxxxxx/evaluation_lane_change"
```

---

## 参考文献

1. Sharma, V. K. et al. "Globally Optimal Policy Gradient Algorithms for Reinforcement Learning with PID Control Policies." NeurIPS, 2025.
2. Wiedemann, N. et al. "Training Efficient Controllers via Analytic Policy Gradient." ICRA, 2023.
3. Kim, G. et al. "Autonomous PID Tuning: SRAIL." IEEE TASE, 2025.
4. Haarnoja, T. et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning." ICML, 2018.
5. Fujimoto, S. et al. "Addressing Function Approximation Error in Actor-Critic Methods (TD3)." ICML, 2018.
6. Amos, B. et al. "Differentiable MPC for End-to-end Planning and Control." NeurIPS, 2018.
7. Romero, A. et al. "Actor-Critic Model Predictive Control (AC4MPC)." arXiv:2306.09852v7, 2025.
