"""
SAC 网络结构探索脚本 — 加载训练好的 RL 模型，逐层展示网络结构。

运行:
    cd sim
    python tests/explore_sac_network.py
"""
import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np

# 1. 加载模型 ------------------------------------------------------------------
MODEL_PATH = 'results/rl/kinematic/20260519_062846/sac_model_final'
from stable_baselines3 import SAC
model = SAC.load(MODEL_PATH, device='cpu')

print("=" * 70)
print("第一部分：模型顶层结构")
print("=" * 70)
print(model.policy)

# 2. 演员网络 (Actor) ---------------------------------------------------------
print("\n" + "=" * 70)
print("第二部分：Actor (策略网络)")
print("作用：输入观测 → 输出动作均值 + 标准差")
print("=" * 70)

actor = model.policy.actor
print(actor)

print("\n逐层参数:")
total_params_actor = 0
for name, param in actor.named_parameters():
    shape = list(param.shape)
    num = param.numel()
    total_params_actor += num
    print(f"  {name:35s}  shape={str(shape):20s}  params={num:,}")

# 3. Critical 网络 (两个 Q 网络) ----------------------------------------------
print("\n" + "=" * 70)
print("第三部分：Critic (两个 Q 网络)")
print("作用：输入(观测+动作) → 输出该动作的期望价值 Q(s,a)")
print("SB3 用双 Q 技巧：两个独立的 Q 网络，取 min 防止过估计")
print("=" * 70)

critic = model.policy.critic
print(critic)

total_params_critic = 0
for name, param in critic.named_parameters():
    shape = list(param.shape)
    num = param.numel()
    total_params_critic += num
    print(f"  {name:35s}  shape={str(shape):20s}  params={num:,}")

# 4. 总参数量 ------------------------------------------------------------------
total = total_params_actor + total_params_critic
print("\n" + "=" * 70)
print(f"第四部分：参数量汇总")
print(f"  Actor:                     {total_params_actor:>8,}")
print(f"  Critic (含两个Q+目标网络):  {total_params_critic:>8,}")
print(f"  总计:                       {total:>8,}")
print("=" * 70)

# 5. 网络是如何自动创建的？-----------------------------------------------------
print("\n" + "=" * 70)
print("第五部分：SAC 如何根据你的环境自动创建网络")
print("=" * 70)

obs_dim = model.observation_space.shape[0]
act_dim = model.action_space.shape[0]
print(f"  你的观测空间: Box(shape=({obs_dim},))  → 输入维度 = {obs_dim}")
print(f"  你的动作空间: Box(shape=({act_dim},))  → 输出维度 = {act_dim}")
print()
print("  SAC('MlpPolicy', env) 内部自动执行:")
print()
print(f"  1. 读取 env.observation_space → 确定输入维度 = {obs_dim}")
print(f"  2. 读取 env.action_space → 确定输出维度 = {act_dim}")
print(f"  3. 默认配置 net_arch = [256, 256]（2个隐藏层，每层256神经元）")
print(f"  4. 调用 get_actor_critic_arch() → actor和critic都用 [256, 256]")
print()
print(f"  5. 构建 Actor:")
print(f"     nn.Flatten()                           # 展平输入")
print(f"     nn.Linear({obs_dim}, 256) + ReLU()     # 第1隐藏层")
print(f"     nn.Linear(256, 256) + ReLU()           # 第2隐藏层")
print(f"     nn.Linear(256, {act_dim})              # μ 输出层(均值)")
print(f"     nn.Linear(256, {act_dim})              # σ 输出层(对数标准差)")
print(f"     输出: μ({act_dim}) + log_σ({act_dim}) → 采样 → 动作({act_dim})")
print()
print(f"  6. 构建 Critic (两个相同的Q网络):")
print(f"     NN1: nn.Linear({obs_dim+act_dim}, 256) → ReLU → Linear(256,256) → ReLU → Linear(256,1)")
print(f"     NN2: nn.Linear({obs_dim+act_dim}, 256) → ReLU → Linear(256,256) → ReLU → Linear(256,1)")
print(f"     训练时取 min(Q1, Q2)，防止 Q 值过估计")
print()
print(f"  7. 目标网络: Critic 的拷贝，软更新 τ=0.005")
print(f"     θ_target ← τ·θ + (1-τ)·θ_target")

# 6. 演示一次前向推理 ----------------------------------------------------------
print("\n" + "=" * 70)
print("第六部分：一次前向推理演示")
print("=" * 70)

# 创建一个假观测（21维）
dummy_obs = torch.randn(1, 21)

# Actor 前向
with torch.no_grad():
    action = actor(dummy_obs, deterministic=True)
print(f"  输入: torch.randn(1, 21)")
print(f"  输出: action = {action.squeeze().numpy()}")
print(f"  动作范围: [{action.min().item():.3f}, {action.max().item():.3f}]")

# 8. SAC 如何输出动作？数学说明 -------------------------------------------------
print("\n" + "=" * 70)
print("第七部分：SAC 动作输出的数学")
print("=" * 70)
print("""
  SAC 策略是 Squashed Gaussian：
    1. 网络输出 μ(均值) 和 log_σ(对数标准差)
    2. log_σ → σ = exp(log_σ)，clamp 在 [e^-20, e^2] ≈ [2e-9, 7.4]
    3. 从 N(μ, σ²) 采样得到 u
    4. 通过 tanh 压缩到 [-1, 1]：a = tanh(u)
    5. 训练时：a = tanh(μ + σ·ε)，ε ~ N(0,1)
    6. 推理时 (deterministic=True)：a = tanh(μ)

  你的动作空间是 [-0.3, 0.3] × 4 + [-0.1, 0.1] × 3 + ...
  SAC 内部输出在 [-1, 1]（tanh），env.step() 时会线性缩放到实际范围。
""")

print("=" * 70)
print("网络结构探索完成！")
print("=" * 70)
