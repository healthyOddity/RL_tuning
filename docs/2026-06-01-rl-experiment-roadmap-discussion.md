# RL 参数整定实验路线与工程优先级讨论

日期：2026-06-01

本文记录一次关于 RL 参数自动整定后续路线的讨论结论。核心背景是：`sim/results/rl/hybrid_v2/20260526_195622/` 已经包含一次 **DC tuned baseline vs DC+RL** 的对比实验，因此后续工作不应重复证明同一件事，而应补齐消融矩阵、泛化验证和训练加速能力。

2026-06-11 更新：当前论文/实验主线已切换到 `truck_trailer`。最新 RL 训练目录为 `sim/results/rl/truck_trailer/20260609_175233/`，其中 `sac_model_final.zip`、`best_model.zip` 和 `sac_model_17000_steps.zip` 的模型内部 `num_timesteps` 均为 `17000`，且 final replay buffer 已保存。该目录尚未生成正式 `evaluation/`，因此下一步应先对 `sac_model_final.zip` 执行 48 条标准轨迹评估，再单独执行 `--include-park-route` 的 OOD/综合路线诊断。当前不应直接进入算法改进或论文结论撰写。

补充背景：另一个调研 session 的成果位于 `C:\Users\huangjiangyu\Desktop\硕士毕设材料\RL参数整定\调研\`。该调研给出的主线不是“普通 SAC 调参”，而是 **BPTT warm-start + Gradient-Informed SAC + 安全/泛化验证**。因此本文的实验路线需要同时满足两件事：

1. 先用已有实现证明 BPTT+SAC 的场景自适应增量；
2. 再逐步推进到调研定义的完整 A+B 组合，即 `BPTT warm-start + 梯度信息增强 SAC 观测`。

## 1. 对 20260526_195622 实验的定位

`sim/results/rl/hybrid_v2/20260526_195622/evaluation/log.txt` 中的评估结果显示：

- DC avg loss = 59.1975
- DC+RL avg loss = 40.8389
- 平均改善 = -31.0%
- DC+RL 优于 DC：48/48 条轨迹

因此，这一次实验可以视为 **DC 与 DC+RL 的整定实验**。它已经回答了一个重要问题：在同一批 48 条标准轨迹上，以 DC tuned 参数作为 baseline，再由 SAC 输出场景相关参数偏移，可以进一步降低 tracking loss。

但这还不是完整消融。它主要证明的是“在当前评估集上，DC+RL 优于 DC”，尚未证明：

- RL 相比 default 的总体收益；
- 不使用 DC warm-start 的纯 RL 是否足够强；
- DC warm-start 对 RL 是否必要；
- 结果是否跨随机种子稳定；
- 结果是否能泛化到未参与训练的轨迹类型、速度档或真实日志场景。

## 2. 建议补齐的消融矩阵

建议不要只做零散 pairwise 对比，而是定义四个系统版本，再统一评估。

| 版本 | 名称 | 参数来源 | 作用 |
|---|---|---|---|
| A | Default | `configs/default.yaml` | 原始基线 |
| B | DC | BPTT/Adam tuned YAML | 验证可微整定收益 |
| C | Pure RL | SAC 从 default 或 random/theory 初值训练 | 验证不依赖 DC 的 RL 能力 |
| D | DC+RL | SAC 以 DC tuned YAML 为 baseline 学偏移 | 验证 RL 场景自适应增量 |

需要报告的对比：

| 对比 | 回答的问题 | 优先级 |
|---|---|---|
| A vs B | DC 相比原始参数提升多少 | 高 |
| A vs C | 纯 RL 相比原始参数是否有效 | 高 |
| A vs D | 完整 pipeline 总收益 | 高 |
| B vs D | RL 在 DC 之上的增量价值 | 已有一组，需多 seed/holdout 补强 |
| C vs D | DC warm-start 是否必要 | 高 |
| B vs C | 梯度优化和纯策略搜索谁更强 | 中 |

这里的“纯 RL”需要明确定义。当前 `RLTuningEnv` 的动作是相对 baseline 的小范围增量，如果 `config_path=None` 或 `config_path=configs/default.yaml`，它仍然是“围绕 default 的 RL 微调”，不是完全随机参数空间里的黑箱 RL。若要更严格地测试纯 RL，可以增加 `random.yaml` 或 `theory.yaml` 初值组，对应已有 `test-initial-param-impact` spec。

调研材料中的主消融更偏论文化，定义为 E0-E6：

| 实验 | 含义 | 当前实现支撑度 |
|---|---|---|
| E0: 纯 BPTT | DC tuned 参数 | 已支持 |
| E1: 纯 SAC | 无 BPTT warm-start 的 RL | 部分支持，需严格定义初值和动作空间 |
| E2: BPTT+SAC | warm-start，无梯度观测 | 当前主要已实现 |
| E3: BPTT+Gradient-Informed SAC | warm-start，有梯度方向/幅值观测 | 尚未实现，是调研定义的完整方法 |
| E4: BPTT+SAC 大动作空间 | 检查动作范围影响 | 尚未系统实现 |
| E5: 随机初始化+Gradient-Informed SAC | 检查梯度信号独立价值 | 尚未实现 |
| E6: BPTT+TD3 | 算法对比 | 尚未实现 |

因此，四臂矩阵适合作为**近期工程消融**；E0-E6 适合作为**论文最终消融矩阵**。两者不是冲突关系，四臂矩阵可以看作 E0/E1/E2 的前置简化版。

## 3. 当前实现对调研主线的支撑度

当前 `sim/optim/rl_env.py`、`rl_train.py`、`rl_evaluate.py` 已经能够支撑调研方案中的一部分核心内容：

| 调研要求 | 当前状态 | 说明 |
|---|---|---|
| BPTT warm-start | 已支持 | `rl_train.py --config configs/tuned/xxx.yaml` 可加载 DC tuned 参数作为 baseline |
| 场景几何特征 | 已支持 | 10 维连续几何特征：曲率、速度、长度、曲率峰值/符号变化等 |
| 参数增量动作 | 已支持 | 11 维动作，围绕 baseline 调 T2/T3/T4/T6 和纵向 PID |
| reward 与 DC loss 对齐 | 已基本支持 | 已有 per-traj baseline loss 归一化和 L2 penalty |
| OOD 检测 | 已支持雏形 | Mahalanobis 距离基于 10 维几何特征 |
| DC vs DC+RL 评估 | 已支持且已有实验 | `20260526_195622` 给出 48/48 胜出结果 |
| Gradient-Informed SAC | 未实现 | 尚未把 BPTT 梯度方向/幅值注入 observation |
| 多算法对比 SAC/TD3/PPO | 未实现 | 当前训练脚本固定 SAC |
| 并行 RL 仿真 | 未实现 | 当前单 `RLTuningEnv` 串行跑完整仿真 |
| safety supervisor | 未实现 | 目前主要是动作范围和参数 clamp，缺少闭环仿真前置检查/回退 |
| OOD/holdout 泛化矩阵 | 未系统实现 | 需要速度、轨迹类型、plant holdout |

结论：当前系统已经能支撑 **方式 B：BPTT -> SAC 递进式场景精调** 的研究内容，也已经有一组有效结果。但它还不能完整支撑调研成果中最强调的 **A+B 组合：BPTT warm-start + Gradient-Informed SAC**。如果论文最终要按调研成果叙事，后续必须补上梯度观测、泛化测试、安全机制和多算法/强基线对比。

## 4. 单场景 BPTT vs 全局 BPTT 是关键前置实验

调研成果中提出了一个很重要的前置判定：**单场景独立 BPTT vs 全局 BPTT**。

目的不是直接证明 RL，而是量化“全局 BPTT 参数是否确实是多场景折中”。具体做法：

1. 对 48 条轨迹分别单独跑 BPTT，得到 48 组 single-trajectory optimum；
2. 与全局 BPTT tuned 参数比较 tracking loss 和参数差异；
3. 统计不同轨迹的最优参数分散程度；
4. 若单场景最优参数差异很大，说明 RL 场景自适应有明确空间；
5. 若差异很小，说明 BPTT 已接近所有场景共同最优，RL 增量可能有限。

建议把这个实验放在完整 E0-E6 之前。它能回答“是否值得继续做 RL”这个根问题，也能为论文中“BPTT 全局折中 vs RL 场景精调”的动机提供直接数据。

## 5. 泛化验证应早于部署验证

当前 48/48 胜出很有价值，但训练和评估如果使用同一批轨迹，就不能排除策略记住了固定场景集合。建议在部署 RL policy 之前补三个 holdout：

1. **速度 holdout**：例如不训练 45/55 kph，只评估 45/55 kph。
2. **轨迹类型 holdout**：例如留出 `s_curve` 或 `clothoid_decel`。
3. **plant holdout**：例如 kinematic/hybrid_v2 训练，truck_trailer 或 dynamic 上评估。

每组至少跑 3 个 seed，报告均值、标准差和逐轨迹胜率。这样才能区分“训练集内调优效果”和“场景自适应泛化能力”。

调研材料还建议增加 OOD 构造集，包括未见曲率、未见速度、曲率+速度复合扰动、轨迹噪声、初始横向/航向偏差等。这部分可以作为第二阶段，不必阻塞当前四臂消融，但应纳入最终论文实验。

## 6. RL policy 作为产物的应用含义

讨论中的“RL policy 应用”指的是：不再把 SAC 最终导出的某一个 YAML 当成主要产物，而是把 `sac_model_final.zip` 看作真正的参数调度器。

典型流程是：

1. 从仿真轨迹或真实日志片段中提取几何/速度特征；
2. 将特征与 baseline 参数快照拼成 observation；
3. SAC policy 推理得到 11 维动作；
4. 将动作解码成控制器参数偏移；
5. 在仿真或实车回放中评估这组参数；
6. 若用于在线控制，则增加参数平滑、变化率限制、OOD 回退和人工确认机制。

所以用户理解是对的：这一步本质上是在真实数据轨迹中推理参数，然后通过回放仿真、离线评估或控车实验验证。它属于“部署 RL 产物 agent”的阶段，应放在基础消融和泛化实验之后。

调研成果强调部署产出形式分为四类：静态全局参数、离散参数组+手动切换、Agent 输出参数、Agent 在线输出增量。当前实现属于“Agent 按轨迹/场景输出参数增量”的离线 episodic 版本；未来若接真实日志或在线控制，需要额外实现参数平滑、推理频率管理和回退机制。

## 7. “RL 只负责调度参数表”与 Safe RL 的关系

“事先整定好参数表，RL 只负责调度”这个理解基本正确，但更准确的说法是：

> 这是 constrained policy / gain scheduling / parameter library selection，比让 RL 连续输出任意参数更安全、更可解释。

如果只让 RL 在一组已验证安全的参数表之间选择或插值，风险会显著小于连续动作 SAC。它可以进一步发展成 safe RL，但二者不是完全等价：

- 只调度安全参数表：安全来自离线验证过的候选集合；
- safe RL：还需要显式约束、shield、风险惩罚、失败回退、约束违规证明或监控；
- 工程部署时推荐二者结合：RL 负责选择/小范围插值，外层 safety shield 负责参数范围、变化率、控制输出和 OOD 回退。

这条路线也更容易落地成量产可解释形式：将 SAC policy 蒸馏成 `speed × curvature × scene_type` 的查表或规则，而不是直接部署神经网络。

调研材料中对应的是“离散安全参数目录”或“安全动作集合”：先离线生成多组已验证安全的候选参数，RL 只在候选集合中选择或插值。这可以作为连续 SAC 的保守 baseline，也可以作为工程部署版本。

## 8. 并行仿真加速的优先级

并行仿真应提升为 P0 或 P0.5。原因很直接：如果一次 20000 epochs/timesteps 需要十几个小时，那么多 seed、holdout、多算法和消融矩阵都会被训练时间卡住。

当前代码中，`rl_train.py` 直接创建单个 `RLTuningEnv` 和单个 `eval_env`，没有使用 `DummyVecEnv` 或 `SubprocVecEnv`。每个 RL step 的主要开销是 `RLTuningEnv.step()` 内部调用 `run_simulation()` 跑完整轨迹，而不是 SAC 网络前向或反向。因此并行化的方向判断是正确的：

- 仿真主要吃 CPU；
- SAC 网络训练可以用 GPU，但当前瓶颈大概率不在 GPU；
- 多环境并行可以让多个 CPU worker 同时跑不同轨迹；
- GPU 只需要消费并行环境返回的 batch transition。

本机通过 `[Environment]::ProcessorCount` 读取到 24 个逻辑处理器。WMI 查询 CPU 型号和内存时权限被拒绝，因此还需要用任务管理器或其他方式确认物理核心数和内存。但从 24 逻辑处理器看，先试 `n_envs=4/8/12` 是合理的。

预期收益不应按 24 倍估计。考虑 Python 进程开销、PyTorch/NumPy 线程竞争、环境初始化、baseline loss 预计算和内存复制，现实目标可以设为：

- `n_envs=4`：2.5-3.5 倍加速；
- `n_envs=8`：4-6 倍加速；
- `n_envs=12`：需实测，可能受内存和调度开销限制。

## 9. 并行加速好不好搞

难度中等，不是纯改一行。

主要风险点：

1. **环境可 pickle**：`SubprocVecEnv` 要把环境构造函数发到子进程，`RLTuningEnv` 里的 controller、trajectory cache、torch tensor 是否都能在 Windows spawn 模式下干净初始化，需要实测。
2. **baseline loss 预计算会重复发生**：当前每个 env 初始化都会预计算 48 条 baseline loss。若 `n_envs=8`，初始化会重复 8 次，成本很高。需要把 baseline loss cache 做成可复用文件，或在并行训练时关闭子环境重复预计算。
3. **轨迹采样要拆开**：如果每个 env 都随机采样 48 条，能工作，但效率不一定最佳。更好的方式是支持固定轨迹子集或按 env rank 分配轨迹。
4. **评估 callback 成本**：`EvalCallback` 也会定期跑评估，可能抵消一部分训练加速。需要调低 eval 频率或单独评估。
5. **Windows 多进程启动开销**：需要使用 `if __name__ == '__main__'` 保护，避免子进程重复执行训练入口。

建议先做一个最小性能探针，而不是直接重构完整训练：

1. 写 `bench_rl_env_step.py`：单 env 跑 20 step，记录平均 step 时间。
2. 用 `DummyVecEnv(n=4)` 跑同样步数，确认接口兼容。
3. 用 `SubprocVecEnv(n=4/8)` 跑同样步数，确认 Windows 下可启动。
4. 比较 wall time、CPU 占用、内存占用。
5. 若 `n_envs=8` 至少有 3 倍加速，再接入 `rl_train.py --n-envs`。

## 10. discussion.md 是否满足调研成果

修订后的本文基本满足“近期实验路线讨论”的需求，但不应替代完整调研方案。

满足的部分：

- 明确承认 `20260526_195622` 已经是 DC vs DC+RL 实验；
- 补齐 Default / DC / Pure RL / DC+RL 的近期消融矩阵；
- 说明 RL policy 应用是在真实/仿真轨迹中推理参数并评估；
- 说明安全参数表调度与 safe RL 的关系；
- 将并行仿真提升到高优先级；
- 补充当前实现对调研目标的支撑度判断；
- 补充单场景 BPTT vs 全局 BPTT 这个关键前置实验；
- 补充 E0-E6 与四臂矩阵的对应关系。

仍不足的部分：

- 没有展开 Gradient-Informed SAC 的具体实现设计；
- 没有定义梯度方向/幅值 observation 的编码和归一化；
- 没有定义 safety supervisor 的接口；
- 没有定义 OOD 数据生成脚本；
- 没有给出多 seed 统计检验模板；
- 没有把真实日志 pipeline 的特征提取接口写成任务。

因此，本文适合作为**实验路线讨论记录**。如果要作为执行 spec，还需要另写一份更工程化的 implementation plan。

## 11. 建议下一步顺序

2026-06-11 更新后的判断：用户提出的“补算法对比、做 rolling agent 部署、做纯 RL/纯 DC/RL+DC 消融，然后进入论文写作”方向总体正确，但执行顺序需要约束。当前最重要的不是立即铺开 DDPG/TD3/PPO，而是先证明当前 SAC 版 DC+RL 在 truck_trailer 48 条标准轨迹上确实优于 DC，并把四臂消融钉住。

### 11.1 近期 P0：先闭合主链路证据

1. **E01：当前 SAC final 模型正式评估**  
   对 `sim/results/rl/truck_trailer/20260609_175233/sac_model_final.zip` 执行 48 条标准轨迹评估，回答“DC+RL 是否优于 DC”。

2. **E01P：park_route 单独诊断**  
   `park_route` 不应混入 48 条标准轨迹结论。它是强 OOD/复合路线压力测试，用来判断 one-shot trajectory-level agent 的能力边界。

3. **E03/E04：四臂消融**  
   尽快补齐 Default / DC / Pure RL / DC+RL。当前 E02 Default vs DC 已由 `20260608_203406_mlp0525` 支撑，因此剩余关键是 Pure RL 和 DC+RL evaluation。

这一轮结束后，才能严谨回答：

```text
DC 是否有效？RL 是否在 DC 之上有增量？DC warm-start 是否必要？当前主线是否值得继续做更复杂方法？
```

### 11.2 P1：泛化、稳定性和诊断能力

如果 P0 成立，再补：

1. **per-scenario 诊断输出**：`rl_evaluate.py` 保存 `rl_loss/dc_loss/delta/lat/head/is_ood/ood_distance/rl_action`。
2. **多 seed 稳定性**：至少 3 个 seed，避免单次训练偶然性。
3. **holdout 泛化**：轨迹类型 holdout、速度 holdout、随机曲率/随机换道长度、双 S 弯。
4. **OOD 策略**：评估 `none/scale/fallback_dc` 等 OOD action 策略。

没有这一层，论文只能声称“固定场景集合上有效”，不能声称“泛化”。

### 11.3 P1/P2：Stable-Baselines3 算法对比

算法对比应该作为支撑实验，而不是主线。推荐顺序：

1. **SAC**：当前主方法，先稳定。
2. **TD3**：最值得优先比较，连续动作、off-policy，与 SAC 接近。
3. **DDPG**：可作为较弱/经典 baseline，但训练可能更不稳定。
4. **PPO**：可做，但当前环境更接近 contextual bandit/one-step episodic，PPO 样本效率可能较差。

算法对比回答的问题是：

```text
收益来自“BPTT warm-start + RL 场景自适应”框架，还是 SAC 特定算法偶然表现好？
```

它不应替代四臂消融。若四臂消融不成立，算法对比的论文价值会很弱。

### 11.4 P2：rolling preview / 在线参数调度部署

rolling 参数调度是部署方向，也是解释 `park_route` 的自然后续。建议先做 inference-only 原型，而不是立刻重训 rolling RL：

1. 把 `park_route` 切成未来 5s 局部窗口。
2. 对每个窗口提取局部特征。
3. 用当前 SAC agent 输出 action timeline。
4. 检查 action 是否贴边、是否跳变。
5. 加参数变化率限制、平滑切换和 OOD gating。
6. 做 rolling simulation，对比 `DC tuned / global RL / rolling RL`。

若 rolling RL 明显改善 `park_route`，再考虑构建真正的多步 rolling RL environment。此时 observation 应包含当前状态、当前参数、未来预瞄窗口特征；reward 应包含局部跟踪误差、控制平顺性和参数变化惩罚。

### 11.5 什么时候进入论文写作

可以现在就开始准备论文材料，但完整论文结论应等以下证据齐备：

| 证据 | 作用 |
|---|---|
| 四臂消融表 | 支撑主 claim：DC warm-start + RL 是否必要 |
| 至少一组泛化/holdout 或 park_route+rolling 结果 | 支撑“场景自适应/复杂路线应用” |
| 算法对比表 | 支撑 SAC 不是偶然选择 |
| 失败/边界分析 | 支撑 limitation 和未来工作 |

因此，`ml-paper-writing` 可以先用于整理方法、实验设计、图表模板和论文大纲；真正写摘要和实验结论时，应等待 E01/E03/E04/E09 或 rolling 关键结果完成。

### 11.6 原始路线顺序

原始推荐顺序如下，现保留为中长期路线：

1. **先做并行仿真 benchmark**：只测 step 吞吐，不训练完整 SAC。
2. **实现 `--n-envs` 并行训练**：先支持 SAC，别同时引入多算法。
3. **补单场景 BPTT vs 全局 BPTT**：判断是否存在足够大的场景最优参数差异。
4. **补齐四臂消融矩阵**：Default / DC / Pure RL / DC+RL。
5. **做 holdout 泛化实验**：速度、轨迹类型、plant。
6. **实现 Gradient-Informed SAC**：在 E2 已成立后推进 E3。
7. **补 safety supervisor / 安全参数目录 baseline**：作为部署和论文安全性章节支撑。
8. **再进入 RL policy 应用阶段**：真实日志特征提取、policy 推理、参数平滑、离线回放评估。
9. **最后考虑安全参数表调度/蒸馏**：把连续 SAC policy 转成可解释、可部署的参数调度表。

这个顺序的理由是：并行加速会直接降低所有后续实验成本；消融矩阵决定 RL 是否值得继续投入；真实数据应用和 safe scheduling 是部署阶段问题，应该在基础结论成立后再做。
