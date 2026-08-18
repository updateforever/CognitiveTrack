# Trajectory-Utility GRPO 设计

## 1. 目标与差异

现有 GRPO 跟踪工作已经覆盖严格格式、当前帧 bbox IoU 和文本更新带来的当前帧辅助
tracker 增益。[R1-Track](https://arxiv.org/abs/2506.21980) 直接对 VLM tracking 使用规则
reward；[ReasoningTrack](https://arxiv.org/abs/2508.05221) 比较更新前后文本在当前帧的
辅助 tracker IoU，但使用固定文本更新间隔。

CognitiveTrack 真正要学的是：一条状态更新写入后，是否在之后一段轨迹中帮助模型维持
同一身份。项目暂用 Trajectory-Utility GRPO（TU-GRPO）表示这个方向。它仍需实现、
消融和更全面检索，当前不能作为已经验证的创新结论。

## 2. 训练样本与生成

每个 GRPO 样本是一个带 GT 的事件中心短序列：

- 在线可见输入：初始身份、当前维护状态、过去三图上下文；
- 当前 GT：presence 与 bbox；
- 标注参考：memory SFT 的人工/银标事件；
- reward-only：当前之后长度为 `H` 的 GT presence/bbox；
- 同组从同一策略采样 `G=4–8` 个严格 JSON 候选。

未来帧只由 reward 计算器读取，绝不进入当前模型消息或 assistant reference。输出仍是
直接结构化答案，不加入公开 CoT。VideoAuto-R1 对视频感知任务的结果表明，强制长推理
不一定优于直接回答；因此“思考文本”不是当前贡献前提。

## 3. 奖励定义

对候选输出 `y` 定义：

```text
R(y) = w_fmt   * R_format
     + w_cur   * R_current
     + w_evt   * R_event
     + w_ground* R_ground
     + w_traj  * Delta-U-H
     - l_churn * P_update
     - l_len   * P_length
     - l_drift * P_identity_drift
```

- `R_format`：唯一合法 JSON、字段顺序、状态/bbox/memory 一致；
- `R_current`：当前 presence 判别和 present bbox IoU；
- `R_event`：与人工/银标 update/keep 决策的一致性，只作冷启动辅助；
- `R_ground`：新状态对 GT target region 的对齐、对干扰物的 margin、与初始身份一致；
- `Delta-U-H`：接受候选状态相对保留旧状态的未来轨迹效用差；
- `P_update`：无收益更新、过密更新和持续缺失时的重复 absent 更新；确认的消失转折不
  因非空 memory 自动受罚；
- `P_length`：超过简洁状态预算；
- `P_identity_drift`：类别、颜色/部件等永久身份冲突或更匹配干扰物。

当前跟踪 reward 使用客观 GT，不用 caption 教师代替。伪标签只影响事件决策分支，从而
避免教师偏差主导全部优势。

## 4. 反事实轨迹效用

对同一候选状态 `m'` 做两条 reward-only 回放：

```text
U_accept = U(future clip | initial identity, m')
U_keep   = U(future clip | initial identity, old state)
Delta-U-H = U_accept - U_keep
```

`U` 初始采用 future presence F1、可见帧 IoU/AUC 和重现恢复的加权和。两条回放必须使用
相同未来帧、历史框、采样设置和冻结 evaluator；唯一变量是接受 `m'` 还是保留旧状态。
这样奖励的是“状态文本对未来身份保持的边际作用”，不是新文本碰巧描述当前帧。

为了控制算力，分两级实现：

1. **缓存代理回放**：在大部分样本上用冻结 region-text encoder、target/distractor crop
   和未来 GT 计算文本持续对齐差；
2. **真实短轨迹回放**：在事件丰富子集上，用冻结 core/memory-SFT evaluator 分别运行
   accept/keep 分支，得到真实 tracking utility。

真实回放可缓存视觉特征和旧状态分支。训练中的 actor 与 reward evaluator 分离，后者
固定 revision，避免 reward 随 actor 同步漂移。

## 5. GRPO 稳定性与防投机

- 同组样本必须来自同一事件；只在组内做优势归一化；
- 全组 reward 完全相同时跳过该组，避免零方差数值问题；
- 格式错误和越界框直接给确定性低分；`absent + 非空 memory` 只有在消失转折语义与
  时序证据一致时合法，持续缺失的重复更新降分；
- 空泛文本如 “same object” 不应因短而得高分，必须经过 target/distractor margin；
- 复制初始描述只有在 keep 真正更优时获益，不能一律奖励或惩罚；
- 对同一状态的近义改写做语义去重，防止更新 churn；
- reward-only future horizon 按序列边界裁剪，并记录实际 `H`；
- 训练/验证按完整序列划分，不把相邻事件拆到两侧。

## 6. 训练阶梯

不直接从 base 做 GRPO。固定顺序为：

1. 混合 SFT 中的 tracking 数据：稳定存在性、bbox 和格式；
2. 混合 SFT 中的 state-update 数据：让模型先学会可接受的 update/null 分布；
3. `format + current` reward smoke；
4. 加入 `event + ground`，验证没有身份漂移；
5. 在小型事件集加入缓存 `Delta-U-H`；
6. 最后加入真实双分支短轨迹回放。

每一级都先做 100–500 样本 reward replay，人工检查 reward 排序，再执行短训练。现有
`cogtrack/training/rewards.py` 可复用格式、presence 和 bbox reward，但 trajectory reward
与 ms-swift GRPO 配置尚待实现。

## 7. 必做消融与成功标准

至少比较：

- tracking/state-update 混合 SFT；
- `+ current-frame GRPO`；
- `+ event/ground reward`；
- `+ cached trajectory reward`；
- `+ true counterfactual trajectory reward`；
- 去掉 update、length 或 drift penalty；
- 固定间隔更新 vs 模型自主更新；
- `H` 和 `G` 的敏感性。

成功不能只看 reward 上升。完整方法需要同时满足：CognitiveBench-Tiny 的主跟踪指标不
下降，Full 的 hold-last/observation-only 有增益，presence 与重现恢复改善，over-update
和身份矛盾率受控，并且接受更新后的 `Delta-AUC@H` 显著优于 forced-null。若只改善格式
而不改善轨迹效用，应将 GRPO 降级为工程增强，不作为核心论文贡献。
