# 统一时序跟踪与记忆混合训练设计（归档）

> 历史设计说明：当前首轮训练已改为 VLT-v6 core SFT，只监督存在性与 bbox，并对
> `memory_update` 值做 loss mask；不再要求在坐标训练前完成记忆标签或 GRPO。当前入口
> 与监督边界见 [VLT-v6 core SFT](../vlt_v6_core_sft.md)。本文保留作为后续记忆训练与
> 长时 case 分桶参考。

> 2026-08-13 设计草案，供后续 AI 和研究讨论使用。本文替代原先依次训练
> Stage-1 pair、Stage-2 mosaic、Stage-3 memory 的规划。已经完成的 Stage-1/2
> 实验仍是有效历史基线，但下一版正式数据与模型不再按能力拆成三次 SFT。

## 1. 新的核心决策

下一版采用一次统一混合 LoRA SFT，让训练数据直接覆盖正式在线推理会遇到的上下文：

- 所有用于指认目标的参考图和可信历史图都直接画框；不再在用户 Prompt 中提供
  reference bbox 数值或 ms-swift `<bbox>` 输入占位符。
- “所有图都画框”只指过去的目标参考图。当前待预测的完整搜索图绝对不能画框，
  否则构成当前标签泄漏。
- 首帧身份锚点保留，但不再依赖坐标文本。最近可信观测必须作为显式历史证据；
  稀疏推理中它是“上一次实际 VLM 观测”，不一定是视频的上一帧。
- 输入始终使用完整图像，不裁成目标模板。画框只是视觉指代标记，不能抹除场景，
  也不能把历史位置当作当前位置先验。
- 主实验不输入数据集额外提供的自然语言描述，只用首帧视觉框定义实例；语言描述作为
  单独消融，避免不同数据源的标注覆盖率造成混杂。
- 统一监督三个字段：`target_status`、当前帧 bbox、`memory_update`。普通样本的
  `memory_update` 为 `null`，只有经过可靠来源或审核的稳定外观变化才为短文本。
- pair、mosaic、长间隔、消失/重现、干净/扰动历史以及记忆样本在一个训练集中
  混合，一次训练完成；不再顺序叠加三轮 LoRA。
- 仍只维护一个 LoRA adapter。现有 Stage-2 adapter 已包含旧 Stage-1 能力，可作为
  新统一训练的初始化对照，但是否从基座或 Stage-2 初始化必须通过验证集消融决定。

## 2. 正式在线输入范式

### 2.1 推荐的三图表示

为控制视觉 token，首版采用三张模型输入：

1. **Image 1: identity anchor**：完整首帧，目标框直接绘制在图上；永久保留，不更新。
2. **Image 2: trusted history mosaic**：0–4 个过去可信观测按时间排序组成的 mosaic，
   每个 panel 都画对应的历史预测框；最后一个 panel 是最近可信观测。没有动态历史时
   可省略 Image 2，退化为两图输入。
3. **Final image: current search frame**：未裁剪、未画框的当前完整帧，是唯一需要输出
   bbox 的图像。

模型不读取绝对帧号。图片顺序已经表达时间关系；运行时 metadata 仍记录真实 frame ID
用于审计。Prompt 必须明确最后一张图才是当前搜索图，所有有框图片都来自过去。

### 2.2 为什么同时保留首帧与最近可信观测

- 首帧框提供不会漂移的实例身份锚点；
- 最近可信观测提供当前外观、姿态和局部时序状态；
- 更早 mosaic 帮助跨越视角、服饰、构型和光照变化；
- 当前帧仍要求全图搜索，尤其是目标消失、镜头切换和重现时，不能围绕旧框局部搜索。

只滚动上一预测会把一次误跟踪自我强化；只看首帧又难以适应长时外观变化。正式范式
因此是“永久首帧锚点 + 门控动态历史”，不是二选一。

### 2.3 历史更新

历史框只能来自过去预测或训练时对过去 GT 的模拟，不得来自 current/future GT。在线时：

- `execution=ok + present + 合法 bbox` 才能成为候选；
- 最近一次预测不等于最近一次可信预测，须经过身份/连续性门控；
- `parse_error`、`model_error`、`skipped`、`absent` 都不能生成历史框；
- absent 期间保留最后可信历史，重现时仍全局搜索；
- 历史 mosaic 允许轻微框误差，训练必须包含与线上预测误差相似的样本。

## 3. 统一输出协议

Qwen3-VL 统一使用三字段严格 JSON：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":null}
```

或：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":"Rear view reveals two stable white stripes."}
```

约束保持不变：

- `present` 必须有当前搜索图上的 bbox；`absent` 必须是 bbox `null`；
- absent 的 `memory_update` 必须为 `null`；
- 普通位移、尺度变化、模糊、遮挡、动作或场景变化都不触发语义更新；
- 非空文本只描述新的、稳定的、可视觉核验且对未来身份判别有用的增量；
- 不输出解释、置信度、旧六分类、身份标签或额外字段。

输入框已经作为图像像素的一部分，不再走 Qwen grounding 输入对象；输出 bbox 仍使用
Qwen3 官方 `[0,1000] xyxy` 协议，并在 ms-swift assistant 侧使用官方 `<bbox>` 与
`objects.image_id` 绑定最后一张当前图。输入指代方式改变不等于输出坐标协议改变。

## 4. 已冻结的 v5 Prompt 语义

下面语义已落实为 `cogtrack/prompts/visual_tracking.py` 的 v5.0.0。代码会按两图退化或
三图 mosaic 生成准确的图片编号；本节保留可读版，运行时以版本化代码为准。

```text
System:
You are a rigorous long-term single-object tracking model. Track the exact instance
marked by boxes in the past reference images. Similar-looking objects are distractors.
Return exactly one JSON object without Markdown or additional text.

User:
Task: find the referenced target in the final unmarked search image.

All boxed images are past observations. The boxes identify the target in those past
images; they are not location hints for the final image.
- Image 1 is the permanent identity anchor.
- If a history mosaic is provided, its panels are chronological trusted observations;
  the last panel is the most recent trusted observation. History boxes may be imperfect.
- The final image is the current full search frame and contains no annotation.

Decision order:
1. Anchor identity to Image 1.
2. Use later boxed history only as supporting evidence for appearance evolution.
3. Search the final image globally and reject same-category distractors.
4. Output present with a box only if the exact target is visible and localizable;
   otherwise output absent with a null box.
5. Set memory_update to a short new visual delta only after a stable, materially useful
   identity cue appears. Otherwise use JSON null.

Return exactly:
{"target_status":"present | absent",
 "bbox_norm1000_xyxy":[x1,y1,x2,y2] or null,
 "memory_update":"short new stable visual delta" or null}
```

已经冻结：历史称为 `accepted past observations whose boxes may be imperfect`；首版
使用版本化红框与自适应线宽；两图和三图复用同一决策语义与输出协议。颜色/线宽增强
保留为后续消融，不混进第一个可复现 baseline。

## 5. 统一训练数据组成

所有 split 仍按完整物理序列划分。建议先以 case 类型分桶，再按配额混合，而不是先生成
海量普通帧后随机采样。首版建议比例是待验证的起点：

| 数据桶 | 建议占比 | 主要监督 |
| --- | ---: | --- |
| 单参考图 / 无动态历史 | 20% | 冷启动、历史不可用时不退化 |
| 干净历史 mosaic | 35% | 常规时序定位与长间隔外观适应 |
| 扰动历史 mosaic | 15% | jitter/stale/drop/sparse 历史鲁棒性 |
| 消失、消失边界与首次重现 | 20% | absent、全局重定位、拒绝旧位置先验 |
| 经审核的语义变化事件与 hard-null | 10% | memory_update 时机和文本 |

这五类不是独立数据集；同一 case 还要按 LaSOT/TNL2K/MGIT 来源、present/absent、
reference-current gap、历史长度分层。最终比例根据 val 的 presence、bbox、过度更新率和
漏更新率调整。

### 5.1 历史框来源混合

训练不能永远看到完美 GT 框，否则线上预测历史产生 exposure bias。历史框建议混合：

- 50% 过去 GT 框，建立清晰任务语义；
- 25% 对过去 GT 做小幅、仍覆盖目标主体的 jitter；
- 15% 使用旧 Stage-2 模型在训练序列上的真实 rollout 预测，且保存模型/commit 来源；
- 10% stale/drop/sparse 等困难历史。

这些比例是草案。严重错到另一实例的框不能标成 `trusted target`；这类 case 若要加入，
必须显式设计为“历史可能错误、首帧优先”的纠错任务，并单独评测，不能混入普通样本。

### 5.2 参考图采样

- Image 1 首选真实序列首个合法 present 初始化帧，最大程度贴合 benchmark；
- 为防首帧质量过差和增加覆盖，可保留少量“严格更早的 present anchor”增强，但必须
  单独标记，验证集只用真实首帧范式；
- 最近可信观测优先靠近 current，但同时覆盖稀疏 observation 的长 gap；
- 所有历史帧严格早于 current，未来确认帧永远不能进入模型输入。

## 6. MGIT 文本与记忆标签

MGIT 的 `action/activity/story/appearance` 继续只用于候选挖掘、审核上下文和弱监督：

- `action` 用于定位事件边界并生成动作/场景变化的 hard-null；
- `activity` 帮助判断变化是否持续或重复；
- `story` 只供离线审核理解完整视频，绝不能进入在线模型输入；
- `appearance` 差异只召回正候选，必须经过当前/过去图像核验和未来帧稳定性确认。

未来确认帧和未来文本只能放在 `annotation_only`，导出训练 JSONL 时物理删除。普通 bbox
数据不能伪造非空记忆文本，也不能把所有未审核普通帧机械标成 hard-null。建议建立
500–1,000 个双人审核的 memory val/test case；训练正例可采用高置信规则加分层抽检，
但需保留标签来源、规则版本和审核状态。

## 7. 训练与消融

下一版正式实验只进行一次统一三字段 LoRA SFT。至少比较：

1. Qwen3-VL-4B 基座；
2. 已完成的旧二阶段 Stage-2 adapter；
3. 从基座开始的统一混合 SFT；
4. 从旧 Stage-2 adapter 继续的统一混合 SFT。

主模型由验证集选择，不能因旧 Stage-2 loss 较低就默认继续训练一定更好。统一数据必须
先做两步 smoke，再做小规模过拟合和 processor 回放，确认输入图画框没有污染当前输出
grounding。GRPO 仅在 SFT benchmark、memory 时机标签和 reward 回放可靠后进行。

必须保留的消融包括：

- 首帧框 + 最近历史 vs 仅首帧框 vs 仅最近历史；
- 视觉画框 vs 旧坐标文本输入；
- clean history vs rollout/noisy history；
- memory 输入/输出开启与关闭；
- pair-only、mosaic-only 与统一混合。

## 8. 验收标准与未完成工作

数据验收至少统计来源、序列、case 桶、present/absent、历史长度、gap、历史框来源、
扰动类型、非空 memory 比例、重复文本率和 split 泄漏。评测至少报告：

- presence precision/recall 与工程错误率；
- bbox AUC/OP50/OP75/P/Pnorm；
- 消失、首次重现和长 gap；
- clean/corrupted history 配对；
- memory 更新 precision/recall、过度更新率、漏更新率；
- 一条记忆进入后对后续跟踪的因果收益，而不只是文本是否通顺。

截至 2026-08-13，在线 anchor/history 共享绘框、v5 Prompt、tracker 配置、三字段
canonical/ms-swift 导出和跨帧 semantic proposal 确认门控已经实现，并完成小样本图片、
processor 和单 case 模型 smoke。仍未完成：

- 论文版五类 case 的精确配额 planner 与 rollout-history 生成；
- 带可靠 provenance 的正式 memory label manifest；
- visual-v5 正式三字段混合数据包与 LoRA；
- CognitiveBench-Tiny 聚合对比和 Full 主表。

任何 AI 不得把工程 smoke、旧 Stage-1/2 或 `feasibility_null` 小数据描述为 visual-v5
正式训练已经完成。
