# Stage-2 时序上下文与 Stage-3 记忆数据设计

本文承接 Stage-1 的 `present/absent + bbox` 监督，定义后续两阶段的数据单位、
标签来源和泄漏边界。Stage-2 可以由跟踪 GT 自动构造；Stage-3 优先复用 MGIT 已有
的多粒度文本，采用“文本弱监督 + 视觉/时间过滤 + 小规模抽检”的低成本路线。
普通 bbox 标签仍不能伪造语义记忆。

## 1. 共同约束

- 数据划分单位始终是完整物理序列，同一变化事件附近的帧不得跨 split。
- reference 必须来自同序列、严格早于 current 的真实 present 帧。
- current 或未来帧 GT 不得进入模型输入；GT 只用于生成监督答案和离线审计。
- reference/current 都保持完整图像，不裁剪、不画 GT 框。reference bbox 通过
  ms-swift `<bbox> + objects.image_id=0` 传入。
- absent 负样本只来自同序列真实消失帧，不使用错配序列或人工抹除。
- Qwen2.5-VL 与 Qwen3-VL 继续使用各自官方 grounding 坐标视图。

## 2. Stage-2：Temporal Context

### 2.1 训练单位

每个 case 使用三张模型输入：

1. Image 1：较早的完整 present reference，以及文本传入的 reference bbox；
2. Image 2：位于 reference 与 current 之间、按时间排序的 2–4 个可信 present
   观测组成的 mosaic；
3. Image 3：未标注的完整 current frame。

输出仍严格只有 `target_status` 和 bbox 两个字段，不加入 `memory_update`。
历史 mosaic 是 teacher-forced 的过去可信观测，可使用过去 GT bbox模拟在线已接受的
预测框；它不是 current 标签泄漏。绘制样式必须与在线 `ContextBuilder` 一致。

### 2.2 case 桶

同一份 sampling plan 应记录 case 类型，至少覆盖：

- 普通 present：尺度、姿态、视角、照明和背景变化；
- 长间隔 present：reference/current 间隔覆盖不同时间分位数；
- disappearance boundary：消失区间起点及区间内部；
- first reappearance：每段 absent 后第一个可定位 present；
- post-reappearance：重现后若干稳定 present；
- history sparse：只有 1–2 个可用历史时退化或稀疏上下文。

不凭 bbox 标签制造“同类干扰物”或身份困难负例。若要加入这类样本，必须有独立、
可审核的实例身份标注。

### 2.3 推荐配方

- 先复用 Stage-1 的 current cases，保证 pair/mosaic 可做严格配对消融；
- 每个 current 同时保留 pair 和 mosaic 的 case key，但训练时采用约
  `25% pair replay + 75% mosaic`，避免遗忘无历史场景；
- history panel 数在 2、3、4 之间分层，不固定为同一种长度；
- history mosaic 不写绝对帧号或时间戳；1–2 帧纵向排列，3–4 帧使用两列网格，
  避免横向长条在视觉 processor resize 后把目标压得过小；
- 第一版只使用正确、过去、present 的 teacher-forced history；后续鲁棒性版本只做
  history drop/stale/sparse，不注入未经标注的错误身份框；
- 验证集同时报告 pair 与 mosaic，按相同序列和 current frame 配对比较。

### 2.4 历史框鲁棒性版本

干净 mosaic 之外，正式 Stage-2 训练再加入约 15% 的 corrupted-history case。每条
只扰动一个历史 panel，current/reference 和监督 bbox 保持干净；扰动类型固定为：

- `jitter_box`：平移并轻微改变历史框尺度；
- `stale_box`：把相邻历史 panel 的旧框错套到当前 panel。

扰动由序列、reference、current 的稳定 hash 决定，保证跨机重建一致。prompt 明确
历史框只是辅助观测、可能有定位噪声，Image 1 才是身份锚点。训练集保留 clean case
和 corrupted case，验证集额外建立同 current 的 clean/corrupt 配对，分别报告指标，
避免把“模型学会忽略历史”误判成时序能力提升。第一版不随机框到跨序列目标上，也不
伪造同类 distractor 身份。

## 3. Stage-3：Memory Update

### 3.1 标签定义

Stage-3 输出增加第三字段：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":null}
```

`memory_update=null` 表示不更新；非空字符串只描述新出现、视觉可核验、稳定且对后续
身份判别有帮助的短语义增量。以下情况必须为 `null`：

- 普通位移、尺度变化或相机运动；
- 短时模糊、局部遮挡或目标消失；
- 仅位置、方向或动作的瞬态描述；
- 已存在记忆的重复表述；
- 无法从输入图像验证的推测、因果解释或身份文字标签。

absent case 的 `memory_update` 必须为 `null`。

### 3.2 事件挖掘与审核

候选事件可以用视觉 embedding 距离、颜色/纹理变化、视角变化和长时间间隔自动召回，
但这些信号只能用于候选排序，不能直接生成最终标签。标注界面应展示：

- reference 与事件前可信历史；
- 事件 current；
- 若干事件后确认帧，仅供标注员判断变化是否持续。

事件后确认帧属于离线标注证据，不得放进该 case 的模型输入。每个候选由人工选择：

1. `null`；或
2. 一条短、可见、增量式 memory text。

非空候选建议双人审核；冲突交由第三次复核。模型可起草文本，但不得把模型提议直接
当作真值。

### 3.3 数据比例与训练

- 记忆更新本应稀疏，第一版非空更新建议控制在全部 Stage-3 cases 的 5–15%；
- 必须人工确认大量 hard-null：外观明显但不稳定、普通视角切换、遮挡、模糊和重复
  记忆，避免模型学成每帧更新；
- 同一事件只保留一个首次稳定更新点，后续相似帧标为重复信息 `null`；
- 从 Stage-2 checkpoint 开始训练，并混入已审核的 pair/mosaic replay；未审核样本不能
  自动补 `memory_update=null`；
- SFT 先学习三字段协议和稀疏更新，GRPO 只在格式、presence、bbox、更新时机及文本
  质量 reward 都经回放验证后启用。

## 4. 验收与评测

Stage-2 数据验收至少统计 reference/current gap、history 数量、case 桶、三来源占比和
present/absent 比例。Stage-3 还应统计非空更新率、事件持续长度、重复文本率、审核者
一致性，以及每条语义标签的来源和修订记录。

实验报告必须分别回答：

- Stage-2 的视觉历史是否改善长间隔、消失和重现；
- Stage-3 是否在不显著损伤 presence/bbox 的前提下降低漏更新与过度更新；
- 记忆是否真实改善后续帧，而不仅是当前帧输出了一段合理文字。

## 5. MGIT 多粒度文本的使用边界

MGIT 的 `attribute/description/<sequence>.json` 提供三层时间标注：

- `action`：局部时间段，包含目标类别、外观、动作、交互对象、场景和一句描述；
- `activity`：合并若干 action 的中程事件摘要；
- `story`：覆盖整个视频的故事级摘要。

这些标注适合 Stage-3 的**候选挖掘和人工审核上下文**，但不能直接复制为
`memory_update`。action/activity/story 主要描述目标正在做什么、与谁交互以及场景如何
变化；这些大多是瞬态事件，不是稳定、身份相关的新外观。story 还概括了当前帧之后
才发生的情节，若进入模型输入会造成显式未来泄漏。

对当前 Stage-1 sampling plan 中 91 条可用 MGIT tiny/train 序列的只读审计得到：

- 513 个 action 段、334 个 activity 段和 90 条 story；一条序列缺少 story；
- 相邻 action 中没有非空 `object_class` 的真实类别变化；
- 只有序列 `420` 出现一次非空 `appearance` 字段变化；
- 大量边界只改变 action 或 scene，适合作为“视觉/叙事发生明显变化，但记忆不应
  更新”的 hard-null 候选。

序列 `420` 的结构化外观从 frame 3251 起由 `with a white shirt` 变为
`with red coat and grey short skirt`，但 action 描述、activity 和 story 仍继续描述
`white shirt`。边界又恰逢 shot cut，抽查边界和后续可见帧也不能稳定核验“红外套和灰短裙”。
因此即使这一条唯一的结构化正候选也只能送人工复核，不能自动生成正标签。这同时
说明 story 的语言流畅度不能替代逐帧视觉真实性。

三层文本的推荐用途如下：

| 标注层级 | 推荐用途 | 禁止用途 |
| --- | --- | --- |
| action | 定位事件边界；挖掘 action/scene hard-null；用 `appearance` 召回少量待审正候选 | 直接把动作句写成记忆；假定边界就是更新点 |
| activity | 帮助标注员理解中程上下文并判断变化是否重复或瞬态 | 作为在线模型输入；整句复制为增量记忆 |
| story | 帮助标注员核对目标身份和完整情节；发现跨段标注矛盾 | 进入任何在线 case 输入；作为帧级或事件级真值 |

### 5.1 MGIT Stage-3 pilot

第一版先做小规模、可审计的 MGIT pilot，不自动把全部 action 边界转成训练样本。候选
按以下优先级召回：

1. `appearance` 变化或视觉 embedding/颜色纹理发生持续变化；
2. action/scene 边界附近的 shot cut、restart、消失、重现和遮挡；
3. 外观距离很大但 action/appearance 文本保持不变的 hard-null；
4. 已经产生过相同记忆后的重复帧 hard-null。

每个候选 manifest 应把模型可见信息与标注员证据物理分开：

```json
{
  "dataset": "mgit",
  "sequence": "420",
  "candidate_frame_id": 3251,
  "candidate_reason": ["appearance_transition", "shotcut"],
  "model_input": {
    "reference_frame_id": 2162,
    "history_frame_ids": [2800, 3100],
    "current_frame_id": 3251
  },
  "annotation_only": {
    "future_confirmation_frame_ids": [3300, 3400, 3585],
    "action_before": "...",
    "action_after": "...",
    "activity": "...",
    "story": "...",
    "appearance_before": "with a white shirt",
    "appearance_after": "with red coat and grey short skirt"
  },
  "review": {
    "memory_update": null,
    "null_reason": "annotation_visual_conflict",
    "reviewer_ids": [],
    "status": "pending"
  }
}
```

`annotation_only` 中的未来确认帧和三层文本只用于离线审核，导出 ms-swift 训练视图时
必须整块删除。最终非空文本由标注员依据当前及过去可见证据写成短增量，并用未来帧
确认其稳定性；未来帧本身不能成为模型输入。hard-null 也必须经过审核并保留原因，
不能把未审核候选批量填成 `null`。

### 5.2 低成本文本弱监督版（当前推荐）

为了尽量减少新增标注，Stage-3 第一版按三档标签质量导出：

1. **高置信自动正例**：相邻 action 段的 `appearance` 字段存在明确、非空且可解析的
   增量（如颜色或服饰变化），并且变化在边界后连续若干采样帧仍可见。由字段差异
   生成一句短增量文本，例如 `appearance changed to a red coat`；不复制完整 story。
2. **自动 hard-null**：只有动作、场景、交互对象或 shot cut 变化，而 appearance
   没变，或变化不能通过边界后帧稳定确认。这些样本只作为候选负例，先经过规则和
   小模型一致性过滤，再从中抽取约 10% 做人工抽检，抽检不通过的整组剔除。
3. **人工金标准**：从自动正例、hard-null 以及外观距离高但文本未变化的候选中按
   来源和事件类型分层抽取约 500–1,000 个 case 双人审核，用作 Stage-3 val/test，
   不参与自动标签扩散。

自动正例必须同时满足：事件边界不在 absent 区间、目标在当前及确认帧均 present、
   文本增量只涉及可观察外观、同一事件只保留首次稳定帧。模型输入仍只放 reference、
   过去 history 和 candidate current；`story/activity`、确认帧和边界后的文本只留在
   `annotation_only` 审计字段中。这样大部分训练样本可零新增标注生成，但验证集仍有
   可量化的人工质量锚点。

推荐的初始混合比例是：约 10% 高置信自动正例、约 30% 经过抽检的 hard-null、约
   60% 从 Stage-2 replay 的 `memory_update=null` 样本中抽取。后续根据非空更新率和
   过度更新率再调整，不把所有普通帧机械标成 null。
