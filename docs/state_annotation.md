# 目标状态记忆标签冷启动方案

## 1. 标签究竟监督什么

状态标签不是逐帧 caption，也不是旧六分类。它只回答：当前可维护文本是否已经不足以
描述同一目标的稳定可见状态，是否应当用一条更有利于后续身份判别的状态快照替换它。

固定语义：

- `initial_identity_description`：首帧确定的类别和稳定实例线索，永久不覆盖；
- `current_target_state_before`：进入当前样本前最近一条已接受状态；
- `memory_update=null`：维持原状态；
- `memory_update="..."`：完整、自包含的替换状态，不是文本增量；
- `target_status=absent`：只改变当前判定，不写记忆。

建议英文状态不超过 30 词，优先包含类别、稳定颜色/纹理/部件和当前视角或构型。例如：

```text
Initial identity: a white dog with black ears and a red collar
State before:     a white dog with black ears and a red collar, seen from the side
Memory update:    the same white dog, now seen from the rear; black ears and red collar remain visible
```

禁止写入：bbox 坐标、帧号、背景相对位置、模型推理、置信度、短时模糊、单纯尺度变化、
未经证实的遮挡后属性或“目标消失”。

## 2. 为什么不对每一帧直接 caption

通用 caption 容易只说类别、混入背景或把同类干扰物当成目标；逐帧生成还会造成海量
“无变化”样本。已有 VLT 工作提供了更合适的路线：

- [DTLLM-VLT](https://arxiv.org/abs/2405.12139) 从 bbox 出发，用 SAM 分割和 Osprey
  生成目标区域描述；
- [CaptionFormer](https://arxiv.org/abs/2510.14904) 使用框视觉提示构造时空目标 caption，
  同时提示了描述过于泛化和混淆同类实例的风险；
- [DUTrack](https://arxiv.org/abs/2503.06621) 表明动态文本有效，但详细描述可能引入噪声，
  且人工更新阈值是局限；
- [ChatTracker](https://arxiv.org/abs/2411.01756) 用目标区域和跟踪/grounding 反馈修正
  MLLM 描述；
- [ATCTrack](https://arxiv.org/abs/2507.19875) 说明可以用强 MLLM 冷启动缺失文本标签，
  但必须明确标签定义并做质量检查。

因此采用“先挖事件、再做 region caption、最后跨帧验标”，而不是全帧盲目 caption。

## 3. 输入来源与数据隔离

只使用 LaSOT、TNL2K、MGIT 的 train split，先按完整序列划分 train/validation，随后才
生成标签。CognitiveBench test/val 不参与训练标签生产。

离线 annotator 可以读取：

1. 带 GT 框的初始化完整图；
2. 带 GT 框的当前候选完整图；
3. 目标 crop 或 SAM2 mask 视图；
4. 当前已维护状态；
5. 候选之后的少量 present 支持帧，用来确认变化是否稳定；
6. 同帧同类/高相似干扰物 crop，用来检查描述是否有区分力。

第 3、5、6 项只服务离线标注和奖励计算，不进入学生在线推理。输出训练 JSONL 前必须
物理删除未来帧、GT 框、mask、teacher reason 和验证分数，防止未来信息泄漏。

### 3.1 初始身份描述单独生成

初始身份与动态状态不能共用一套未来可见输入。`initial_identity_description` 的生成器
只能看首帧带框全图、首帧 crop/mask 和被确认安全的初始化语言；它不能看第二帧或整段
视频。LaSOT/TNL2K 的现有初始化描述可作为 seed，MGIT story 不可用；缺少安全文本时，
由 region-caption 教师仅根据首帧生成类别和可见稳定线索。

初始描述验收只检查它是否被首帧证据支持、是否对应红框区域、是否比同类干扰物更
具体。即使后续帧能看到新部件，也不能把这些未来属性倒灌到初始身份。动态状态教师才
允许使用 future support 做“变化是否持续”的 reward-only/annotation-only 验证。

## 4. 事件候选挖掘

先从 GT present 段抽取目标区域特征，建议使用冻结的 DINOv2 或 SigLIP，并缓存到序列级
特征文件。候选包括：

- 与上次已接受状态相比，目标区域 embedding 明显变化；
- 消失后重现；
- 视角、姿态、构型、可见部件或稳定携带物发生改变；
- 长时间间隔后的首次可靠观测；
- 数据集属性提示的稳定变化边界。

同时构造 hard-null：普通运动、仅尺度/位置变化、短时模糊、瞬时遮挡、单帧光照突变、
重复状态，以及所有 absent 帧。候选阈值不写死为跨数据集绝对值；先按序列内变化分位数
召回，再由教师和稳定性验证筛选。初始建议每个候选至少有两个后续 present 支持观测，
但最终窗口和阈值由人工审核集选择并写入版本化配置。

这里的 7:3 是 core 跟踪样本的 present/absent 比例，不是记忆更新比例。memory 子集初始
可按 25% update、75% hard-null 组织；最终根据 validation 的 over-update 与漏更新调整。

## 5. 教师与 verifier

主教师优先使用比 Qwen3-VL-4B 学生更强的本地模型，例如 Qwen3-VL-32B；独立 verifier
尽量使用另一模型族或冻结的 region-text encoder，避免同源偏差。专用 ref/grounding
模型负责检查文本是否对应 GT 区域及能否排除干扰物，不负责产生 bbox 真值。

每个候选执行两次不同采样种子的教师生成，并由独立 verifier 裁决。教师提示词可以比
在线跟踪 prompt 更详细，因为它只用于离线构造标注。建议核心约束为：

```text
You are annotating target-state memory for long-term single-object tracking.
Preserve the initialized identity. Decide whether the maintained state should be replaced.
Update only for a stable, visually supported and future-useful target-state change.
Return one strict JSON object; do not infer invisible attributes.
```

不要要求教师输出长 CoT。若需要审计理由，使用有限枚举 `reason_codes`，而不是把自由
推理文本混入学生答案。

## 6. 原始标注 schema

建议将带全部证据的 annotator 结果保存为独立 JSONL：

```json
{
  "schema_version": "cogtrack.memory_labels.v1",
  "dataset": "lasot",
  "sequence": "airplane-1",
  "frame_id": 420,
  "target_status": "present",
  "initial_identity_description": "a white airplane with red wing tips",
  "current_target_state_before": "a white airplane with red wing tips, seen from the side",
  "decision": "update",
  "memory_update": "the same white airplane, now viewed from below; red wing tips remain visible",
  "reason_codes": ["stable_viewpoint_change", "identity_cues_preserved"],
  "evidence": {
    "candidate_reason": ["embedding_shift", "long_gap"],
    "past_frame_ids": [300, 360],
    "support_frame_ids": [424, 431],
    "distractor_frame_ids": []
  },
  "teacher": {
    "model": "teacher-model-id",
    "revision": "fixed-revision",
    "prompt_version": "memory-annotator-v1",
    "generation_seed": 17
  },
  "quality": {
    "teacher_agreement": true,
    "target_text_score": 0.0,
    "distractor_margin": 0.0,
    "identity_consistent": true,
    "stable_on_support": true,
    "human_review": "pending"
  }
}
```

分发给 SFT 的精简视图只保留在线可得输入、GT presence/bbox 和最终
`memory_update`；`support_frame_ids` 等 future-only 字段永远留在 annotation 包。

## 7. 自动验收与人工审核

非空状态只有同时满足以下条件才进入 silver train：

1. 两次教师的 update/keep 决策一致，文本语义近似；
2. 类别和永久身份线索不与初始化描述矛盾；
3. target crop 的 text alignment 达标，并高于同类干扰物 margin；
4. 新状态在至少两个后续 present 支持观测中成立；
5. 文本满足长度、字符集、禁词、无坐标、无背景依赖等规则；
6. 当前 GT 为 present 且 bbox 有效。

keep/null 也要验收，重点保留接近阈值的 hard-null，避免模型学习“只要画面变化就更新”。
无法确认的样本标记 `ambiguous` 并排除，不强行二值化。

validation/test 的记忆事件全部人工复核。train 先审核所有非空候选的小规模种子集，再按
数据集、事件类型、目标类别、teacher agreement 和 verifier margin 分层抽检；首轮建议
人工看 500–1000 个事件，据此校准阈值。审核员只看证据包，不看模型训练结果。

## 8. 冷启动数据配方

首轮 memory SFT 建议：

- 总 batch 中 70% 保留 core 跟踪样本，且继续只监督跟踪字段；
- 30% 使用显式 memory 样本并对三字段全量监督；
- memory 样本内部约 25% update、75% hard-null；
- absent 一律为 null，并保留消失前/重现后的事件对；
- 同一序列的状态链必须按时间顺序重放，`state_before` 只能来自更早已接受标签。

这些是首轮超参数，不是论文结论。必须用 memory validation 上的 update F1、over-update、
身份矛盾率和下游轨迹收益共同选择，不能只看 teacher agreement。

## 9. 数据发布与可重放性

建议发布 Hugging Face/ModelScope 兼容目录：

```text
cogtrack_memory_v1/
├── README.md
├── dataset_info.json
├── manifests/
│   ├── sequence_split.json
│   ├── candidate_plan.json
│   └── checksums.json
├── annotation/
│   ├── train_full.jsonl
│   └── validation_human.jsonl
├── ms_swift/qwen3_vl/
│   ├── train.jsonl
│   └── val.jsonl
└── images/ or image_shards/
```

每版固定：原始数据 revision、序列划分、候选 plan、教师与 verifier revision、prompt
版本、随机种子、接受阈值和文件 SHA-256。任何阈值变化都生成新 dataset version，禁止
在原版本上静默覆盖。

## 10. 实现顺序

后续代码按以下小步实现并逐步验收：

1. `tracking/mine_memory_events.py`：只生成候选 plan 和缓存特征；
2. `tracking/annotate_target_states.py`：调用本地教师并保存未裁剪原始响应；
3. `tracking/verify_target_states.py`：双教师、region-text、稳定性和规则验收；
4. `tracking/export_memory_sft.py`：重放状态链并导出无 future 信息的 ms-swift JSONL；
5. 小规模人工审核、processor 回放、两步训练和过拟合；
6. 冻结 `memory_labels.v1` 后才生成全量数据。

上述文件是明确的下一阶段计划；当前仓库尚未声称它们已经实现。
