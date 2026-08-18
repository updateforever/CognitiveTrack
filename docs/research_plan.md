# CognitiveTrack 当前研究方案

## 1. 研究问题与完整主线

CognitiveTrack 研究纯视觉语言模型在长时单目标跟踪中的三个基础能力：全图定位、同一
目标存在性判别，以及在外观和可见状态长期变化时维护身份。当前论文主线固定为：

```text
GT presence/bbox 数据
        ↓
VLT-v6.4 tracking_sft（存在性 + 定位）
        ↓
MGIT 官方分段 + 额外 Aliyun Qwen3-VL-Plus API 状态标签
        ↓
tracking/state-update 单次混合全参 SFT（冻结 ViT）
        ↓
TU-GRPO（优化更新对未来短轨迹的真实收益）
        ↓
CognitiveBench-Tiny 迭代 → CognitiveBench Full 主表
```

这条研究故事已经闭环，但实验尚未全部完成。旧 Stage-1/2 的数据、训练和 Tiny 指标是
有效历史证据；它们采用旧输入/输出协议，不能替代 VLT-v6.4 的混合 SFT、TU-GRPO 或
Full 评测。真实进度见
[`project_status.md`](project_status.md)。

## 2. 固定在线跟踪协议

所有主实验共享同一套推理方式，不随训练阶段改变：

- Image 1：带红框的永久初始模板，提供不可覆盖的身份锚点；
- Image 2：近期三次可信观测组成的单行带框条带，从左到右由旧到新，panel 间使用白色
  竖向分隔带；不足三帧时在右侧复制最近可用观测，尚无动态历史时复制初始化观测；
- Image 3：无框当前完整图，模型必须全图搜索；
- 文本 1：原文不可变的 `initial_identity_description`（审计 provenance，可能粗糙）；
- 文本 2：可替换的 `current_target_state`，初始值等于身份描述；
- 输出：`bbox_2d`、`status`、可空的 `memory_update`。

`memory_update=null` 表示沿用当前状态。非空值不是只写“变成背面”一类增量，而是一个
短、可独立理解、保留身份线索的完整替换状态；这样重放任意帧时不需要拼接一串可能
矛盾的历史文本。永久 identity anchor 始终不被动态状态覆盖；刚刚消失时动态 memory
可以替换为明确的消失描述，持续缺失通常保持 `null`，重新出现时再替换为带重现语义的
当前指代表达。

训练模型使用 `prompt_version=6.4.0` 的 native 极简 System Prompt：只定义不可覆盖的
初始身份、历史轨迹利用、当前目标存在性与状态分析，以及稳定变化时的记忆更新。双图
兼容、padding、JSON schema、坐标和格式惩罚均不写入 System Prompt，而由固定输入协议
和 SFT 答案内化。未经跟踪训练的通用 VLM 可以使用独立版本化的 strict comparison
Prompt 补充可解析性约束，但必须在 manifest 中记录，不能与 native Prompt 混为同一
实验条件。

## 3. 两类 SFT 数据与后续 GRPO

### 3.1 tracking_sft：大规模跟踪监督

使用 LaSOT/TNL2K/MGIT train 的 GT presence 和 bbox。正式 release 的 case-level
present/absent 为精确 80:20，
reference/history 均严格早于 current。三字段结构全部保留，但仅屏蔽
`memory_update` 的值，不把普通跟踪样本伪监督为“永不更新”。

### 3.2 state_update_sft：状态更新监督

MGIT 官方 action 分段尽量全量使用；LaSOT/TNL2K 再独立采样并通过 Aliyun
`qwen3-vl-plus` OpenAI-compatible API 生成约 1,500 条标签。当前策略是单次强模型生成加确定性质量门，
不是 independent verifier。学生训练输入仍严格使用在线可得的三图上下文；GT 负责
bbox/presence 和消失转折，强模型只负责 present 状态指代表达，不能替代定位真值。

身份与状态分成两层：

- `initial_identity_description`：来自首帧的初始化文本，原文永久保留但不凌驾于 Image 1
  的视觉身份；状态 memory 可以在证据充分时完整纠正它；
- `current_target_state`：由最近一次已接受状态更新得到，可稀疏替换。

具体候选挖掘、教师 schema、自动质检和人工审核见
[`data.md`](data.md)。

tracking/state-update 最终混入同一次 Qwen3-VL-4B 全参 SFT；冻结 ViT，训练 LLM 与视觉
merger/aligner，不再维护或串联多套 adapter。

首轮两类 release 独立生成。若数据覆盖审计显示状态转折监督不足，可以再从
`tracking_sft` 中选择
GT 可明确证明的消失/重现 case 做同类 API overlay 补标；overlay 不覆盖原始 release，
是否纳入正式混合训练在两类主 release 完成后根据覆盖率与 API 成本冻结，基础链路 smoke
不依赖这一步。

### 3.3 TU-GRPO：学习“更新是否真的帮助未来”

单帧 IoU reward 能训练当前定位，却不能证明一条新状态值得写入长期记忆。当前候选创新
是 Trajectory-Utility GRPO（TU-GRPO）：对同一个候选状态分别执行“接受更新”和“保留
旧状态”的短未来轨迹回放，以后续 presence、IoU/AUC、重现恢复的差值作为核心奖励，
同时惩罚频繁更新、身份矛盾、冗长文本和不必要的重复 absent 更新；真正的消失转折仍可
产生非空状态更新。完整定义见
[`grpo.md`](grpo.md)。

## 4. 与已有工作的关系

| 工作 | 可复用经验 | CognitiveTrack 的取舍 |
| --- | --- | --- |
| [DTLLM-VLT](https://arxiv.org/abs/2405.12139) | 用 bbox、SAM 与 region captioner 生成多粒度目标文本 | 采用区域化标注思路，但文本分成永久身份与动态状态 |
| [CaptionFormer](https://arxiv.org/abs/2510.14904) | 用框视觉提示合成时空目标 caption，并显示合成 caption 可支持下游学习 | 借鉴事件级时空 caption，但显式防止同类目标混淆和过于泛化的描述 |
| [DUTrack](https://arxiv.org/abs/2503.06621) | 动态语言能改善 VLT；简洁描述优于含噪详细描述，更新依赖人工阈值 | 状态文本保持短小，并把固定阈值更新改为可学习决策 |
| [ChatTracker](https://arxiv.org/abs/2411.01756) | 用 grounding/跟踪反馈反思描述，避免通用 caption 幻觉 | 教师输出必须经过目标区域与干扰物对比验证 |
| [ATCTrack](https://arxiv.org/abs/2507.19875) | 用强 MLLM 产生缺失的目标/上下文词伪标签并审核准确性 | 所有银标保存教师、prompt、证据和验收 provenance |
| [MemVLT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1af3e0bf5905e33789979f666c31192d-Abstract-Conference.html) | 长短期记忆可用于动态提示 | 保留永久模板，状态记忆只作可审计的辅助证据 |
| [R1-Track](https://arxiv.org/abs/2506.21980) | 直接 VLM 跟踪可用规则 reward 做 GRPO；其公开实验中 no-think 更强 | 主协议只输出结构化答案，不监督可见 CoT |
| [ReasoningTrack](https://arxiv.org/abs/2508.05221) | 用更新文本带来的当前帧辅助 tracker IoU 增益做 GRPO | 不重复同帧 IoU 设计，重点优化未来轨迹的反事实收益和自适应更新时机 |
| [VideoAuto-R1](https://arxiv.org/abs/2601.05175) | 视频感知任务不一定从强制长推理获益 | 先保证直接回答；`memory_update` 末字段自然形成 null 快路径 |
| [RELO](https://arxiv.org/abs/2605.07379) | 跟踪 RL 可同时使用帧级 IoU 与序列级 AUC | 轨迹级 reward 是合理方向，但记忆反事实差值仍需消融证明 |

“TU-GRPO”目前是项目内工作名和候选贡献，不应在完成更全面检索、实现及消融前直接
宣称为已证明的新方法。

## 5. 论文实验矩阵

主表固定同一 VLT-v6.4 推理协议：

| 模型 | tracking 数据 | state-update 数据 | TU-GRPO | 目的 |
| --- | :---: | :---: | :---: | --- |
| Qwen3-VL-4B Base |  |  |  | 零样本下限 |
| Tracking-only ablation | ✓ |  |  | 跟踪监督收益 |
| Mixed SFT | ✓ | ✓ |  | 状态标签收益 |
| Mixed SFT + TU-GRPO | ✓ | ✓ | ✓ | 完整方法 |

必要消融：

- initial identity only / 加 current state；
- generic caption / region caption / Aliyun Qwen3-VL-Plus quality-gated caption；
- 所有帧 caption / 事件候选采样；
- 非空增量 / 完整替换状态；
- 当前帧 IoU reward / 未来轨迹差值 reward；
- 无更新惩罚、无身份漂移惩罚、无 hard-null；
- direct answer / 显式 CoT；
- 状态记忆关闭、强制 null、oracle event。

## 6. 评测闭环

跟踪主指标使用 hold-last 与 observation-only 的 AUC、P、Pnorm、OP50/OP75；目标判别
报告 presence precision/recall/F1、absent FPR、present miss 和重现恢复。额外报告：

- 状态事件：update precision/recall/F1、over-update、miss-update；
- 文本质量：目标区域对齐、目标—干扰物 margin、身份矛盾率；
- 因果收益：`Delta-AUC-after-update@H`、`Delta-presence-F1@H`、重现恢复增益；
- 代价：更新率、生成 token、单次延迟和序列吞吐；
- 安全性：错误更新后的身份漂移率与恢复时间。

Tiny 用于正式快速迭代，Full 只在配方冻结后运行。memory validation/test 的事件标签应
全部人工复核；训练银标只做分层抽检。任何结果都必须记录代码 commit、模型 revision、
数据 manifest、prompt version、观察策略和随机种子。
