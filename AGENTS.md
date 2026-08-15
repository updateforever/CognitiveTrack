# CognitiveTrack AI 执行与恢复指南

本文件是新 AI、新会话和新训练服务器接手项目时的首要上下文。开始修改前完整阅读本
文件，再按任务读取对应文档。不要要求用户重复已经记录的研究决策。

## 1. 项目定位与隔离边界

CognitiveTrack 是独立、可开源的 pytracking 长时单目标跟踪框架，研究纯 VLM 的全图
搜索、目标存在性、同实例定位、时序状态记忆和稀疏执行。标准生命周期为：

```text
Sequence -> Tracker.initialize -> Tracker.track -> ResultWriter -> Evaluator
```

SUTrack 是传统强基线，Hybrid 是工程方向；当前论文主线是纯 VLM。SOIBench 属于另一个
尚未发表项目，必须彻底隔离，禁止把其代码、数据、实验名或结论迁入本仓库。

## 2. 当前唯一论文主线

当前协议是 VLT-v6.3，完整方案见 [`docs/research_plan.md`](docs/research_plan.md)：

1. VLT-v6.3 core SFT：只学习 presence 与 bbox；
2. region-caption/ref 教师冷启动目标状态事件标签；
3. memory SFT：学习何时更新及完整状态快照；
4. TU-GRPO：优化状态更新对未来短轨迹的反事实收益；
5. CognitiveBench-Tiny 迭代，配方冻结后运行 Full。

旧 pair Stage-1、mosaic Stage-2、visual-v5 和 Qwen v4 probe 都已移入
[`docs/archive/`](docs/archive/README.md)。旧结果是有效历史对照，但不能代表新主线已经
训练或评测。真实完成度以 [`docs/project_status.md`](docs/project_status.md) 为准。

## 3. 冻结的在线输入与输出

所有 Base/Core/Memory/GRPO 主实验使用同一推理范式，不根据训练阶段切换：

- Image 1：带红框的永久首帧完整图，身份锚点永不覆盖；
- Image 2：近期三次可信观测组成的单行条带，从左到右由旧到新，panel 间使用白色竖向
  分隔带；不足三帧时在右侧复制最近可用观测，尚无动态历史时将初始化观测复制三次；
- Image 3：无框当前完整图，必须全图搜索；
- `initial_identity_description`：首帧身份描述，不可变；
- `current_target_state`：当前维护状态，可被稀疏替换，初值等于身份描述。

训练模型的 System Prompt 固定为 `6.3.0` 极简任务定义：说明初始身份不可被历史或状态
覆盖，要求结合历史轨迹在当前帧判断同一目标是否存在、分析当前状态、存在时定位，并仅在
稳定且有助于未来跟踪的状态变化时更新记忆。它**不写**双图兼容、padding、JSON schema、
坐标规则或格式惩罚；这些能力由 SFT 样本内化。Prompt 唯一来源是
`cogtrack/prompts/vlt_tracking.py`，禁止复制到 tracker 或数据脚本。

模型与运行时之间的输出协议仍严格为：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":null}
```

`memory_update=null` 沿用当前状态。非空字符串必须是短小、自包含、保留身份线索的完整
替换状态，不是只写变化量；`absent` 时必须为 null。目标消失不能清空记忆。字段固定放在
最后，null 是快速生成路径，非空是慢路径。

对未经本任务训练的通用 VLM 做比较时，可以使用单独版本化的 strict comparison Prompt
追加 JSON、坐标和格式说明以保证结果可解析；不得把这些约束写回训练模型的 native
Prompt，也不得在实验记录中混淆两种 Prompt profile。该 comparison profile 尚未冻结，
实现前需要先补配置、manifest 字段和公平性说明。

训练和在线推理不提供 reference bbox 坐标文本：过去参考/历史图直接画框，current 永远
不画框。模型输出当前 bbox 仍遵循 Qwen 官方 grounding 形式。

## 4. 监督边界

- GT 只监督 `present/absent` 与 present bbox；负样本只来自同一真实视频的消失帧；
- 不监督旧六分类、数值置信度、身份匹配文字、自由解释或公开 CoT；
- Core SFT 保留三字段，但仅 mask `memory_update` 的 JSON 值；字段名和闭合仍受监督；
- Core checkpoint 推理时关闭 semantic write，不能声称已经学会记忆；
- Memory SFT 只使用有 provenance 的人工/银标事件，不能给普通样本机械追加 null；
- 非空状态标签只来自 train split 的离线标注与验收；future 支持帧只能用于 annotator/
  reward，导出学生输入前必须物理移除；
- GRPO 只能从稳定 Memory SFT 初始化，未来 GT 只进入 reward 计算器。

状态标注执行规范见 [`docs/state_annotation.md`](docs/state_annotation.md)，TU-GRPO 定义见
[`docs/grpo.md`](docs/grpo.md)。

## 5. 初始化文本与坐标安全

LaSOT `nlp.txt` 和 TNL2K `language.txt` 可作初始化身份描述。MGIT 当前 story 可能包含
未来事件，禁止直接作为在线文本；回退到 object class 或首帧红框视觉指代。

两代 Qwen 坐标不可混用：

| 模型族 | 数据视图 | 输出字段 | tracker 协议 |
| --- | --- | --- | --- |
| Qwen3-VL | 当前图 `[0,1000]` 相对 `xyxy` | `bbox_norm1000_xyxy` | `norm1000` |
| Qwen2.5-VL | processor resize 后绝对像素 `xyxy` | `bbox_pixel_xyxy` | `qwen_abs_pixel` |

ms-swift 继续使用 `<bbox> + objects.bbox + image_id` 和 `QWENVL_BBOX_FORMAT=new`。Qwen3
与 Qwen2.5 的 JSONL 目录不能互换，训练前必须用真实 processor 回放。

## 6. 代码结构与修改位置

- `pytracking/`：dataset、tracker lifecycle、runner 和结果写入；
- `pytracking/trackers/cognitive_vlm.py`：VLM 跟踪器编排；
- `cogtrack/context/`：三图上下文、视觉画框和历史选择；
- `cogtrack/prompts/`：版本化 prompt；
- `cogtrack/protocol/`：状态、bbox 与严格 JSON；
- `cogtrack/cognition/`、`cogtrack/memory/`：状态机与可审计更新门控；
- `cogtrack/vlm/`：本地 Hugging Face 与 OpenAI-compatible/vLLM backend；
- `cogtrack/training/`：样本、Qwen 导出、loss mask 与 reward；
- `cogtrack/models/sutrack/`：已做数值保真的 SUTrack runtime；
- `tracking/`：数据、训练、推理和评测 CLI；
- `configs/`：模型、tracker、dataset 和训练配置。

新增/修改 tracker 时同步核对 tracker、配置、Prompt、parser、结果 schema 和评测器。路径
只写 `configs/env.local.yaml` 或环境变量，禁止向可提交文件写机器绝对路径。

## 7. 数据和推理不可越过的边界

- 第一帧只用 GT 初始化，后续 tracker 禁止读取当前/future GT；
- reference/history 必须是同序列、严格早于 current 的 accepted present 观测；
- 训练、验证、测试按完整序列划分，禁止帧级泄漏；
- 稀疏推理的“最近历史”是最近一次可信观测，不是字面上一帧；
- 动态预测必须门控，不能让一次误跟踪立刻覆盖永久身份；
- `parse_error`、`model_error`、`skipped` 与模型预测 `absent` 严格分开；
- 结果 TXT/JSONL 帧数必须与序列一致，无框按协议写 `NaN`/`null`；
- 同一训练模型系列的 benchmark 不因 checkpoint 改 prompt、历史数量、观察策略或评测
  口径；通用 VLM 的 strict comparison profile 必须单独报告，不能混入 Base/Core 消融。

## 8. 当前执行顺序

除非用户明确改变优先级，按以下顺序推进：

1. 训练服务器生成并冻结 VLT-v6.3 core plan；
2. 渲染全量数据，运行监督审计与真实 Qwen processor mask 回放；
3. 从 Qwen3-VL-4B-Instruct 做 core LoRA smoke、过拟合和正式训练；
4. 用固定 VLT-v6.3 在 CognitiveBench-Tiny 比较 Base/Core；
5. 实现 `mine → annotate → verify → export` 状态标签流水线，先建人工审核集；
6. Memory SFT，并做 memory-on/forced-null 因果评测；
7. 实现缓存版再到真实双分支的 TU-GRPO；
8. 配方冻结后运行 CognitiveBench Full 与全部消融。

训练服务器从零恢复见 [`docs/l40_setup.md`](docs/l40_setup.md)，训练配置见
[`docs/training.md`](docs/training.md)。

## 9. 常用验证命令

```bash
python scripts/verify_env.py --verbose
python -m ruff check .
python -m pytest -q
python tools/verify_cognitivebench.py
git diff --check
```

涉及训练数据时额外执行：

```bash
python tracking/validate_sft_supervision.py \
  --profile tracking_core --dataset "$TRAIN_DATA" --dataset "$VAL_DATA"
python tools/verify_qwen_grounding_templates.py \
  --dataset-root "$DATASET_ROOT" --qwen3-model "$MODEL_PATH" \
  --verify-tracking-core-mask
```

修改后至少跑针对性测试，再跑完整 Ruff/pytest。benchmark 改动至少做单序列 smoke 和
冻结标注验证。不要用单帧 case、训练 loss 或 reward 均值代替正式聚合指标。

## 10. 文档与提交规则

README 只保留公开项目定位、快速安装、主协议和入口，不写内部讨论、机器账户、私有
prompt 记录或阶段流水账。当前研究设计更新 `research_plan.md`；标签更新
`state_annotation.md`；GRPO 更新 `grpo.md`；完成度更新 `project_status.md`；旧方案移动
到 `docs/archive/`，不要删除历史证据。

Python 使用 4 空格、函数/模块 `snake_case`、类 `PascalCase`，中文注释解释设计原因。
提交信息使用明确祈使句，如 `feat: add state annotation verifier`、
`docs: define trajectory-utility GRPO`。不要提交数据、权重、cache、大结果或
`configs/env.local.yaml`。
