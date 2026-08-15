# VLT-v6.3 固定三图与核心跟踪 SFT

代码中的 `vlt_v6` profile 和配置文件名为兼容已生成数据保持不变；实际 Prompt 版本由
manifest 中的 `prompt_version=6.3.0` 和 `history_layout_version=recent_strip_3_v2`
区分，禁止只凭目录名判断实验协议。

## 1. 当前结论

第一轮不需要 GRPO，也不需要为语义记忆伪造标签。先用普通 SFT 学习两个已有可靠
监督的能力：

1. 当前帧中初始化目标是否存在；
2. 目标存在时的 Qwen3 官方 `norm1000 xyxy` 坐标框。

训练和最终推理仍共用三字段输出。`memory_update` 字段保留，但首轮只将它的 JSON 值
做 token-level loss mask。这样不会把所有样本机械监督成“永不更新记忆”，也无需先上
GRPO。该模型只能称为“VLT-v6 core SFT”，不能声称已经学会生成或使用语义记忆。

## 2. 固定输入协议

正式 VLT-v6.3 输入始终包含三张完整图，不做目标局部裁剪：

- Image 1：永久初始化模板，红框标出目标；
- Image 2：近期三次已接受观测组成的单行条带，从左到右由旧到新，各 panel 用红框标出
  历史目标，相邻 panel 使用高度约 3% 的纯白竖向分隔带；
- Image 3：当前待搜索完整图，不画框。

不足三次动态历史时，在右侧复制最近可用观测：`[h1] -> [h1,h1,h1]`，
`[h1,h2] -> [h1,h2,h2]`。尚无动态历史时用初始化观测复制三次。这些重复 panel 只是
padding，不表示新增观测，不读取 current GT，也不伪造未来预测。后续 Image 2 只能来自
tracker 已经接受的过去预测。

分隔带不携带序号、文字或箭头，避免引入 OCR 负担和固定时间间隔的错误暗示。旧的
`recent_strip_3_v1` 无分隔条带只用于复现已有实验，新数据和正式推理使用
`recent_strip_3_v2`。

动态文本分为不可变身份与可替换状态。没有已接受更新时，当前状态直接初始化为身份
描述，而不是额外写 `none`。实际 user message 为：

```text
<image><image><image>
Initial target identity: {initial_identity_description}
Current maintained target state: {accepted_state_or_initial_identity}
Track the initialized target in the final image.
```

训练模型的 native System Prompt 只说明四件事：初始身份不可被历史或状态覆盖、结合
历史轨迹、在当前帧判断同一目标是否存在并分析状态、稳定变化时才更新记忆。它不描述
双图、padding、JSON schema、坐标或格式限制；这些由 SFT 答案学习。中文语义审核稿为：

```text
你是一个长时视觉语言单目标跟踪器。始终以图1中红框标记的初始目标及其文本描述作为
身份锚点，初始身份不得被历史预测或状态记忆覆盖。结合图2中按时间排列的历史轨迹和
当前维护的目标状态，在图3中判断同一目标是否存在，分析目标当前状态，并在目标存在时
完成定位；仅在观察到稳定且有助于后续跟踪的目标状态变化时更新状态记忆。
```

实际英文 Prompt 和版本号唯一来源是 `cogtrack/prompts/vlt_tracking.py`，不要在数据脚本
或 tracker 中复制另一份。通用未训练 VLM 所需的 JSON、坐标和格式说明必须放在单独的
strict comparison Prompt profile；该 profile 尚未冻结。

初始身份永不覆盖。后续非空 `memory_update` 是完整、自包含的替换状态，不是只写变化
量；`null` 沿用当前状态，`absent` 也保留消失前最后一条可信状态。

初始化文本的在线边界如下：

- LaSOT `nlp.txt`：可用，标记为 `initial_target`；
- TNL2K `language.txt`：可用，标记为 `initial_target`；
- MGIT 当前 `story`：描述整段视频，可能含初始化之后事件，禁止作为在线输入；优先退回
  object class，没有类别时使用“Image 1 红框目标”的无额外语义指代。

## 3. 输出与 loss mask

Qwen3-VL 输出保持：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":null}
```

absent 样本使用 `bbox_norm1000_xyxy:null`。首轮各段 loss 为：

| 输出部分 | Loss | 说明 |
| --- | ---: | --- |
| `target_status` 及其值 | 1 | 学习目标存在性 |
| bbox 字段、`<bbox>` 展开的坐标 token | 1 | 学习定位 |
| `"memory_update":` | 1 | 保持最终三字段结构 |
| `memory_update` 的值（首轮为 `null`） | 0 | 不提供真假记忆监督 |
| 最终 `}` 和模板结束 token | 1 | 保持输出闭合 |

实现位于：

- `cogtrack/training/loss_mask.py`：纯 Python 无损分段；
- `cogtrack/training/ms_swift_plugin.py`：注册 `cogtrack_tracking_core`；
- `tracking/validate_sft_supervision.py`：训练前核对数据 metadata 与 mask 档位；
- `scripts/train_sft.sh`：根据 `SFT_SUPERVISION_PROFILE` 自动加载插件。

不能简单按字符长度截掉“后半段”，否则可能同时屏蔽 bbox 或遗漏不同长度的记忆值。
本实现按最终 JSON 字段边界定位，只屏蔽值。由于 bbox 和状态在记忆字段之前，因果语言
模型不会让被 mask 的后置记忆值反向成为 bbox 的输入；最终 `}` 虽继续监督，但只用于
约束合法闭合。

## 4. 数据合成

训练服务器先生成冻结 sampling plan：

```bash
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --max-samples-per-sequence 20 --absent-ratio 0.3 \
  --output-dir data/plans/cogtrack_vlt_v6_core --plan-only
```

再严格重放并渲染图片：

```bash
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --sampling-plan data/plans/cogtrack_vlt_v6_core/sampling_plan.json \
  --max-samples-per-sequence 20 --absent-ratio 0.3 \
  --output-dir data/releases/cogtrack_vlt_v6_core
```

默认已经固定：`context_mode=mosaic`、`reference_mode=visual_box`、
`prompt_profile=vlt_v6`、`force_history_image=true`、`memory_supervision=masked_null` 和
Qwen3-VL 训练视图。同一序列真实 present/absent 仍按约 7:3 采样；训练、验证按完整
序列划分。

## 5. 训练前验证与 SFT

```bash
export DATASET_ROOT=/datasets/cogtrack_vlt_v6_core
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"

python tracking/validate_sft_supervision.py \
  --profile tracking_core \
  --dataset "$TRAIN_DATA" --dataset "$VAL_DATA"

python tools/verify_qwen_grounding_templates.py \
  --dataset-root "$DATASET_ROOT" \
  --qwen3-model /models/Qwen3-VL-4B-Instruct \
  --verify-tracking-core-mask
```

第二条命令使用真实 Qwen3 processor 和 ms-swift 模板，必须确认 bbox 坐标仍受监督，且
`masked_text` 恰好为 `null`。之后启动 LoRA：

```bash
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_vlt_v6_core
export QWEN_MODEL_FAMILY=qwen3_vl

bash scripts/train_qwen3vl_4b_vlt_v6_core.sh
```

通用入口也可直接使用，但必须显式设置
`SFT_SUPERVISION_PROFILE=tracking_core`。训练脚本会在加载模型和占用 GPU 前拒绝
`full`/`tracking_core` 混用的数据。

## 6. 评测边界与后续记忆训练

Base 与 core SFT 使用相同的 VLT-v6 输入协议，分别对应：

- `configs/trackers/qwen3vl_4b_vlt_v6_base_vllm.yaml`；
- `configs/trackers/qwen3vl_4b_vlt_v6_core_sft_vllm.yaml`。

core SFT 配置保留视觉历史，但设置 `semantic_enabled:false`。模型生成的未监督记忆值只
记录、不落入长期语义记忆，避免污染后续帧。效果以 CognitiveBench-Tiny 的固定指标
比较 Base/core SFT，而不是训练 loss 或人工挑选 case。

后续有可靠记忆事件标签时，可改用 `memory_supervision=explicit` 和 `full` loss 训练同一
协议。标签构造见 [state_annotation.md](state_annotation.md)；轨迹效用 GRPO 见
[grpo.md](grpo.md)。二者都不是 core 坐标 SFT 的前置条件。
