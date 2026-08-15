# ms-swift 分阶段训练指引

## 1. 当前训练顺序

CognitiveTrack 不再按旧 pair/mosaic 的模型能力名称切换推理模式。所有当前 checkpoint
共享 VLT-v6.3 三图输入和三字段输出，只改变哪些字段已有可靠监督：

| 阶段 | 初始化 | 监督 | 推理时 memory |
| --- | --- | --- | --- |
| Base | Qwen3-VL-4B-Instruct | 无 | 关闭写入 |
| Core SFT | Base | presence + bbox；memory 值 mask | 关闭写入 |
| Memory SFT | Core | core + update/null + 完整状态快照 | 开启并门控 |
| TU-GRPO | Memory SFT | 当前 GT + 文本 groundedness + 轨迹效用 reward | 开启并门控 |

旧 Stage-1/2 训练只作历史对照，配方和结果见 [`archive/`](archive/README.md)。

## 2. Core SFT

VLT-v6.3 固定使用三张完整图：带框初始模板、按时间从左到右且由白色竖向分隔带隔开的
近期三帧带框历史条带和无框当前图；不足三帧时复制最近可用历史进行右侧 padding。文本包含
不可变 initial identity 和当前 maintained state。Core 数据没有记忆真值，因此答案仍
带 `memory_update:null`，但训练只将该 JSON 值的 token loss 设为 0：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":null}
```

状态、bbox、字段名和 JSON 闭合继续受监督。实现和完整命令见
[`vlt_v6_core_sft.md`](vlt_v6_core_sft.md)。标准入口：

训练模型的 `6.3.0` native System Prompt 不重复 JSON、坐标或格式要求；三字段协议由
assistant 监督内化。对通用未训练 VLM 的 strict comparison Prompt 必须使用独立 profile
和 manifest 标记，不能拿它生成 native SFT 数据。

```bash
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct
export DATASET_ROOT=/datasets/derived/cogtrack_vlt_v6_core
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_vlt_v6_core
export QWEN_MODEL_FAMILY=qwen3_vl

bash scripts/train_qwen3vl_4b_vlt_v6_core.sh
```

正式训练前必须执行 `tracking/validate_sft_supervision.py` 和真实 processor mask 回放；
不能只检查字符串长度或假设 `<bbox>` 展开后仍受监督。

## 3. Memory SFT

Memory SFT 继续使用完全相同的在线输入和输出。差别是显式 memory 样本来自
`memory_labels.v1`，其 `memory_update` 值参与 loss：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":"the same white dog, now seen from the rear; black ears and red collar remain visible"}
```

`null` 是经过事件筛选的 hard-null，不是机械给所有普通 bbox 样本补空标签。非空内容是
完整替换状态；`absent` 必须为 null。建议首轮 batch 混合 70% core 和 30% memory，
memory 内约 25% update、75% hard-null，最终比例由 validation 选择。

状态标签生成工具仍待实现；在 [`state_annotation.md`](state_annotation.md) 的审核集、
provenance 和无未来泄漏校验完成前，不提供虚假的“一键全量训练”命令。

## 4. TU-GRPO

GRPO 只从已稳定的 Memory SFT 初始化。现有 reward 模块可复用格式、presence 和 bbox
IoU；下一步新增 target/distractor groundedness、身份漂移惩罚和 accept/keep 双分支
短轨迹效用。未来 GT 只进入 reward，不得进入模型输入。

先做离线 reward replay，再按 `format/current → event/ground → cached trajectory → true
trajectory` 逐级打开。完整定义、消融和成功标准见 [`grpo.md`](grpo.md)。在 trajectory
reward 和 ms-swift 插件尚未落地前，不将现有 presence GRPO 入口称为完整方法。

## 5. Qwen 坐标与数据视图

两代模型的训练 JSONL 不能交叉使用：

| 模型族 | JSONL | 坐标 | 输出字段 |
| --- | --- | --- | --- |
| Qwen3-VL | `ms_swift/qwen3_vl/` | 当前图 `[0,1000]` 相对 `xyxy` | `bbox_norm1000_xyxy` |
| Qwen2.5-VL | `ms_swift/qwen2_5_vl/` | processor resize 后绝对像素 `xyxy` | `bbox_pixel_xyxy` |

两者均保留 `QWENVL_BBOX_FORMAT=new`，并通过 ms-swift 的 `<bbox>` 与
`objects.image_id` 将 assistant bbox 绑定最后一张 current 图。visual-box 输入不会为
Image 1 创建 reference `<bbox>` object。

```bash
python tracking/validate_qwen_training_view.py \
  --model "$MODEL_PATH" \
  --dataset "$TRAIN_DATA" --dataset "$VAL_DATA" \
  --expected-family "$QWEN_MODEL_FAMILY"
```

## 6. LoRA 与常规全参 SFT

主实验优先 LoRA，便于快速比较标签与 reward 设计。用户所说的“常规全参微调”定义为：
语言模型和对齐层可训练、视觉编码器冻结。只有专门研究视觉表征适配时才解冻视觉侧，
并在实验名中明确标记。

8×24GB GPU 做 3B/4B 全参对照时必须用 FSDP 或 ZeRO-3 分片参数、梯度和优化器状态：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
TUNER_TYPE=full FSDP_MODE=fsdp2 \
FREEZE_LLM=false FREEZE_VIT=true FREEZE_ALIGNER=false \
LEARNING_RATE=1e-5 EPOCHS=1 BATCH_SIZE=1 GRAD_ACC_STEPS=4 \
MODEL_PATH="$MODEL_PATH" TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$VAL_DATA" \
DATASET_ROOT="$DATASET_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
bash scripts/train_sft.sh
```

LoRA 与 full 不是两个不同推理协议，不能因训练方式改变测试 prompt、历史长度或观察策略。

## 7. 数据与训练验收

每次训练前后至少保存：

- 完整序列 train/validation split 和 sampling/annotation plan；
- JSONL、图片 manifest、模型权重的 SHA-256；
- model family、prompt name/version、supervision profile；
- ms-swift/transformers/torch revision 与 Git commit；
- 两步 smoke、小样本过拟合、正式 loss 和 validation 记录；
- 可训练参数量、world size、global batch、学习率、显存和吞吐；
- 最终 adapter、trainer state 和恢复命令。

训练 loss 只证明优化过程运行，效果结论必须来自固定 CognitiveBench-Tiny/Full。Core
checkpoint 关闭 semantic write；Memory/TU-GRPO checkpoint 同时报告 memory-on 与
forced-null，证明提升确实来自状态维护而非其他训练差异。
