# Qwen3-VL-4B 全参 SFT 训练说明

> 更新日期：2026-08-18。当前目标是冻结 ViT，全参训练 LLM 与视觉 merger/aligner，
> 在一次训练中混合大规模跟踪监督与小规模状态更新监督。

## 1. 两种逐样本监督

| 数据档位 | `bbox_2d/status` | `memory_update` | 用途 |
| --- | ---: | ---: | --- |
| `tracking_sft` present | 1 | 未知占位 `null` 的值为 0 | 大规模定位、存在性和结构 |
| `tracking_sft` absent | 1 | 未知占位 `null` 的值为 0 | 目标缺失时的 bbox/status/结构 |
| `state_update_sft` update | 1 | verified 非空快照为 1 | 学何时、如何替换状态 |
| `state_update_sft` hard-null | 1 | verified `null` 为 1 | 学何时保持状态 |

`tracking_sft` 只 mask `memory_update` 的未知值；字段名、逗号、JSON 闭合和 EOS 仍监督。
因此它不强制模型推理时永远输出 null。`state_update_sft` 不允许
`masked_unknown`，每一行的状态值都必须有证据。

## 2. 坐标与数据视图

- 当前只训练 Qwen3-VL-4B；
- assistant 字段为 `bbox_2d`，值是 Image 3 上 `[0,1000] xyxy`；
- source metadata 可继续使用明确的审计名 `bbox_norm1000_xyxy`，但它不是模型可见键；
- ms-swift JSONL 使用官方 `<bbox> + objects.bbox + image_id` 与
  `QWENVL_BBOX_FORMAT=new`；
- Qwen2.5-VL 的 absolute-pixel 视图只作历史兼容，不能与 Qwen3 数据交叉加载。

## 3. 数据入口

大规模跟踪数据：

```bash
bash scripts/generate_tracking_sft_data.sh \
  cogtrack_vlt_v640_tracking_sft_r80_20_case20_mgit200_robust15_v1
```

状态数据由两部分组成，详见 [`data.md`](data.md)：

```text
MGIT 官方分段可用标签（734）
+ `qwen3-vl-plus` API/QC 合格标签（2,329）
= 统一 state_update_sft release（3,063，已完成）
```

这两类数据先独立生成，不能把 API 状态标签回填进原始 `tracking_sft`。如果后续确实
需要利用第一部分中少量明确的消失/重现样本，再单独生成 `tracking_sft` 的 memory
overlay；overlay 只在最终训练打包时替换对应 masked 行，不改变原始 release，也不构成
第三种数据类型。

## 4. 训练前预检

每个数据视图先分别检查模型族与监督边界：

```bash
python tracking/validate_qwen_training_view.py \
  --model /root/public/models/Qwen/Qwen3-VL-4B-Instruct \
  --dataset /absolute/path/to/train.jsonl \
  --dataset /absolute/path/to/val.jsonl \
  --expected-family qwen3_vl

python tracking/validate_sft_supervision.py \
  --profile tracking_sft \
  --dataset /absolute/path/to/tracking/train.jsonl \
  --dataset /absolute/path/to/tracking/val.jsonl

python tracking/validate_sft_supervision.py \
  --profile state_update_sft \
  --dataset /absolute/path/to/state/train.jsonl \
  --dataset /absolute/path/to/state/val.jsonl
```

`state_update_sft` 检查中出现一个 `masked_unknown` 都必须失败。真实 processor replay 还要
确认 `<bbox>` 绑定 Image 3 并解码为原始 norm1000 坐标。

## 5. 单次混合全参 SFT

最终训练直接更新 Qwen3-VL-4B 的 LLM 与视觉 merger/aligner，ViT 保持冻结。
tracking/state-update 两种行需要保留各自的
per-message loss scale；不能先把所有 assistant 文本统一成 full loss，也不能把状态行
统一套 tracking mask。

两个独立 release 已统一打包为：

```text
data/releases/cogtrack_v640_mixed_sft_full_v1
```

正式训练使用其中的 `train.jsonl`（79,110 行，90/4/6）；发布统计使用
`train_unique.jsonl`，验证使用 `val.jsonl`。目录完全自包含，不依赖原始 release 路径。
可重放入口是 `scripts/build_mixed_sft_release.sh`。

两卡 smoke（正式混合数据路径冻结后替换下列变量）使用：

```bash
export MODEL_PATH=/root/public/models/Qwen/Qwen3-VL-4B-Instruct
export DATASET_ROOT=/absolute/path/to/cogtrack_v640_mixed_sft_full_v1
export TRAIN_DATA="$DATASET_ROOT/train.jsonl"
export VAL_DATA="$DATASET_ROOT/val.jsonl"
export OUTPUT_DIR=/absolute/path/to/outputs/qwen3vl_4b_v640_tracking_smoke
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export TUNER_TYPE=full
export FREEZE_VIT=true FREEZE_LLM=false FREEZE_ALIGNER=false
export DEEPSPEED=zero2
export SFT_SUPERVISION_PROFILE=mixed_sft
export SAVE_STRATEGY=no EVAL_STRATEGY=no REPORT_TO=none

bash scripts/train_qwen3vl_4b_tracking_sft.sh \
  --max_steps 2 --save_strategy no --eval_strategy no
```

默认使用 BF16、FlashAttention 2、DeepSpeed ZeRO-2 和 activation/gradient checkpointing。
模型共 44.38 亿参数，其中冻结 ViT 后 41.32 亿可训练。训练图片导出长边 648，ms-swift
`max_pixels=200704` 控制每张图视觉 token 上限。1×L40S 实测会在首次 AdamW 状态创建时
达到 45,449 MiB 并 OOM；推荐 4×L40S 或 2×H100 80GB，2×L40S 可先测 ZeRO-2，必要时
退回 ZeRO-3。smoke 必须记录每卡峰值显存、tokens/sample、steps/s、有限 loss 和 ETA。

## 6. 评测边界

训练后至少在同一 Prompt 6.4.0 / CognitiveBench-Tiny 配置比较 Base、旧 Stage-2 和新
全参 SFT 模型，并报告 tracking 指标、presence、reappearance、memory update rate、over-update 与
forced-null 对照。两步 loss 只证明链路可训练，不能证明跟踪提升。

TU-GRPO 必须在混合 SFT 和状态更新评测之后，详见 [`grpo.md`](grpo.md)。
