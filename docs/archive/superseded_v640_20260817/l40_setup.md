# L40/L40S 环境与当前执行入口

> 更新日期：2026-08-17。当前主线是 VLT-v6.4 数据生成与 Qwen3-VL-4B LoRA；旧
> v6.3.1 Core、Stage-1/2 命令只在历史归档中使用。

## 1. 本机环境与路径

本机 Conda 环境：

```text
/root/nas/user-mhf/user-data/wyp/CognitiveTrack/.conda/envs/cogtrack-l40
```

公共数据与模型根：

```text
/root/nas/user-mhf/user-data/PUBLIC_DATASETS/
/root/public/models/Qwen/
```

路径以 `configs/env.local.yaml` 或环境变量为准，不写入 Git。恢复后运行：

```bash
.conda/envs/cogtrack-l40/bin/python scripts/verify_env.py --verbose
.conda/envs/cogtrack-l40/bin/python -m pytest -q
```

## 2. 当前模型职责

- `Qwen3-VL-4B-Instruct`：学生模型，LoRA SFT 与后续评测；
- 闭源 Qwen3.6：在优惠远端通过 OpenAI-compatible API 生成约 1,500 条离线状态标签；
- 数据集 GT：负责 bbox/presence、消失转折和重现时序，不由语言模型猜测；
- 旧 Qwen2.5-VL/Qwen3-VL-32B candidate：只保留历史比较，不进入当前正式状态数据。

本机缺少 Qwen3-VL-32B 权重不再是标签生成 blocker。便携 bundle 通过 ModelScope 搬到
远端，API key 只通过环境变量注入；本机 L40S 主要用于 Qwen3-VL-4B processor replay、
LoRA smoke 和正式训练。API 标签采用单次强模型生成加确定性质量门，不能在报告中称为
independent verification。

## 3. 数据生成

大规模跟踪数据：

```bash
bash scripts/generate_tracking_sft_data.sh \
  cogtrack_vlt_v640_tracking_sft_r80_20_case20_mgit200_robust15_v1
```

小规模预检：

```bash
DRY_RUN=1 DRY_RUN_SEQS=12 MAX_CASES_PER_SEQ=3 MGIT_CAP=3 \
  bash scripts/generate_tracking_sft_data.sh smoke_tracking_sft
```

MGIT 状态分段、额外约 1,500 条 Qwen3.6 API 标签和最终合并命令见
[`state_annotation.md`](state_annotation.md)。

## 4. Qwen3/ms-swift preflight

```bash
export DATASET_ROOT=/absolute/path/to/release
export MODEL_PATH=/root/public/models/Qwen/Qwen3-VL-4B-Instruct
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"

.conda/envs/cogtrack-l40/bin/python tracking/validate_qwen_training_view.py \
  --model "$MODEL_PATH" --dataset "$TRAIN_DATA" --dataset "$VAL_DATA" \
  --expected-family qwen3_vl

.conda/envs/cogtrack-l40/bin/python tracking/validate_sft_supervision.py \
  --dataset "$TRAIN_DATA" --dataset "$VAL_DATA" --profile tracking_sft
```

状态 release 将 profile 改成 `state_update_sft`。后者必须零 `masked_unknown`。

## 5. 两卡 LoRA smoke

当前服务器是两卡时：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export OUTPUT_DIR=/absolute/path/to/outputs/qwen3vl_4b_v640_tracking_smoke
export SAVE_STRATEGY=no EVAL_STRATEGY=no REPORT_TO=none
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

bash scripts/train_qwen3vl_4b_tracking_sft.sh \
  --max_steps 2 --save_strategy no --eval_strategy no
```

smoke 记录每卡峰值显存、LoRA 可训练参数、tokens/sample、steps/s、有限 loss 和 ETA，再
决定正式使用 2/4/8 卡。ms-swift 原生通过 `NPROC_PER_NODE` 启动多卡；项目没有修改
site-packages 中的 swift 源码。

当前最终目标是把 tracking/state-update 数据打包后单次训练同一套 LoRA。统一混合入口
尚未冻结前，上述命令只用于验证 `tracking_sft` 链路，不应被称为最终模型训练。

## 6. 评测边界

训练完成后使用同一 Prompt 6.4.0 和 CognitiveBench-Tiny 比较 Base、旧 Stage-2 和新
模型。数据生成、processor replay、有限 loss 或显存利用率都不是精度证据。
