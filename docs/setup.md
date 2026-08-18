# CognitiveTrack 环境与执行入口

> 更新日期：2026-08-18。本文统一通用环境、本机 L40/L40S 路径、数据预检和训练启动
> 说明。数据生成细节见 [`data.md`](data.md)，训练配方见 [`training.md`](training.md)。

## 1. 当前服务器

项目根目录：

```text
/root/nas/user-mhf/user-data/wyp/CognitiveTrack
```

已配置的 Conda 环境：

```text
/root/nas/user-mhf/user-data/wyp/CognitiveTrack/.conda/envs/cogtrack-l40
```

公共数据与模型根：

```text
/root/nas/user-mhf/user-data/PUBLIC_DATASETS/
/root/public/models/Qwen/
```

路径只写入 `configs/env.local.yaml` 或环境变量，不进入 Git。恢复服务器后运行：

```bash
.conda/envs/cogtrack-l40/bin/python scripts/verify_env.py --verbose
.conda/envs/cogtrack-l40/bin/python -m pytest -q
```

## 2. 通用独立环境

在其他机器从头建立最小环境：

```bash
conda env create -f environment.yml
conda activate cogtrack
pip install -e . --no-build-isolation
```

如果已有经过 Qwen-VL 验证的 Python 3.10 环境，可以克隆后离线安装：

```bash
conda create --name cogtrack --clone <source-env>
conda run -n cogtrack python -m pip install -e . --no-deps --no-build-isolation
```

不要在公用 baseline 环境中直接升级 `transformers`、`torch` 或 `ms-swift`。精确
CUDA/PyTorch/FlashAttention ABI 由集群镜像或已验证环境负责，`environment.yml` 只提供
通用最小依赖。

## 3. 路径配置

```bash
cp configs/env.example.yaml configs/env.local.yaml
```

可用环境变量覆盖 YAML：

- `COGTRACK_MODEL_ROOT`
- `COGTRACK_DATASET_ROOT`
- `COGTRACK_COGNITIVEBENCH_ROOT`
- `COGTRACK_LASOT_ROOT`
- `COGTRACK_TNL2K_ROOT`
- `COGTRACK_MGIT_ROOT`
- `COGTRACK_OUTPUT_ROOT`

模型 YAML 只记录 checkpoint 目录名，runner 通过 `model_root` 解析；可提交配置中不写
开发机绝对路径。

## 4. 当前模型职责

- `Qwen3-VL-4B-Instruct`：学生模型、冻结 ViT 的全参 SFT 和评测；
- Aliyun `qwen3-vl-plus`：本机 OpenAI-compatible API 状态标注；
- 数据集 GT：bbox/presence、消失转折和重现时序；
- 旧 Qwen2.5-VL/Qwen3-VL-32B candidate：只作历史比较，不进入正式状态数据。

本机缺少 Qwen3-VL-32B 不再是 blocker。当前 Aliyun API key 只通过当前 shell 的环境变量
注入，不写入 bundle 或 Git。本机单张 L40S 主要负责 processor replay 和评测；正式全参
SFT 使用多卡 DeepSpeed。

## 5. 长任务与日志

数据生成、训练和长测试必须放在具名 tmux 中，并将日志写入 `data/logs/` 或输出目录。
每个任务结束时记录 `EXIT_CODE`，这样对话或 SSH 中断不会丢失状态。

当前没有仍在运行的正式数据生成任务。两份 v6.4 主 release 已完成；后续混合打包、训练
和完整评测继续使用具名 tmux，并在任务目录记录退出码。

## 6. 数据与 Qwen3 preflight

数据生成入口统一记录在 [`data.md`](data.md)。生成完成后：

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

状态 release 把 profile 改为 `state_update_sft`，并要求零 `masked_unknown`。

## 7. 全参 SFT smoke 与多卡

两卡时：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export TUNER_TYPE=full
export FREEZE_VIT=true FREEZE_LLM=false FREEZE_ALIGNER=false
export DEEPSPEED=zero2
export SFT_SUPERVISION_PROFILE=mixed_sft
export OUTPUT_DIR=/absolute/path/to/outputs/qwen3vl_4b_v640_tracking_smoke
export SAVE_STRATEGY=no EVAL_STRATEGY=no REPORT_TO=none
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

bash scripts/train_qwen3vl_4b_tracking_sft.sh \
  --max_steps 2 --save_strategy no --eval_strategy no
```

ms-swift 原生通过 `NPROC_PER_NODE` 启动多卡，项目没有修改 site-packages 中的 swift。
当前环境使用 FlashAttention 2.8.3.post1 与 DeepSpeed 0.19.5。正式默认是 ZeRO-2：推荐
4×L40S 或 2×H100 80GB；2×L40S 先烟测 ZeRO-2，若 OOM 再切 `DEEPSPEED=zero3`。
smoke 必须记录每卡峰值显存、tokens/sample、steps/s、有限 loss、41.32 亿可训练参数和 ETA。

正式 mixed 数据入口已经冻结为
`data/releases/cogtrack_v640_mixed_sft_full_v1/{train.jsonl,val.jsonl}`；训练前仍先跑两步
smoke，记录显存与有限 loss 后再启动完整 epoch。

## 8. SUTrack checkpoint

SUTrack 权重不写入 YAML：

```bash
export COGTRACK_SUTRACK_CHECKPOINT=/path/to/SUTRACK_ep0180.pth.tar
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/sutrack_b384.yaml \
  --dataset-config configs/datasets/cognitivebench.yaml \
  --sequence 005 --debug-frames 2
```

内置 B384 runtime 需要 `torch`、`torchvision` 和 `timm`，可用
`pip install -e '.[sutrack]'` 安装。manifest 会记录网络配置、checkpoint 路径、大小与
SHA-256。

## 9. 评测边界

训练完成后使用同一 Prompt 6.4.0 和 CognitiveBench-Tiny 比较 Base、旧 Stage-2 与新
模型。数据生成、processor replay、有限 loss、显存利用率或吞吐都不是精度证据。
