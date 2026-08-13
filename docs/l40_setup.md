# L40 训练服务器部署与 Stage-1 复现

> **历史配方提示（2026-08-13）：** 本文保留用于复现已经完成的旧 Stage-1 二字段、
> reference 坐标文本实验。下一版正式研究方案已改为“过去参考/历史图视觉画框 + 一次
> 统一三字段混合 LoRA SFT”，尚未实现；不要直接用本文命令启动下一版正式训练。最新
> 状态见 [`project_status_20260813.md`](project_status_20260813.md)，完整设计见
> [`stage2_stage3_data.md`](stage2_stage3_data.md)。

本文是面向训练服务器部署与实验复现的操作说明。目标是在不复制现有 Conda
目录、不依赖开发机绝对路径的前提下，用固定 Git commit、官方模型权重以及已有的
LaSOT/TNL2K/MGIT 原始训练集复现 Qwen3-VL-4B Stage-1 数据和训练环境。

## 1. 同步边界

| 内容 | 首选方式 | 说明 |
| --- | --- | --- |
| CognitiveTrack 代码与配方 | Git | 固定 commit，不复制上层 VLMTrack 的其他文件 |
| CognitiveBench v1 标注 | Git | 约 35MB，不含图片；随代码 clone |
| Qwen3-VL-4B 权重 | L40 从 ModelScope 官方仓库重新下载 | 约 8.89GB，不进入 Git |
| Stage-1 数据 | 用 L40 已有 LaSOT/TNL2K/MGIT 重建 | 必须重放固定 pair64 plan |
| 固定 sampling plan | 从训练服务器或后续数据发布包同步 | 重建时传给 `--sampling-plan` |
| Stage-1 LoRA | ModelScope 模型仓库 | 约 127MB，不进入 Git |
| Conda 环境 | 在 L40 从零安装 | 禁止复制另一台机器的环境目录 |
| 旧 outputs/cache | 不同步 | 不属于训练输入 |

至少为本次实验预留 100GB。LoRA adapter 很小，但完整图片数据、基座模型、数据缓存和
可恢复训练 checkpoint 仍会占用较多空间。

## 2. 推荐目录

```text
/workspace/CognitiveTrack/                         # Git clone
/datasets/raw/LaSOT/                               # 已有官方原始训练集
/datasets/raw/TNL2K/TNL2K_train_subset/            # 已有官方原始训练集
/datasets/derived/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1/  # 重建结果
/models/Qwen3-VL-4B-Instruct/                      # 官方权重
/models/CognitiveTrack-Qwen3VL-4B-Stage1-LoRA/    # 发布的 Stage-1 adapter
/outputs/cogtrack/qwen3vl_4b_stage1/               # 训练结果
/cache/cogtrack/                                   # 可删除缓存
```

代码同步：

```bash
git clone https://github.com/updateforever/CognitiveTrack.git /workspace/CognitiveTrack
cd /workspace/CognitiveTrack
git fetch --all --tags
git checkout <本次实验固定的 commit 或 tag>
git status --short
```

验收要求：`git status --short` 为空，并把 `git rev-parse HEAD` 写进实验记录。

## 3. GPU 与驱动预检

Qwen3-VL-4B 当前验证组合为 Python 3.10、PyTorch 2.8.0、CUDA runtime 12.8。
L40 是 Ada SM89，支持 BF16。宿主机驱动应支持 CUDA 12.8；推荐 570 系列或更新。
PyTorch wheel 自带 CUDA runtime，通常不需要另外安装 CUDA toolkit。

```bash
nvidia-smi
nvidia-smi topo -m
```

不要从 4090 机器照搬 `NCCL_P2P_DISABLE=1` 或 `NCCL_IB_DISABLE=1`。L40 第一轮先
保持 P2P/IB 默认开启；只有拓扑或 NCCL 日志证明不可用时再禁用。

## 4. 独立环境安装

仓库中的 `scripts/setup_env.sh` 会新建环境，不修改任何已有环境，并按兼容矩阵安装
PyTorch、transformers、ms-swift、vLLM 和预编译 flash-attn wheel：

```bash
cd /workspace/CognitiveTrack
ENV_NAME=cogtrack-l40 bash scripts/setup_env.sh
conda activate cogtrack-l40
```

已验证的关键版本：

```text
Python             3.10.20
torch              2.8.0+cu128
torchvision        0.23.0+cu128
transformers       4.57.1
ms-swift           4.3.1
flash-attn         2.8.3.post1
accelerate         1.14.0
datasets           4.8.4
peft               0.19.1
qwen-vl-utils      0.0.14
numpy              2.2.6
opencv-python      5.0.0.93
Pillow             11.3.0
```

安装完成后必须执行：

```bash
python scripts/verify_env.py --verbose
python -m pytest -q
python - <<'PY'
import torch, transformers, swift
print("torch", torch.__version__, "runtime CUDA", torch.version.cuda)
print("transformers", transformers.__version__, "ms-swift", swift.__version__)
print("cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
PY
```

`scripts/verify_env.py` 会真正执行 flash-attn kernel；仅 `import flash_attn` 不算验收。

## 5. 本机路径配置

```bash
cp configs/env.example.yaml configs/env.local.yaml
```

将 `configs/env.local.yaml` 改为：

```yaml
project_root: /workspace/CognitiveTrack
dataset_root: /datasets/raw
model_root: /models
output_root: /outputs/cogtrack

datasets:
  lasot: /datasets/raw/LaSOT
  tnl2k: /datasets/raw/TNL2K
  cognitivebench: /workspace/CognitiveTrack/benchmarks/cognitivebench/v1
  mgit: /datasets/raw/MGIT
```

`env.local.yaml` 已被 Git 忽略，禁止提交。TNL2K 也可直接指向
`TNL2K_train_subset`，但推荐指向其父目录，由 loader 按 `split=train` 解析。

CognitiveBench 标注已经随 Git clone，无需另行下载。评测它时还需要 LaSOT-test、
TNL2K-test 和 MGIT-val 原始图像；只进行 Stage-1 SFT 时可暂不准备这些测试图像。clone
后先运行：

```bash
python tools/verify_cognitivebench.py
```

## 6. 原始数据验收

正式 v1 使用：

- LaSOT train：1120 序列；
- TNL2K train：1300 序列；
- MGIT tiny/train：当前镜像 95 条有帧序列，10 条空目录显式排除；
- 固定 plan 最终包含 2511 条有合法 case 的序列、160,049 个不重复
  `(reference,current)` pairs；
- present 112,034、absent 48,015，各来源内部均约为 70:30。

先做 loader 级检查和正式源摘要检查：

```bash
python tracking/inspect_dataset.py \
  --dataset lasot --config configs/env.local.yaml --split train --limit 1
python tracking/inspect_dataset.py \
  --dataset tnl2k --config configs/env.local.yaml --split train --limit 1
python tracking/inspect_dataset.py \
  --dataset mgit --config configs/env.local.yaml --split train --limit 1
python tools/verify_stage1_sources.py \
  --lasot-root /datasets/raw/LaSOT \
  --tnl2k-root /datasets/raw/TNL2K \
  --output /datasets/manifests/stage1_source_verification.json
```

开发机正式数据的源摘要如下。若服务器上的镜像不匹配，应停止本地重建，改用第 8
节的 ModelScope 成品包，不要一边换源数据一边沿用同一个实验版本名。

```text
LaSOT training_set.txt:
0ae7df00644ee36794ac1d67f123612eb6341b4e94fef933169607d650ade893

LaSOT train 标注集合:
2c665501c9ee752f6ab2df3d61b806c2f1dc6a04f0e0209cb4582a989b987ed4

TNL2K train groundtruth 集合:
f551c0afaa9d9811b20dab162a535869326e2ef0576426f5109e8ca983b1db94

TNL2K train 排序后序列名:
db25334d28f236c0cd9c40e89edf34b529d4ba0d6cd93c8936b2a9a144fe8516
```

`verify_stage1_sources.py` 的聚合流只使用相对路径和单文件摘要，不把绝对路径写进
hash；任一摘要或序列数不符都会以非零状态退出。这里不直接采用 GNU
`sha256sum` 的文本输出作为聚合输入，因为含反斜杠的序列名会触发 GNU 转义，导致
相同文件集合在不同工具实现间出现摘要口径差异。

## 7. 使用固定计划重建数据

取得当前三来源 `sampling_plan.json`。正式 pair64 v1 的 plan SHA-256：

```text
158372c68e82918d9460826d89f601d1278a3f97e6980e2069718755689c03a7
```

执行确定性重建：

```bash
export STAGE1_ROOT=/datasets/derived/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1
export STAGE1_PLAN=/datasets/manifests/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1/sampling_plan.json

sha256sum "$STAGE1_PLAN"
python tracking/synthesize_stage1_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny \
  --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --context-mode pair \
  --frame-stride 1 \
  --max-samples-per-sequence 64 \
  --absent-ratio 0.3 \
  --history-size 4 \
  --mosaic-panel-height 240 \
  --max-image-side 648 \
  --jpeg-quality 95 \
  --val-ratio 0.05 \
  --seed 20260809 \
  --sampling-plan "$STAGE1_PLAN" \
  --qwen-model-families qwen2_5_vl qwen3_vl \
  --output-dir "$STAGE1_ROOT"
```

`--sampling-plan` 不会重新抽帧，并会拒绝数据集、seed、正负比例、每序列上限、
序列数量、case 数或 present/absent 标注发生变化。该正式导出已经用于完成 Stage-1
LoRA 训练；换服务器时必须取得原 plan/成品包并重新核对统计，禁止沿用旧
48,400-case v1 的摘要。

JPEG 会经过 OpenCV/libjpeg 重新编码。若源图片或底层编解码库不同，图片字节可能
不一致；此时至少要求 sample ID、状态、bbox、图片尺寸与 train/val 序列划分一致。

## 8. ModelScope 成品包兜底

新三来源 pair64 v1 尚未发布 ModelScope 成品包。发布后必须填入真实 dataset ID、固定
revision 和 SHA256SUMS；在此之前禁止使用下面的旧占位命令冒充正式来源。

```bash
modelscope download <OWNER/cogtrack-stage1-lasot-tnl2k-mgit-tiny-pair64-v1> \
  --repo-type dataset \
  --revision <固定 revision> \
  --local-dir /datasets/derived/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1 \
  --max-workers 4

cd /datasets/derived/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1
sha256sum -c SHA256SUMS
for shard in image_shards/images-*.tar; do tar -xf "$shard"; done
```

ModelScope 仓库 ID 和 revision 必须写入实验记录。私有仓库不自动消除 LaSOT/TNL2K
的上游再分发限制，对外发布前仍需检查许可。

## 9. 模型下载与协议检查

```bash
modelscope download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir /models/Qwen3-VL-4B-Instruct \
  --max-workers 4
```

只做推理或继续 Stage-2 时，同时下载已完成的 Stage-1 LoRA：

```bash
modelscope download updateforever/CognitiveTrack-Qwen3VL-4B-Stage1-LoRA \
  --local-dir /models/CognitiveTrack-Qwen3VL-4B-Stage1-LoRA \
  --max-workers 4

cd /models/CognitiveTrack-Qwen3VL-4B-Stage1-LoRA
sha256sum -c SHA256SUMS
```

`adapter_model.safetensors` 的 SHA-256 应为：

```text
732ff15f4791f75c1ca16b2a72163fe59ff8a8059e87e765caf22382ddd07131
```

模型分片 SHA-256：

```text
model-00001-of-00002.safetensors:
30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9

model-00002-of-00002.safetensors:
046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6
```

检查模型族与训练视图：

```bash
python tracking/validate_qwen_training_view.py \
  --model /models/Qwen3-VL-4B-Instruct \
  --dataset "$STAGE1_ROOT/ms_swift/qwen3_vl/train.jsonl" \
  --dataset "$STAGE1_ROOT/ms_swift/qwen3_vl/val.jsonl" \
  --expected-family qwen3_vl
```

禁止让 Qwen3 使用 `qwen2_5_vl` JSONL；两代坐标协议不同。

## 10. 单机多卡配置与两步 smoke

以下先用 2 卡复现正式 LoRA 的 global batch 8；扩展卡数时应保持 global batch，避免
首轮同时改变优化轨迹。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29501
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset NCCL_P2P_DISABLE NCCL_IB_DISABLE

export SWIFT_BIN="$(command -v swift)"
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct
export DATASET_ROOT="$STAGE1_ROOT"
export TRAIN_DATA="$STAGE1_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$STAGE1_ROOT/ms_swift/qwen3_vl/val.jsonl"
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_stage1_smoke
export REPORT_TO=none
export SAVE_STRATEGY=no
export EVAL_STRATEGY=no
export BATCH_SIZE=4
export GRAD_ACC_STEPS=1
export LEARNING_RATE=5e-5

bash scripts/train_qwen3vl_4b_stage1.sh \
  --max_steps 2 \
  --save_strategy no \
  --eval_strategy no
```

验收标准：

- 两张 GPU 均有训练进程；
- `model_type=qwen3_vl`、`tuner_type=lora`；
- `freeze_vit=true`、`freeze_aligner=true`；
- 可训练参数约 33.0301M/4.471B（0.7388%）；
- loss、grad norm 有限，无 NaN、图片路径或 bbox family 错误；
- L40 显存有安全余量。

## 11. 正式训练、断点与回传

已发布 Stage-1 的可复现实验参数为：2 卡、单卡 batch 4、梯度累积 1、global batch 8、
一轮、学习率 5e-5、cosine、warmup 0.05。原运行关闭了在线 evaluation；保留了按序列
隔离的 8,010 条 val，但没有 validation loss。

```bash
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_stage1
export SAVE_STRATEGY=steps
export SAVE_STEPS=1000
export SAVE_TOTAL_LIMIT=2
export EVAL_STRATEGY=no
export LOGGING_STEPS=20
export BATCH_SIZE=4 GRAD_ACC_STEPS=1 LEARNING_RATE=5e-5 EPOCHS=1

bash scripts/train_qwen3vl_4b_stage1.sh
```

恢复：

```bash
export RESUME_FROM_CHECKPOINT=/outputs/cogtrack/qwen3vl_4b_stage1/<run>/checkpoint-250
bash scripts/train_qwen3vl_4b_stage1.sh
```

恢复时尽量保持 world size、PyTorch、ms-swift 和数据不变。回传至少包含最终完整
checkpoint、`args.json`、`logging.jsonl`、trainer state、Git commit、ModelScope
revision、数据 checksum、GPU 拓扑和 NCCL 环境。不要在 checkpoint 正在写入时同步。

已完成运行的记录为 19,005/19,005 steps、约 4h45m41s、最终 train loss
0.29283377、token accuracy 0.882494、峰值记录 38.66GiB/卡。该结果只证明训练完成；
性能结论必须来自冻结 CognitiveBench 的零样本/LoRA 同协议完整指标。

## 12. 交付验收清单

另一位 AI 完成部署后必须回报：

1. `git rev-parse HEAD`；
2. `nvidia-smi` 与 `nvidia-smi topo -m`；
3. Python/PyTorch/CUDA/transformers/ms-swift/flash-attn 版本；
4. `scripts/verify_env.py --verbose` 和 `pytest -q` 结果；
5. 原始数据序列数及 sampling plan SHA-256；
6. 重建后的样本数、正负数、train/val 数及 JSONL SHA-256；
7. 模型权重分片 SHA-256；
8. smoke 的 trainable 参数、loss、显存和吞吐；
9. 正式输出目录、最终 adapter 和 ModelScope revision；
10. CognitiveBench 的 hold-last、observation-only、observation rate 与 presence 指标。
