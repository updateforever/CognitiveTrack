# 训练服务器恢复与当前主线执行

本文用于在新服务器从固定 Git commit 恢复 CognitiveTrack，并继续 VLT-v6.3 的 core
SFT、状态标签和后续 GRPO。旧 Stage-1 复现配方已移至
[`archive/l40_stage1_reproduction.md`](archive/l40_stage1_reproduction.md)。

## 1. 同步边界

| 内容 | 推荐方式 | 约束 |
| --- | --- | --- |
| 代码、配置、CognitiveBench 标注 | Git | 固定 commit/tag，仓库不含原图 |
| Qwen 权重与 adapter | ModelScope | 记录仓库 ID、revision、SHA-256 |
| LaSOT/TNL2K/MGIT | 使用服务器已有官方数据 | 先由 loader 验收 split |
| 派生训练数据 | 固定 plan 重建或私有数据仓库 | 不通过 Git 同步图片 |
| Conda 环境 | 每台服务器独立安装 | 不复制另一台机器的环境目录 |
| outputs/cache | 不作为代码同步内容 | checkpoint 单独上传并校验 |

SOIBench 不属于本项目，禁止同步到 CognitiveTrack 目录或训练包。

## 2. 拉取代码与创建环境

```bash
git clone https://github.com/updateforever/CognitiveTrack.git /workspace/CognitiveTrack
cd /workspace/CognitiveTrack
git fetch --all --tags
git checkout <固定 commit 或 tag>
git status --short

ENV_NAME=cogtrack-l40 bash scripts/setup_env.sh
conda activate cogtrack-l40
python scripts/verify_env.py --verbose
python -m pytest -q
```

如 GitHub HTTPS 连接被代理中断，可以使用 SSH-over-443：

```bash
git clone ssh://git@ssh.github.com:443/updateforever/CognitiveTrack.git /workspace/CognitiveTrack
```

不要通过关闭 SSL 校验规避网络问题。GPU/NCCL 先保持机器默认设置；只有拓扑和日志证明
P2P/IB 不可用时才禁用，不能从另一台 4090 机器复制 NCCL 环境变量。

## 3. 配置本机路径

```bash
cp configs/env.example.yaml configs/env.local.yaml
```

示例：

```yaml
project_root: /workspace/CognitiveTrack
model_root: /models
output_root: /outputs/cogtrack

datasets:
  cognitivebench: /workspace/CognitiveTrack/benchmarks/cognitivebench/v1
  lasot: /datasets/LaSOT
  tnl2k: /datasets/TNL2K
  mgit: /datasets/MGIT
```

`configs/env.local.yaml` 已被 Git 忽略，禁止提交。先检查 loader 和冻结 benchmark：

```bash
python tracking/inspect_dataset.py --dataset lasot --config configs/env.local.yaml --split train --limit 1
python tracking/inspect_dataset.py --dataset tnl2k --config configs/env.local.yaml --split train --limit 1
python tracking/inspect_dataset.py --dataset mgit --config configs/env.local.yaml --split train --limit 1
python tools/verify_cognitivebench.py
```

## 4. 下载基座模型

```bash
modelscope download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir /models/Qwen3-VL-4B-Instruct \
  --max-workers 4
```

旧 Stage-1/2 adapter 只作历史对照或初始化消融。VLT-v6.3 core 主实验默认从官方 Base
初始化，不能把旧 adapter 当成新协议已经训练好的权重。

## 5. 生成 VLT-v6.3 core 数据

先只生成 sampling plan，并把 plan、统计和 SHA-256 写入实验记录：

```bash
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --max-samples-per-sequence 20 --absent-ratio 0.3 \
  --output-dir /datasets/plans/cogtrack_vlt_v6_core \
  --plan-only
```

人工核对 sequence split、present/absent、reference/current 时间关系后，再严格重放：

```bash
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --sampling-plan /datasets/plans/cogtrack_vlt_v6_core/sampling_plan.json \
  --max-samples-per-sequence 20 --absent-ratio 0.3 \
  --output-dir /datasets/derived/cogtrack_vlt_v6_core
```

该阶段仅监督 presence/bbox；`memory_update` 的值被 mask。MGIT story 可能包含未来信息，
构造器必须回退到安全类别或视觉指代。

## 6. 训练前审计与 core SFT

```bash
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct
export DATASET_ROOT=/datasets/derived/cogtrack_vlt_v6_core
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_vlt_v6_core
export QWEN_MODEL_FAMILY=qwen3_vl

python tracking/validate_sft_supervision.py \
  --profile tracking_core \
  --dataset "$TRAIN_DATA" --dataset "$VAL_DATA"

python tools/verify_qwen_grounding_templates.py \
  --dataset-root "$DATASET_ROOT" \
  --qwen3-model "$MODEL_PATH" \
  --verify-tracking-core-mask

bash scripts/train_qwen3vl_4b_vlt_v6_core.sh \
  --max_steps 2 --save_strategy no --eval_strategy no
```

确认两步 smoke 无 NaN、bbox token 仍有 loss、只有 memory 值被 mask，再移除覆盖参数执行
正式训练：

```bash
bash scripts/train_qwen3vl_4b_vlt_v6_core.sh
```

训练细节和全参对照见 [`training.md`](training.md)。

## 7. 固定 Tiny 评测

启动 vLLM 服务后，Base 与 core adapter 必须使用相同的 VLT-v6.3 tracker 配置，只改变
模型权重。先运行 24 条完整序列 Tiny：

```bash
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/qwen3vl_4b_vlt_v6_base_vllm.yaml \
  --dataset-config configs/datasets/cognitivebench_tiny.yaml
```

core 模型尚未学习记忆生成，配置中应关闭 semantic memory 写入。必须保存 summary、
逐序列指标、运行 manifest、prompt version 和完整错误统计。

## 8. 状态标签与 GRPO

状态标签工具当前仍是下一阶段开发项，不能在服务器上假装已有入口。实现后按
[`state_annotation.md`](state_annotation.md) 的 `mine → annotate → verify → export`
顺序执行，先冻结人工审核集，再全量生成。Memory SFT 通过后，才按
[`grpo.md`](grpo.md) 增加 TU-GRPO；未来帧只能进入 reward 计算器。

## 9. 交付清单

每次远端任务至少回传：

1. Git commit 与 `git status --short`；
2. GPU 拓扑、环境版本和环境验证结果；
3. 原始数据序列统计、sampling plan、数据 JSONL SHA-256；
4. 模型/adapter revision 与 SHA-256；
5. 训练参数、可训练参数量、loss、显存、吞吐和异常日志；
6. 最终 checkpoint、trainer state 与数据/prompt 版本；
7. Tiny/Full 的固定协议指标及错误统计；
8. 记忆实验额外提供标签版本、update rate、身份矛盾率和轨迹效用。
