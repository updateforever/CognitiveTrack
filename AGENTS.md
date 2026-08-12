# CognitiveTrack AI 执行与进度恢复指南

本文件是新服务器、新会话或新代码 AI 接手 CognitiveTrack 时的首要上下文。开始工作
前必须完整阅读本文件；环境和数据的逐命令部署细节继续阅读
[`docs/l40_setup.md`](docs/l40_setup.md)。不要要求用户重新讲述已经记录在这里的项目
历史。

## 1. 项目是什么

CognitiveTrack 是建立在 pytracking 标准生命周期上的视觉语言模型长时单目标跟踪
研究框架：

```text
Sequence -> Tracker.initialize -> Tracker.track -> ResultWriter -> Evaluator
```

研究目标是让 VLM 在完整图像中完成目标判别与全局定位，并进一步研究稀疏执行、
时序上下文和语义变化记忆。当前论文目标是先回答一个更基础的问题：纯 VLM 是否能
从首帧身份参考出发，在长视频中判断目标是否存在，并在存在时定位同一实例。

本仓库是独立项目，不依赖上层 `/data2/wyp/VLMTrack` 的运行时代码。SOIBench 是另一个
尚未发表的项目，必须彻底隔离，禁止把其代码、数据、实验名称或结论迁入本仓库。

## 2. 当前研究主线与已冻结决策

除非用户明确改变研究设计，后续 AI 必须遵守以下决策：

- 当前优先级是纯 VLM 稀疏跟踪评测，不先做 SUTrack/VLM 工程融合。Hybrid 已有可用
  实现，但不是现阶段论文主实验。
- Stage-1 主 baseline 是 `Qwen3-VL-4B-Instruct`；Qwen2.5-VL-3B 只作代际/成本
  对照，Qwen3-VL-32B 只考虑 LoRA 对照。
- Stage-1/2 只监督 `present/absent` 与存在时的 bbox，不监督旧六分类、身份文字标签、
  解释文本、细粒度状态或数值置信度。
- 负样本只能来自同一视频中真实的目标消失帧，禁止使用跨序列错配或人工抹除目标。
- 输入图片保持完整，不做目标局部裁剪，不在图片上画 GT 框。Image 1 是完整初始化帧，
  初始框通过文本/官方 grounding 对象传入；Image 2 是未标注的完整当前帧。
- Stage-1 输出严格是两个字段。Qwen3 示例：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520]}
```

- 在线认知跟踪协议可以有第三字段 `memory_update`，其值同时表示“是否更新”和短语义
  增量；`null` 表示不更新。但记忆监督属于 Stage-3，当前 Stage-1 数据和 tracker
  配置必须关闭该字段。
- 记忆目标是模型自主学习“发生稳定重大外观变化时才更新”，不能从普通 bbox 标签
  伪造记忆文本。
- `parse_error`、`model_error`、`skipped` 和模型预测的 `absent` 必须严格分离；工程
  失败不得伪装成目标消失。
- 第一帧只用 GT 初始化，后续帧禁止把当前或未来 GT 传入 tracker。GT 只能在推理
  结束后用于评测。
- 训练、验证、测试按完整序列划分，禁止帧级泄漏。

## 3. Qwen 坐标协议：不可混用

Qwen2.5-VL 与 Qwen3-VL 不是同一套官方 grounding 坐标：

| 模型族 | 数据视图 | 输出字段 | tracker 协议 |
| --- | --- | --- | --- |
| Qwen2.5-VL | processor resize 后绝对像素 `xyxy` | `bbox_pixel_xyxy` | `qwen_abs_pixel` |
| Qwen3-VL | `[0,1000]` 相对 `xyxy` | `bbox_norm1000_xyxy` | `norm1000` |

训练 JSONL 使用 ms-swift 官方 `<bbox> + objects.bbox + image_id` 格式，并设置
`QWENVL_BBOX_FORMAT=new`。不得手工发明新的坐标解析方式，也不得把
`ms_swift/qwen2_5_vl/` 数据交给 Qwen3，反之亦然。训练脚本会在加载模型前检查模型
族和数据视图；不要绕过检查。

## 4. 当前代码结构

- `pytracking/`：数据集、tracker 生命周期、runner、结果写入和实验配置装配。
- `pytracking/trackers/cognitive_vlm.py`：纯 VLM pair/mosaic tracker 编排。
- `cogtrack/vlm/`：本地 Hugging Face Qwen 与 OpenAI-compatible API backend。
- `cogtrack/prompts/`：所有长 prompt；禁止重新硬编码到 tracker。
- `cogtrack/protocol/`：状态、bbox 和严格 JSON 协议。
- `cogtrack/cognition/`、`cogtrack/memory/`：状态机与可审计记忆门控。
- `cogtrack/training/`：数据构造、官方 Qwen grounding 导出、GRPO reward。
- `cogtrack/models/sutrack/`：独立迁入且已做数值保真验证的 SUTrack runtime。
- `tracking/`：命令行入口。
- `configs/models/`、`configs/trackers/`、`configs/training/`：模型、实验与训练配方。
- `scripts/train_qwen3vl_4b_stage1.sh`：当前主训练入口。
- `docs/l40_setup.md`：换服务器恢复的完整操作手册。

新增 tracker 时至少同步检查 tracker、parameter/config、prompt 三层。路径只放在
`configs/env.local.yaml` 或环境变量中，禁止把本机绝对路径写入可提交配置。

## 5. 截至 2026-08-10 的真实进度

### 工程与基线

- pytracking loader、tracker、结果 TXT/JSONL、评测闭环已完成。
- SUTrack-B384 官方 epoch 180 checkpoint 已做严格键匹配和真实前向；迁入实现与原版
  两条语言分支逐帧 bit 级一致。
- 相同预测经原版和本项目指标聚合，15 条序列五项指标最大绝对差为 `0.00e+00`。
- Qwen2.5/Qwen3 本地多图推理、pair/mosaic、二分类严格解析均已接通。
- Qwen3-VL-4B 在 CognitiveBench `005` 的真实双图零样本样例执行成功：预测约
  `[669.0,422.0,40.0,100.0]`，GT 为 `[668,422,38,100]`。
- 真实 Qwen processor 回放已经证明：Qwen2.5 使用 resize 后绝对像素，Qwen3 使用
  norm1000，且多图 assistant bbox 正确绑定当前帧。
- CognitiveBench v1 的 995 序列冻结标注已纳入
  `benchmarks/cognitivebench/v1/`，共 1,408,438 帧和 343,616 个 0-based 关键帧；
  benchmark 不含图像，运行时仍依赖 LaSOT-test、TNL2K-test 和 MGIT-val。

### Stage-1 正式数据

- 来源：LaSOT train + TNL2K train；MGIT 尚未纳入正式 v1。
- 2420 个序列，48,400 cases；present 33,880、absent 14,520，严格 70:30。
- train 45,980、val 2,420，按完整序列、按数据来源分层划分。
- 图片 50,820 张；解包数据约 3.63GB，Hub/ModelScope 发布包约 3.66GB。
- 正式本地构建目录曾为：
  `data/releases/cogtrack_stage1_lasot_tnl2k_fullref_v1`。
- 可迁移发布包曾为：
  `data/releases/cogtrack_stage1_lasot_tnl2k_fullref_v1_hub`。
- `sampling_plan.json` SHA-256：
  `08c7a271fed2562b8692dfe6a198c8ac6199b4022e45e392ff4dd04c9f13dd31`。
- 构造 CLI 已支持 `--sampling-plan` 严格重放，源数据验收工具为
  `tools/verify_stage1_sources.py`。

数据和 outputs 被 `.gitignore` 排除，不会随 Git clone 出现。固定 sampling plan 应从
私有 ModelScope 数据仓库或原服务器取得；如果文档中仍是 `<OWNER/...>` 占位符，先向
用户询问一次实际 ModelScope dataset ID/revision，不要静默重新抽样。

### Qwen3-VL-4B SFT 冒烟

- 环境：8×RTX 4090 24GB、PyTorch 2.8.0+cu128、transformers 4.57.1、ms-swift
  4.3.1、flash-attn 2.8.3.post1。
- 训练定义：`tuner_type=full`，冻结视觉主干，全参训练 LLM、主 merger 和三个
  deepstack merger；这就是本项目所说的“常规全量微调”。
- 总参数 4.438B，可训练 4.132B（93.10%）。
- FSDP2 full shard，BF16，单卡 batch 2，全局 batch 16。
- 两步峰值 16.29GiB/卡，吞吐约 4.64 samples/s；首步 loss 1.536，两步平均 1.366。
- 只完成了两步 smoke，尚未宣称完成正式一轮 Stage-1 训练或获得训练后 benchmark
  提升。

## 6. 新服务器最快恢复流程

以下是摘要流程；路径、checksum、L40/NCCL 和故障处理以
[`docs/l40_setup.md`](docs/l40_setup.md) 为准。

### 6.1 拉取固定代码

```bash
git clone https://github.com/updateforever/CognitiveTrack.git /workspace/CognitiveTrack
cd /workspace/CognitiveTrack
git checkout <用户指定的 commit 或 tag>
git status --short
```

如果服务器的 HTTPS/GnuTLS 被代理提前断开，可按 GitHub 支持的 SSH-over-443 使用：

```bash
git clone ssh://git@ssh.github.com:443/updateforever/CognitiveTrack.git /workspace/CognitiveTrack
```

这仍需要该服务器 SSH key 已加入 GitHub。不要关闭 SSL 校验规避网络问题。

### 6.2 新建独立环境

```bash
cd /workspace/CognitiveTrack
ENV_NAME=cogtrack-l40 bash scripts/setup_env.sh
conda activate cogtrack-l40
python scripts/verify_env.py --verbose
python -m pytest -q
```

当前完整测试基线是 119 个通过。若测试数量随后续 commit 变化，以该 commit 的测试
集合为准，但不能忽略失败项。

### 6.3 配置机器路径

```bash
cp configs/env.example.yaml configs/env.local.yaml
```

编辑 `env.local.yaml`，至少设置 `project_root`、`model_root`、`output_root`、LaSOT 和
TNL2K 路径。该文件禁止提交。

### 6.4 准备数据

首选使用服务器已有的 LaSOT/TNL2K 和固定 plan 重建：

```bash
python tools/verify_stage1_sources.py \
  --lasot-root /datasets/raw/LaSOT \
  --tnl2k-root /datasets/raw/TNL2K

python tracking/synthesize_stage1_dataset.py \
  --datasets lasot tnl2k \
  --env-config configs/env.local.yaml \
  --context-mode pair \
  --frame-stride 1 \
  --max-samples-per-sequence 20 \
  --absent-ratio 0.3 \
  --max-image-side 648 \
  --jpeg-quality 95 \
  --val-ratio 0.05 \
  --seed 20260809 \
  --sampling-plan /path/to/sampling_plan.json \
  --qwen-model-families qwen2_5_vl qwen3_vl \
  --output-dir /datasets/derived/cogtrack_stage1_lasot_tnl2k_v1
```

如果原始数据摘要不匹配，禁止继续沿用 v1 名称；改为下载 ModelScope 成品包并运行
`sha256sum -c SHA256SUMS`。不要同时保留 unpacked 和 Hub 两份数据浪费空间。

### 6.5 下载主模型

```bash
modelscope download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir /models/Qwen3-VL-4B-Instruct \
  --max-workers 4
```

### 6.6 训练前验证

```bash
export DATASET_ROOT=/datasets/derived/cogtrack_stage1_lasot_tnl2k_v1

python tracking/validate_qwen_training_view.py \
  --model /models/Qwen3-VL-4B-Instruct \
  --dataset "$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl" \
  --dataset "$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl" \
  --expected-family qwen3_vl
```

有 Qwen2.5 权重时再运行双代 processor 回放；没有 Qwen2.5 时不要为了这项审计阻塞
Qwen3 smoke。

### 6.7 两步 SFT smoke

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset NCCL_P2P_DISABLE NCCL_IB_DISABLE

export SWIFT_BIN="$(command -v swift)"
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_stage1_smoke
export SAVE_STRATEGY=no EVAL_STRATEGY=no REPORT_TO=none

bash scripts/train_qwen3vl_4b_stage1.sh \
  --max_steps 2 --save_strategy no --eval_strategy no
```

L40 首轮不要照搬 4090 的 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`；先根据
`nvidia-smi topo -m` 和 NCCL 日志判断。smoke 必须确认模型族、冻结范围、可训练参数、
有限 loss、显存和吞吐。

### 6.8 正式 Stage-1

只有 smoke 通过后才启动：

```bash
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_stage1
export SAVE_STRATEGY=steps SAVE_STEPS=250 SAVE_TOTAL_LIMIT=2
export EVAL_STRATEGY=steps EVAL_STEPS=250 LOGGING_STEPS=5
bash scripts/train_qwen3vl_4b_stage1.sh
```

恢复使用 `RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N`。不要在 checkpoint 正在写入时
同步。回传至少包含最终 checkpoint、`args.json`、`logging.jsonl`、trainer state、
Git commit、数据/模型 revision 与 checksum、GPU 拓扑和 NCCL 环境。

## 7. 接手后的正确工作顺序

新 AI 必须先做只读确认，再执行改动：

1. 阅读本文件、`README.md`、`docs/l40_setup.md`、`docs/training.md`。
2. 运行 `git status -sb`，保护用户已有修改，不使用破坏性 reset/checkout。
3. 确认当前 Git commit、GPU 型号/数量、数据根、模型根和输出根。
4. 运行 `python tools/verify_cognitivebench.py`、`pytest -q` 和与任务相关的最小验证。
5. 恢复任务时先检查现有 `logging.jsonl`、checkpoint 完整性和 GPU 残留进程，禁止
   重复启动同一训练。
6. 先完成两步 smoke，再开始昂贵实验；先纯 VLM，再讨论 hybrid。
7. 结果汇报必须区分“工程链路已跑通”“零样本单 case 合理”“正式 benchmark 有提升”
   三种不同证据等级。

## 8. 下一阶段建议

当前安全的推进顺序是：

1. 在 L40 恢复固定 commit、环境、正式数据和 Qwen3-VL-4B。
2. 复现两步 smoke，记录 L40 显存/吞吐，不改变 global batch 16。
3. 完成一轮 Stage-1 SFT，并保存可恢复 checkpoint。
4. 先跑纯 VLM 稀疏评测，比较零样本、SFT 后和 Qwen2.5 对照。
5. 分析 present/absent、定位、消失/重现和长间隔 case，而不只看单一 AUC。
6. Stage-1 确认有效后再构建 Stage-2 mosaic/时序上下文。
7. 只有取得可靠更新时机和语义增量标签后才进入 Stage-3 memory SFT/GRPO。

## 9. 开发与提交约束

- Python 使用 4 空格、`snake_case` 函数/模块、`PascalCase` 类；中文注释说明研究
  约束和易错逻辑，不为显然代码堆注释。
- Prompt 集中在 `cogtrack/prompts/`；关键帧策略放 evaluation/observation policy，
  不在 tracker 内重复实现。
- 数据、模型、checkpoint、outputs、缓存和 `configs/env.local.yaml` 不进入 Git。
- 修改 bbox、状态、NaN 填充或结果格式后，必须确认传统 TXT、逐帧 JSONL 和评测器
  兼容。
- 提交前至少运行 Ruff、完整测试和一个任务相关 smoke。
- 使用明确提交信息，例如 `feat: replay fixed stage1 sampling plan`、
  `docs: add L40 recovery guide`，不要使用含糊的 `add`。
- 不要声称未运行的实验已经完成，不要根据两步 loss 推断正式精度提升。
