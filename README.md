# CognitiveTrack

CognitiveTrack 是一个建立在 [pytracking](https://github.com/visionml/pytracking) 推理与
评测范式上的视觉语言模型长时单目标跟踪框架。项目研究 VLM 能否从完整参考图像和
初始目标框出发，在长视频中完成全图搜索、目标存在性判断、同实例定位、稀疏执行与
时序记忆。

框架同时提供纯 VLM、SUTrack 和 Hybrid 跟踪器，训练侧使用 ms-swift 支持 SFT 与
GRPO。所有实验遵循：第一帧仅用 GT 初始化，后续推理不读取当前或未来 GT；训练、
验证和测试按完整序列划分。

当前 VLT-v6.3 基线使用固定三图输入：带框首帧身份锚点、从左到右排列的近期三帧带框
历史条带和无框当前全图。历史 panel 由白色竖向分隔带隔开；不足三帧时复制最近可用
历史进行右侧 padding。文本状态分为不可变初始身份与可替换当前目标状态，输出目标
存在性、当前框和可空状态更新。首轮 SFT 只监督存在性与 bbox；完整研究路线见
[docs/research_plan.md](docs/research_plan.md)。

## 主要特性

- 标准 `Sequence -> initialize -> track -> ResultWriter -> Evaluator` 生命周期；
- Qwen3-VL、Qwen2.5-VL 本地多图推理，以及可扩展的 API/vLLM backend；
- 完整图像输入，不依赖局部搜索裁剪；
- 严格区分 `present`、`absent`、`skipped`、解析错误和模型错误；
- pair、mosaic、视觉历史和语义记忆上下文；
- dense、关键帧 sparse、SUTrack 与 VLM hybrid 执行策略；
- TXT bbox 与逐帧 JSONL 双结果格式；
- pytracking 定位指标、稀疏执行指标和目标存在性诊断；
- 基于 LaSOT、TNL2K、MGIT 的确定性训练数据构造；
- ms-swift LoRA/全参 SFT，以及可扩展的 GRPO reward 接口。

## CognitiveBench

仓库包含 CognitiveBench v1 的冻结标注：995 个长时序列、1,408,438 帧和 343,616
个关键帧。标注由 LaSOT test、TNL2K test 和 MGIT val 组成，不包含原始图像。

项目同时冻结 `CognitiveBench-Tiny v1`：从三个来源选取 24 条完整序列，共 39,251 帧
和 9,892 个关键帧。Tiny 用于快速正式迭代，Full 用于最终主表；Tiny 不是截帧 smoke，
两者复用完全相同的标注、评测器与推理协议。

完整数据格式和使用约束见
[benchmarks/cognitivebench/v1/README.md](benchmarks/cognitivebench/v1/README.md)。clone 后可
离线检查标注完整性：

```bash
python tools/verify_cognitivebench.py
```

## 安装

推荐创建独立 Python 3.10 环境：

```bash
conda create -n cogtrack python=3.10 -y
conda activate cogtrack

pip install -e '.[qwen,sutrack,test]'
# 需要训练时再安装：
pip install -e '.[train]'
```

环境兼容矩阵、CUDA 和 FlashAttention 检查见
[docs/environment.md](docs/environment.md)。安装后运行：

```bash
python scripts/verify_env.py --verbose
python -m pytest -q
```

## 本机路径

复制配置模板：

```bash
cp configs/env.example.yaml configs/env.local.yaml
```

编辑本机路径：

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

`configs/env.local.yaml` 不进入 Git。CognitiveBench loader 会按照每条序列的
`source_dataset` 从上述三个公开数据集根目录解析原始帧。

## 模型

### Qwen3-VL-4B 基座

```bash
modelscope download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir /models/Qwen3-VL-4B-Instruct \
  --max-workers 4
```

已发布的 Stage-1/2 adapter 属于旧协议，只用于历史复现。当前 checkpoint 与实验完成度
统一记录在 [docs/project_status.md](docs/project_status.md)。

## 推理

先做不计入正式指标的链路检查：

```bash
python tracking/smoke_test_qwen.py \
  --tracker-config configs/trackers/qwen3vl_4b_vlt_v6_base.yaml \
  --dataset-config configs/datasets/cognitivebench_tiny.yaml \
  --env-config configs/env.local.yaml
```

正式 VLT-v6.3 推理固定使用初始化文本、首帧身份锚点、三帧可信历史条带、最近状态
记忆、当前关键帧全图，以及 `target_status + bbox + memory_update` 三字段协议。先在
Tiny 上评测：

```bash
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/qwen3vl_4b_vlt_v6_base_vllm.yaml \
  --dataset-config configs/datasets/cognitivebench_tiny.yaml
```

确认协议和性能趋势后，仅将数据配置换成 `configs/datasets/cognitivebench.yaml` 即可
运行 Full。核心 SFT adapter 使用 `qwen3vl_4b_vlt_v6_core_sft_vllm.yaml`；旧坐标文本、pair、
无历史和无语义记忆配置只用于历史复现或消融。SUTrack 与 Hybrid 配置位于
`configs/trackers/`；SUTrack checkpoint 通过
`COGTRACK_SUTRACK_CHECKPOINT` 注入，不在 YAML 中写绝对路径。

## 评测

```bash
python tracking/evaluate.py \
  --input /outputs/cogtrack/<tracker-run>/cognitivebench
```

实验结论以冻结数据、固定观察策略和完整序列聚合指标为准，不根据人工挑选的单帧
case 判断提升。错误 case 仅用于分析失败原因。

稠密跟踪器沿用 pytracking 的 AUC、OP50、OP75、P@20 和 Pnorm@0.2。纯 VLM 稀疏
评测同时报告：

- `hold_last`：非观察帧沿用最近一次合法框，衡量任意时刻的跟踪状态；
- `observation_only`：只在模型实际观察的关键帧上计分，衡量 VLM 本身的判别与定位；
- observation rate：实际模型调用比例。

`dense_zero` 为兼容传统结果保留，但其上限受关键帧率限制，不能单独用于比较不同
稀疏策略。目标判别同时报告 presence precision/recall/F1、absent false-positive
rate、present miss rate 和 decision coverage。

`--debug-frames` 生成的截断结果默认不进入正式聚合。评测会输出 `summary.json`、
`sequence_metrics.csv` 和 `report.md`。

## 数据与训练

VLT-v6.3 使用 LaSOT train、TNL2K train 和 MGIT train 中同一序列的真实
present/absent 帧。所有 reference/history 严格早于 current；负样本只来自原视频的
真实目标消失帧。下面命令先生成可重放的采样计划：

```bash
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --max-samples-per-sequence 20 \
  --absent-ratio 0.3 \
  --output-dir data/plans/cogtrack_vlt_v6_core \
  --plan-only
```

首轮三字段数据无需伪造 memory label；训练时只屏蔽 `memory_update` 的值。完整渲染、
字段级 loss 校验与训练命令见 [docs/vlt_v6_core_sft.md](docs/vlt_v6_core_sft.md)。
Qwen3-VL 使用 `[0,1000]` 相对 `xyxy`；
Qwen2.5-VL 使用 processor resize 后绝对像素 `xyxy`，两代 JSONL 不能交叉使用。

```bash
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct
export DATASET_ROOT=/datasets/cogtrack_vlt_v6_core
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_vlt_v6_core
export QWEN_MODEL_FAMILY=qwen3_vl

bash scripts/train_qwen3vl_4b_vlt_v6_core.sh
```

详细监督边界、数据格式和训练说明见 [docs/training.md](docs/training.md)。状态标签冷启动
与轨迹效用 GRPO 分别见 [docs/state_annotation.md](docs/state_annotation.md) 和
[docs/grpo.md](docs/grpo.md)。当前完成度见 [docs/project_status.md](docs/project_status.md)，
旧尝试统一保存在 [docs/archive/](docs/archive/README.md)。

## 代码结构

```text
pytracking/             数据集、tracker 生命周期、runner 与结果写入
cogtrack/vlm/           VLM backend 与模型缓存
cogtrack/prompts/       pair、mosaic 和 memory prompt
cogtrack/protocol/      bbox、状态和严格 JSON 协议
cogtrack/cognition/     目标存在性与状态机
cogtrack/memory/        可审计时序记忆
cogtrack/evaluation/    pytracking、稀疏执行和认知指标
cogtrack/training/      数据构造、Qwen grounding 与 GRPO reward
cogtrack/models/        SUTrack runtime 与传统跟踪接口
tracking/               训练、推理和评测命令行入口
configs/                模型、tracker、数据与训练配置
```

架构与协议详见 [docs/architecture.md](docs/architecture.md) 和
[docs/protocol.md](docs/protocol.md)。

## 许可

项目采用 MIT License。pytracking、SUTrack、Qwen 模型及各数据集遵循其原始许可证和
使用条款；发布或再分发权重、标注与图像前，请分别核对对应上游许可。
