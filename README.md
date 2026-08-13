# CognitiveTrack

代码 AI 或新服务器接手时，先阅读根目录的
[AGENTS.md](AGENTS.md)。其中记录了研究边界、已验证进度、关键设计决策、换服务器
恢复步骤、下一阶段顺序和禁止事项。

2026-08-13 的统一视觉画框与混合训练讨论摘要见
[docs/project_status_20260813.md](docs/project_status_20260813.md)，完整设计草案见
[docs/stage2_stage3_data.md](docs/stage2_stage3_data.md)。

CognitiveTrack 是一个建立在 **pytracking 标准推理范式**之上的大模型长时认知跟踪框架。项目关注的不是让视觉语言模型被迫逐帧画框，而是显式研究：

1. 初始化目标当前是否可见且可定位；
2. 存在时能否在当前全图中输出目标框；
3. 模型能否自主判断何时产生一条新的语义变化记忆；
4. 稀疏执行下如何利用视觉历史和语义记忆支持长时跟踪。

本目录是可独立安装的新实现，不依赖目录外的运行时代码。范围严格限定为单目标
认知跟踪；其他研究子系统、数据路径和实验产物均不进入本项目。

## 设计原则

- 遵循 `Sequence -> Tracker.initialize -> Tracker.track -> ResultWriter -> Evaluator` 流程。
- 第一帧只使用 GT 初始化，后续推理禁止读取当前或未来 GT。
- 内部 bbox 统一为像素级 `xywh`；训练和推理都按模型代际显式选择官方坐标协议。
- 模型与 GT 均只使用 `present/absent`；不监督身份、细粒度状态或数值置信度。
- 在线推理第三字段 `memory_update` 同时承担更新开关和短语义增量；`null` 表示不更新。
- `parse_error`、`model_error`、`skipped` 与 `absent` 严格分离。
- 传统 TXT 结果和包含认知信息的逐帧 JSONL 同时保存。
- 训练、验证、测试必须按序列划分，禁止帧级数据泄漏。

## 首批数据集

- CognitiveBench v1（995 序列的冻结标注已随仓库发布，原始图像仍从下面数据集解析）
- LaSOT
- TNL2K
- MGIT

CognitiveBench 的格式、规模、原始图像依赖和评测约束见
[benchmarks/cognitivebench/v1/README.md](benchmarks/cognitivebench/v1/README.md)。clone 后
可先运行 `python tools/verify_cognitivebench.py` 离线检查标注完整性。

使用 Qwen3-VL 4B 运行稀疏关键帧快速验证：

```bash
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/qwen3vl_4b_pair_cognitivebench_sparse.yaml \
  --dataset-config configs/datasets/cognitivebench.yaml \
  --sequence 005 --debug-frames 5
```

该配置只在 `keyframes.txt` 指定帧执行纯 VLM，未融合 SUTrack；非关键帧的
`skipped/NaN` 是稀疏执行协议的一部分，debug 截断结果不能作为正式稠密 SOT 指标。

## 环境

推荐创建独立环境，避免影响已有实验环境：

```bash
conda create -n cogtrack python=3.10 -y
conda activate cogtrack
pip install -e '.[qwen,sutrack,test]'
# 需要训练时再安装：
pip install -e '.[train]'
```

当前开发机采用更保守的离线隔离方式：从已验证的 Python 3.10 环境克隆为
`cogtrack`，随后使用 `pip install -e . --no-deps --no-build-isolation` 安装本
项目。推理和训练脚本默认只使用 `cogtrack`，不会向其他 Conda 环境安装依赖。

将 `configs/env.example.yaml` 复制为 `configs/env.local.yaml` 并修改本机路径。真实配置不会被提交。
完整的新建、离线克隆和环境自检见 [docs/environment.md](docs/environment.md)。
在 L40 服务器上进行 Stage-1 训练时，使用
[docs/l40_setup.md](docs/l40_setup.md) 中的 Git、原始数据重建、ModelScope 兜底、
NCCL、两步 smoke 和断点恢复流程。

## 预期命令

```bash
# 检查数据集
python tracking/inspect_dataset.py --dataset cognitivebench --config configs/env.local.yaml

# 本地 Qwen 单样本检查
python tracking/smoke_test_qwen.py \
  --tracker-config configs/trackers/qwen3vl_4b_pair.yaml \
  --dataset-config configs/datasets/cognitivebench.yaml \
  --env-config configs/env.local.yaml

# 标准跟踪
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/qwen25vl_7b_pair.yaml \
  --dataset-config configs/datasets/cognitivebench.yaml \
  --sequence 005 --debug-frames 5

# 公开 SUTrack-B384 checkpoint 基线
export COGTRACK_SUTRACK_CHECKPOINT=/path/to/SUTRACK_ep0180.pth.tar
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/sutrack_b384.yaml \
  --dataset-config configs/datasets/cognitivebench.yaml \
  --sequence 005 --debug-frames 5

# SUTrack 稠密定位 + Qwen 关键帧认知判别
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/hybrid_sutrack_b384_qwen25vl.yaml \
  --dataset-config configs/datasets/cognitivebench.yaml \
  --sequence 005 --debug-frames 20

# 评测
python tracking/evaluate.py \
  --result-root outputs/cognitive_vlm/qwen25vl_7b_pair_v4_memory/cognitivebench
```

评测主指标是 pytracking 口径（AUC / OP50 / OP75 / P / Pnorm），与 SUTrack 已发表
数字同源。CognitiveTrack 自定义的 presence / identity / reappearance 指标下沉到
`cognitive_diagnostics`，只作诊断，不参与对外比较。

`--debug-frames N` 跑出的截断结果默认**不计入**指标：主指标按序列宏平均，一条
20 帧的截断序列会和几千帧的完整序列等权，实测能把 AUC 抬高 7 个点。评测器读
manifest 的 `extra.debug_limited` 自动跳过并打印被排除的文件；确实需要纳入时加
`--include-debug-runs`。

具体可用参数以各脚本的 `--help` 为准。

移植保真度有两个独立复核工具，详见 [docs/migration.md](docs/migration.md)：

- [tools/verify_sutrack_parity.py](tools/verify_sutrack_parity.py)：逐帧比较内置
  SUTrack 与原版仓库的预测 bbox（已核对：两条语言分支均 bit 级一致）。
- [tools/verify_metrics_parity.py](tools/verify_metrics_parity.py)：同一份预测分别用
  原版 `extract_results.py` 的函数和本项目评测路径聚合（已核对：15 条序列五个指标
  最大绝对差 0.00e+00）。

## VLM bbox 坐标协议

Qwen2.5-VL 与 Qwen3-VL **不是同一套坐标协议**，不能共用一份已写死坐标文本的
训练 JSONL：

| 模型族 | 官方 grounding 坐标 | tracker 配置 | 输出字段 |
| --- | --- | --- | --- |
| Qwen2.5-VL | processor resize 后图像的绝对像素 `xyxy` | `qwen_abs_pixel` | `bbox_pixel_xyxy` |
| Qwen3-VL | `[0,1000]` 相对 `xyxy` | `norm1000` | `bbox_norm1000_xyxy` |

因此 Qwen2.5-VL 评测配置为：

```yaml
bbox_protocol: qwen_abs_pixel   # 模型 JSON 字段名为 bbox_pixel_xyxy
```

要求 Qwen2.5-VL 输出 norm1000 会让它做未按该代际协议训练的格式，实测把 68/69 个观测帧的
IoU 压成恰好 0，而模型的文字推理其实是对的。参考系是 processor 两级缩放之后的
尺寸（`930x510 -> 648x355 -> 644x364`），由 `image_grid_thw` 如实回推，不能用
`max_image_side` 反算。Qwen3-VL 则使用 `norm1000`。完整分析见
[docs/migration.md](docs/migration.md)。

训练数据不重新发明坐标解析：JSONL 使用 ms-swift 官方
`<bbox> + objects.bbox + image_id`，由实际模型模板完成代际专属转换，并设置
`QWENVL_BBOX_FORMAT=new`。启动脚本会在占用 GPU 前拒绝模型族与数据目录不匹配。

## 训练数据与 ms-swift

```bash
# 从官方训练集构建 Stage-1 真实 present/absent 数据（case 比例约 7:3）
python tracking/synthesize_stage1_dataset.py \
  --datasets lasot tnl2k mgit --env-config configs/env.local.yaml \
  --context-mode pair --max-samples-per-sequence 20 --absent-ratio 0.3 \
  --output-dir data/stage1_tracking_presence_v1

MODEL_PATH=/path/to/Qwen2.5-VL \
TRAIN_DATA=/abs/path/stage1_tracking_presence_v1/ms_swift/qwen2_5_vl/train.jsonl \
VAL_DATA=/abs/path/stage1_tracking_presence_v1/ms_swift/qwen2_5_vl/val.jsonl \
DATASET_ROOT=/abs/path/stage1_tracking_presence_v1 \
bash scripts/train_sft.sh
```

训练 Qwen3-VL 时同时换成 Qwen3-VL 模型和 `ms_swift/qwen3_vl/` 两个 JSONL，不能
只换模型路径。两套 JSONL 共享同一批图像和样本划分，不会增加图片存储。

上面的命令记录已经完成的旧 Stage-1 二字段数据：约 70% `present+bbox`、30%
`absent+null`，reference bbox 通过官方 grounding 坐标传入。2026-08-13 起下一版正式
方案改为一次统一混合三字段 LoRA SFT：所有过去的参考/历史完整图直接画框，不再提供
reference 坐标文本；当前完整搜索图始终无框；单参考、mosaic、消失/重现、历史扰动和
语义记忆样本在一份数据中混合。完整草案与未完成清单见
[docs/stage2_stage3_data.md](docs/stage2_stage3_data.md)，旧命令不能直接生成新范式数据。
本地 7B 模型的 v4.1 三字段零样本探针及由此得到的数据优先级见
[docs/qwen_v4_probe.md](docs/qwen_v4_probe.md)。

## 当前完成度

- pytracking 数据集、tracker 生命周期、结果落盘和评测闭环已可用。
- Qwen2.5-VL/Qwen3-VL 本地后端、pair/mosaic、二分类严格解析、视觉正记忆与模型控制的语义记忆已接通。
- SUTrack-B384 Fast-iTPN、CLIP 文本塔、裁剪预处理、checkpoint runtime 和进程级模型缓存已经独立迁入；同时保留插件契约，便于替换其他 SUTrack 变体。
- “每帧 SUTrack + 关键帧 VLM” hybrid 已用真实 SUTrack checkpoint 跑通。
- ms-swift SFT 的 Qwen2.5/Qwen3 官方 grounding 视图、二/三字段校验和序列划分已可用；GRPO 的格式、presence、一致性 reward 已接入，bbox reward 需按模型族单独验证。
- 已在 2×L40 完成旧范式 Qwen3-VL-4B Stage-1 和 Stage-2 的同一 LoRA 连续训练；
  Stage-2 最终为 28,819 steps、train loss 0.25926472、token accuracy 0.89407277。该结果
  只证明训练完成，尚无正式 benchmark 提升结论。
- Stage-1/2 adapter 已上传私有 ModelScope 仓库
  `updateforever/CognitiveTrack-Qwen3VL-4B-Stage1-LoRA`；Stage-2 位于
  `stage2-mosaic-robust-v2/`，权重 SHA-256 为
  `7437dd2be3bae21070059ef5ce704da7bbd5008607f7e04eb18b3638f04930a1`。
- 新的视觉画框统一混合范式尚未实现或训练；当前 tracker 首帧仍走坐标文本，不能把
  旧 Stage-2 权重描述为新范式模型。

当前主 baseline 选用 `Qwen3-VL-4B-Instruct`：规模适中，输出使用 Qwen3 官方
`norm1000` grounding 协议，训练默认采用 LoRA 并冻结视觉塔和 aligner。Qwen2.5-VL-3B
保留为代际/成本对照；Qwen3-VL-32B 只考虑 LoRA 对照。

## 已验证的本地 Qwen 链路

2026-08-10 下载并验证官方 `Qwen3-VL-4B-Instruct` 后，使用完整
`qwen3vl_4b_pair` 配置完成 CognitiveBench 双图零样本跟踪。模型严格输出二字段
JSON；样例预测框约为 `[669.0, 422.0, 40.0, 100.0]`，对应 GT 为
`[668, 422, 38, 100]`，执行状态为 `ok`。同一权重还完成了 8×4090 FSDP2
两步 SFT 冒烟，确认冻结视觉塔、全参训练 LLM 与 Qwen3 merger 的链路可运行。
结构化推理报告保存在 `outputs/qwen3vl_4b_zero_shot_smoke.json`，训练实测见
[docs/training.md](docs/training.md)。

2026-08-05 使用本地 Qwen2.5-VL-7B-Instruct 在 CognitiveBench 单序列上完成了
真实权重加载、双图生成、结构化 JSON 严格解析、坐标反归一化、标准 CLI
落盘与离线评测。该 smoke test 只证明工程链路正确，不代表未微调模型的跟踪精度。

同日使用公开 SUTrack-B384 epoch 180 checkpoint 完成严格权重键匹配、CPU 两帧
真实前向、Hybrid 非关键帧连续定位和标准评测，运行结果为 0 工程错误。checkpoint
路径只通过 `COGTRACK_SUTRACK_CHECKPOINT` 注入，并在 manifest 中记录 SHA-256。

## 代码结构

- `pytracking/`：数据、tracker 生命周期、运行调度和结果保存。
- `cogtrack/protocol/`：统一状态与 bbox 协议。
- `cogtrack/vlm/`：本地 Qwen-VL 及后续 API/vLLM backend。
- `cogtrack/cognition/`：目标存在性与时序状态机。
- `cogtrack/memory/`：带来源审计、连续确认和几何一致性门控的记忆管理。
- `cogtrack/evaluation/`：传统、存在性、身份和恢复指标。
- `cogtrack/training/`：ms-swift SFT/GRPO 数据与奖励。
- `cogtrack/models/`：SUTrack 等传统跟踪底座的稳定插件契约。

## 开源与许可

pytracking/SUTrack 派生部分保留 MIT 许可与原作者版权说明。新增 CognitiveTrack 代码同样采用 MIT 许可。
