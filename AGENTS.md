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
- 主 baseline 是 `Qwen3-VL-4B-Instruct`；Qwen2.5-VL-3B 只作代际/成本
  对照，Qwen3-VL-32B 只考虑 LoRA 对照。
- 2026-08-13 起下一版正式训练不再按 Stage-1/2/3 顺序做三次 SFT，而是把单参考、
  mosaic、长间隔、消失/重现、干净/扰动历史和记忆样本混合后做一次统一三字段 LoRA
  SFT。旧 Stage-1/2 是已经完成的历史实验和初始化候选，不代表新范式已训练。
- 统一训练监督 `present/absent`、存在时 bbox 和可空 `memory_update`；仍不监督旧六分类、
  身份文字标签、解释文本、细粒度状态或数值置信度。
- 负样本只能来自同一视频中真实的目标消失帧，禁止使用跨序列错配或人工抹除目标。
- 输入图片保持完整、不做目标局部裁剪。下一版中所有过去的身份参考图和可信历史图
  统一直接画框，不再向模型提供 reference bbox 坐标文本；当前待预测的完整搜索图
  永远不画框。“所有图画框”只指过去的参考/历史图，不能给 current 画框造成泄漏。
- 正式在线输入保留永久首帧身份锚点，并显式提供最近可信 VLM 观测及更早历史；稀疏
  推理中的“最近”不是字面上一视频帧。首帧不能被滚动预测替换，动态历史必须门控，
  防止一次误跟踪自我强化。
- 下一版 Qwen3 输出严格是三个字段。例如：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":null}
```

- `memory_update` 同时表示“是否更新”和短语义增量；`null` 表示不更新。非空标签只能
  来自可靠现有标注、视觉/时间过滤和必要审核，普通 bbox 标签不得伪造记忆文本。
- 记忆目标是模型自主学习“发生稳定重大外观变化时才更新”，不能从普通 bbox 标签
  伪造记忆文本。
- `parse_error`、`model_error`、`skipped` 和模型预测的 `absent` 必须严格分离；工程
  失败不得伪装成目标消失。
- 在线推理第一帧只用 GT 初始化并在身份锚点图上画框，后续帧禁止把当前或未来 GT
  传入 tracker。训练 reference/history 只能来自同序列严格更早的 present 帧；GT 只能
  在推理结束后用于评测。
- 训练、验证、测试按完整序列划分，禁止帧级泄漏。

统一数据与候选 Prompt 的完整设计见
[`docs/stage2_stage3_data.md`](docs/stage2_stage3_data.md)。截至 2026-08-13，该设计尚未
实现到 tracker、Prompt 和数据生成代码；不要把规划当成已完成工程事实。

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

## 5. 截至 2026-08-13 的真实进度

### 工程与基线

- pytracking loader、tracker、结果 TXT/JSONL、评测闭环已完成。
- SUTrack-B384 官方 epoch 180 checkpoint 已做严格键匹配和真实前向；迁入实现与原版
  两条语言分支逐帧 bit 级一致。
- 相同预测经原版和本项目指标聚合，15 条序列五项指标最大绝对差为 `0.00e+00`。
- Qwen2.5/Qwen3 本地多图推理、pair/mosaic、二分类严格解析均已接通。
- Qwen3-VL-4B 的零样本与 Stage-1 LoRA 双图推理均已真实跑通。单帧结果只能用于
  工程和错误分析；是否提升必须以固定 CognitiveBench 协议的完整聚合指标判断。
- 真实 Qwen processor 回放已经证明：Qwen2.5 使用 resize 后绝对像素，Qwen3 使用
  norm1000，且多图 assistant bbox 正确绑定当前帧。
- CognitiveBench v1 的 995 序列冻结标注已纳入
  `benchmarks/cognitivebench/v1/`，共 1,408,438 帧和 343,616 个 0-based 关键帧；
  benchmark 不含图像，运行时仍依赖 LaSOT-test、TNL2K-test 和 MGIT-val。

### Stage-1 正式数据

- 来源：LaSOT train + TNL2K train + MGIT tiny/train。
- 当前服务器可用来源为 LaSOT 1120、TNL2K 1300、MGIT tiny/train 95 条；MGIT
  另有 10 条空帧目录被显式排除，不能声称使用完整 105 条 tiny/train。
- 正式首版采用最多 64 个 `(reference,current)` pairs/序列；reference 是同序列中
  严格更早的真实 present 帧，同一真实 current 可与不同 reference 组成不重复 pair。
  present/absent 在每个数据来源内部均约为严格 70:30。
- 固定 sampling plan 包含 2,511 条有效序列、160,049 pairs：present 112,034、
  absent 48,015；LaSOT/TNL2K/MGIT 分别为 71,680/82,545/5,824 cases。
- plan 中 16,213 个 case 复用 current 但使用不同 reference，精确重复 pair 为 0；
  reference/current gap 中位数 161 帧、90% 分位 1,170 帧、最大 23,738 帧。
- `sampling_plan.json` SHA-256：
  `158372c68e82918d9460826d89f601d1278a3f97e6980e2069718755689c03a7`。
- 新版本目录为
  `data/releases/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1`。旧范式导出已完成：
  train 152,039、val 8,010，完整序列划分。`source_samples.jsonl`、Qwen3 train/val
  JSONL SHA-256 分别为
  `e23c61307a132fee5d7ffc83d06bbaf072eda22292491815b21d432191cf9915`、
  `f5db098150e950013dfefdf2db29be76fe65ca31a757e69aeaa607379d544dce`、
  `e4b04c79acd6fffb422ac2fcdd3b1f3321c6707ba1977d8c45dd0089c2454543`。
- 旧 LaSOT+TNL2K 48,400-case v1 从未用于实际训练，其统计和 sampling plan 不再作为
  当前正式数据输入，只保留为历史记录。
- 构造 CLI 已支持 `--sampling-plan` 严格重放，源数据验收工具为
  `tools/verify_stage1_sources.py`。

数据和 outputs 被 `.gitignore` 排除，不会随 Git clone 出现。固定 sampling plan 应从
私有 ModelScope 数据仓库或原服务器取得；如果文档中仍是 `<OWNER/...>` 占位符，先向
用户询问一次实际 ModelScope dataset ID/revision，不要静默重新抽样。

### 已完成的旧范式 Stage-1/Stage-2 LoRA SFT

- 正式训练环境为本服务器 2×L40；基座为 `Qwen3-VL-4B-Instruct`，BF16，LoRA rank
  16、alpha 32、dropout 0.05，语言模型线性层注入，视觉塔和 aligner 冻结。
- Stage-1 旧 pair64 数据共 160,049 cases。最终 checkpoint：
  `outputs/qwen3vl_4b_stage1_lora_pair64_v2_2gpu/v0-20260812-010602/checkpoint-19005`；
  19,005/19,005 steps，4h45m41s，train loss 0.29283377，token accuracy 0.882494。
- Stage-2 旧 mosaic robust v2 数据共 181,969 cases，其中 train 172,915、val 9,054、
  corrupted-history 21,920。`source_samples.jsonl`、Qwen3 train/val JSONL SHA-256 分别为
  `08d1ad9c8591fba5924cfb749d172ee7fc86768f0a0bce93d29d3177058a6055`、
  `758c4e8f04c9dec8df695874ef30f871cf66e9dcc837fa25503df32fb18b6204`、
  `602241dc4c7cbf316b3904f30ba069339ea990c3cb428b983c929a2d99a7d309`。
- Stage-2 从 Stage-1 `checkpoint-19005` 继续训练同一个 adapter，没有叠加第二个 LoRA。
  最终 checkpoint：
  `outputs/qwen3vl_4b_stage2_lora_mosaic_robust_v2_2gpu/v0-20260812-100910/checkpoint-28819`；
  28,819/28,819 steps，7h15m14s，train loss 0.25926472，token accuracy 0.89407277，
  峰值显存 38.13GiB/卡。
- Stage-1 和 Stage-2 adapter 已发布到私有 ModelScope 模型仓库
  `updateforever/CognitiveTrack-Qwen3VL-4B-Stage1-LoRA`。Stage-1 位于仓库根目录；
  Stage-2 位于 `stage2-mosaic-robust-v2/`，权重 SHA-256 为
  `7437dd2be3bae21070059ef5ce704da7bbd5008607f7e04eb18b3638f04930a1`，已做远端
  回下载校验。
- 上述两次训练使用旧范式：参考图框通过坐标文本/grounding 对象传入，Stage-1/2 输出
  是二字段。它们只证明训练完成，尚未通过正式 CognitiveBench 证明指标提升，也不等于
  新的“历史图视觉画框 + 一次统一三字段混合训练”已经实现或训练。

### 当前代码与新规划之间的差距

- 当前 `TrackingContextBuilder` 的首帧仍不画框，并在 Prompt 中传 reference bbox 坐标；
  只有 history mosaic panel 会画框。
- 当前正式 Qwen3 tracker 配置仍为 pair、二字段并关闭 memory；三字段 parser、语义记忆
  回灌和门控代码虽已存在，但尚未冻结为新范式配置。
- 新规划需要同步修改 context builder、pair/mosaic Prompt、训练导出、tracker 配置与测试，
  之后重新生成统一混合数据并训练。

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

测试数量随 commit 增长，以当前 commit 的完整测试集合为准，不能忽略失败项。

### 6.3 配置机器路径

```bash
cp configs/env.example.yaml configs/env.local.yaml
```

编辑 `env.local.yaml`，至少设置 `project_root`、`model_root`、`output_root`、LaSOT、
TNL2K 和 MGIT 路径。该文件禁止提交。

### 6.4 准备数据

首选使用服务器已有的 LaSOT/TNL2K/MGIT tiny 和固定 plan 重建：

```bash
python tools/verify_stage1_sources.py \
  --lasot-root /datasets/raw/LaSOT \
  --tnl2k-root /datasets/raw/TNL2K

python tracking/synthesize_stage1_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny \
  --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --context-mode pair \
  --frame-stride 1 \
  --max-samples-per-sequence 64 \
  --absent-ratio 0.3 \
  --max-image-side 648 \
  --jpeg-quality 95 \
  --val-ratio 0.05 \
  --seed 20260809 \
  --sampling-plan /path/to/sampling_plan.json \
  --qwen-model-families qwen2_5_vl qwen3_vl \
  --output-dir /datasets/derived/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1
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
export DATASET_ROOT=/datasets/derived/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1

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

1. 先讨论并冻结 `docs/stage2_stage3_data.md` 的视觉画框输入范式与 Prompt；明确框样式、
   两图退化和最近可信观测的门控语义。
2. 同步修改 context builder、Prompt、tracker 配置和训练导出，使训练/推理使用同一绘框
   实现；当前搜索图必须保持无框。
3. 构建小规模混合数据，完成 processor 回放、两步 smoke 和小样本过拟合；验证三字段
   JSON、最后一张图 bbox 绑定和没有 current/future 泄漏。
4. 生成正式统一混合数据，统计各数据桶、历史框来源和 memory 标签来源，随后只做一次
   Qwen3-VL-4B LoRA SFT。
5. 同时评测基座、旧 Stage-2 adapter、从基座统一训练、从 Stage-2 继续统一训练，按
   CognitiveBench 的 presence、bbox、消失/重现、长 gap 和 memory 指标选择主模型。
6. 只有 SFT 与 memory reward 回放可靠后再做 GRPO；先纯 VLM，不转向 hybrid 主实验。

## 9. 开发与提交约束

- 根目录 `README.md` 是公开项目主页，只保留项目定位、安装、数据、模型、标准推理、
  评测和训练入口。AI 接手提示、内部服务器路径、未整理的实验流水账和 prompt 探针
  结论放在本文件或 `docs/`，不要重新写回公开 README。
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
