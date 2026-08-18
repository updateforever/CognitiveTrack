# CognitiveTrack 数据说明

> 更新日期：2026-08-18。本文是 VLT-v6.4 数据设计、生成与监督边界的唯一顶层说明。
> 实际完成度以 [`project_status.md`](project_status.md) 为准，训练参数以
> [`training.md`](training.md) 为准。

## 1. 统一输入与输出

学生模型是 `Qwen3-VL-4B-Instruct`，所有正式 SFT 行使用 Prompt 6.4.0 的三图输入：

1. Image 1：同序列更早的 present reference，红框提供永久身份锚点；
2. Image 2：三个更早可信 present 观测组成的带框历史条带，按时间从左到右排列；
3. Image 3：当前完整搜索图，永远不画框。

历史不足时复制最近观测或 reference 做右侧 padding。条带不含帧号、箭头或文本，panel
之间只有白色分隔带。图片默认长边 648。

模型输出固定为：

```json
{"bbox_2d":[100,120,400,520],"status":"present","memory_update":null}
```

目标缺失时 `bbox_2d` 为 `null`。bbox 是 Image 3 上 `[0,1000]` 归一化 `xyxy`。非空
`memory_update` 是完整、自包含的动态指代表达替换快照，不是追加日志；永久 identity
anchor 不被动态状态覆盖。

## 2. 两类独立数据

| 数据 | 来源 | 主要监督 | memory 监督 |
| --- | --- | --- | --- |
| `tracking_sft` | LaSOT/TNL2K/MGIT tiny train | bbox、presence、JSON 结构 | present/absent 的占位 `null` 都 mask |
| `state_update_sft` | MGIT 官方分段 + 约 1,500 条 Aliyun Qwen3-VL-Plus API 标签 | bbox、presence、JSON 结构 | update/hard-null 全监督 |

两类 release 独立采样、独立渲染、独立审计，不要求 case 一一对应，也不把状态标签直接
回填进原始 `tracking_sft`。最终只在一次冻结 ViT 的 Qwen3-VL-4B 全参 SFT 训练包中混合。

## 2.1 自包含 mixed release

可直接训练和发布的统一目录是：

```text
data/releases/cogtrack_v640_mixed_sft_full_v1
```

它包含两种视图：`train_unique.jsonl` 保留两份源 release 的每一行一次；`train.jsonl`
保留全部 71,199 条 tracking train 行，并确定性重采样为 90% tracking、4% verified update、
6% verified hard-null，共 79,110 行。`val.jsonl` 不做重采样。两份源数据共享视觉 case 的
不同监督行按设计同时保留，不做 sample-id 去重。

统一 split 以 tracking release 为准，共处理 22 个共享序列冲突、移动 214 条 state 行，
最终 train/val sequence overlap 为零。目录内图片路径全部相对 release 根目录，约 21 万
图片、逻辑大小约 11.24GB；根目录 `README.md` 同时包含 10 类真实 case 可视化。

## 3. tracking_sft

### 3.1 正式规模与来源

正式 release 位于：

```text
data/releases/cogtrack_vlt_v640_tracking_sft_r80_20_case20_mgit200_robust15_v1
```

它使用 2,511 个 train 序列，共 66,600 个 unique cases：

- present：53,280；
- absent：13,320；
- case-level 比例：精确 80:20；
- MGIT tiny/train 单序列上限：200；
- LaSOT/TNL2K 默认单序列上限：20。

reference/history 必须严格早于 current。负样本只使用同一序列真实 absent 帧；current
永远不画框。加入 8,433 个 jitter/stale 历史视图后共有 75,033 行，train/val 为
71,199/3,834。train/val 按完整序列切分；监督、taxonomy 和真实 Qwen3 processor replay
均已通过。

### 3.2 27 种合法视觉组合

时间事件：

- `continuous_present`：最近可信历史之后没有 absent；
- `absent`：current 是真实目标缺失帧；
- `reappearance`：可信历史之后经历 absent，current 再次 present。

历史质量：

- `clean`：历史框全部正确；
- `jitter_box`：恰好一个 panel 的框发生平移/缩放扰动；
- `stale_box`：恰好一个 panel 使用另一历史观测的旧框。

历史完整度：

- H0：仅 reference；
- H1：一个动态观测；
- H2：两个动态观测；
- H3：三个动态观测。

合法形式为 clean×H0/H1/H2/H3、jitter×H1/H2/H3、stale×H2/H3，共九种；与三个
时间事件组合后得到 27 种。生成器与 taxonomy preflight 会逐行复算这些条件。

### 3.3 loss 边界

`tracking_sft` 不读取 MGIT 文本，也不调用状态教师。present 与 absent 都写
`memory_update:null`，并只 mask 这个 value；bbox、status、字段名、逗号、JSON 闭合和
EOS 继续监督。因此数据不会强制模型推理时永远输出 null，也不会把未知状态伪装成
hard-null。

正式入口：

```bash
bash scripts/generate_tracking_sft_data.sh \
  cogtrack_vlt_v640_tracking_sft_r80_20_case20_mgit200_robust15_v1
```

中断恢复可复用同名图片资产：

```bash
REUSE_EXISTING_ASSETS=1 bash scripts/generate_tracking_sft_data.sh \
  cogtrack_vlt_v640_tracking_sft_r80_20_case20_mgit200_robust15_v1
```

## 4. state_update_sft

### 4.1 状态规则

只收有证据的两类标签：

- `verified_update`：可靠非空完整快照；
- `verified_hard_null`：明确确认不应更新的 `null`。

核心时序规则：

```text
持续可见、当前描述仍有效 → null
显著且有用的状态变化     → 新的当前指代表达
刚刚消失                 → 明确的消失状态快照
持续缺失                 → null
重新出现                 → 带重现语义的当前指代表达
```

动态 memory 可以描述消失与重现，但永久视觉 identity anchor 始终独立保存。bundle 中的
`initial_identity` 原文保持不变，作为可审计的初始 provenance；它可能是粗粒度或视觉上不
准确的描述。教师允许在后续 `memory_update` 中对其做完整、大幅的视觉纠正，只要红框目标
与 Image 1 是同一物理目标，这不算 identity drift。状态链按序列时间
推进，只有接受的非空 update 才替换 `current_target_state`。

### 4.2 MGIT 官方分段

MGIT tiny/train 官方名单有 105 条，10 条缺帧目录、4 条空目录，实际可用 91 条。正式
release：

```text
data/releases/cogtrack_vlt_v640_state_update_mgit_segments_v1
```

它包含 734 行：350 个 update、384 个 present hard-null，train/val 为 645/89，零
`masked_unknown`。重建入口：

```bash
bash scripts/generate_state_update_sft_data.sh \
  cogtrack_vlt_v640_state_update_mgit_segments_v1
```

### 4.3 约 1,500 条 Aliyun Qwen3-VL-Plus API 标签

LaSOT/TNL2K 重新独立采样固定视觉锚点的状态链，打包三图资产后使用 Aliyun 的
OpenAI-compatible API。本机当前选择 `qwen3-vl-plus`，每个 present case 单次生成
`update/keep/uncertain` 决策；低置信度、uncertain、格式错误、身份漂移和空更新丢弃。

数据集 GT 直接标记消失转折，不浪费 API；持续 absent 保持 null。输入状态为“已消失”
后的 present case 必须产生带重现语义的 update。当前方案是单次强模型生成加确定性质量
门，不得描述成 independent verifier。

本机生成与上传：

```bash
bash scripts/generate_state_update_api_bundle.sh <bundle-release-name>
bash scripts/modelscope_state_update_transfer.sh upload \
  <owner/dataset-repo> data/annotation_bundles/<bundle-release-name> \
  inputs/<bundle-release-name>
```

2026-08-17 的正式 bundle 已上传到：

```text
历史 Qwen3.6 bundle：
repo: updateforever/CognitiveTrack-sft-memory
path: inputs/cogtrack_vlt_v640_state_update_api_qwen36_1500_v1

当前本地 Aliyun bundle：
data/annotation_bundles/cogtrack_vlt_v640_state_update_api_aliyun_qwen3vlplus_1500_v1
```

两版都包含同一批 228 条序列、4,107 个候选和 8,442 张图片，大小约 405 MiB。历史版本
上传的 8,449 个文件全部成功，且 README、manifest、Prompt、sampling plan、SHA256SUMS
与便携标注脚本均已从远端回读并与本地逐字节核对。当前本地版本使用 Prompt 2.1.0，
默认从候选中按序列时间前缀选择 3,000 条：2,264 条 present 调用 API，736 条 absent 由
GT 零成本生成监督。

当前 bundle 内运行（从已经加载 Aliyun 环境变量的 shell 启动）：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export API_MODEL=...
python tools/annotate_state_update_openai_api.py \
  --bundle . --output-dir annotation_result --workers 8 --resume
```

脚本使用 JSONL journal、逐行 `fsync` 和断点续跑。正式报告必须满足
`annotation_policy=single_pass_frontier_api_v1`、`quality_gate_applied=true`、
`dry_run=false`、`minimum_output_reached=true`；默认最低接受 1,200 条。

### 4.4 联合 release

不能直接拼接标签 JSONL。合并器需要重算 sampling plan，检查固定锚点、因果状态链、
重复键和 API 报告：

```bash
bash scripts/build_state_update_sft_release.sh \
  <final-release-name> \
  <mgit-plan> <mgit-labels> \
  <api-plan> <api-labels> <api-report>
```

首版已经完成：`734 + 2,329 = 3,063` 行，位于
`data/releases/cogtrack_vlt_v640_state_update_sft_combined_3063_v1`。其中 2,253 条为
verified update、810 条为 verified hard-null，`masked_unknown=0`。

## 5. 可选 tracking memory overlay

两份主 release 完成后、冻结最终混合配方前，可以选择性补标 `tracking_sft` 中 GT 能
明确证明状态转折的 case，例如消失、重现或少量显著外观变化候选。

补标必须：

- 保存为按 `sample_id` 对齐的 overlay 或派生 release；
- 不覆盖、改名或重写原始 `tracking_sft`；
- 训练时替换对应 masked 行，不能同时采样冲突的 masked/labelled 两个版本；
- 不确定 case 继续 mask；
- 保存教师、Prompt、原始响应、质量门和 provenance。

这一步仍是 `tracking_sft` 的可选监督增强，不是第三类数据，也不与约 1,500 条独立
`state_update_sft` 生成耦合。基础混合训练 smoke 不依赖它。

## 6. 统一预检

每份 release 都先验证 Qwen3 模型族：

```bash
python tracking/validate_qwen_training_view.py \
  --model /root/public/models/Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/releases/<release>/ms_swift/qwen3_vl/train.jsonl \
  --dataset data/releases/<release>/ms_swift/qwen3_vl/val.jsonl \
  --expected-family qwen3_vl
```

再按数据类型验证监督：

```bash
python tracking/validate_sft_supervision.py \
  --profile tracking_sft \
  --dataset data/releases/<tracking-release>/ms_swift/qwen3_vl/train.jsonl \
  --dataset data/releases/<tracking-release>/ms_swift/qwen3_vl/val.jsonl

python tracking/validate_sft_supervision.py \
  --profile state_update_sft \
  --dataset data/releases/<state-release>/ms_swift/qwen3_vl/train.jsonl \
  --dataset data/releases/<state-release>/ms_swift/qwen3_vl/val.jsonl
```

正式 tracking 还必须通过 27 类 taxonomy 审计；真实 processor replay 要确认 `<bbox>`
绑定 Image 3，并解码回原始 norm1000 坐标。数据生成、回放和有限 loss 不能证明精度提升，
只有冻结 CognitiveBench 对照可以建立性能结论。
