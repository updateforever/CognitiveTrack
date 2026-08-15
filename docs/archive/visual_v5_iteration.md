# Visual-v5 迭代与训练服务器执行边界（归档）

> 状态说明：本文保留 visual-v5 的历史协议和已完成证据。当前正式迭代已经切换到
> “初始化文本 + 固定三图 + 最近状态记忆”的 VLT-v6 core SFT，见
> [VLT-v6 core SFT](../vlt_v6_core_sft.md)。不要修改 v5 配置来冒充 v6。

本文记录 2026-08-13 起的新视觉指代基线，以及“本机可行性验证、训练服务器正式合成与
训练”的固定分工。旧 Stage-1/Stage-2 结果仍保留作历史对照，但不得与 visual-v5 结果
混写。

## 1. 已冻结的正式推理协议

- Image 1 是永久首帧身份锚点：完整图像，目标用红框标出；
- 可选 Image 2 是 0–4 条已接受历史观测的时序 mosaic，每个 panel 使用同一红框样式；
- 最后一张图始终是未画框、未裁剪的当前完整搜索图；
- 无历史时直接退化为 anchor + current 两图，不构造空白占位图；
- Prompt 不再包含 reference bbox 数字，输入也不再创建 reference `<bbox>` object；
- 主实验设置 `use_init_language: false`，只评估首帧视觉框指代，不利用部分数据集额外
  提供的自然语言描述；
- Qwen3-VL 输出固定为 `target_status`、`bbox_norm1000_xyxy`、`memory_update` 三字段；
- `memory_update` 的非空提议在线上至少需要两次跨帧相似确认，才能写入长期语义记忆；
  语义确认窗口独立设为 300 个视频帧，以适配稀疏关键帧，不复用视觉历史的 30 帧窗口。

视觉框由 `cogtrack/context/visual.py` 统一渲染：RGB 红色、线宽随图像短边自适应，版本为
`red_box_v1`。训练和在线推理必须复用该实现；改变颜色、线宽或 mosaic 布局时必须提升
marker 版本。

旧坐标文本范式仍可通过 `reference_mode: bbox_text` 显式复现，缺省值不会被代码升级
静默改变。新实验必须使用独立的 `visual_v5` tracker 配置。

## 2. 当前证据等级

已经完成的是工程与单 case 可行性，不是正式 benchmark 提升：

1. 真实 TNL2K train 小样本生成成功：2 条序列、16 条 pair/mosaic 样本；anchor/history
   有框，current 无框；
2. Qwen3-VL 官方 ms-swift processor 回放成功，assistant bbox 只绑定最后一张 current
   图，输入没有 reference bbox object；
3. 本地 Qwen3-VL-4B-Instruct 在 CognitiveBench-Tiny 的真实两图 case 上完成推理，严格
   JSON 解析和 norm1000 坐标转换成功；
4. 基座在首个观测帧主动产生了非空 `memory_update`，新门控正确将其保留为 `1/2`
   待确认，没有立即污染长期记忆；
5. visual-v5 像素、Prompt、导出、parser、memory gate 与 tracker 测试已进入完整测试集。

尚未完成：visual-v5 正式大数据、SFT、CognitiveBench-Tiny 聚合对比、Full 主表与 GRPO。

## 3. 机器分工

本机只负责：

- 协议、Prompt、渲染器、采样器和训练导出代码；
- 1–2 条序列的图片审计、processor 回放和真实模型 smoke；
- 单元测试、CognitiveBench-Tiny 推理与结果分析；
- 提交代码、配置、固定 sampling plan 和不含图片的统计 manifest。

训练服务器负责：

- 读取 LaSOT/TNL2K/MGIT 官方 train split；
- 根据 Git 中的确定性代码或冻结 sampling plan 生成图片与 JSONL；
- 执行小规模过拟合、正式 LoRA SFT 和必要的初始化消融；
- 保存数据 checksum、训练日志、最终 adapter 与环境信息；
- 将 adapter 发布到 ModelScope，Git 中不提交数据、权重和 outputs。

## 4. 本机或新服务器的最小闭环

下面命令只验证协议，`feasibility_null` 会在 metadata 中标记为不可用于正式训练：

```bash
python tracking/synthesize_visual_v5_dataset.py \
  --datasets tnl2k --limit-sequences-per-dataset 2 \
  --env-config configs/env.local.yaml \
  --max-samples-per-sequence 4 --absent-ratio 0.0 \
  --memory-supervision feasibility_null \
  --output-dir data/feasibility/cogtrack_visual_v5_tnl2k

python tools/verify_qwen_grounding_templates.py \
  --dataset-root data/feasibility/cogtrack_visual_v5_tnl2k \
  --qwen3-model /models/Qwen3-VL-4B-Instruct

CUDA_VISIBLE_DEVICES=0 python tracking/smoke_test_qwen.py \
  --tracker-config configs/trackers/qwen3vl_4b_visual_v5_base.yaml \
  --dataset-config configs/datasets/cognitivebench_tiny.yaml \
  --env-config configs/env.local.yaml
```

这里的 `absent-ratio=0` 只是让极小 dry-run 不因选中数据没有消失段而失败。正式 probe
仍使用同序列真实消失帧并保持 7:3。

## 5. 训练服务器的 visual-v5 probe

先只生成采样计划，不编码图片：

```bash
python tracking/synthesize_visual_v5_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --max-samples-per-sequence 20 --absent-ratio 0.3 \
  --output-dir data/plans/cogtrack_visual_v5_probe --plan-only
```

该入口会把 `reference_policy` 固定为 `fixed_identity_anchor`：每条序列的 Image 1 永远是
首个合法初始化锚点，同一个 current 不会再像旧 Stage-1 那样通过更换 reference 重复采样。
重放旧 `sampled_prior_present` plan 会直接报错，防止两种任务定义静默混用。
命令会输出 `sampling_plan.json` 的 SHA-256；跨服务器传递 plan 时必须同时记录并复核。

可用于 SFT probe 的三字段数据需要逐帧标签 manifest。每行格式为：

```json
{"dataset":"tnl2k","sequence":"example","frame_id":120,"memory_update":null,"source":"verified_hard_null_v1","reviewed":true}
```

非空标签必须是稳定、可视觉核验且有助于后续身份判别的外观增量；absent 帧必须为
`null`。`source` 不能为空，重复 `(dataset, sequence, frame_id)` 会被拒绝。得到标签后，
严格重放 plan 并自动导出 Qwen3 ms-swift train/val：

```bash
python tracking/synthesize_visual_v5_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --sampling-plan data/plans/cogtrack_visual_v5_probe/sampling_plan.json \
  --memory-labels data/labels/cogtrack_visual_v5_memory.jsonl \
  --max-samples-per-sequence 20 --absent-ratio 0.3 \
  --history-corruption-ratio 0.15 \
  --output-dir data/releases/cogtrack_visual_v5_probe
```

在占用 GPU 前必须先跑 processor 回放和两步小样本过拟合。正式 LoRA 命令为：

```bash
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct
export DATASET_ROOT=/datasets/cogtrack_visual_v5_probe
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_visual_v5_probe
export QWEN_MODEL_FAMILY=qwen3_vl

bash scripts/train_sft.sh
```

## 6. 下一轮优化顺序

1. 先比较基座、旧 Stage-2 在 visual-v5 输入上的零样本兼容性；旧 adapter 没接受过这种
   输入，只是诊断对照；
2. 用同一 probe 数据分别从基座和旧 Stage-2 初始化 LoRA，其他超参数完全相同；
3. 只在 CognitiveBench-Tiny 上比较 presence、bbox、消失/重现、长 gap 和结构化错误，
   选择是否扩大数据；
4. probe 有明确增益后，再实现论文版确定性 case-bucket planner，固定
   pair/clean-mosaic/corrupted-mosaic/disappearance/memory 的配额；
5. 通过旧模型 rollout 生成一部分预测历史，降低始终使用 GT history 的 exposure bias；
6. 冻结正式统一数据包后跑完整 LoRA，并只把 dataset config 换为 Full 做最终主表；
7. SFT、记忆标签和 reward 回放稳定前不启动 GRPO，也不把 Hybrid 作为论文主线。

probe 的目的只是快速回答“视觉指代统一协议是否值得继续”。其 metadata 会标为
`data_tier=sft_probe`、`paper_full_eligible=false`；它不替代论文版分桶数据，也不能用
训练 loss 代替固定 benchmark 指标。当前 probe 只监督第三字段输出，metadata 中
`uses_semantic_memory_input=false`；论文版数据还必须加入“过去已接受记忆 -> 后续跟踪”的
回放 case，才能评估记忆的因果收益。
