# ms-swift 训练指引

## 1. 当前监督边界与统一训练

2026-08-13 起，下一版正式训练改为一次统一混合 LoRA SFT，不再依次做三次能力阶段：

- 混合单参考、mosaic、长间隔、消失/重现、干净/扰动历史和记忆事件；
- 所有过去的参考/历史图直接视觉画框，不再在 Prompt 中传 reference bbox 坐标；
- 当前完整搜索图始终无框；
- 所有样本统一输出三字段，普通样本的 `memory_update` 为 `null`，非空文本只来自可靠
  标签与审核；
- 不监督旧六分类、解释文本、身份标签、细粒度状态或数值置信度。

完整输入范式、候选 Prompt、混合比例、MGIT 使用边界和消融见
[stage2_stage3_data.md](stage2_stage3_data.md)。该方案目前是待实现规划，现有数据生成器
仍使用旧 reference 坐标文本范式。

统一训练使用 v4 三字段样本：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":"Rear view reveals two stable white stripes."}
```

上例是 Qwen3-VL；Qwen2.5-VL 使用 `bbox_pixel_xyxy` 和 processor-resize 后绝对
像素值。

同一正式训练批次不混用二字段和三字段 Prompt。旧二字段 Stage-1/2 数据只能作为历史
基线；若用于新混合训练，必须重新渲染参考图并补充经过规则/审核的第三字段，而不能
机械给所有旧样本追加 `null`。导出校验器必须严格拒绝缺字段、额外字段、空字符串以及
`absent + 非空 memory_update`。

## 2. 已完成的旧 Stage-1 数据（历史基线）

以下入口记录已经完成的旧二字段实验，不能直接生成新统一画框数据。它固定读取三个
数据集的官方 `train` split，并在生成后再次审计每条样本的
来源。它会同时产出图片、源 manifest、校验报告，以及 Qwen2.5-VL/Qwen3-VL 各自
可直接交给 ms-swift 的 `train.jsonl` / `val.jsonl`：

```bash
python tracking/synthesize_stage1_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --context-mode pair --max-samples-per-sequence 64 \
  --absent-ratio 0.3 \
  --output-dir data/releases/cogtrack_stage1_lasot_tnl2k_mgit_tiny_pair64_v1
```

构造器先扫描同序列连续状态区间，再在全数据集层面分配 absent 配额。负样本只来自
LaSOT 的 `full_occlusion/out_of_view`、TNL2K 的零框帧和 MGIT 的 `absent` 属性；
不会跨序列配对或人工抹除目标。absent 区间优先覆盖首尾，present 则优先覆盖消失前
和重现后的边界帧，再做时间均匀采样。默认将图片长边限制为 648。

旧范式初始化遵循 pytracking 的 ``initialize(full_image, init_bbox)`` 接口。在线 benchmark
仍只用第一帧 GT 初始化；Stage-1 训练 pair 则从同序列选择严格早于当前帧的真实
present reference。传给 VLM 的 Image 1 是未画框、未裁剪的完整 reference 帧，Image 2
始终是未画 GT 的当前完整帧。reference 框通过 Prompt 中的 `<bbox>` 和
`objects.image_id=0` 绑定 Image 1，当前 GT 仅在 assistant 答案中通过另一枚 `<bbox>`
绑定最后一张图。reference/current 的时间顺序写入 sampling plan，并严格禁止
reference 使用 current 或未来帧。下一版会改为在完整 reference/history 图上直接画框，
current 仍无框，输出 bbox 仍绑定最后一张图。

## 3. Qwen 官方坐标训练视图

Qwen2.5-VL 与 Qwen3-VL 不是同一套 grounding 坐标：

| 模型族 | JSONL 目录 | 模型实际看到的坐标 | 输出字段 |
| --- | --- | --- | --- |
| Qwen2.5-VL | `ms_swift/qwen2_5_vl/` | processor resize 后绝对像素 `xyxy` | `bbox_pixel_xyxy` |
| Qwen3-VL | `ms_swift/qwen3_vl/` | `[0,1000]` 相对 `xyxy` | `bbox_norm1000_xyxy` |

两套 JSONL 都不预先复刻 `smart_resize`，而是保存导出 JPEG 上的真实框：

```json
{
  "messages": [
    {"role": "user", "content": "<image><image>... Initialization target bbox: <bbox> ..."},
    {"role": "assistant", "content": "{\"target_status\":\"present\",\"bbox_pixel_xyxy\":<bbox>}"}
  ],
  "objects": {
    "bbox": [[308.8,172.4,377.7,186.6], [309.3,172.9,378.2,187.1]],
    "bbox_type": "real",
    "image_id": [0,1]
  }
}
```

ms-swift 根据实际模型模板处理 `<bbox>`；训练必须保留
`QWENVL_BBOX_FORMAT=new`。同一 canonical 数据只复用图片和 split，不能把某一代
JSONL 交给另一代模型。`scripts/train_sft.sh` 会在加载模型前读取 Hugging Face
`config.json:model_type` 和数据元数据，发现交叉使用就直接退出。

在新服务器上开始训练前，可用真实 ms-swift processor 对任意一条 present 样本做
坐标回放（不加载模型权重）：

```bash
python tools/verify_qwen_grounding_templates.py \
  --dataset-root /path/to/dataset \
  --qwen25-model /path/to/Qwen2.5-VL \
  --qwen3-model /path/to/Qwen3-VL
```

先做小规模管线检查：

```bash
python tracking/synthesize_stage1_dataset.py \
  --datasets tnl2k --limit-sequences-per-dataset 50 \
  --max-samples-per-sequence 4 --absent-ratio 0.3 --context-mode pair \
  --output-dir data/stage1_debug
```

LaSOT 使用 `training_set.txt`；TNL2K 使用 `TNL2K_train_subset`；MGIT 使用官方
`videocube.json[full][train]` 和 `data/train`。任何训练帧未解压或 split 不匹配都会
立即失败，不会回退到测试集。

## 4. 通用源样本构造器

`tracking/build_tracking_dataset.py` 直接消费 pytracking `Sequence`，支持 pair/mosaic、稳定抽样、
关键帧和流式 JSONL。输出图片全部使用相对路径，可连同数据集整体移动。

```bash
python tracking/build_tracking_dataset.py \
  --dataset cognitivebench --env-config configs/env.local.yaml \
  --output-dir data/cognitive_pair --mode pair \
  --frame-stride 10 --max-samples-per-sequence 64
```

Mosaic 历史只能使用当前帧之前的有效正帧；无历史时与在线 tracker 一样自动退化为 pair。

旧 identity 困难负样本构造器暂时隔离，其输出不能通过 v4 主协议校验，也不能
与当前 presence 训练集混合。

## 5. 校验与序列划分

```bash
python tracking/export_qwen_grounding_dataset.py \
  --input data/cognitive_pair/source_samples.jsonl \
  --dataset-root data/cognitive_pair \
  --output-dir data/cognitive_pair/ms_swift \
  --model-families qwen2_5_vl qwen3_vl
```

划分单位是完整序列，不是帧。导出器会检查图片、`<image>`/`<bbox>` 数量、
`objects.bbox`、`image_id`、版本化二/三字段 JSON 和二分类状态；默认任何非法样本
都会阻止导出。

GRPO 模式会删除 assistant 消息，并把参考答案移入 `solution`，避免标签泄漏。

## 6. SFT 与 GRPO

```bash
MODEL_PATH=/path/to/Qwen2.5-VL \
TRAIN_DATA=/path/to/dataset/ms_swift/qwen2_5_vl/train.jsonl \
VAL_DATA=/path/to/dataset/ms_swift/qwen2_5_vl/val.jsonl \
DATASET_ROOT=/path/to/dataset \
bash scripts/train_sft.sh
```

旧 baseline 快捷入口为 `scripts/train_qwen3vl_4b_stage1.sh`，默认采用 LoRA。已在 2×L40
完成旧 Stage-1 pair64 与 Stage-2 mosaic robust v2 正式训练；Stage-2 从 Stage-1 最终
checkpoint 继续同一个 adapter，没有叠加 LoRA。最终结果为：

- Stage-1：19,005 steps，4h45m41s，train loss 0.29283377，token accuracy 0.882494；
- Stage-2：28,819 steps，7h15m14s，train loss 0.25926472，token accuracy 0.89407277，
  峰值 38.13GiB/卡。

二者都尚未通过正式 benchmark 证明跟踪提升。下一版改为一次统一三字段混合 LoRA SFT，
不再新建 Stage-1/2/3 三轮 adapter；从基座或旧 Stage-2 adapter 初始化由验证集消融决定。
需要全参对照时必须显式设置 `TUNER_TYPE=full`。

正式 Stage-1 LoRA 已完成一轮训练：152,039 条 train、8,010 条按序列隔离的 val，
19,005 steps，global world size 2、单卡 batch 4、global batch 8、学习率 5e-5。模型共
约 4.471B 参数，其中 33.0301M（0.7388%）可训练；最终 train loss 为 0.29283377，
token accuracy 为 0.882494。该次运行设置了 `eval_strategy=no`，因此没有 validation
loss，不能仅根据训练曲线宣称泛化提升。

发布的 adapter 为 ModelScope
`updateforever/CognitiveTrack-Qwen3VL-4B-Stage1-LoRA`，核心权重 SHA-256：

```text
732ff15f4791f75c1ca16b2a72163fe59ff8a8059e87e765caf22382ddd07131
```

训练效果必须在冻结 CognitiveBench 全集上比较零样本与 LoRA 的相同观察策略，并列
报告 hold-last、observation-only 和 presence 指标。单帧 case 只用于定位错误原因。

此前的全参参考结果为：8×4090、单卡
batch=2、全局 batch=16 的两步实测中，模型共 4.438B 参数、可训练 4.132B
（93.10%），峰值 16.29GiB/卡，吞吐约 4.64 samples/s。视觉主干保持冻结，Qwen3
新增的主 merger 与三个 deepstack merger 均参与训练。

代际对照入口为 `scripts/train_qwen25vl_3b_stage1.sh`。相同机器设置下，Qwen2.5
单卡 batch=4、全局 batch=32，实测峰值 18.63GiB/卡、吞吐约 9.75 samples/s；
完整 45,980 条训练集预计约 1.3 小时，另加验证与保存时间。

Qwen3-VL 训练时必须同时替换模型路径和 `qwen3_vl` 数据目录。可选
`QWEN_MODEL_FAMILY=qwen2_5_vl|qwen3_vl` 会再增加一层显式断言。

8×24GB GPU 上做 3B/4B 全参数微调时必须分片参数、梯度和优化器状态。可使用
PyTorch FSDP（无需额外安装）或 DeepSpeed ZeRO-3：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
TUNER_TYPE=full FSDP_MODE=fsdp2 \
FREEZE_LLM=false FREEZE_VIT=true FREEZE_ALIGNER=false \
LEARNING_RATE=1e-5 EPOCHS=1 BATCH_SIZE=1 GRAD_ACC_STEPS=4 \
MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
TRAIN_DATA=/path/to/dataset/ms_swift/qwen2_5_vl/train.jsonl \
VAL_DATA=/path/to/dataset/ms_swift/qwen2_5_vl/val.jsonl \
DATASET_ROOT=/path/to/dataset \
bash scripts/train_sft.sh
```

这里的“常规全参 SFT”指语言模型和对齐层全参更新、视觉编码器冻结。这样优先保留
基座已有的视觉/grounding 表征，只让语言侧学习跨帧判别和结构化输出。只有专门研究
视觉表征适配时才设置 `FREEZE_VIT=false`，并在实验名中明确标记为全模型微调。

GRPO 基础设施提供严格格式、存在性、bbox IoU 和内部一致性四个独立 reward，但
旧 Stage-1 包首先用于二字段 SFT。Qwen3-VL 的归一化 bbox reward 可复用 canonical
监督；Qwen2.5-VL 的 bbox reward 必须拿到该次 rollout processor 的真实缩放尺寸后再
计算，不能用原图框或 norm1000 框冒充。完成这项 processor-aware reward 回放前，
不把 Qwen2.5-VL bbox GRPO 列为已验证训练入口。

统一 SFT 仍不监督身份、细粒度状态、解释文本或数值置信度，但会在同一三字段协议中
监督稀疏语义记忆。只有经过审核或可靠规则确认的样本才能提供非空记忆；在加入专门的
更新时机和文本质量 reward 之前，不把现有 presence GRPO 误称为记忆训练。
奖励仅读取 `solution`/监督列，不读取跟踪运行时的未来信息。
