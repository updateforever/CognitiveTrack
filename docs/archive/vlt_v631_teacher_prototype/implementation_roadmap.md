# CognitiveTrack 工程实现路线图（归档草案）

> **归档说明（2026-08-15）：** 本文混合了已完成实现、伪代码和早期计划，不再作为
> 执行清单。当前进度见 [`../../implementation_progress.md`](../../implementation_progress.md)。

> 目标：聚焦 presence-aware tracking，推进数据生成、SFT、GRPO 和评测
> 更新日期：2026-08-14

---

## 一、数据生成框架（Core SFT）

### 1.1 当前状态 ✅

**已有实现**：
- `tracking/synthesize_stage1_dataset.py`：通用入口
- `tracking/synthesize_vlt_v6_dataset.py`：VLT-v6.3 wrapper
- `cogtrack/training/tracking_samples.py`：核心构建引擎
- 支持 profile：`legacy_stage1`, `visual_v5`, `vlt_v6`

**VLT-v6.3 关键配置**：
```python
profile = "vlt_v6"
context_mode = "mosaic"  # 固定三图
reference_mode = "visual_box"  # 视觉画框
memory_supervision = "masked_null"  # 只屏蔽 memory_update 值
prompt_profile = "vlt_v6"
force_history_image = True  # 不足三帧时复制补齐
history_size = 3  # 固定三帧历史
```

### 1.2 执行命令（Step-by-Step）

#### **Step 1: 生成 Sampling Plan（可重放）**

```bash
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny \
  --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --max-samples-per-sequence 20 \
  --absent-ratio 0.3 \
  --output-dir data/plans/cogtrack_vlt_v63_core \
  --plan-only
```

**产出**：
- `data/plans/cogtrack_vlt_v63_core/sampling_plan.json`
- 包含每个序列的采样帧 ID、present/absent 分布
- 可重放，保证跨服务器一致性

**验证**：
```bash
# 检查 plan 统计
python -c "
import json
plan = json.load(open('data/plans/cogtrack_vlt_v63_core/sampling_plan.json'))
print(f'Total sequences: {len(plan[\"sequences\"])}')
print(f'Total cases: {plan[\"summary\"][\"total_cases\"]}')
print(f'Present: {plan[\"summary\"][\"present_cases\"]}')
print(f'Absent: {plan[\"summary\"][\"absent_cases\"]}')
print(f'Absent ratio: {plan[\"summary\"][\"absent_cases\"] / plan[\"summary\"][\"total_cases\"]:.3f}')
"
```

**Expected**：
- Total sequences: ~1500-2000（LaSOT + TNL2K + MGIT tiny train）
- Absent ratio: ~0.30
- 每序列 ≤20 cases

---

#### **Step 2: 渲染完整数据（重放 plan）**

```bash
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny \
  --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --sampling-plan data/plans/cogtrack_vlt_v63_core/sampling_plan.json \
  --max-samples-per-sequence 20 \
  --absent-ratio 0.3 \
  --output-dir data/releases/cogtrack_vlt_v63_lasot_tnl2k_mgit_core
```

**产出**：
```
data/releases/cogtrack_vlt_v63_lasot_tnl2k_mgit_core/
├── manifest.json                    # 版本、配置、校验和
├── source.jsonl                     # 源标注（带图片路径）
├── train_val_split.json             # 序列级划分
├── ms_swift/
│   └── qwen3_vl/
│       ├── train.jsonl              # ms-swift 格式
│       └── val.jsonl
└── images/
    ├── <sequence>/
    │   ├── reference_<frame>.jpg
    │   ├── history_<hash>.jpg
    │   └── current_<frame>.jpg
```

**时间估计**：
- Plan only: ~5-10 分钟
- 完整渲染: ~2-4 小时（取决于 I/O 和 JPEG 编码）

---

#### **Step 3: 训练前验证**

```bash
export DATASET_ROOT=/path/to/data/releases/cogtrack_vlt_v63_lasot_tnl2k_mgit_core
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"

# 验证监督档位
python tracking/validate_sft_supervision.py \
  --profile tracking_core \
  --dataset "$TRAIN_DATA" \
  --dataset "$VAL_DATA"

# Qwen processor 回放（验证 bbox 和 mask 正确）
python tools/verify_qwen_grounding_templates.py \
  --dataset-root "$DATASET_ROOT" \
  --qwen3-model /models/Qwen3-VL-4B-Instruct \
  --verify-tracking-core-mask
```

**预期输出**：
```
✓ All records have valid supervision profile
✓ tracking_core: 'memory_update' value masked, other fields supervised
✓ Bbox coordinates within [0,1000]
✓ No future information leakage
✓ Processor replay: bbox tokens supervised, masked_text exactly 'null'
```

**如果验证失败**：
- 检查 `manifest.json` 中的 `memory_supervision` 是否为 `masked_null`
- 检查 `prompt_version` 是否为 `6.3.0`
- 重新生成数据

---

### 1.3 数据质量检查清单

**必须通过的检查**：

1. **Present/Absent 比例**：
   ```python
   # 应该接近 7:3
   absent_ratio = absent_cases / total_cases
   assert 0.25 <= absent_ratio <= 0.35
   ```

2. **Reference 严格早于 Current**：
   ```python
   for record in train_data:
       ref_frame = record['metadata']['reference_frame_id']
       cur_frame = record['metadata']['current_frame_id']
       assert ref_frame < cur_frame
   ```

3. **History 严格早于 Current**：
   ```python
   for record in train_data:
       if 'history_frame_ids' in record['metadata']:
           history = record['metadata']['history_frame_ids']
           current = record['metadata']['current_frame_id']
           assert all(h < current for h in history)
   ```

4. **三字段结构完整**：
   ```python
   for record in train_data:
       assert 'target_status' in record['response']
       assert 'bbox_norm1000_xyxy' in record['response']
       assert 'memory_update' in record['response']
   ```

5. **Loss mask 正确**：
   ```python
   # tracking_core: memory_update 值被 mask
   # 验证脚本会自动检查
   ```

---

## 二、Core SFT 训练

### 2.1 训练配置（已有）

**脚本位置**：
- `scripts/train_qwen3vl_4b_vlt_v6_core.sh`
- 或通用入口 `scripts/train_sft.sh`

**关键参数**：
```bash
export MODEL_PATH=/models/Qwen3-VL-4B-Instruct
export DATASET_ROOT=/path/to/cogtrack_vlt_v63_lasot_tnl2k_mgit_core
export TRAIN_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"
export OUTPUT_DIR=/outputs/cogtrack/qwen3vl_4b_vlt_v63_core_sft
export QWEN_MODEL_FAMILY=qwen3_vl
export SFT_SUPERVISION_PROFILE=tracking_core

# 训练超参数
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=1e-4
BATCH_SIZE=8  # 实际 = per_device * gradient_accumulation
LORA_RANK=64
LORA_ALPHA=128
```

### 2.2 执行命令

```bash
bash scripts/train_qwen3vl_4b_vlt_v6_core.sh
```

**或直接调用 ms-swift**：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
  --model_type qwen3-vl-4b-instruct \
  --model_id_or_path "$MODEL_PATH" \
  --dataset "$TRAIN_DATA" \
  --val_dataset "$VAL_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --sft_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 10 \
  --dataloader_num_workers 4 \
  --max_length 4096 \
  --lora_target_modules ALL \
  --gradient_checkpointing true \
  --save_total_limit 3
```

### 2.3 训练监控

**必须记录的指标**：
```python
# 每 epoch 记录
{
  "epoch": 1,
  "train_loss": 0.45,
  "train_loss_presence": 0.12,  # target_status loss
  "train_loss_bbox": 0.28,       # bbox loss
  "train_loss_structure": 0.05,  # 字段结构 loss
  "val_loss": 0.52,
  "val_presence_accuracy": 0.89,
  "val_bbox_iou": 0.71,
  "format_error_rate": 0.03,     # JSON 解析失败率
  "learning_rate": 1e-4
}
```

**Early stopping 条件**：
```python
if val_loss 不降 for 3 epochs:
    stop_training()
```

**Over-fitting 检查**：
```python
if (train_loss 持续下降) and (val_loss 开始上升):
    warning("可能过拟合，考虑 early stop")
```

### 2.4 Checkpoint 验证

**训练完成后立即验证**：

```bash
# 在 Tiny 上快速评测
python tracking/test.py \
  --config configs/env.local.yaml \
  --tracker-config configs/trackers/qwen3vl_4b_vlt_v6_core_sft_vllm.yaml \
  --dataset-config configs/datasets/cognitivebench_tiny.yaml \
  --output-name core_sft_smoke
```

**预期指标（Tiny，24 序列）**：
```
Base (zero-shot):
  AUC (hold-last): ~40-50
  Presence F1: ~65-70
  
Core SFT (expected):
  AUC (hold-last): ~58-65
  Presence F1: ~80-85
  Absent FPR: <10%
```

**如果指标异常低**：
- 检查 checkpoint 是否加载成功
- 检查 `semantic_enabled: false`（Core 不能写入记忆）
- 检查训练 loss 是否收敛
- 人工抽查 10-20 个预测，看格式和逻辑

---

## 三、Presence-Aware 评测体系

### 3.1 核心指标定义

#### **3.1.1 Presence Precision/Recall/F1**

**定义**：模型对"目标在不在"的判别准确性

```python
def compute_presence_metrics(predictions, ground_truth):
    """
    predictions: List[str]  # ["present", "absent", "present", ...]
    ground_truth: List[str] # ["present", "absent", "present", ...]
    """
    tp = sum(1 for p, g in zip(predictions, ground_truth) 
             if p == "present" and g == "present")
    fp = sum(1 for p, g in zip(predictions, ground_truth) 
             if p == "present" and g == "absent")
    fn = sum(1 for p, g in zip(predictions, ground_truth) 
             if p == "absent" and g == "present")
    tn = sum(1 for p, g in zip(predictions, ground_truth) 
             if p == "absent" and g == "absent")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }
```

**解释**：
- **Precision**: 模型说 present，实际也 present 的比例
- **Recall**: 实际 present，模型正确判断的比例
- **F1**: 综合指标

---

#### **3.1.2 Absent False-Positive Rate (FPR)**

**定义**：目标真实不在，但模型误判为 present 的比例

```python
def compute_absent_fpr(predictions, ground_truth):
    """
    FPR = FP / (FP + TN)
    即：在所有真实 absent 的帧中，模型错误预测为 present 的比例
    """
    absent_frames = [i for i, g in enumerate(ground_truth) if g == "absent"]
    if len(absent_frames) == 0:
        return 0.0
    
    false_positives = sum(1 for i in absent_frames if predictions[i] == "present")
    fpr = false_positives / len(absent_frames)
    return fpr
```

**关键**：
- 传统 tracker 的 FPR = **100%**（永远预测 present）
- 这是 presence-aware tracking 与传统方法的**根本差异**

---

#### **3.1.3 Present Miss Rate**

**定义**：目标真实存在，但模型误判为 absent 的比例

```python
def compute_present_miss_rate(predictions, ground_truth):
    """
    Miss Rate = FN / (TP + FN)
    即：在所有真实 present 的帧中，模型错误预测为 absent 的比例
    """
    present_frames = [i for i, g in enumerate(ground_truth) if g == "present"]
    if len(present_frames) == 0:
        return 0.0
    
    false_negatives = sum(1 for i in present_frames if predictions[i] == "absent")
    miss_rate = false_negatives / len(present_frames)
    return miss_rate
```

**Trade-off**：
- 降低 FPR（少误报） vs 降低 Miss Rate（少漏报）
- 理想模型：FPR 和 Miss Rate 都低

---

#### **3.1.4 Re-Identification After Occlusion**

**定义**：目标消失后重现，模型能否正确判别为 present

**实现步骤**：

**Step 1: 自动挖掘消失-重现配对**

```python
def mine_reappearance_pairs(sequence):
    """
    自动从 GT presence/absence 挖掘消失-重现事件
    """
    pairs = []
    status = sequence['target_status']  # List[str]: ["present", "absent", ...]
    
    i = 0
    while i < len(status):
        # 找到消失开始
        if status[i] == "absent":
            disappear_start = i
            disappear_end = i
            
            # 找到消失段结束
            while disappear_end < len(status) and status[disappear_end] == "absent":
                disappear_end += 1
            
            # 如果有重现
            if disappear_end < len(status) and status[disappear_end] == "present":
                occlusion_length = disappear_end - disappear_start
                last_before = disappear_start - 1
                first_after = disappear_end
                
                pairs.append({
                    "sequence": sequence['name'],
                    "last_before_frame": last_before,
                    "first_after_frame": first_after,
                    "occlusion_length": occlusion_length,
                    "disappear_start": disappear_start,
                    "disappear_end": disappear_end
                })
            
            i = disappear_end
        else:
            i += 1
    
    return pairs
```

**Step 2: 计算 Re-ID Rate**

```python
def compute_reid_rate(pairs, predictions, ground_truth, bboxes_pred, bboxes_gt):
    """
    对每个重现事件，检查模型是否正确判别为 present 且 IoU > 0.5
    """
    reid_success = 0
    
    for pair in pairs:
        first_after = pair['first_after_frame']
        
        # 模型预测
        pred_status = predictions[first_after]
        
        if pred_status == "present":
            # 计算 bbox IoU
            pred_bbox = bboxes_pred[first_after]
            gt_bbox = bboxes_gt[first_after]
            
            if pred_bbox is not None and gt_bbox is not None:
                iou = compute_iou(pred_bbox, gt_bbox)
                if iou > 0.5:
                    reid_success += 1
    
    reid_rate = reid_success / len(pairs) if len(pairs) > 0 else 0.0
    return reid_rate, reid_success, len(pairs)
```

**Step 3: 按遮挡长度分组报告**

```python
def compute_reid_by_occlusion_length(pairs, predictions, ground_truth, bboxes_pred, bboxes_gt):
    """
    分别报告短期、中期、长期遮挡后的 re-ID 率
    """
    bins = {
        "short (<20f)": [],
        "medium (20-50f)": [],
        "long (>50f)": []
    }
    
    for pair in pairs:
        length = pair['occlusion_length']
        if length < 20:
            bins["short (<20f)"].append(pair)
        elif length <= 50:
            bins["medium (20-50f)"].append(pair)
        else:
            bins["long (>50f)"].append(pair)
    
    results = {}
    for bin_name, bin_pairs in bins.items():
        if len(bin_pairs) > 0:
            rate, success, total = compute_reid_rate(
                bin_pairs, predictions, ground_truth, bboxes_pred, bboxes_gt
            )
            results[bin_name] = {
                "rate": rate,
                "success": success,
                "total": total
            }
    
    return results
```

---

### 3.2 评测代码实现位置

**建议文件结构**：

```
cogtrack/evaluation/
├── __init__.py
├── evaluator.py                    # 已有：主评测流程
├── metrics.py                      # 已有：基础指标
├── presence_metrics.py             # 新增：presence-aware 指标
└── report.py                       # 已有：报告生成
```

**新增文件**：`cogtrack/evaluation/presence_metrics.py`

```python
"""Presence-aware tracking 专用指标。"""

from __future__ import annotations
from typing import Any


def compute_presence_classification_metrics(
    predictions: list[str],
    ground_truth: list[str]
) -> dict[str, Any]:
    """计算 presence 判别的 precision/recall/F1。"""
    # 实现见 3.1.1
    pass


def compute_absent_false_positive_rate(
    predictions: list[str],
    ground_truth: list[str]
) -> float:
    """计算 absent FPR（传统 tracker = 100%）。"""
    # 实现见 3.1.2
    pass


def compute_present_miss_rate(
    predictions: list[str],
    ground_truth: list[str]
) -> float:
    """计算 present miss rate。"""
    # 实现见 3.1.3
    pass


def mine_reappearance_pairs(
    sequence_name: str,
    target_status: list[str],
    frame_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """自动挖掘消失-重现事件配对。"""
    # 实现见 3.1.4 Step 1
    pass


def compute_reidentification_rate(
    pairs: list[dict[str, Any]],
    predictions: list[str],
    bboxes_pred: list[list[float] | None],
    bboxes_gt: list[list[float] | None],
    iou_threshold: float = 0.5
) -> tuple[float, int, int]:
    """计算 re-ID 成功率。
    
    Returns:
        (rate, success_count, total_pairs)
    """
    # 实现见 3.1.4 Step 2
    pass


def compute_reidentification_by_occlusion_length(
    pairs: list[dict[str, Any]],
    predictions: list[str],
    bboxes_pred: list[list[float] | None],
    bboxes_gt: list[list[float] | None],
    iou_threshold: float = 0.5
) -> dict[str, dict[str, Any]]:
    """按遮挡长度分组计算 re-ID 率。
    
    Returns:
        {
            "short (<20f)": {"rate": 0.85, "success": 42, "total": 50},
            "medium (20-50f)": {"rate": 0.68, "success": 28, "total": 41},
            "long (>50f)": {"rate": 0.52, "success": 15, "total": 29}
        }
    """
    # 实现见 3.1.4 Step 3
    pass
```

---

### 3.3 集成到评测流程

**修改**：`cogtrack/evaluation/evaluator.py`

```python
# 在现有 evaluate_frames() 中添加

from .presence_metrics import (
    compute_presence_classification_metrics,
    compute_absent_false_positive_rate,
    compute_present_miss_rate,
    mine_reappearance_pairs,
    compute_reidentification_by_occlusion_length,
)


def evaluate_frames(frames: list[CanonicalFrame], ...) -> dict[str, Any]:
    """已有函数，添加 presence-aware 指标。"""
    
    # ... 现有代码 ...
    
    # 提取 predictions 和 ground_truth
    predictions = [f.target_status for f in frames]
    ground_truth = [f.gt_status for f in frames]
    bboxes_pred = [f.bbox_xyxy if f.target_status == "present" else None for f in frames]
    bboxes_gt = [f.gt_bbox_xyxy if f.gt_status == "present" else None for f in frames]
    
    # 新增：Presence 分类指标
    presence_metrics = compute_presence_classification_metrics(predictions, ground_truth)
    
    # 新增：Absent FPR
    absent_fpr = compute_absent_false_positive_rate(predictions, ground_truth)
    
    # 新增：Present Miss Rate
    present_miss = compute_present_miss_rate(predictions, ground_truth)
    
    # 新增：Re-ID 指标
    pairs = mine_reappearance_pairs(
        sequence_name=frames[0].sequence,
        target_status=ground_truth
    )
    reid_metrics = compute_reidentification_by_occlusion_length(
        pairs, predictions, bboxes_pred, bboxes_gt
    )
    
    return {
        # ... 现有指标 ...
        "presence_precision": presence_metrics["precision"],
        "presence_recall": presence_metrics["recall"],
        "presence_f1": presence_metrics["f1"],
        "absent_fpr": absent_fpr,
        "present_miss_rate": present_miss,
        "reappearance_pairs": len(pairs),
        "reid_overall": reid_metrics.get("overall", {}).get("rate", 0.0),
        "reid_short": reid_metrics.get("short (<20f)", {}).get("rate", 0.0),
        "reid_medium": reid_metrics.get("medium (20-50f)", {}).get("rate", 0.0),
        "reid_long": reid_metrics.get("long (>50f)", {}).get("rate", 0.0),
    }
```

---

### 3.4 报告生成

**修改**：`cogtrack/evaluation/report.py`

添加 presence-aware 专用表格：

```python
def generate_presence_aware_report(aggregated_metrics: dict[str, Any]) -> str:
    """生成 presence-aware 专用报告。"""
    
    report = []
    report.append("# Presence-Aware Tracking Report\n")
    report.append("## Overall Presence Discrimination\n")
    report.append(f"- Presence Precision: {aggregated_metrics['presence_precision']:.3f}")
    report.append(f"- Presence Recall: {aggregated_metrics['presence_recall']:.3f}")
    report.append(f"- Presence F1: {aggregated_metrics['presence_f1']:.3f}")
    report.append(f"- Absent FPR: {aggregated_metrics['absent_fpr']:.3f}")
    report.append(f"- Present Miss Rate: {aggregated_metrics['present_miss_rate']:.3f}")
    report.append("")
    
    report.append("## Re-Identification After Occlusion\n")
    report.append(f"- Total reappearance events: {aggregated_metrics['reappearance_pairs']}")
    report.append(f"- Overall Re-ID rate: {aggregated_metrics['reid_overall']:.3f}")
    report.append(f"- Short occlusion (<20f): {aggregated_metrics['reid_short']:.3f}")
    report.append(f"- Medium occlusion (20-50f): {aggregated_metrics['reid_medium']:.3f}")
    report.append(f"- Long occlusion (>50f): {aggregated_metrics['reid_long']:.3f}")
    report.append("")
    
    return "\n".join(report)
```

---

## 四、Memory SFT 数据生成（待实现）

### 4.1 流水线设计

```
tracking/mine_memory_events.py           # 事件候选挖掘
         ↓
tracking/annotate_target_states.py      # 双教师生成
         ↓
tracking/verify_target_states.py        # 一致性 + 稳定性验证
         ↓
tracking/export_memory_sft.py           # 重放状态链 + 导出 JSONL
```

### 4.2 核心实现要点

#### **4.2.1 事件候选挖掘**

**目标**：找到"值得更新状态"的帧

**策略**：
1. **视觉特征变化**：目标区域 embedding 变化超过阈值
2. **重现事件**：absent → present
3. **长时间间隔**：距离上次可信观测 >N 帧
4. **Hard-null 采样**：普通运动、仅尺度变化（负样本）

**伪代码**：
```python
def mine_memory_event_candidates(sequence, feature_extractor):
    candidates = []
    
    # 提取所有 present 帧的目标区域特征
    features = []
    for frame_id, bbox in enumerate(sequence.gt_bboxes):
        if sequence.target_status[frame_id] == "present":
            crop = extract_crop(sequence.frames[frame_id], bbox)
            feat = feature_extractor.encode(crop)  # DINOv2 or SigLIP
            features.append((frame_id, feat))
    
    # 候选 1: 特征变化超过阈值
    for i in range(1, len(features)):
        prev_feat = features[i-1][1]
        curr_feat = features[i][1]
        distance = cosine_distance(prev_feat, curr_feat)
        
        # 阈值基于序列内分位数，不是全局固定值
        threshold = np.percentile(all_distances_in_sequence, 75)
        
        if distance > threshold:
            candidates.append({
                "frame_id": features[i][0],
                "reason": "embedding_shift",
                "distance": distance
            })
    
    # 候选 2: 重现事件
    for i in range(1, len(sequence.target_status)):
        if sequence.target_status[i-1] == "absent" and sequence.target_status[i] == "present":
            candidates.append({
                "frame_id": i,
                "reason": "reappearance"
            })
    
    # 候选 3: 长时间间隔
    # ...
    
    # Hard-null: 普通帧（负样本）
    # ...
    
    return candidates
```

---

#### **4.2.2 双教师标注**

**输入**：
- 当前帧（带 GT bbox）
- 目标 crop
- 当前已维护状态
- 支持帧（未来 2-3 帧，验证稳定性）
- 干扰物 crop（验证区分度）

**Prompt**（给教师模型）：
```python
teacher_prompt = f"""
You are annotating target-state memory for long-term tracking.

Initial target identity (permanent): {initial_identity}
Current maintained state: {current_state}

[Image 1: Current frame with GT bbox]
[Image 2: Target region crop]
[Image 3-5: Support frames (future 2-3 frames)]

Task: Decide whether the maintained state should be updated.

Update only if:
1. Stable appearance/viewpoint change (verified on support frames)
2. New state helps future discrimination
3. Identity cues (class, key attributes) preserved

Output JSON:
{{
  "decision": "update" or "keep",
  "memory_update": "<complete self-contained state>" or null,
  "reason_codes": ["stable_viewpoint_change", "identity_cues_preserved"]
}}

Do NOT infer invisible attributes. Be concise (<30 words).
"""
```

**双教师验证**：
```python
teacher1_output = call_teacher(prompt, seed=17)
teacher2_output = call_teacher(prompt, seed=42)

# 检查一致性
if teacher1_output['decision'] == teacher2_output['decision']:
    if teacher1_output['decision'] == "update":
        # 检查文本语义相似度
        similarity = semantic_similarity(
            teacher1_output['memory_update'],
            teacher2_output['memory_update']
        )
        if similarity > 0.8:
            consensus = True
else:
    consensus = False
```

---

#### **4.2.3 自动验收**

**验收条件**（全部满足才接受）：

1. **双教师一致性**：决策相同 + 文本相似
2. **区域对齐**：新状态描述对应 GT target region
3. **干扰物 margin**：target-text score > distractor-text score + 0.15
4. **支持帧稳定性**：新状态在后续帧仍成立
5. **身份一致性**：不与 initial_identity 矛盾
6. **规则检查**：长度、字符集、无坐标、无禁词

```python
def verify_candidate(candidate, teacher_outputs, support_frames):
    # 1. 双教师一致性
    if not check_teacher_agreement(teacher_outputs):
        return False, "teacher_disagreement"
    
    # 2. 区域对齐
    target_crop = candidate['target_crop']
    new_state = teacher_outputs[0]['memory_update']
    target_text_score = compute_clip_score(target_crop, new_state)
    if target_text_score < 0.7:
        return False, "low_target_alignment"
    
    # 3. 干扰物 margin
    if candidate.get('distractor_crops'):
        for distractor in candidate['distractor_crops']:
            distractor_score = compute_clip_score(distractor, new_state)
            if distractor_score + 0.15 >= target_text_score:
                return False, "low_distractor_margin"
    
    # 4. 支持帧稳定性
    for support_frame in support_frames:
        support_crop = extract_crop(support_frame, gt_bbox)
        support_score = compute_clip_score(support_crop, new_state)
        if support_score < 0.65:
            return False, "unstable_on_support"
    
    # 5. 身份一致性
    if contradicts_identity(new_state, candidate['initial_identity']):
        return False, "identity_contradiction"
    
    # 6. 规则检查
    if not passes_rule_checks(new_state):
        return False, "rule_violation"
    
    return True, "accepted"
```

---

### 4.3 数据配方

**Memory SFT 数据混合**：
```python
# 70% Core 跟踪样本（记忆值仍 mask）
core_samples = load_jsonl("core_train.jsonl")
for sample in core_samples:
    sample['supervision_profile'] = 'tracking_core'  # 保持 mask

# 30% Memory 样本（全量监督）
memory_samples = load_jsonl("memory_events.jsonl")
for sample in memory_samples:
    sample['supervision_profile'] = 'full'  # 三字段全监督

# 混合
mixed_data = []
mixed_data.extend(random.sample(core_samples, k=int(0.7 * total)))
mixed_data.extend(random.sample(memory_samples, k=int(0.3 * total)))
random.shuffle(mixed_data)
```

**Memory 样本内部比例**：
```python
# 25% update, 75% hard-null
update_samples = [s for s in memory_samples if s['memory_update'] is not None]
null_samples = [s for s in memory_samples if s['memory_update'] is None]

memory_batch = []
memory_batch.extend(random.sample(update_samples, k=int(0.25 * memory_size)))
memory_batch.extend(random.sample(null_samples, k=int(0.75 * memory_size)))
```

---

## 五、TU-GRPO 设计（核心创新）

### 5.1 反事实轨迹设计

**核心思想**：对同一候选状态，做两条未来回放

```python
def compute_trajectory_utility_reward(
    candidate_state: str,
    current_state: str,
    sequence: Sequence,
    event_frame: int,
    horizon: int = 30,
    evaluator_checkpoint: str = "memory_sft_frozen"
) -> float:
    """
    计算接受 vs 保留的未来轨迹效用差
    
    Args:
        candidate_state: 候选新状态
        current_state: 当前已维护状态
        event_frame: 当前帧 ID
        horizon: 未来轨迹长度
        evaluator_checkpoint: 冻结的评测器
    
    Returns:
        Delta_U = U_accept - U_keep
    """
    
    # 未来帧范围
    future_start = event_frame + 1
    future_end = min(event_frame + horizon + 1, len(sequence))
    future_frames = range(future_start, future_end)
    
    # 分支 1: 接受新状态
    tracker_accept = load_frozen_tracker(evaluator_checkpoint)
    tracker_accept.initialize(sequence, frame=0)
    tracker_accept.update_state_memory(candidate_state, frame=event_frame)
    
    results_accept = []
    for frame_id in future_frames:
        result = tracker_accept.track(sequence.frames[frame_id])
        results_accept.append(result)
    
    # 分支 2: 保留旧状态
    tracker_keep = load_frozen_tracker(evaluator_checkpoint)
    tracker_keep.initialize(sequence, frame=0)
    tracker_keep.update_state_memory(current_state, frame=event_frame)
    
    results_keep = []
    for frame_id in future_frames:
        result = tracker_keep.track(sequence.frames[frame_id])
        results_keep.append(result)
    
    # 计算效用
    U_accept = compute_utility(results_accept, future_frames, sequence)
    U_keep = compute_utility(results_keep, future_frames, sequence)
    
    return U_accept - U_keep
```

### 5.2 效用函数定义

```python
def compute_utility(results, frame_ids, sequence):
    """
    效用 = 未来 presence F1 + 定位精度 + 重现恢复
    """
    # Presence 判别
    pred_status = [r['target_status'] for r in results]
    gt_status = [sequence.target_status[i] for i in frame_ids]
    presence_f1 = compute_f1(pred_status, gt_status)
    
    # 定位精度（只在 present 时计算）
    ious = []
    for r, frame_id in zip(results, frame_ids):
        if r['target_status'] == "present" and sequence.target_status[frame_id] == "present":
            iou = compute_iou(r['bbox'], sequence.gt_bbox[frame_id])
            ious.append(iou)
    avg_iou = np.mean(ious) if ious else 0.0
    
    # 重现恢复（如果未来有重现事件）
    reid_bonus = 0.0
    for i in range(1, len(gt_status)):
        if gt_status[i-1] == "absent" and gt_status[i] == "present":
            # 有重现事件
            if pred_status[i] == "present":
                reid_iou = compute_iou(results[i]['bbox'], sequence.gt_bbox[frame_ids[i]])
                if reid_iou > 0.5:
                    reid_bonus = 1.0  # 成功重识别
    
    # 加权组合
    utility = 0.4 * presence_f1 + 0.4 * avg_iou + 0.2 * reid_bonus
    return utility
```

### 5.3 完整 GRPO Reward

```python
def compute_grpo_reward(candidate_output, metadata):
    """
    R = w_fmt * R_format 
      + w_cur * R_current
      + w_evt * R_event
      + w_ground * R_ground
      + w_traj * Delta_U
      - l_update * P_over_update
      - l_len * P_length
      - l_drift * P_identity_drift
    """
    
    # 1. 格式 reward
    R_format = 1.0 if is_valid_json(candidate_output) else 0.0
    
    # 2. 当前帧 reward
    R_current = 0.0
    if metadata['gt_status'] == "present":
        if candidate_output['target_status'] == "present":
            R_current += 0.5  # 正确判别
            iou = compute_iou(candidate_output['bbox'], metadata['gt_bbox'])
            R_current += 0.5 * iou  # 定位精度
    else:
        if candidate_output['target_status'] == "absent":
            R_current = 1.0  # 正确判别 absent
    
    # 3. 事件一致性 reward（与 memory SFT 标签一致）
    R_event = 0.0
    if 'memory_label' in metadata:
        if (candidate_output['memory_update'] is not None) == (metadata['memory_label'] is not None):
            R_event = 0.5
            if candidate_output['memory_update'] is not None:
                # 文本相似度
                similarity = semantic_similarity(
                    candidate_output['memory_update'],
                    metadata['memory_label']
                )
                R_event += 0.5 * similarity
    
    # 4. Grounding reward
    R_ground = 0.0
    if candidate_output['memory_update'] is not None:
        # Target-text 对齐
        target_crop = metadata['target_crop']
        text = candidate_output['memory_update']
        target_score = compute_clip_score(target_crop, text)
        R_ground += 0.5 * min(target_score / 0.7, 1.0)
        
        # Distractor margin
        if 'distractor_crops' in metadata:
            distractors = metadata['distractor_crops']
            distractor_scores = [compute_clip_score(d, text) for d in distractors]
            max_distractor = max(distractor_scores) if distractor_scores else 0.0
            margin = target_score - max_distractor
            R_ground += 0.5 * min(margin / 0.15, 1.0)
    
    # 5. 轨迹效用 reward（核心创新）
    Delta_U = metadata.get('trajectory_utility_delta', 0.0)
    R_traj = Delta_U  # 已经归一化到 [-1, 1]
    
    # 6. Over-update penalty
    P_over_update = 0.0
    if candidate_output['memory_update'] is not None:
        # 无收益更新
        if Delta_U < 0.05:
            P_over_update += 0.5
        # Absent 时更新
        if candidate_output['target_status'] == "absent":
            P_over_update += 0.5
    
    # 7. 长度 penalty
    P_length = 0.0
    if candidate_output['memory_update'] is not None:
        word_count = len(candidate_output['memory_update'].split())
        if word_count > 30:
            P_length = 0.5
    
    # 8. 身份漂移 penalty
    P_identity_drift = 0.0
    if candidate_output['memory_update'] is not None:
        if contradicts_identity(
            candidate_output['memory_update'],
            metadata['initial_identity']
        ):
            P_identity_drift = 1.0
    
    # 加权组合
    reward = (
        0.1 * R_format +
        0.2 * R_current +
        0.1 * R_event +
        0.1 * R_ground +
        0.4 * R_traj -  # 最高权重给轨迹效用
        0.3 * P_over_update -
        0.1 * P_length -
        0.5 * P_identity_drift
    )
    
    return reward
```

### 5.4 缓存优化

**问题**：真实双分支回放成本高

**方案**：两级实现

**Tier 1: 缓存代理（大部分样本）**
```python
def compute_cached_trajectory_utility(
    candidate_state: str,
    current_state: str,
    future_frames_data: list,
    frozen_encoder
):
    """
    用冻结的 region-text encoder 近似轨迹效用
    不需要真实运行 tracker
    """
    scores_accept = []
    scores_keep = []
    
    for frame_data in future_frames_data:
        if frame_data['gt_status'] == "present":
            target_crop = frame_data['target_crop']
            
            # 新状态的对齐分数
            score_new = frozen_encoder.score(target_crop, candidate_state)
            scores_accept.append(score_new)
            
            # 旧状态的对齐分数
            score_old = frozen_encoder.score(target_crop, current_state)
            scores_keep.append(score_old)
    
    avg_accept = np.mean(scores_accept) if scores_accept else 0.0
    avg_keep = np.mean(scores_keep) if scores_keep else 0.0
    
    return avg_accept - avg_keep
```

**Tier 2: 真实回放（事件丰富子集）**
```python
# 选择标准：
# 1. 有重现事件
# 2. 长时间遮挡后
# 3. 缓存代理的 Delta_U 接近 0（难以判断）
```

---

## 六、执行时间线与里程碑

### Week 1-2: Core SFT + Presence 评测

**任务**：
- [ ] 生成 VLT-v6.3 core 数据（sampling plan + 完整渲染）
- [ ] 训练前验证（validate_sft_supervision + processor replay）
- [ ] 训练 Core SFT checkpoint
- [ ] 实现 presence-aware 评测指标
- [ ] Tiny 评测：Base vs Core

**产出**：
- Core SFT checkpoint
- Presence F1, Absent FPR, Re-ID rate 基线数据
- 评测代码 `cogtrack/evaluation/presence_metrics.py`

**验收标准**：
- Core SFT 的 Presence F1 > 80%
- Absent FPR < 10%
- Re-ID rate (overall) > 50%

---

### Week 3-4: Memory 标签 + Memory SFT

**任务**：
- [ ] 实现 `tracking/mine_memory_events.py`
- [ ] 实现 `tracking/annotate_target_states.py`（双教师）
- [ ] 实现 `tracking/verify_target_states.py`（验收）
- [ ] 人工审核 100-200 validation events
- [ ] 生成 train 银标（分层抽检）
- [ ] 导出 Memory SFT 混合数据
- [ ] 训练 Memory SFT checkpoint
- [ ] Tiny 评测：Core+Memory vs Core alone

**产出**：
- Memory 标签工具链
- Memory SFT checkpoint
- 人工审核质量报告

**验收标准**：
- Teacher agreement > 85%
- Region-text score > 0.7
- Distractor margin > 0.15
- Memory SFT 的 Re-ID rate > Core SFT +5%

---

### Week 5-6: TU-GRPO 验证 + 训练

**任务**：
- [ ] Reward replay（不训练）：100-200 events
- [ ] 手动检查 Delta_U 排序是否合理
- [ ] **决策点**：继续或降级
- [ ] 实现缓存代理版本
- [ ] 实现真实回放版本（如果缓存不够准）
- [ ] 训练 TU-GRPO checkpoint
- [ ] Tiny 评测：TU-GRPO vs Memory SFT

**产出**：
- TU-GRPO checkpoint
- Reward 组件分析报告

**验收标准**：
- Delta_U 与真实 future AUC 正相关
- TU-GRPO 的 Re-ID rate > Memory SFT +5%
- Over-update rate < Memory SFT

---

### Week 7-8: Full 评测 + 核心消融

**任务**：
- [ ] 在 CognitiveBench Full 995 序列运行所有 checkpoints
- [ ] 完整的 presence-aware 指标矩阵
- [ ] 核心消融实验
- [ ] Failure case 分析

**产出**：
- 完整评测报告
- 论文 Table 1, 2, 3 数据

---

## 七、Quick Start 执行清单

**今天立即做的**：

1. **确认环境**：
   ```bash
   python scripts/verify_env.py --verbose
   python -m pytest tests/ -q
   ```

2. **准备数据生成配置**：
   ```bash
   # 检查 configs/env.local.yaml 是否配置正确
   # 检查 LaSOT/TNL2K/MGIT 数据集路径
   ```

3. **生成 Sampling Plan**：
   ```bash
   python tracking/synthesize_vlt_v6_dataset.py \
     --datasets lasot tnl2k mgit \
     --mgit-version tiny \
     --allow-missing-mgit-sequences \
     --env-config configs/env.local.yaml \
     --history-size 3 \
     --max-samples-per-sequence 20 \
     --absent-ratio 0.3 \
     --output-dir data/plans/cogtrack_vlt_v63_core \
     --plan-only
   ```

4. **实现 Presence 评测指标**：
   - 创建 `cogtrack/evaluation/presence_metrics.py`
   - 实现本文档 3.1 节的所有函数
   - 集成到 `evaluator.py`

**本周完成的**：

5. **渲染完整数据**
6. **训练 Core SFT**
7. **Tiny 评测**

需要我协助哪个部分的详细实现代码？
