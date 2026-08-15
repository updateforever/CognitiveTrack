# CognitiveTrack 数据生成完整方案

> 目标：确定 Prompt、MGIT 文本处理、Memory 监督策略
> 日期：2026-08-14

---

## 一、优化后的 Prompt（v6.3.1）

### **System Prompt**

```text
You are a long-term vision-language single-object tracker. Always use the target
marked by the red box in Image 1 and its initial description as the identity anchor; 
neither history predictions nor state memory may overwrite the initialized identity. 

Using the temporally ordered trajectory in Image 2 and the maintained target state, 
first determine whether the same target is present in Image 3:
- present: the exact initialized target is visible and localizable
- absent: the target is not visible, occluded, out-of-view, or cannot be localized

When present, localize the target precisely. Update the target-state memory only 
when a stable state change would help future tracking.
```

**改进点**：
1. ✅ "first determine"（强调判别优先）
2. ✅ 明确 present/absent 定义
3. ✅ 保持简洁（+35 tokens）

### **User Prompt**（不变）

```text
Initial target identity: {initial_identity_description}
Current maintained target state: {current_target_state}
Track the initialized target in the final image.
```

---

## 二、MGIT 文本处理方案

### **2.1 MGIT 描述结构**

```json
{
  "action": "...",
  "activity": "...",
  "story": {
    "story_1": {
      "description": "A male secret agent wearing a black suit walks..."
    }
  }
}
```

**问题**：
- `story` 描述整段视频，可能包含初始化之后的事件
- 例如："walks in the washroom, and stands near..." → 包含未来动作

**当前代码已标记**：
```python
metadata={
    "language_scope": "full_video_story",  # 已标记不安全
}
```

### **2.2 推荐方案：提取 Object Class**

**方案 A：从 Story 提取首句主语（推荐）**

```python
def extract_mgit_initial_identity(story_text: str) -> str:
    """
    从 MGIT story 提取安全的初始化描述
    
    策略：只保留第一句的主语部分，去掉动作
    
    Example:
        Input:  "A male secret agent wearing a black suit walks in..."
        Output: "a male secret agent wearing a black suit"
        
        Input:  "A brown fur dog waits his owner..."
        Output: "a brown fur dog"
    """
    if not story_text:
        return None
    
    # 取第一句
    first_sentence = story_text.split('.')[0].strip()
    
    # 简单启发式：找第一个动词（walks, waits, plays 等）之前的部分
    import re
    # 匹配：A/An/The + 名词短语 + 动词
    match = re.match(r'^(An?\s+[^.]+?)\s+(walks|waits|plays|is|stands|runs|goes|comes)', 
                     first_sentence, re.IGNORECASE)
    
    if match:
        subject = match.group(1).strip()
        # 去掉末尾可能的逗号
        subject = subject.rstrip(',')
        return subject.lower()
    
    # Fallback: 取前 10 个词
    words = first_sentence.split()[:10]
    return ' '.join(words).lower()
```

**方案 B：使用 action/activity 字段（如果可用）**

```python
def extract_mgit_identity_from_metadata(json_data: dict) -> str:
    """
    优先使用 action/activity，Fallback 到 story 提取
    """
    # 1. 尝试 activity
    if 'activity' in json_data and json_data['activity']:
        return json_data['activity'].lower()
    
    # 2. 尝试 action
    if 'action' in json_data and json_data['action']:
        return json_data['action'].lower()
    
    # 3. Fallback: 从 story 提取
    story = json_data.get('story', {})
    if isinstance(story, dict):
        for key in ['story_1'] + sorted(story.keys()):
            item = story.get(key)
            if isinstance(item, dict) and item.get('description'):
                return extract_mgit_initial_identity(item['description'])
    
    return None
```

**方案 C：最保守（回退到视觉指代）**

```python
def get_mgit_identity_safe(json_data: dict) -> str:
    """
    最保守：如果无法安全提取，回退到视觉指代
    """
    identity = extract_mgit_identity_from_metadata(json_data)
    
    if not identity or len(identity) < 5:
        # 回退到视觉指代
        return "the target marked by the red box in Image 1"
    
    return identity
```

**推荐**：✅ 方案 A + Fallback C

---

### **2.3 实现代码修改**

**位置**：`pytracking/datasets/mgit.py`

```python
def _load_description(self, name: str) -> str | None:
    path = self.base_path / "attribute" / "description" / f"{name}.json"
    if not path.is_file():
        return None
    
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"MGIT 描述文件损坏: {path}") from exc
    
    # 新增：提取安全的初始化描述
    identity = self._extract_safe_identity(payload)
    return identity

def _extract_safe_identity(self, json_data: dict) -> str | None:
    """从 MGIT JSON 提取安全的初始化身份描述"""
    
    # 尝试从 story 提取主语
    story = json_data.get("story", {})
    if isinstance(story, dict):
        for key in ["story_1"] + sorted(story.keys()):
            item = story.get(key)
            if isinstance(item, dict) and item.get("description"):
                desc = str(item["description"]).strip()
                if desc:
                    # 提取第一句的主语部分
                    identity = self._extract_subject_from_story(desc)
                    if identity:
                        return identity
    
    # Fallback: 返回 None，让 tracker 使用视觉指代
    return None

def _extract_subject_from_story(self, story_text: str) -> str | None:
    """从 story 文本提取主语（去掉动作）"""
    import re
    
    first_sentence = story_text.split('.')[0].strip()
    
    # 匹配模式：A/An/The + 名词短语 + 动词
    match = re.match(
        r'^(An?\s+[^.]+?)\s+(walks|waits|plays|is|are|stands|runs|goes|comes|wakes|slides)',
        first_sentence,
        re.IGNORECASE
    )
    
    if match:
        subject = match.group(1).strip().rstrip(',')
        # 首字母小写（与 LaSOT/TNL2K 一致）
        return subject[0].lower() + subject[1:] if subject else None
    
    # Fallback: 取前 8 个词
    words = first_sentence.split()[:8]
    result = ' '.join(words)
    return result[0].lower() + result[1:] if result else None
```

---

## 三、Memory 监督策略分析

### **3.1 你的问题：Memory 训练是否可以 mask 监督？**

**回答**：✅ 可以，但不推荐作为主要方案

**三种 Memory 监督策略对比**：

| 策略 | Core SFT | Memory SFT | 何时用 |
|------|----------|-----------|--------|
| **A. Masked (tracking_core)** | ✅ mask memory 值 | ✅ 继续 mask | 不训练 memory，只靠 GRPO |
| **B. Mixed (推荐)** | ✅ mask memory 值 | ✅ 部分样本 full 监督 | SFT 冷启动 + GRPO 优化 |
| **C. Full (激进)** | ✅ mask memory 值 | ✅ 所有样本 full 监督 | 重度依赖银标质量 |

---

### **3.2 策略 A：Pure Masked（只靠 GRPO）**

**流程**：
```
Core SFT (mask memory) 
    ↓
直接 GRPO（从头学习何时更新）
    ↓
无需 Memory SFT
```

**优点**：
- ✅ 不需要生成 memory 银标
- ✅ 避免银标质量问题
- ✅ GRPO 完全从 reward 学习

**缺点**：
- ❌ GRPO 冷启动困难（没有 SFT warm-up）
- ❌ 可能需要更多 GRPO 样本和训练时间
- ❌ 更新率可能不稳定（要么过度更新，要么从不更新）

**可行性**：⚠️ 理论可行，但风险高

---

### **3.3 策略 B：Mixed Supervision（推荐）**

**流程**：
```
Core SFT (mask memory)
    ↓
Memory SFT (70% core mask + 30% memory full)
    ↓
TU-GRPO（优化更新时机和质量）
```

**数据配方**：
```python
# Core 样本（保持 mask）
core_samples = load_core_data()  # ~20K samples
for s in core_samples:
    s['supervision_profile'] = 'tracking_core'  # mask memory 值

# Memory 事件样本（full 监督）
memory_samples = load_memory_events()  # ~6K samples
for s in memory_samples:
    s['supervision_profile'] = 'full'  # 三字段全监督

# Memory 样本内部：25% update, 75% hard-null
update_samples = [s for s in memory_samples if s['memory_update'] is not None]
null_samples = [s for s in memory_samples if s['memory_update'] is None]

# 混合比例
train_data = []
train_data.extend(random.sample(core_samples, k=14000))  # 70%
train_data.extend(random.sample(update_samples, k=1500))  # 7.5%
train_data.extend(random.sample(null_samples, k=4500))   # 22.5%
random.shuffle(train_data)
```

**优点**：
- ✅ Memory SFT 提供 warm-up（模型知道"什么是更新"）
- ✅ GRPO 有更好的初始化（不是完全随机）
- ✅ 银标质量问题影响有限（只占 30%）

**缺点**：
- ⚠️ 需要生成 memory 银标（但可以少量高质量）

**推荐理由**：
1. 平衡了 SFT 和 GRPO 的优势
2. 即使银标有噪声，70% core 样本保持跟踪能力
3. GRPO 可以纠正 SFT 的错误更新倾向

---

### **3.4 策略 C：Full Supervision（激进）**

**流程**：
```
Core SFT (mask memory)
    ↓
Memory SFT (100% full 监督)
    ↓
可选 GRPO（微调）
```

**缺点**：
- ❌ 重度依赖银标质量
- ❌ 模型可能学会"盲目跟随银标模式"
- ❌ 泛化能力可能不如 GRPO

**适用场景**：
- 银标质量极高（双教师一致性 >95%）
- 时间紧张，无法做 GRPO
- 作为 baseline 对比

---

### **3.5 你的判断：重点通过 GRPO 优化**

**我的建议**：✅ 完全正确

**最优路线**：
```
Week 1-2: Core SFT (mask memory)
    ↓
Week 3: 生成少量高质量 memory 事件（~3K-5K，重质不重量）
    ↓
Week 4: Memory SFT (70% core + 30% memory mixed)
    ↓
Week 5-6: TU-GRPO (核心优化)
    ↓
Week 7-8: Full 评测
```

**关键**：
- Memory SFT 只是 warm-up，不追求完美
- GRPO 是真正学习"何时更新有用"的阶段
- 银标数量可以少（3K-5K），但质量要高

---

## 四、实现代码

### **4.1 Prompt 更新代码**

```python
# cogtrack/prompts/vlt_tracking.py

VLT_TRACKING_PROMPT_VERSION = "6.3.1"

def _system_prompt() -> str:
    """渲染训练模型使用的极简任务定义，不重复其已经学过的输出协议。"""
    
    return """You are a long-term vision-language single-object tracker. Always use the target
marked by the red box in Image 1 and its initial description as the identity anchor; neither
history predictions nor state memory may overwrite the initialized identity. 

Using the temporally ordered trajectory in Image 2 and the maintained target state, first 
determine whether the same target is present in Image 3:
- present: the exact initialized target is visible and localizable
- absent: the target is not visible, occluded, out-of-view, or cannot be localized

When present, localize the target precisely. Update the target-state memory only when a stable 
state change would help future tracking."""
```

### **4.2 MGIT 文本提取代码**

```python
# pytracking/datasets/mgit.py

import re

def _extract_safe_identity(self, json_data: dict) -> str | None:
    """从 MGIT JSON 提取安全的初始化身份描述"""
    
    # 尝试从 story 提取主语
    story = json_data.get("story", {})
    if isinstance(story, dict):
        for key in ["story_1"] + sorted(k for k in story if k != "story_1"):
            item = story.get(key)
            if isinstance(item, dict) and item.get("description"):
                desc = str(item["description"]).strip()
                if desc:
                    # 提取第一句的主语部分
                    identity = self._extract_subject_from_story(desc)
                    if identity:
                        return identity
    
    # Fallback: 返回 None，让 tracker 使用视觉指代
    return None

def _extract_subject_from_story(self, story_text: str) -> str | None:
    """从 story 文本提取主语（去掉动作）"""
    
    first_sentence = story_text.split('.')[0].strip()
    
    # 匹配模式：A/An/The + 名词短语 + 动词
    match = re.match(
        r'^(An?\s+[^.]+?)\s+(walks|waits|plays|is|are|stands|runs|goes|comes|wakes|slides)',
        first_sentence,
        re.IGNORECASE
    )
    
    if match:
        subject = match.group(1).strip().rstrip(',')
        # 首字母小写
        return subject[0].lower() + subject[1:] if subject else None
    
    # Fallback: 取前 8 个词
    words = first_sentence.split()[:8]
    result = ' '.join(words)
    return result[0].lower() + result[1:] if result else None

def _load_description(self, name: str) -> str | None:
    """加载并提取安全的初始化描述"""
    path = self.base_path / "attribute" / "description" / f"{name}.json"
    if not path.is_file():
        return None
    
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"MGIT 描述文件损坏: {path}") from exc
    
    # 提取安全的初始化描述
    return self._extract_safe_identity(payload)
```

---

## 五、数据生成执行脚本

### **5.1 完整生成脚本**

创建 `scripts/generate_vlt_v631_core_data.sh`：

```bash
#!/bin/bash
# VLT-v6.3.1 Core SFT 数据生成脚本

set -e

# ============ 配置 ============
DATASETS="lasot tnl2k mgit"
MGIT_VERSION="tiny"
HISTORY_SIZE=3
MAX_SAMPLES_PER_SEQ=20
ABSENT_RATIO=0.3
OUTPUT_BASE="data/releases/cogtrack_vlt_v631_core"
ENV_CONFIG="configs/env.local.yaml"

# ============ Step 1: 生成 Sampling Plan ============
echo "========================================="
echo "Step 1: 生成 Sampling Plan"
echo "========================================="

python tracking/synthesize_vlt_v6_dataset.py \
  --datasets ${DATASETS} \
  --mgit-version ${MGIT_VERSION} \
  --allow-missing-mgit-sequences \
  --env-config ${ENV_CONFIG} \
  --history-size ${HISTORY_SIZE} \
  --max-samples-per-sequence ${MAX_SAMPLES_PER_SEQ} \
  --absent-ratio ${ABSENT_RATIO} \
  --output-dir ${OUTPUT_BASE} \
  --plan-only

echo ""
echo "✓ Sampling Plan 生成完成"
echo ""

# ============ Step 2: 检查 Plan 统计 ============
echo "========================================="
echo "Step 2: 检查 Plan 统计"
echo "========================================="

python -c "
import json
plan = json.load(open('${OUTPUT_BASE}/sampling_plan.json'))
print(f'总序列数: {len(plan[\"sequences\"])}')
print(f'总 cases: {plan[\"summary\"][\"total_cases\"]}')
print(f'Present: {plan[\"summary\"][\"present_cases\"]}')
print(f'Absent: {plan[\"summary\"][\"absent_cases\"]}')
ratio = plan['summary']['absent_cases'] / plan['summary']['total_cases']
print(f'Absent ratio: {ratio:.3f}')
print()
print('各数据集分布:')
for ds_stats in plan['summary']['per_dataset']:
    print(f'  {ds_stats[\"dataset\"]}: {ds_stats[\"sequences\"]} seqs, {ds_stats[\"cases\"]} cases')
"

echo ""
read -p "Plan 统计正常？按 Enter 继续，Ctrl+C 取消..." 

# ============ Step 3: 渲染完整数据 ============
echo "========================================="
echo "Step 3: 渲染完整数据"
echo "========================================="

python tracking/synthesize_vlt_v6_dataset.py \
  --datasets ${DATASETS} \
  --mgit-version ${MGIT_VERSION} \
  --allow-missing-mgit-sequences \
  --env-config ${ENV_CONFIG} \
  --history-size ${HISTORY_SIZE} \
  --sampling-plan ${OUTPUT_BASE}/sampling_plan.json \
  --max-samples-per-sequence ${MAX_SAMPLES_PER_SEQ} \
  --absent-ratio ${ABSENT_RATIO} \
  --output-dir ${OUTPUT_BASE}

echo ""
echo "✓ 数据渲染完成"
echo ""

# ============ Step 4: 验证数据 ============
echo "========================================="
echo "Step 4: 验证数据质量"
echo "========================================="

export DATASET_ROOT="${OUTPUT_BASE}"
export TRAIN_DATA="${DATASET_ROOT}/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="${DATASET_ROOT}/ms_swift/qwen3_vl/val.jsonl"

# 验证监督档位
echo "验证监督档位..."
python tracking/validate_sft_supervision.py \
  --profile tracking_core \
  --dataset "${TRAIN_DATA}" \
  --dataset "${VAL_DATA}"

echo ""
echo "验证 Qwen processor 回放..."
python tools/verify_qwen_grounding_templates.py \
  --dataset-root "${DATASET_ROOT}" \
  --qwen3-model /models/Qwen3-VL-4B-Instruct \
  --verify-tracking-core-mask

echo ""
echo "========================================="
echo "✓ 数据生成完成！"
echo "========================================="
echo "输出目录: ${OUTPUT_BASE}"
echo "训练数据: ${TRAIN_DATA}"
echo "验证数据: ${VAL_DATA}"
echo ""
echo "下一步："
echo "1. 人工抽查 10-20 个样本"
echo "2. 启动 Core SFT 训练"
```

### **5.2 可视化检查脚本**

创建 `scripts/visualize_training_samples.py`：

```python
#!/usr/bin/env python3
"""可视化训练样本，用于人工检查数据质量"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def visualize_sample(sample, dataset_root, output_path=None):
    """可视化一个训练样本"""
    
    # 读取图像
    images = []
    for img_path in sample['images']:
        full_path = Path(dataset_root) / img_path
        img = cv2.imread(str(full_path))
        if img is not None:
            images.append(img)
    
    if len(images) != 3:
        print(f"Warning: Expected 3 images, got {len(images)}")
        return
    
    # 拼接三图
    canvas_width = sum(img.shape[1] for img in images) + 40
    canvas_height = max(img.shape[0] for img in images) + 100
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    x_offset = 10
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        canvas[50:50+h, x_offset:x_offset+w] = img
        
        # 添加标签
        label = f"Image {i+1}"
        cv2.putText(canvas, label, (x_offset, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        x_offset += w + 10
    
    # 添加文本信息
    prompt = sample['conversations'][0]['value']
    response = sample['conversations'][1]['value']
    
    # 解析 response
    try:
        resp_data = json.loads(response)
        status = resp_data['target_status']
        bbox = resp_data.get('bbox_norm1000_xyxy', 'null')
        memory = resp_data.get('memory_update', 'null')
    except:
        status = "parse_error"
        bbox = "error"
        memory = "error"
    
    # 添加文本到画布底部
    y_text = canvas_height - 80
    cv2.putText(canvas, f"Status: {status}", (10, y_text), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y_text += 25
    cv2.putText(canvas, f"Bbox: {str(bbox)[:60]}", (10, y_text), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_text += 20
    cv2.putText(canvas, f"Memory: {str(memory)[:60]}", (10, y_text), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
    
    # 保存或显示
    if output_path:
        cv2.imwrite(output_path, canvas)
        print(f"Saved to {output_path}")
    else:
        cv2.imshow("Training Sample", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='ms-swift JSONL 文件')
    parser.add_argument('--dataset-root', required=True, help='数据根目录')
    parser.add_argument('--count', type=int, default=10, help='可视化样本数')
    parser.add_argument('--output-dir', help='输出目录（不指定则交互显示）')
    parser.add_argument('--random', action='store_true', help='随机采样')
    args = parser.parse_args()
    
    # 加载数据
    with open(args.dataset) as f:
        samples = [json.loads(line) for line in f]
    
    print(f"Total samples: {len(samples)}")
    
    # 采样
    if args.random:
        samples = random.sample(samples, min(args.count, len(samples)))
    else:
        samples = samples[:args.count]
    
    # 可视化
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples):
        print(f"\n=== Sample {i+1}/{len(samples)} ===")
        
        output_path = None
        if args.output_dir:
            output_path = str(Path(args.output_dir) / f"sample_{i:03d}.jpg")
        
        visualize_sample(sample, args.dataset_root, output_path)


if __name__ == '__main__':
    main()
```

**使用方法**：
```bash
python scripts/visualize_training_samples.py \
  --dataset data/releases/cogtrack_vlt_v631_core/ms_swift/qwen3_vl/train.jsonl \
  --dataset-root data/releases/cogtrack_vlt_v631_core \
  --count 10 \
  --output-dir data/visualization/vlt_v631_samples \
  --random
```

---

## 六、执行清单

### **今天完成（代码修改）**

- [ ] 更新 Prompt 到 6.3.1（`cogtrack/prompts/vlt_tracking.py`）
- [ ] 实现 MGIT 文本提取（`pytracking/datasets/mgit.py`）
- [ ] 创建数据生成脚本（`scripts/generate_vlt_v631_core_data.sh`）
- [ ] 创建可视化脚本（`scripts/visualize_training_samples.py`）
- [ ] 测试代码（本地快速验证）

### **训练服务器执行（数据生成）**

- [ ] 同步代码到训练服务器
- [ ] 确认环境和数据集路径
- [ ] 运行 `scripts/generate_vlt_v631_core_data.sh`
- [ ] 验证数据质量
- [ ] 可视化 10-20 个样本检查
- [ ] 启动 Core SFT 训练

---

## 七、Memory 训练策略（Week 3-4）

**推荐路线**：Mixed Supervision

```
1. 少量高质量事件（3K-5K）
2. 70% core (mask) + 30% memory (full)
3. Memory SFT 作为 GRPO warm-up
4. GRPO 是真正优化阶段
```

**不推荐**：Pure masked（风险太高，GRPO 冷启动困难）

---

需要我现在：
1. **生成完整的代码修改 patch**？
2. **帮你在本地测试这些改动**？
3. **准备训练服务器的部署脚本**？
