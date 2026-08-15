# VLT-v6.3 Prompt 优化分析与数据生成确认

> 目标：确认数据生成的 Prompt 设计，聚焦 presence-aware tracking
> 日期：2026-08-14

---

## 一、当前 VLT-v6.3 Prompt 完整内容

### **System Prompt (Version 6.3.0)**

```text
You are a long-term vision-language single-object tracker. Always use the target
marked by the red box in Image 1 and its initial description as the identity anchor; 
neither history predictions nor state memory may overwrite the initialized identity. 
Using the temporally ordered trajectory in Image 2 and the maintained target state, 
determine whether the same target is present in Image 3, analyze its current state, 
and localize it when present. Update the target-state memory only when a stable 
state change would help future tracking.
```

### **User Prompt (动态部分)**

```text
Initial target identity: {initial_identity_description}
Current maintained target state: {current_target_state}
Track the initialized target in the final image.
```

### **输入图像**
- Image 1: 带红框的永久初始化模板
- Image 2: 近期三帧带框历史条带（白色竖向分隔）
- Image 3: 无框当前完整图

### **输出格式**（由 SFT 内化，不写入 Prompt）
```json
{
  "target_status": "present",
  "bbox_norm1000_xyxy": [100, 120, 400, 520],
  "memory_update": null
}
```

---

## 二、当前设计的优点 ✅

### **1. 极简且聚焦**
- 只说"判断目标在不在"、"定位"、"按需更新"
- 不重复 JSON schema、bbox format（由 SFT 内化）
- Token 效率高

### **2. 永久锚点机制清晰**
- "Always use... as the identity anchor"
- "neither history predictions nor state memory may overwrite"
- 明确防止记忆覆盖初始化信息

### **3. 任务定义准确**
- "determine whether the same target is present"（判别在不在）
- "localize it when present"（存在时定位）
- "Update... only when stable state change would help future tracking"（有条件更新）

### **4. 与 presence-aware 故事完美契合**
- 核心任务就是"是否 present"
- 不强调"身份维护"（那是扩展）
- 符合当前重点

---

## 三、可能的优化方向

### **优化 1: 强化 Presence Judgment 的重要性**

**当前**（隐含）：
> "determine whether the same target is present in Image 3"

**优化后**（显式）：
> "First determine whether the same target is present in Image 3. If absent, 
> the target is not visible or cannot be localized in the current frame."

**理由**：
- 明确 absent 的含义（不可见 or 不可定位）
- 强调"先判别，再定位"的顺序
- 与传统 tracker"盲目预测"形成对比

**Trade-off**：
- 增加了 ~20 tokens
- 但让任务定义更清晰

**建议**：✅ 采纳（小改进，收益明显）

---

### **优化 2: 明确 Absent vs Present 的定义**

**当前**（简略）：
> "determine whether the same target is present"

**优化后**（详细）：
> "Determine whether the same target is present in Image 3:
> - present: the exact initialized target is visible and localizable
> - absent: the target is not visible, occluded, out-of-view, or cannot be localized
> - Similar-category objects are NOT the target"

**理由**：
- 明确 present/absent 的边界
- 防止模型把"同类物体"当作 present
- 强调"exact initialized target"

**Trade-off**：
- 增加 ~40 tokens
- 可能过于详细，SFT 后应该内化

**建议**：⚠️ 可选（如果 Base 模型 absent FPR 高，可加入）

---

### **优化 3: 简化 Memory Update 说明**

**当前**（training 版本）：
> "Update the target-state memory only when a stable state change would help 
> future tracking."

**问题**：
- Core SFT 阶段 memory_update 值被 mask，模型看不到真实更新
- 这句话可能让模型困惑"什么是有用的更新"

**优化 A（Core SFT 专用）**：
> "A memory_update field is reserved for future state tracking but not used 
> in this task."

或者干脆**省略这句**，让 System Prompt 只聚焦 presence + localization。

**优化 B（Memory SFT 之后）**：
保持当前版本，或改为：
> "Update the target-state memory when the target shows a stable appearance 
> or viewpoint change that would help discriminate it in future frames."

**建议**：
- ✅ Core SFT: 省略或简化 memory 说明
- ✅ Memory SFT: 使用更具体的更新条件

---

### **优化 4: 添加 Negative Case 说明**

**当前**（无）：
系统 prompt 没有说明"什么情况下应该输出 absent"

**优化后**：
> "Output absent when:
> - The target is occluded or out-of-view
> - The target cannot be confidently localized
> - Only similar-category distractors are visible (not the exact target)"

**理由**：
- 明确 absent 的触发条件
- 帮助模型区分"目标消失"vs"定位不准"

**Trade-off**：
- 增加 ~30 tokens
- 可能让 prompt 过于详细

**建议**：⚠️ 可选（如果 present miss rate 高，可加入）

---

## 四、针对 Presence-Aware 的推荐 Prompt

### **方案 A: 最小改动（保守）**

**System Prompt (v6.3.1 - Conservative)**：
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

**改动**：
- ✅ 强化 presence judgment（"first determine"）
- ✅ 明确 present/absent 定义
- ✅ 保持简洁（增加 ~35 tokens）

---

### **方案 B: Core SFT 专用（激进）**

**System Prompt (v6.3.1-core - Aggressive)**：
```text
You are a long-term vision-language single-object tracker. Always use the target
marked by the red box in Image 1 and its initial description as the identity anchor.

Using the temporally ordered trajectory in Image 2 and any maintained target state, 
determine whether the exact initialized target is present in Image 3:
- present: the target is visible and can be localized
- absent: the target is not visible, occluded, or out-of-view

When present, localize the target precisely.
```

**改动**：
- ✅ 移除 memory update 说明（Core SFT 不需要）
- ✅ 聚焦 presence + localization
- ✅ 更简洁（减少 ~20 tokens）

**适用**：
- Core SFT 训练专用
- Memory SFT 时再加回 memory 说明

---

### **方案 C: 保持现有（不改）**

**理由**：
- 当前 6.3.0 已经很好
- 改动风险 > 收益
- SFT 会内化细节

**建议场景**：
- 如果 Base 模型 presence F1 已经 >60%
- 如果团队希望快速推进，避免重新验证

---

## 五、User Prompt 优化

### **当前 User Prompt**：
```text
Initial target identity: {initial_identity_description}
Current maintained target state: {current_target_state}
Track the initialized target in the final image.
```

### **可能的优化**：

**选项 1: 强化任务触发（显式）**
```text
Initial target identity: {initial_identity_description}
Current maintained target state: {current_target_state}

Question: Is the initialized target present in Image 3? If yes, where is it?
```

**选项 2: 简化触发语（极简）**
```text
Initial target identity: {initial_identity_description}
Current maintained target state: {current_target_state}

Track the target in the final image.
```
（去掉 "initialized"，因为 System Prompt 已经说明）

**建议**：✅ 保持当前版本（已经很好，无需改动）

---

## 六、数据生成中的 Prompt 一致性检查

### **6.1 当前实现检查**

```python
# cogtrack/prompts/vlt_tracking.py
def build_vlt_tracking_prompt(
    history_count: int = 0,
    target_text: str = "",
    semantic_memory: str = "",
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
    include_memory_update: bool = True,
) -> PromptSpec:
    """构造 VLT-v6.3 的极简动态输入。"""
    
    # ... 省略 ...
    
    initial_identity = _dynamic_value(
        target_text,
        empty="the target marked by the red box in Image 1",
    )
    current_state = _dynamic_value(semantic_memory, empty=initial_identity)
    
    user_prompt = "\n".join((
        "Initial target identity: " + initial_identity,
        "Current maintained target state: " + current_state,
        "Track the initialized target in the final image.",
    ))
    
    return PromptSpec(
        name=prompt_name,
        version=VLT_TRACKING_PROMPT_VERSION,  # "6.3.0"
        system_prompt=_system_prompt(),
        user_prompt=user_prompt,
        expected_image_count=expected_image_count,
        include_memory_update=include_memory_update,
        bbox_protocol=bbox_protocol,
    )
```

**检查项**：
- ✅ Prompt version 明确（6.3.0）
- ✅ 动态文本正确处理（initial_identity 有 fallback）
- ✅ Current state 初始化为 initial_identity
- ✅ 三图协议一致（expected_image_count=3）

---

### **6.2 数据生成时的 Prompt 使用**

```python
# cogtrack/training/tracking_samples.py
# 在 build_tracking_samples() 中

if prompt_profile == PROMPT_PROFILE_VLT_V6:
    # VLT-v6.3 使用固定三图 + 双文本
    prompt = build_vlt_tracking_prompt(
        history_count=len(history_panels) if history_panels else 0,
        target_text=initial_identity_text,
        semantic_memory=current_semantic_memory,
        bbox_protocol=bbox_protocol,
        include_memory_update=(memory_supervision != MEMORY_SUPERVISION_DISABLED)
    )
```

**检查项**：
- ✅ history_count 正确传递（3 或 0）
- ✅ target_text 从 LaSOT/TNL2K 获取
- ✅ semantic_memory 初始化为空（Core SFT）
- ✅ include_memory_update 根据 supervision 模式控制

---

### **6.3 Core SFT 的特殊处理**

**关键**：Core SFT 阶段，`memory_update` 字段保留，但值被 mask

```python
# cogtrack/training/loss_mask.py
def compute_tracking_core_loss_mask(response_text: str) -> list[tuple[int, int, int]]:
    """
    tracking_core: 
    - target_status: loss=1
    - bbox: loss=1
    - "memory_update": (字段名) loss=1
    - null (值) loss=0
    - 最终 }: loss=1
    """
    # ... 实现省略 ...
```

**验证**：
```bash
python tracking/validate_sft_supervision.py \
  --profile tracking_core \
  --dataset train.jsonl
```

**Expected output**：
```
✓ All 'memory_update' values are masked (loss=0)
✓ Field name '"memory_update":' is supervised (loss=1)
✓ Other fields fully supervised
```

---

## 七、LaSOT/TNL2K/MGIT 初始化文本处理

### **7.1 文本来源**

| 数据集 | 文本来源 | 可用性 | 处理方式 |
|--------|---------|--------|---------|
| LaSOT | `nlp.txt` | ✅ 全部可用 | 直接使用 |
| TNL2K | `language.txt` | ✅ 全部可用 | 直接使用 |
| MGIT | `story` | ⚠️ 可能含未来事件 | 回退到 class or 视觉指代 |

### **7.2 当前实现**

```python
# pytracking/datasets/lasot.py
def _load_sequence(self, sequence_name: str) -> Sequence:
    # ...
    nlp_file = sequence_path / "nlp.txt"
    if nlp_file.exists():
        with nlp_file.open("r", encoding="utf-8") as f:
            nlp = f.read().strip()
            metadata["language_description"] = nlp
    # ...
```

```python
# pytracking/datasets/tnl2k.py
def _load_sequence(self, sequence_name: str) -> Sequence:
    # ...
    language_file = sequence_path / "language.txt"
    if language_file.exists():
        with language_file.open("r", encoding="utf-8") as f:
            language = f.read().strip()
            metadata["language_description"] = language
    # ...
```

```python
# pytracking/datasets/mgit.py
def _load_sequence(self, sequence_name: str) -> Sequence:
    # ...
    # MGIT 的 story 可能包含整段视频描述
    # 当前代码是否安全处理？需要检查
    metadata["language_description"] = story if story_is_safe else object_class
    # ...
```

### **7.3 需要验证的点**

**Action Item**：
```bash
# 检查 MGIT 序列的 story 是否含未来事件
python -c "
import sys
sys.path.insert(0, '.')
from pytracking.datasets import iter_dataset
from pytracking.evaluation.environment import load_environment

env = load_environment()
mgit = iter_dataset('mgit', environment=env, split='train', version='tiny', limit=10)

for seq in mgit:
    story = seq.metadata.get('language_description', '')
    print(f'{seq.name}: {story[:100]}...')
"
```

**如果发现 story 不安全**：
- 在 `tracking/synthesize_vlt_v6_dataset.py` 中添加过滤
- 或者直接回退到 object class

---

## 八、输出格式与解析

### **8.1 训练样本的输出格式**

**Core SFT**：
```json
{
  "target_status": "present",
  "bbox_norm1000_xyxy": [100, 120, 400, 520],
  "memory_update": null
}
```

**三字段全部保留**，但 `null` 值被 loss mask。

### **8.2 Absent 样本的输出**

```json
{
  "target_status": "absent",
  "bbox_norm1000_xyxy": null,
  "memory_update": null
}
```

**检查项**：
- ✅ absent 时 bbox 必须为 null
- ✅ absent 时 memory_update 也为 null
- ✅ 三字段结构完整

### **8.3 解析时的严格性**

```python
# cogtrack/protocol/schema.py
# 解析器会检查：
# 1. JSON 格式正确
# 2. target_status in ["present", "absent"]
# 3. present 时 bbox 必须有效
# 4. absent 时 bbox 必须为 null
# 5. 字段顺序（可选）
```

**验证命令**：
```bash
python tools/verify_qwen_grounding_templates.py \
  --dataset-root /path/to/data \
  --qwen3-model /models/Qwen3-VL-4B-Instruct \
  --verify-tracking-core-mask
```

---

## 九、最终推荐方案

### **推荐：方案 A（最小改动）**

**理由**：
1. ✅ 当前 6.3.0 已经很好，聚焦 presence-aware
2. ✅ 小改进（明确 present/absent 定义）收益明显
3. ✅ 避免大改动带来的验证成本
4. ✅ 与论文故事（presence-aware tracking）完美契合

**具体改动**：

**New System Prompt (v6.3.1)**：
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

**User Prompt**：保持不变

**Version bump**：`6.3.0` → `6.3.1`

---

### **实施步骤**

1. **修改 Prompt 代码**（10 分钟）：
```python
# cogtrack/prompts/vlt_tracking.py
VLT_TRACKING_PROMPT_VERSION = "6.3.1"

def _system_prompt() -> str:
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

2. **生成新数据**（2-4 小时）：
```bash
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny \
  --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --max-samples-per-sequence 20 \
  --absent-ratio 0.3 \
  --output-dir data/releases/cogtrack_vlt_v631_core \
  --plan-only  # 先生成 plan

# 验证 plan 后再渲染
python tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --mgit-version tiny \
  --allow-missing-mgit-sequences \
  --env-config configs/env.local.yaml \
  --history-size 3 \
  --sampling-plan data/releases/cogtrack_vlt_v631_core/sampling_plan.json \
  --max-samples-per-sequence 20 \
  --absent-ratio 0.3 \
  --output-dir data/releases/cogtrack_vlt_v631_core
```

3. **验证数据**（30 分钟）：
```bash
# 监督档位
python tracking/validate_sft_supervision.py \
  --profile tracking_core \
  --dataset data/releases/cogtrack_vlt_v631_core/ms_swift/qwen3_vl/train.jsonl

# Processor 回放
python tools/verify_qwen_grounding_templates.py \
  --dataset-root data/releases/cogtrack_vlt_v631_core \
  --qwen3-model /models/Qwen3-VL-4B-Instruct \
  --verify-tracking-core-mask

# 人工抽查 10 个样本
python tools/inspect_training_samples.py \
  --dataset data/releases/cogtrack_vlt_v631_core/source.jsonl \
  --count 10 \
  --random
```

---

## 十、可选：如果不改 Prompt

**如果你决定保持 6.3.0 不变**，那么当前设计已经完全可用：

**优点**：
- ✅ 已经验证可行
- ✅ 避免重新测试
- ✅ 快速推进训练

**风险**：
- ⚠️ Base 模型可能对 "present/absent" 理解有偏差
- ⚠️ 但 SFT 会纠正

**建议**：
- 先用 6.3.0 训练 Core SFT
- 如果 Presence F1 < 75%，再考虑改 Prompt 到 6.3.1
- 如果 Presence F1 > 80%，说明 6.3.0 已经够好

---

## 十一、数据生成 Checklist（最终确认）

### **✅ Prompt 设计**
- [x] System Prompt 明确（6.3.0 或 6.3.1）
- [x] User Prompt 简洁
- [x] 三图协议固定
- [x] 输出格式三字段

### **✅ 数据来源**
- [x] LaSOT train (nlp.txt)
- [x] TNL2K train (language.txt)
- [x] MGIT train tiny (需验证 story 安全性)

### **✅ 采样策略**
- [x] Present:absent ≈ 7:3
- [x] 每序列 max 20 samples
- [x] Reference 严格早于 current
- [x] History 严格早于 current

### **✅ 图像处理**
- [x] 三图：reference + history_strip + current
- [x] History strip: 白色竖向分隔带
- [x] 不足三帧时复制补齐
- [x] 视觉画框（红色）

### **✅ 监督设置**
- [x] memory_supervision = "masked_null"
- [x] Loss mask 只屏蔽 memory_update 值
- [x] 字段名和结构仍受监督

### **✅ 验证流程**
- [x] Sampling plan 可重放
- [x] validate_sft_supervision.py
- [x] verify_qwen_grounding_templates.py
- [x] 人工抽查 10-20 样本

---

## 十二、你的决策

**请选择**：

### **选项 A：采用 Prompt 6.3.1（推荐）**
- 小改进，明确 present/absent 定义
- 需要修改代码 + 重新生成数据
- 时间成本：+3 小时

### **选项 B：保持 Prompt 6.3.0**
- 快速推进，避免额外验证
- 立即开始数据生成
- 时间成本：0

### **选项 C：其他定制**
- 你有特定的 Prompt 想法？
- 我可以协助设计和实现

**你倾向哪个？**
