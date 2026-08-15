# VLT-v6.3 Prompt 设计与合理性论证

> 分析日期：2026-08-14
> 目标：为论文 Method 章节准备 Prompt 设计的清晰表述和消融方向

## 一、当前 Prompt 完整内容

### System Prompt (Version 6.3.0)

```text
You are a long-term vision-language single-object tracker. Always use the target
marked by the red box in Image 1 and its initial description as the identity anchor; neither
history predictions nor state memory may overwrite the initialized identity. Using the temporally
ordered trajectory in Image 2 and the maintained target state, determine whether the same target
is present in Image 3, analyze its current state, and localize it when present. Update the
target-state memory only when a stable state change would help future tracking.
```

### User Prompt (动态部分)

```text
Initial target identity: {initial_identity_description}
Current maintained target state: {current_target_state}
Track the initialized target in the final image.
```

**输入图像**：
- Image 1：带红框的永久初始化模板
- Image 2：近期三次可信观测的带框条带（白色竖向分隔）
- Image 3：无框当前完整图

**输出格式**（由 SFT 内化，不写入 Prompt）：
```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":null}
```

## 二、设计决策与理论支撑

### 决策 1：极简 System Prompt，不写 JSON/坐标/格式约束

**Rationale**：
- 训练模型通过 SFT 样本内化输出协议，不需要在每次推理时重复格式说明
- 保持 System Prompt 聚焦于任务语义本身：身份不可覆盖、利用历史、分析状态、按需更新
- 减少 prompt token 开销，提升推理效率

**支持证据**：
- VideoAuto-R1 等工作表明，过度详细的 prompt 不一定优于简洁任务定义
- 现代 VLM 的指令遵循能力已足够强，SFT 后可稳定输出结构化格式

**需要的消融**：
- Minimal vs Detailed prompt (补充 JSON schema、bbox format、格式约束)
- 量化 token 节省 vs 格式错误率 trade-off

### 决策 2：永久身份锚点，不可被历史/状态覆盖

**核心约束**：
> "Always use the target marked by the red box in Image 1 and its initial description as 
> the identity anchor; neither history predictions nor state memory may overwrite the 
> initialized identity."

**Why it matters**：
- 长时跟踪中，累积的历史预测可能包含错误，如果完全覆盖初始身份，容易漂移
- 动态状态可以更新，但核心身份线索（类别、稳定外观特征）必须保持一致
- 这是 CognitiveTrack 与"全动态 caption"方法的关键区别

**实现方式**：
- `initial_identity_description`：首帧确定，永不改变
- `current_target_state`：可稀疏替换，但必须与身份一致
- Image 1 红框模板：永久存在，不被历史图像覆盖

**需要的消融**：
- With vs without "identity anchor cannot be overwritten" 约束
- 量化 identity drift rate、contradiction rate

### 决策 3：分离身份与状态

**文本结构**：
```
Initial target identity:         [永久不变]
Current maintained target state: [可替换，初值 = identity]
```

**Why separate**：
- 身份：类别、稳定特征（颜色、纹理、部件），由首帧确定
- 状态：当前视角、姿态、构型、可见性，随时间变化
- 分离后可以独立审计：状态更新是否与身份矛盾？

**对比已有工作**：
- DTLLM-VLT：多粒度描述，但未显式区分永久身份与动态状态
- DUTrack：动态更新文本，但未保留初始锚点
- MemVLT：有长短期记忆，但未强调身份不可覆盖

**需要的消融**：
- Single text (identity only) vs Disentangled (identity + state)
- 量化重现恢复能力、长时身份保持

### 决策 4：完整替换状态，不是增量

**更新语义**：
- `memory_update=null`：沿用当前状态
- `memory_update="..."`：完整、自包含的替换文本，不是增量描述

**Why complete snapshot**：
- 增量模式："目标转向背面" + "开始奔跑" + "进入阴影" → 重放时需要拼接，可能矛盾
- 完整替换："白色大狗，现在从背面可见，正在奔跑；黑色耳朵和红色项圈仍可见" → 自包含
- absent 时仍保留 null，不清空记忆，便于重现恢复

**对比已有工作**：
- DUTrack：动态更新但未明确说明是否完整替换
- ChatTracker：修正幻觉但未强调状态快照设计

**需要的消融**：
- Complete replacement vs Incremental delta
- 量化状态矛盾率、重放一致性

### 决策 5：末字段快路径

**字段顺序**：
```json
{
  "target_status": "...",        // 先判别
  "bbox_norm1000_xyxy": [...],   // 再定位
  "memory_update": null          // 最后决策是否更新
}
```

**Why last**：
- 因果语言模型从左到右生成，`memory_update` 在最后可以依赖前面的判别与定位
- `null` 是快速生成路径（高频），非空状态是慢路径（低频）
- 与 VideoAuto-R1 观察一致：快速直接回答优于强制长推理

**需要的消融**：
- Field order: memory_update first vs last
- 量化生成延迟、token 开销

### 决策 6：极简用户触发语，不重复帧号/步骤

**User prompt 末尾**：
```
Track the initialized target in the final image.
```

**Why minimal**：
- 不写"这是第 N 帧"：避免模型依赖绝对帧号而非视觉证据
- 不写"步骤 1 判别、步骤 2 定位"：SFT 已内化流程
- 不写"参考 Image 2 的历史"：System Prompt 已说明

**对比**：
- 某些工作在每次 prompt 中重复详细步骤
- CognitiveTrack 依赖 SFT 内化流程，保持推理时 prompt 简洁

## 三、输入协议设计

### 三图固定结构

**设计**：
- Image 1：永久模板 + 红框
- Image 2：三个历史 panel + 白色竖向分隔带
- Image 3：当前完整图，无框

**Why three images consistently**：
- Base/Core/Memory/GRPO 所有实验共用同一协议，保证公平比较
- 不因训练阶段切换 pair/mosaic，避免引入混杂因素

**历史条带设计**：
```
[h1 带框] | [h2 带框] | [h3 带框]
   ↑           ↑           ↑
  旧          中          新
```

**Why 白色分隔带**：
- 视觉清晰分隔，避免 panel 粘连
- 不用箭头/数字/文字：避免引入 OCR 依赖和固定时间间隔的错误暗示
- 历史间隔可能不均匀（稀疏执行），不应假设等间距

**Padding 策略**：
- 不足 3 个历史时，复制最近可用观测填充右侧
- 尚无动态历史时，复制初始化观测三次
- 这些重复不表示新观测，只是视觉 padding

**对比已有工作**：
- MemVLT：长短期记忆，但未固定历史图像数量
- 多数工作：pair（两图）为主，历史数量不固定
- CognitiveTrack：固定三图协议，统一所有训练阶段

### 视觉画框 vs 坐标文本

**当前选择**：视觉红框（`reference_mode=visual_box`）

**Why visual**：
- 更自然：VLM 视觉能力强，直接画框比解析坐标文本更直观
- 避免坐标系混淆：不需要在 prompt 中解释 norm1000 vs pixel 的区别
- 与 Qwen 官方 grounding 协议一致

**需要的消融**：
- Visual box vs Textual bbox coordinates
- 量化定位精度、格式错误率

## 四、与通用 VLM 的 Strict Comparison Prompt

### 两种 Prompt Profile

| Profile | 用途 | System Prompt | JSON/坐标约束 | 格式惩罚 |
|---------|------|---------------|--------------|---------|
| **Native (6.3.0)** | 训练模型推理 | 极简任务定义 | ❌ 由 SFT 内化 | ❌ 内化 |
| **Strict Comparison** | 未训练通用 VLM | 补充详细约束 | ✅ 显式说明 | ✅ 规则验证 |

### Strict Profile 设计原则（待冻结）

**补充内容**：
1. JSON schema 和字段定义
2. Bbox format: `[x1, y1, x2, y2]` in `[0, 1000]` relative coordinates
3. 格式约束：必须输出合法 JSON，不能有额外文字
4. 错误处理：absent 时 bbox 必须为 null

**使用场景**：
- 对比未经本任务训练的 Base VLM（Qwen3-VL-4B, Qwen2.5-VL-7B）
- 不能与 Core/Memory/GRPO 混在同一消融表中
- 必须在实验记录中明确标注 prompt profile

**公平性说明**：
- Strict profile 不是作弊，而是让通用 VLM 能够输出可解析结果
- 训练模型不需要 strict profile，因为已通过 SFT 内化
- 两种 profile 的对比不是"哪个 prompt 更好"，而是"训练是否有效内化协议"

## 五、需要补充的 Prompt Ablation 实验

### 5.1 核心消融

| 实验 | Base Prompt | Variant | 测试假设 |
|------|-------------|---------|---------|
| **Anchor Constraint** | 有 "identity anchor cannot be overwritten" | 无此约束 | 永久锚点是否防止身份漂移 |
| **Identity-State Split** | 双文本（identity + state） | 单文本（只有 dynamic caption） | 分离设计是否改善身份一致性 |
| **Complete Replacement** | 完整替换状态 | 增量描述 | 自包含快照是否减少矛盾 |
| **History Utilization** | "Using temporally ordered trajectory" | 无历史提示 | 显式要求利用历史的效果 |
| **Update Condition** | "stable state change would help future" | "任何变化都更新" | 更新条件是否影响频率和质量 |

### 5.2 Prompt 详细程度

| Level | System Prompt 长度 | 包含内容 | 预期效果 |
|-------|-------------------|---------|---------|
| **Minimal (当前)** | ~70 words | 只说任务语义 | SFT 后格式稳定，token 省 |
| **Moderate** | ~150 words | + 输出字段定义 | 可能降低格式错误 |
| **Detailed** | ~250 words | + JSON schema + bbox format + 示例 | 通用 VLM 可用，但冗长 |

### 5.3 视觉输入

| 实验 | 配置 | 预期影响 |
|------|------|---------|
| **Reference Mode** | Visual box vs Textual bbox | 定位精度、可解释性 |
| **History Layout** | Strip vs Grid vs Overlaid | 历史利用效果 |
| **Separator** | 白色分隔 vs 无分隔 vs 带序号 | 时序理解 |
| **Padding** | 复制 vs 空白 vs 不 padding | 少历史时的行为 |

## 六、论文中的表达策略

### Method 章节建议结构

```markdown
### 3.2 Long-Term Tracking Protocol

**Input**: We adopt a fixed three-image input across all training stages:
- Image 1: permanent initialization template with red bounding box
- Image 2: temporal strip of three recent accepted observations with white separators
- Image 3: current full frame without annotation

**Text**: Disentangled identity and state:
- Initial target identity (permanent anchor)
- Current maintained target state (replaceable snapshot)

**Output**: Structured JSON with presence status, bbox, and nullable state update:
```json
{"target_status":"present","bbox_norm1000_xyxy":[...], "memory_update":null}
```

**Design principles**:
1. **Permanent identity anchor**: Initial description and reference frame are never 
   overwritten by history or state memory, preventing identity drift in long videos.
2. **Complete state replacement**: Updates are self-contained snapshots, not incremental 
   deltas, avoiding contradiction accumulation.
3. **Last-field fast path**: `memory_update` placed at the end enables null as the fast 
   generation path and non-null as the deliberate slow path.
4. **Minimal system prompt**: Training internalizes output protocol via SFT; the prompt 
   focuses on task semantics (identity preservation, history utilization, state analysis).

### 3.3 Prompt Design

We employ a **minimal task-oriented system prompt (version 6.3.0)** that emphasizes:
- Identity anchor immutability
- Temporally ordered history utilization  
- State analysis and selective memory update

The prompt does NOT enumerate JSON schema, bbox format, or output constraints—these are 
internalized through SFT supervision. For fair comparison with untrained base VLMs, we 
maintain a separate **strict comparison profile** that adds explicit format specifications; 
this is never mixed with trained-model results.

### Ablation (in Experiments)

We ablate the following design choices:
- Identity anchor constraint (with vs without)
- Identity-state disentanglement (dual vs single text)
- State update mode (complete replacement vs incremental)
- Prompt detail level (minimal vs detailed)
- Visual reference (red box vs textual bbox)

Results in Table X show that [具体发现待实验后补充].
```

## 七、当前状态与 TODO

### 已完成 ✅
- Prompt 6.3.0 冻结并实现
- 三图协议固定
- 极简 System Prompt + 双文本输入设计
- 完整替换状态语义明确

### 待完成 ⚠️
- [ ] Strict comparison profile 设计与冻结
- [ ] Prompt ablation 实验设计与执行
- [ ] 定量证据：
  - Identity drift rate (有无 anchor 约束)
  - State contradiction rate (完整替换 vs 增量)
  - Format error rate (minimal vs detailed prompt)
  - Token efficiency (prompt 长度 vs 性能)
- [ ] Few-shot case 示例（Appendix）
- [ ] 与 DTLLM-VLT/DUTrack 等的 prompt 对比表格

### 论文撰写优先级 🔥

1. **必须有**：
   - Prompt 完整文本（Method + Appendix）
   - 设计理由的清晰表述（为什么极简、为什么分离、为什么末字段）
   - 与已有工作的对比（谁用固定描述、谁用动态更新、我们的创新）

2. **强烈建议有**：
   - 至少一组 prompt ablation（identity anchor、identity-state split）
   - Minimal vs detailed prompt 的 token 节省与格式错误 trade-off

3. **可选但加分**：
   - 完整 prompt ablation 矩阵
   - Visual vs textual reference 对比
   - Few-shot case 可视化

---

**总结**：VLT-v6.3 的 Prompt 设计是深思熟虑的，每个决策都有理论支撑，但需要实验证据验证假设。当前最缺的是定量消融和与已有工作的直接对比。
