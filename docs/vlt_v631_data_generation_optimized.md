# VLT-v6.3.1 数据生成方案（基于调研优化版）

**更新日期**：2026-08-14  
**基于调研**：VLM Tracking 数据生成方法调研报告

---

## 🎯 核心创新点（调研验证）

### 1. **TU-GRPO (Trajectory-Utility GRPO)** 🔥🔥🔥
**与 ReasoningTrack 的本质区别**：
- ❌ ReasoningTrack: `R = IoU(current, new_text) - IoU(current, old_text)`
- ✅ VLT-v6.3.1: `Delta-U-H = U(future | accept) - U(future | keep)`

**反事实评估**：
```python
# Accept new state
trajectory_accept = replay_future_with_state(new_state)

# Keep old state  
trajectory_keep = replay_future_with_state(old_state)

# Reward = 未来轨迹效用差
reward = compute_utility(trajectory_accept) - compute_utility(trajectory_keep)
```

### 2. **Identity-State Disentangled Memory** 🔥🔥🔥
- **永久身份锚点**：首帧确定，永不覆盖
- **动态状态完整替换**：非增量，避免矛盾累积
- **防漂移机制**：初始帧始终作为 Image 1

### 3. **Event-Driven State Annotation** 🔥🔥
**不逐帧 caption（受 ATCTrack/ChatTracker 启发）**：
```
1. 事件候选挖掘
   └─ DINOv2 特征变化 + 序列内分位数阈值
   
2. 双教师生成
   └─ Qwen3-VL-32B × 2，一致性验收
   
3. 多维验证
   ├─ Region-text alignment (target vs distractor margin)
   ├─ 支持帧稳定性（变化是否持续）
   └─ 初始身份一致性
   
4. Human audit
   └─ 500-1000 种子事件校准阈值
```

### 4. **Presence-aware Protocol** 🔥🔥
- 显式 `target_status: present/absent` 监督
- Present/absent 7:3 比例
- Absent 保留状态（不清空），支持重现恢复

---

## 📊 数据生成策略调整

### **调整 1：数据量**
**原方案**：~30,000 样本  
**调研建议**：Core SFT 50K-60K（参考 DTLLM-VLT 规模）

**新方案**：
- LaSOT train: ~1,120 序列 × 15 样本 = **16,800**
- TNL2K train: ~1,300 序列 × 10 样本 = **13,000**
- MGIT train: ~150 序列 × 20 样本 = **3,000**
- **总计：~32,800 → 调整为 50K+**（增加每序列采样密度）

**实施**：
```python
# 更新采样密度
SAMPLES_PER_SHORT_SEQ = 8    # < 300 帧 (原 5)
SAMPLES_PER_MEDIUM_SEQ = 15  # 300-1000 帧 (原 10)
SAMPLES_PER_LONG_SEQ = 30    # > 1000 帧 (原 20)
```

### **调整 2：状态描述生成策略**
**原方案**：每个样本都调用大模型生成状态描述  
**调研启示**：DUTrack 发现"简洁优于详细"

**新方案**：
1. **初始身份**：优先使用数据集提供的文本（LaSOT/TNL2K/MGIT action layer）
2. **状态更新**：
   - **简洁模式**（< 30 词）
   - **禁止内容**：坐标、帧号、背景、推理
   - **只写**：类别 + 稳定属性 + 当前构型

**Prompt 优化**：
```python
# 状态更新 prompt（简洁版）
prompt = f"""You are tracking: {initial_identity}
Previous state: {previous_state}
Current: {status_description}

Generate a concise state update (< 30 words):
- If present: action + position
- If absent: absence reason
- If stable: keep same
- NO coordinates, frame numbers, or background details

State:"""
```

### **调整 3：History 采样策略**
**原方案**：固定间隔 10 帧  
**调研启示**：事件驱动优于固定间隔

**新方案（保留简化版）**：
- Core SFT 阶段：**保持固定间隔**（训练基础能力）
- Memory SFT/GRPO 阶段：**事件驱动采样**（高级能力）

**理由**：
- 固定间隔更简单、可复现
- 事件驱动留给 Memory SFT 阶段（已有 mining 工具）

### **调整 4：负样本比例**
**原方案**：60/15/15/10  
**调研建议**：7:3 的 present/absent 更合理

**新方案**：
```python
sample_type_ratios = {
    SampleType.PURE_POSITIVE: 0.70,     # 70% (原 60%)
    SampleType.CURRENT_ABSENT: 0.15,    # 15% (不变)
    SampleType.HISTORY_NOISY: 0.10,     # 10% (原 15%)
    SampleType.MIXED_HARD: 0.05,        # 5% (原 10%)
}
# Present = 70% + 10% = 80%
# Absent = 15% + 5% = 20%
# 约等于 8:2，接近 7:3
```

---

## 🔧 实施修正

### 修正 1：采样密度调整

**文件**：[cogtrack/training/sample_builder.py](../cogtrack/training/sample_builder.py)

```python
def get_samples_per_sequence(seq_length: int) -> int:
    """根据序列长度确定采样数量（调研优化版）"""
    if seq_length < 300:
        return 8    # 原 5
    elif seq_length < 1000:
        return 15   # 原 10
    else:
        return 30   # 原 20
```

### 修正 2：简洁状态描述

**文件**：[cogtrack/training/state_generator.py](../cogtrack/training/state_generator.py)

```python
def _build_state_update_prompt(self, ...) -> str:
    """构造状态更新生成的 prompt（简洁版）"""
    if target_status == "present" and bbox:
        status_desc = "The target is present."
    else:
        status_desc = "The target is absent."

    return f"""You are tracking: {initial_identity}

Previous state: {previous_state}

Current frame: {status_desc}

Generate a concise state update (<30 words):
- If present: describe action and position
- If absent: describe absence reason (e.g., "occluded", "out of view")
- If no significant change: keep previous state
- Forbidden: coordinates, frame numbers, background details, reasoning

State:"""
```

### 修正 3：正负样本比例

**文件**：[tracking/synthesize_vlt_v631_core_data.py](../tracking/synthesize_vlt_v631_core_data.py)

```python
sample_builder = SampleBuilder(
    max_frame_span=200,
    history_buffer_size=3,
    history_sample_interval=10,
    sample_type_ratios={
        SampleType.PURE_POSITIVE: 0.70,   # 调整
        SampleType.CURRENT_ABSENT: 0.15,
        SampleType.HISTORY_NOISY: 0.10,   # 调整
        SampleType.MIXED_HARD: 0.05,      # 调整
    },
)
```

---

## 📋 最佳实践（调研总结）

### ✅ 通用原则

1. **Region-based caption 优于全帧**（DTLLM-VLT, ChatTracker）
2. **事件驱动优于逐帧密集**（ATCTrack 启示）
3. **多教师 + 验证机制**（防偏差）
4. **显式 absent 监督**（VLT-v6.3.1 创新）
5. **简洁优于详细**（DUTrack 发现，< 30 词）

### ✅ 质量控制

**数据生成前**：
- [ ] Sampling plan 冻结并版本化
- [ ] 序列 train/val/test 完整划分
- [ ] 教师模型和 revision 固定
- [ ] Prompt 版本号明确（v6.3.1）

**数据生成中**：
- [ ] Present/absent 比例符合预期（8:2 或 7:3）
- [ ] History 严格早于 current
- [ ] 无未来信息泄漏
- [ ] 状态描述长度 < 30 词

**数据生成后**：
- [ ] 可视化检查 50-100 个样本
- [ ] 统计分析（帧间隔、样本类型分布）
- [ ] 文本质量审核（是否包含禁止内容）
- [ ] 计算 SHA-256 并记录 manifest

---

## 🆚 与相关工作的差异（最终版）

| 维度 | DTLLM-VLT | DUTrack | ReasoningTrack | **VLT-v6.3.1** |
|------|-----------|---------|----------------|----------------|
| 数据生成 | SAM + region-caption | 固定阈值更新 | 固定间隔 + IoU GRPO | **事件驱动 + 双教师** |
| 文本结构 | 单一描述 | 动态更新 | CoT + bbox | **身份-状态分离** |
| Absent 监督 | 未明确 | 未明确 | 未明确 | **显式 present/absent** |
| GRPO reward | N/A | N/A | 当前帧 IoU | **未来轨迹反事实 Delta-U** |
| 数据量 | 26K + 214K | 未明确 | 未明确 | **50K Core + 3-5K Memory** |

**核心优势**：
- 🔥🔥🔥 **TU-GRPO**：唯一优化"未来轨迹效用"的方法
- 🔥🔥🔥 **Identity-State 分离**：永久锚点 + 完整替换
- 🔥🔥 **事件驱动标注**：不逐帧 caption，重质不重量
- 🔥🔥 **Presence-aware**：显式 absent 监督，25% 负样本

---

## 📅 实施时间线（12 周）

### Week 1-2: Core SFT 数据生成
- ✅ 修改代码（采样密度、比例、prompt）
- ✅ 快速测试验证
- ✅ 完整数据生成（50K+ 样本）
- ✅ 质量检查

### Week 3-4: Core SFT 训练
- 训练 Qwen3-VL-4B
- CognitiveBench-Tiny 评测
- 验证基础能力（presence + bbox）

### Week 5-6: Memory 标签生成
- 事件候选挖掘（DINOv2 特征）
- 双教师生成（Qwen3-VL-32B × 2）
- 人工审核 500-1000 校准
- 输出 3-5K 高质量事件

### Week 7-8: Memory SFT
- 混合训练：70% core + 30% memory
- 因果评测：memory-on vs forced-null
- Full benchmark 验证

### Week 9-10: TU-GRPO 训练
- Reward replay 验证 Delta-U 分布
- GRPO 训练
- 最终评测

### Week 11-12: 消融与论文
- 完整消融实验
- 错误分析
- 论文撰写

---

## 🔗 相关文档

- [VLM Tracking 数据生成调研报告](/tmp/vlm_tracking_data_generation_research.md)
- [VLT-v6.3.1 数据生成文档](vlt_v631_data_generation.md)
- [实施进度追踪](implementation_progress.md)

---

**最后更新**：2026-08-14  
**状态**：待实施代码修正
