# CognitiveTrack 关键优化建议清单（归档）

> **归档说明（2026-08-15）：** 这是生成正式 Core 数据之前的建议集合；已实现与未实现
> 内容混杂，不得作为当前完成度证据。

> 分析日期：2026-08-14
> 目标：基于已有文档和方法，给出可执行的优化方向和顶会投稿准备

---

## 一、当前方案的核心优势（已确认）

### ✅ 1. 方法学完整性
- **三阶段训练路线清晰**：Core SFT → Memory SFT → TU-GRPO
- **每阶段职责明确**：先学跟踪、再学记忆、最后优化更新决策
- **监督边界严格**：GT 只管 presence/bbox，教师只管状态，不混淆

### ✅ 2. 协议设计合理
- **固定三图输入**：所有实验共用协议，公平可比
- **永久身份锚点**：防止长时漂移
- **完整替换状态**：避免增量矛盾累积
- **极简 System Prompt**：SFT 内化协议，推理时高效

### ✅ 3. 数据合成体系
- **确定性采样计划**：可重放、可审计
- **事件驱动标注**：避免全帧 caption 噪声
- **双教师验证**：提升标签质量
- **字段级 loss mask**：Core 阶段不伪造记忆监督

### ✅ 4. 工程质量
- **代码结构清晰**：pytracking 生命周期 + cogtrack 组件化
- **配置版本化**：prompt version、history layout、bbox protocol 都有版本号
- **实验可复现**：manifest、checksum、processor replay

---

## 二、亮点与创新点（需要实验证明）

### 🔥 核心创新候选：Trajectory-Utility GRPO

**What**: 对同一候选状态做"接受 vs 保留旧状态"的未来轨迹反事实回放

**Why it's novel**:
- ReasoningTrack 用当前帧 IoU 增益 → 无法衡量"记忆对未来的帮助"
- R1-Track 用规则 reward → 未优化状态更新时机
- TU-GRPO 优化的是"未来轨迹效用差" → 真正学习"何时更新对后续有益"

**需要证明的假设**:
1. Delta-U (counterfactual) > current-frame IoU (ReasoningTrack 风格)
2. 学到的更新决策 > 固定阈值
3. 更新后的 Delta-AUC@H 显著为正

**关键消融**:
```
Memory SFT alone                           [baseline]
+ current-frame IoU GRPO                   [ablation 1]
+ event consistency reward                 [ablation 2]
+ cached trajectory reward                 [ablation 3]
+ true counterfactual replay (ours)        [full method]
```

**如果不 work 的降级方案**:
- 降级为 current-frame GRPO + event reward + update penalty
- 强调 "event-driven annotation + disentangled memory" 作为主贡献
- TU-GRPO 作为 "future direction"

---

### ⚡ 重要贡献：Disentangled Identity-State Memory

**What**: 永久身份锚点 + 可替换动态状态

**Why it matters**:
- DUTrack 等动态更新文本，但未显式防止身份覆盖
- MemVLT 有记忆，但未区分永久 vs 临时信息
- CognitiveTrack 明确分离 → 可审计身份一致性

**需要证明**:
1. Identity drift rate: with anchor < without anchor
2. Re-identification after long occlusion: 分离设计 > 单一动态文本
3. Contradiction rate: 完整替换 < 增量描述

**关键消融**:
```
Single dynamic text (no anchor)            [ablation 1]
Identity + incremental delta               [ablation 2]
Identity + complete replacement (ours)     [full method]
```

---

### ⚡ 重要贡献：Event-Driven State Annotation

**What**: 候选挖掘 + 双教师 + 区域验证 + 稳定性检查

**Why better than alternatives**:
- Per-frame caption: 噪声大、效率低、同类混淆
- Single-teacher: 可能幻觉或偏差
- 无区域验证: 描述可能对应背景或干扰物

**需要证明**:
1. Annotation quality: dual-teacher + verification > single-teacher
2. Downstream tracking: event-driven > per-frame baseline
3. Over-update rate: 有验证 < 无验证

**关键对比**:
```
Per-frame single-teacher caption           [baseline 1]
Event-driven single-teacher                [baseline 2]
Event-driven dual-teacher (ours)           [full method]
+ region-text verification                 [full method]
```

---

## 三、当前最缺的内容

### ❌ 1. 定量动机证据

**问题**: Introduction 说"现有方法不够"，但缺乏定量支撑

**需要补充**:
1. **Zero-shot VLM 失败分析**:
   ```
   Qwen3-VL-4B Base on CognitiveBench-Tiny:
   - Presence F1: [X]
   - AUC (hold-last): [Y]
   - Identity drift rate: [Z]%
   - Failure modes: [长时遮挡、重现恢复、同类混淆]
   ```

2. **固定描述 vs 动态状态的 ablation**:
   ```
   Identity only (no state update):        AUC = [A]
   Identity + dynamic state (w/ update):   AUC = [B > A]
   Improvement:                            +[B-A]%
   
   → 证明"动态状态确实有必要"
   ```

3. **固定阈值的局限性量化**:
   ```
   Fixed threshold (IoU drop > 0.3):
   - Over-update rate: [X]%
   - Miss-update rate: [Y]%
   - Update precision: [Z]%
   
   Learned update (TU-GRPO):
   - Over-update rate: [X' < X]%
   - Miss-update rate: [Y' < Y]%  
   - Update precision: [Z' > Z]%
   ```

**执行时机**: Core SFT 完成后，立即做这些 baseline 实验

---

### ❌ 2. 已有工作的详细对比

**问题**: research_plan.md 列了很多工作，但缺乏：
- 他们的详细实验设置（数据、模型、训练方式）
- 在相同 benchmark 上的直接对比
- 明确的优劣势对比

**等待调研 agent 完成后补充**:
1. DTLLM-VLT/DUTrack/R1-Track 的完整方法细节
2. 他们在 LaSOT/TNL2K 上的 SOTA 数据
3. 是否有公开代码/checkpoint 可复现
4. CognitiveTrack 的差异化优势表格

**如果无法复现已有工作**:
- 至少实现 strong baselines: fixed-threshold update、per-frame caption
- 在 Related Work 中详细对比设计差异
- 在 Discussion 中说明复现困难（代码不公开、依赖私有数据等）

---

### ❌ 3. TU-GRPO 的可行性验证

**问题**: 这是核心创新，但尚未实现和验证

**高风险点**:
1. 双分支回放计算成本是否可承受？
2. Delta-U 是否真的比 current-frame reward 更好？
3. 缓存代理是否足够准确？

**建议验证顺序**:
1. **Week 1 (最优先)**:
   - 先做 reward replay（不训练），只计算 Delta-U
   - 在 100-200 个事件上手动检查：
     - Delta-U > 0 的案例：更新后未来 AUC 是否真的更高？
     - Delta-U < 0 的案例：更新是否确实伤害了未来？
   - 如果 Delta-U 排序不合理 → 尽早降级方案

2. **Week 2-3**:
   - 实现缓存代理版本
   - 对比 cached Delta-U vs true Delta-U 的相关性
   - 如果相关性低，必须用真实回放（成本高）

3. **Week 4-5**:
   - 真实训练，看 reward 是否收敛
   - 看最终 AUC 是否比 Memory SFT 提升

4. **如果不 work**:
   - 降级为 current-frame GRPO + event + ground reward
   - 论文中说 "future work: trajectory-utility optimization"
   - 主贡献改为 "disentangled memory + event annotation"

---

### ❌ 4. 消融实验的完整矩阵

**问题**: 设计了很多消融，但尚未执行

**优先级排序**:

**🔥 Must Have (论文主表必须有)**:
```
Table 1: Main Results
- Base
- Core SFT
- Core + Memory SFT
- Core + Memory + TU-GRPO (full)

Table 3: Core Design Ablations
- Full model
- w/o identity anchor constraint
- w/o identity-state split
- Incremental state (not complete replacement)

Table 5: GRPO Reward Ablations
- Memory SFT alone
- + current-frame IoU GRPO
- + event consistency reward
- + trajectory utility (cached)
- + trajectory utility (true replay) [full]
```

**⚡ Strongly Recommended (审稿人可能要求)**:
```
Table 4: Annotation Method Ablations
- Event-driven dual-teacher (ours)
- Per-frame single-teacher
- No region verification
- No stability check

Table 2: Comparison with Prior Work
- DTLLM-VLT (if reproducible)
- DUTrack (if reproducible)
- Fixed-threshold baseline (必须有)
```

**✅ Nice to Have (加分但非必需)**:
```
Prompt ablations
Visual vs textual reference
History layout variants
Cross-dataset generalization
```

---

## 四、数据与训练的优化方向

### 1. Core SFT 数据质量保证

**当前状态**: 方案完善，待训练服务器执行

**建议检查项**:
```bash
# 数据生成后必须验证
python tracking/validate_sft_supervision.py \
  --profile tracking_core \
  --dataset train.jsonl val.jsonl

python tools/verify_qwen_grounding_templates.py \
  --dataset-root /path/to/data \
  --qwen3-model /path/to/model \
  --verify-tracking-core-mask

# 手动抽查 10-20 个样本
- Present/absent 标签是否正确？
- Bbox 是否合理？
- History 是否严格早于 current？
- Loss mask 是否正确（memory_update 值被 mask）？
```

**数据分布检查**:
```python
# 统计并报告
- Present:absent ratio per dataset
- Samples per sequence distribution
- History size distribution (should all be 3 after padding)
- Bbox validity rate
- Text description coverage (LaSOT/TNL2K 有, MGIT 可能缺)
```

---

### 2. Memory 标签的冷启动策略

**当前状态**: 设计完善，代码待实现

**建议执行顺序**:
1. **先做人工审核集 (Week 3)**:
   - 从 validation split 挑选 100-200 个候选事件
   - 人工标注 update/keep + 完整状态文本
   - 作为 gold standard 校准自动标注阈值

2. **然后做双教师自动标注 (Week 3-4)**:
   - 在 train split 生成银标
   - 与人工审核集对比，调整阈值
   - 目标: teacher agreement > 85%, region-text score > 0.7, distractor margin > 0.15

3. **分层抽检 (Week 4)**:
   - 按 dataset / event type / target class 分层
   - 每层抽 50-100 样本人工复核
   - 计算 precision / recall / F1

4. **只在质量达标后生成全量数据**:
   - 如果 precision < 80%, 重新调整阈值或教师 prompt
   - 宁可少而精，不要多而噪

**Quality metrics to track**:
```
Teacher agreement rate:        [target: >85%]
Region-text alignment:         [target: >0.7]
Distractor margin:             [target: >0.15]
Human audit precision:         [target: >80%]
Human audit recall:            [target: >70%]
Identity consistency:          [target: >95%]
```

---

### 3. 训练稳定性与过拟合检查

**当前缺失**: 训练监控和早停策略

**建议补充**:
1. **Core SFT**:
   ```python
   # 每 epoch 记录
   - Train loss (overall, presence, bbox)
   - Val loss + presence F1 + bbox IoU
   - Format error rate
   - Early stop if val loss 不降 3 epochs
   ```

2. **Memory SFT**:
   ```python
   # 除了上述，还要记录
   - Update F1 (precision, recall)
   - Over-update rate (update but no change in GT)
   - Miss-update rate (should update but didn't)
   - Identity contradiction rate
   ```

3. **TU-GRPO**:
   ```python
   # 记录
   - Average reward per component
   - KL divergence vs Memory SFT
   - Update rate distribution
   - Delta-AUC@H on validation events
   ```

4. **Over-fitting 检查**:
   ```bash
   # 每个 checkpoint 都在 Tiny 上快速评测
   # 如果 val 指标下降但 train 还在涨 → early stop
   ```

---

## 五、论文撰写的优化建议

### 1. Title 优化

**当前候选**:
```
1. CognitiveTrack: Memory-Augmented Vision-Language Tracking 
   with Trajectory-Utility Learning
2. Learning When to Remember: Trajectory-Utility GRPO for 
   Long-Term VLM Tracking
3. Disentangled Identity-State Memory for Long-Term 
   Vision-Language Tracking
```

**建议**: 
- 如果 TU-GRPO 实验成功 → Option 1 或 2
- 如果 TU-GRPO 不如预期 → Option 3
- 保持 "Long-Term" + "Vision-Language Tracking" 关键词

---

### 2. Abstract 结构优化

**当前草稿已有**: 问题、方法、贡献、结果（待补充数值）

**建议强化**:
1. **Opening**: 增加一句定量 pain point
   ```
   "While recent VLM trackers show promise, they suffer from X% identity 
   drift in long occlusions and Y% over-update rate with fixed thresholds."
   ```

2. **Key insight**: 更清晰地说明 "为什么 trajectory utility"
   ```
   "Our key insight: a state update should be accepted only if it improves 
   tracking on future frames, not just current-frame correlation."
   ```

3. **Results**: 用具体数值
   ```
   "Experiments on CognitiveBench show 8.3% AUC improvement over base VLMs, 
   5.1% over fixed-threshold updates, with 40% fewer updates and 65% lower 
   identity drift rate."
   ```

---

### 3. Related Work 的对比表格

**建议补充**: 在 Related Work 末尾加一张综合对比表

```markdown
**Table 0: Comparison of VLM Tracking Methods**

| Method | Identity Preservation | Dynamic Update | Update Decision | Training |
|--------|----------------------|----------------|-----------------|----------|
| DTLLM-VLT [2024] | Static multi-granular | ❌ | - | SFT |
| DUTrack [2025] | ❌ | ✅ Fixed threshold | IoU drop | SFT |
| MemVLT [2024] | Short-term only | ✅ | Heuristic | SFT |
| R1-Track [2025] | Static | ❌ | - | GRPO (format) |
| ReasoningTrack [2025] | Static | ✅ Fixed interval | Current IoU | GRPO |
| **CognitiveTrack (ours)** | ✅ Permanent anchor | ✅ Learned | **Future utility** | SFT + GRPO |
```

---

### 4. Method 章节的可视化

**当前缺失**: 方法图

**建议补充**:

**Figure 1: Overall Pipeline**
```
[Video + Init bbox]
        ↓
┌───────────────────────────┐
│  Three-Image Protocol     │
│  ┌─────┐ ┌─────┐ ┌─────┐ │
│  │ Ref │ │Hist │ │Curr │ │
│  └─────┘ └─────┘ └─────┘ │
│  Identity | State         │
└───────────────────────────┘
        ↓
┌───────────────────────────┐
│  VLM Tracker              │
│  Presence + Bbox + Update │
└───────────────────────────┘
        ↓
┌───────────────────────────┐
│  Memory Gate              │
│  Accept? → State Snapshot │
└───────────────────────────┘
```

**Figure 2: TU-GRPO Counterfactual Replay**
```
         Candidate State m'
                ↓
        ┌───────┴───────┐
        ↓               ↓
   Accept m'        Keep old
        ↓               ↓
Future trajectory   Future trajectory
   [t+1...t+H]       [t+1...t+H]
        ↓               ↓
     U_accept        U_keep
        └───────┬───────┘
                ↓
         Delta-U = U_accept - U_keep
                ↓
         GRPO Reward Component
```

**Figure 3: Event-Driven Annotation**
```
Train sequences
        ↓
Candidate mining (embedding shift, re-appearance)
        ↓
Dual-teacher generation (different seeds)
        ↓
Verification (region-text, distractor, stability)
        ↓
Human audit (stratified sampling)
        ↓
High-quality state labels
```

---

## 六、执行 Checklist

### Week 1-2: Core SFT 基线

- [ ] 训练服务器生成 VLT-v6.3 core 全量数据
- [ ] 验证数据质量（validate_sft_supervision.py + processor replay）
- [ ] 训练 Core SFT checkpoint
- [ ] Tiny 评测: Base vs Core
- [ ] **产出**: Table 1 前两行，Figure 3 成功 case 素材

### Week 3-4: Memory 标签与训练

- [ ] 实现 mine_memory_events.py
- [ ] 实现 annotate_target_states.py
- [ ] 实现 verify_target_states.py
- [ ] 人工审核 100-200 validation events
- [ ] 生成 train 银标
- [ ] 训练 Memory SFT checkpoint
- [ ] Tiny 评测: Core+Memory vs forced-null
- [ ] **产出**: Table 1 第三行, Table 4 部分数据

### Week 5-6: TU-GRPO 可行性验证

- [ ] Reward replay (不训练): 计算 100-200 events 的 Delta-U
- [ ] 手动检查 Delta-U 排序是否合理
- [ ] **决策点**: 如果不合理，降级为 current-frame GRPO
- [ ] 实现缓存代理版本
- [ ] 实现真实回放版本（如果缓存不够准）
- [ ] 训练 TU-GRPO checkpoint
- [ ] **产出**: Table 1 第四行, Table 5 完整数据

### Week 7-8: Full 评测与核心消融

- [ ] 在 CognitiveBench Full 995 序列运行所有 checkpoints
- [ ] Identity anchor ablation
- [ ] Identity-state split ablation
- [ ] Complete replacement vs incremental ablation
- [ ] Annotation method ablation
- [ ] **产出**: Table 1, 3, 4, 5 完整数据

### Week 9-10: Baseline 对比与定性分析

- [ ] 固定阈值 baseline 实现与评测
- [ ] 尝试复现 DTLLM-VLT/DUTrack（如果有代码）
- [ ] 挑选成功/失败 case
- [ ] 反事实可视化
- [ ] Efficiency analysis
- [ ] **产出**: Table 2, 6, Figure 3, 4, 5

### Week 11: 论文初稿

- [ ] Introduction
- [ ] Related Work + Table 0 对比
- [ ] Method + Figure 1, 2, 3
- [ ] Experiments + All tables & figures
- [ ] Discussion & Conclusion
- [ ] **产出**: Draft v1

### Week 12-13: Polish & Submission

- [ ] Internal review
- [ ] 补充实验（如果有 gap）
- [ ] Appendix
- [ ] Reproducibility checklist
- [ ] **产出**: Camera-ready draft → Submission

---

## 七、风险与应对

### Risk 1: TU-GRPO 不 work

**Symptom**: Delta-U 与真实 future AUC 不相关

**Mitigation**:
- 降级为 current-frame IoU + event consistency + ground reward
- 主贡献改为 "disentangled memory + event annotation"
- TU-GRPO 作为 "promising future direction"

**Impact**: 仍是 solid 工作，但少了一个 major novelty

---

### Risk 2: 无法复现已有 baseline

**Symptom**: DTLLM-VLT/DUTrack 无公开代码

**Mitigation**:
- 实现 strong baselines: fixed-threshold, per-frame caption
- Related Work 详细对比设计差异
- Discussion 说明复现困难

**Impact**: Table 2 只有 fixed-threshold 对比，不是最 strong

---

### Risk 3: 时间不够

**Symptom**: 11 月临近，实验未完成

**Mitigation**:
- 砍掉 Nice-to-have ablations (Prompt, history layout)
- 只保留 Must-have (Table 1, 3, 5)
- Tiny 先出结果，Full 后补（虽然不理想）

**Impact**: 论文完整度下降，但核心结论仍成立

---

## 八、总结：当前最优先的三件事

### 🔥 Priority 1: 完成 Core SFT 并建立 baseline

**Why**: 这是一切的基础
- 没有 Core checkpoint，无法做 Memory SFT
- 没有 Base vs Core 对比，无法证明训练有效

**What**:
- 生成数据 → 训练 → Tiny 评测
- 得到第一组真实数字：Base AUC [X], Core AUC [X+Δ]

**When**: Week 1-2 (最优先)

---

### 🔥 Priority 2: TU-GRPO 可行性早期验证

**Why**: 这是最大的不确定性
- 如果 Delta-U 不合理，必须尽早知道并降级
- 不要等到 Week 5 才发现不 work

**What**:
- Week 2 结束前，做 100 个事件的 reward replay
- 手动检查 Delta-U > 0 的案例是否真的 future 更好

**When**: Week 2 末（与 Core SFT 并行）

---

### 🔥 Priority 3: 等待调研 agent，补充 Related Work

**Why**: 需要知道最新 SOTA 和可复现 baseline
- 哪些工作可以直接对比？
- 哪些需要自己实现 baseline？
- 社区对 VLM tracking 的接受度如何？

**What**:
- 调研 agent 完成后，更新 paper_outline_v1.md
- 补充 Table 2 baseline 目标
- 更新 Related Work 对比表

**When**: 等待 agent 完成（今天）

---

**Next Step**: 调研 agent 完成后，立即整合结果并更新文档，然后开始 Core SFT 数据生成。
