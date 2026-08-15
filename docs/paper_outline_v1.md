# CognitiveTrack 论文大纲 v1

> 目标会议：CVPR 2027 (假设 deadline ~2026-11-15)
> 当前日期：2026-08-14
> 状态：等待调研 agent 完成，补充最新文献后更新

---

## Title (候选)

1. **CognitiveTrack: Memory-Augmented Vision-Language Tracking with Trajectory-Utility Learning**
2. **Learning When to Remember: Trajectory-Utility GRPO for Long-Term VLM Tracking**
3. **Disentangled Identity-State Memory for Long-Term Vision-Language Tracking**

**倾向**：Option 1（综合性强，三个关键词都在）

---

## Abstract (草稿)

Long-term single-object tracking requires maintaining target identity across appearance changes, occlusions, and disappearances. While vision-language models (VLMs) offer semantic understanding through natural language descriptions, existing methods either use static captions that fail to adapt, or dynamic updates with fixed thresholds that ignore future utility. We present **CognitiveTrack**, a memory-augmented VLM tracker that learns when and how to update target-state memory by optimizing for future trajectory utility. 

Our key contributions include: (1) a **disentangled identity-state architecture** where permanent identity anchors prevent drift while replaceable state snapshots adapt to changes; (2) an **event-driven annotation pipeline** using dual teachers and stability verification to generate high-quality state labels without per-frame captioning; (3) **Trajectory-Utility GRPO (TU-GRPO)**, which learns update decisions by comparing counterfactual future trajectories with and without accepting the proposed state. Extensive experiments on CognitiveBench (995 sequences, 1.4M frames) show that CognitiveTrack achieves [X%] AUC improvement over base VLMs and [Y%] over fixed-threshold baselines, with [Z%] fewer updates and stronger identity consistency.

**关键指标待补充**：X, Y, Z 依赖实际实验结果。

---

## 1. Introduction

### 1.1 Opening: 问题动机

Long-term single-object tracking in unconstrained videos presents fundamental challenges beyond short-term tracking: targets undergo drastic appearance changes (viewpoint, pose, occlusion), temporarily leave the field of view, and reappear after significant gaps. Traditional trackers rely on visual feature matching, which degrades when appearance shifts exceed the model's invariance capacity.

Vision-language models (VLMs) promise a complementary approach: natural language descriptions can encode semantic identity ("a white dog with black ears and a red collar") that remains stable despite viewpoint changes. Recent works [DTLLM-VLT, DUTrack, MemVLT] have shown that language guidance improves tracking robustness. However, two critical questions remain unresolved:

1. **When to update memory?** Fixed-interval [DUTrack] or threshold-based [prior work] updates ignore whether the new state actually helps future frames.
2. **What to update?** Generic captions [prior work] may confuse similar objects; incremental descriptions accumulate contradictions.

### 1.2 Our Approach

We introduce **CognitiveTrack**, a VLM-based tracker that learns adaptive state memory through **trajectory-utility optimization**. Our core insight: a state update should be accepted only if it improves tracking performance on future frames, compared to keeping the old state.

**Key idea 1: Disentangled Identity-State Memory**
- Permanent identity anchor (from first frame): never overwritten
- Replaceable state snapshot: complete self-contained description, not incremental delta
- Prevents identity drift while allowing state adaptation

**Key idea 2: Event-Driven State Annotation**
- Mine candidate events (re-appearance, viewpoint change, long gaps)
- Dual-teacher generation + region-text verification + support-frame stability check
- Avoid per-frame captioning noise and same-class confusion

**Key idea 3: Trajectory-Utility GRPO (TU-GRPO)**
- For each candidate state, run counterfactual replay: accept vs keep old state
- Reward = future trajectory difference (presence F1, IoU, re-identification)
- Learn when updates truly help, not just current-frame correlation

### 1.3 Contributions

1. A unified three-image protocol (permanent template + temporal history strip + current frame) and disentangled identity-state memory that maintains long-term identity while adapting to state changes.

2. An event-driven annotation pipeline with dual-teacher consistency, region-text verification, and stability checks, producing high-quality state labels without per-frame captioning.

3. Trajectory-Utility GRPO: learning update decisions via counterfactual future trajectory comparison, optimizing for long-term utility rather than current-frame reward.

4. CognitiveBench: a long-term VLM tracking benchmark with 995 sequences (1.4M frames, 344K keyframes) from LaSOT, TNL2K, and MGIT, with frozen annotations for presence/absence and keyframe indicators.

5. Comprehensive experiments showing [具体数值待补充]: improved AUC, reduced over-updates, stronger identity consistency, and better re-identification after occlusions.

---

## 2. Related Work

### 2.1 Vision-Language Tracking

**Static language descriptions**:
- Early VLT works [cite] use class names or fixed captions
- DTLLM-VLT [2024]: multi-granularity region captions from SAM + Osprey
- Limitation: static text cannot adapt to appearance changes

**Dynamic language descriptions**:
- DUTrack [2025]: updates descriptions dynamically, shows concise > noisy detailed
- Uses fixed IoU threshold for updates → does not learn when to update
- MemVLT [2024]: long-short term memory, but does not separate identity vs state
- ChatTracker [2024]: uses grounding feedback to fix caption hallucination
- Our work: learns when to update via trajectory utility, separates permanent identity from dynamic state

**Language-guided localization**:
- CaptionFormer [2024]: temporal target caption generation, shows risk of same-class confusion
- ATCTrack [2024]: uses strong MLLM to cold-start missing labels with quality checks
- Our work: event-driven annotation + dual-teacher + region-text verification

### 2.2 Reinforcement Learning for Tracking

**GRPO/RL for tracking**:
- R1-Track [2025]: direct VLM tracking with rule-based GRPO reward
- ReasoningTrack [2025]: GRPO with current-frame IoU gain from updated text
- RELO [2024]: frame-level IoU + sequence-level AUC for RL tracking
- Limitation: current-frame reward does not measure whether memory update helps future
- Our work: counterfactual trajectory replay comparing accept vs keep old state

**Reasoning in video VLMs**:
- VideoAuto-R1 [2025]: video tasks do not always benefit from forced long reasoning
- Our work: direct structured output; memory_update as last field enables fast null path

### 2.3 Long-Term Single-Object Tracking

**Traditional approaches**: [brief survey of SOT, Siamese, transformer trackers]
- Strong short-term performance, degrade on long occlusions and disappearances

**Datasets**:
- LaSOT [2019], GOT-10k [2019], TNL2K [2021]: language annotations available
- Our CognitiveBench: focuses on long-term presence/absence and re-identification

---

## 3. Method

### 3.1 Overview

```
Input: Video frames + initial bbox + initial identity description
       ↓
[VLT-v6.3 Core SFT: learn presence + bbox]
       ↓
[Event-driven state annotation: dual teachers + verification]
       ↓
[Memory SFT: learn when to update + complete state snapshots]
       ↓
[TU-GRPO: optimize update decisions via counterfactual trajectory utility]
       ↓
Output: Per-frame presence, bbox, updated state memory
```

### 3.2 Long-Term Tracking Protocol

**Input modality** (fixed across all training stages):
- Image 1: Permanent initialization template with red bounding box (identity anchor)
- Image 2: Temporal strip of three recent accepted observations, oldest to newest, separated by white vertical bars
- Image 3: Current full frame without box (model must search)
- Text 1: Initial target identity (immutable)
- Text 2: Current maintained target state (replaceable, initialized to identity)

**Output structure**:
```json
{
  "target_status": "present",              // or "absent"
  "bbox_norm1000_xyxy": [x1, y1, x2, y2],  // null when absent
  "memory_update": null                    // or complete replacement state
}
```

**Design principles**:
1. **Permanent identity anchor**: Image 1 and initial text never overwritten → prevents drift
2. **Complete state replacement**: Updates are self-contained snapshots → avoids contradiction accumulation
3. **Last-field fast path**: `memory_update` at end → null is fast, non-null is deliberate
4. **Minimal system prompt**: Output protocol internalized via SFT → prompt focuses on task semantics

**System Prompt (v6.3.0)**:
> You are a long-term vision-language single-object tracker. Always use the target marked 
> by the red box in Image 1 and its initial description as the identity anchor; neither 
> history predictions nor state memory may overwrite the initialized identity. Using the 
> temporally ordered trajectory in Image 2 and the maintained target state, determine 
> whether the same target is present in Image 3, analyze its current state, and localize 
> it when present. Update the target-state memory only when a stable state change would 
> help future tracking.

**Why three images consistently?**
- Unified protocol for Base/Core/Memory/GRPO experiments → fair comparison
- No protocol switching between training stages → isolates contribution of supervision

### 3.3 Three-Stage Training

#### Stage 1: Core SFT (Presence + Localization)

**Supervision**: Ground-truth presence + bbox from LaSOT/TNL2K/MGIT train splits

**Loss masking**:
- Full supervision on `target_status` and `bbox_norm1000_xyxy`
- Field name `"memory_update":` supervised (maintains structure)
- Field value (e.g., `null`) masked out (do not supervise memory at this stage)

**Why mask memory value?**
- Avoids teaching "never update memory" on normal tracking samples
- Leaves room for future memory training without distribution shift

**Data sampling**:
- Present:absent ≈ 7:3 (matching real distribution)
- Reference/history strictly earlier than current frame
- ~20 samples per sequence, stratified by presence

#### Stage 2: Memory SFT (When & How to Update)

**Challenge**: No ground-truth memory labels in original datasets

**Solution: Event-Driven Annotation Pipeline**

**Step 1: Candidate Mining**
- Extract frozen visual features (DINOv2 or SigLIP) from target regions
- Identify events:
  - Significant embedding shift vs last accepted state
  - Re-appearance after absence
  - Long temporal gap since last observation
  - Stable viewpoint/pose change
- Also sample hard-nulls: ordinary motion, scale-only change, short blur

**Step 2: Dual-Teacher Generation**
- Teacher model (stronger than student, e.g., Qwen3-VL-32B)
- Inputs:
  - Current frame with GT bbox
  - Target region crop/mask
  - Current maintained state
  - **Support frames** (future 2-3 present frames) to verify stability
  - Distractor crops (same-class or high-similarity objects)
- Generate with two different seeds → check consistency

**Step 3: Verification**
- Teacher agreement on update/keep decision
- Region-text alignment score > threshold
- Target-text score > distractor-text score + margin
- New state consistent across support frames
- Rule checks: length, no coordinates, no forbidden words, identity consistency

**Step 4: Human Audit**
- Validation/test events: full human review
- Train events: stratified sampling (500-1000 seed events)
- Calibrate thresholds based on audit findings

**Data mixing**:
- 70% core tracking samples (memory value still masked)
- 30% memory samples with full supervision
- Within memory samples: ~25% update, ~75% hard-null

**Why event-driven, not per-frame?**
- Per-frame captioning is noisy and inefficient
- Most frames have no state change worth remembering
- Event-based focuses annotation budget on meaningful changes

#### Stage 3: Trajectory-Utility GRPO

**Motivation**: Current-frame reward (e.g., IoU) does not measure whether memory update helps future tracking.

**Core Innovation: Counterfactual Trajectory Replay**

For each candidate state `m'`, run two future trajectories:
```
Trajectory_accept = track(future_frames | initial_identity, state = m')
Trajectory_keep   = track(future_frames | initial_identity, state = old_state)

Delta-U-H = Utility(Trajectory_accept) - Utility(Trajectory_keep)
```

**Utility components**:
- Presence F1 on future H frames
- IoU/AUC on present frames
- Re-identification success after occlusion
- (weighted sum, tuned on validation)

**Reward function**:
```
R(y) = w_fmt   * R_format             (valid JSON, field order)
     + w_cur   * R_current            (current presence + bbox IoU)
     + w_evt   * R_event_consistency  (agreement with memory SFT labels)
     + w_ground* R_grounding          (region-text score, distractor margin)
     + w_traj  * Delta-U-H            (counterfactual trajectory utility)
     - l_update* P_over_update        (too frequent, no gain, absent update)
     - l_len   * P_length             (exceeds concise budget)
     - l_drift * P_identity_drift     (contradicts initial identity)
```

**Two-tier implementation** (for computational efficiency):

1. **Cached proxy replay** (most samples):
   - Use frozen region-text encoder
   - Compare text alignment on target/distractor crops across future frames
   - Fast approximation of trajectory utility

2. **True trajectory replay** (event-rich subset):
   - Freeze core+memory SFT checkpoint as evaluator
   - Run accept and keep branches with visual feature caching
   - Get real tracking IoU/presence on future H frames

**Why counterfactual?**
- Isolates causal effect of memory update on future performance
- Some updates improve current frame but hurt future (e.g., overfitting to transient)
- Optimizes for what memory is meant to do: help future tracking

**Stability measures**:
- Group samples by event; normalize advantage within group
- Skip if all rewards identical (zero variance)
- Reward evaluator frozen to avoid drift with actor
- Semantic deduplication to penalize update churn

### 3.4 Implementation Details

**Model**: Qwen3-VL-4B-Instruct (base), LoRA fine-tuning (rank 64, alpha 128)

**Training**:
- Core SFT: 3 epochs, lr 1e-4, batch 64 (with gradient accumulation)
- Memory SFT: 2 epochs, lr 5e-5, batch 32 + 32 (core + memory mixed)
- TU-GRPO: 1 epoch, lr 1e-5, group size 4-8, KL penalty 0.01

**Inference**:
- vLLM backend for efficiency
- Sparse observation policy: keyframe-only (follows dataset keyframe annotations)
- Visual history: most recent 3 accepted present observations
- Semantic memory: last accepted non-null state update

**Data splits**: Train/val/test by complete sequences (no frame leakage)

---

## 4. Experiments

### 4.1 Experimental Setup

**Benchmark: CognitiveBench v1**
- 995 sequences: LaSOT-test (280), TNL2K-test (700), MGIT-val (15)
- 1,408,438 total frames; 343,616 keyframes
- Frozen presence/absence annotations
- CognitiveBench-Tiny: 24 sequences for rapid iteration

**Metrics**:
- **Tracking**: AUC, OP50, OP75, Precision@20, Pnorm@0.2
- **Sparse modes**: 
  - hold-last (last valid bbox propagated to non-observed frames)
  - observation-only (score only on VLM-observed keyframes)
- **Presence**: Precision, Recall, F1; Absent FPR; Re-identification rate
- **Memory**: Update F1, Over-update rate, Miss-update rate
- **Quality**: Target-text score, distractor margin, identity contradiction rate
- **Causal**: Delta-AUC@H, Delta-presence-F1@H (after memory update)
- **Efficiency**: Updates per 100 frames, generated tokens, latency

**Baselines**:
- Qwen3-VL-4B Base (zero-shot)
- Core SFT only
- Core + Memory SFT
- Core + Memory + Current-frame GRPO (ablation: not counterfactual)
- Fixed-threshold update (IoU drop > 0.3)
- Oracle event labels (upper bound)

**Comparisons** (if code/checkpoints available):
- DTLLM-VLT [cite]
- DUTrack [cite]
- R1-Track [cite]

### 4.2 Main Results

**Table 1: Main Tracking Performance on CognitiveBench Full**

| Method | AUC (hold-last) | AUC (obs-only) | OP50 | Presence F1 | Update Rate | Identity Drift |
|--------|-----------------|----------------|------|-------------|-------------|----------------|
| Qwen3-VL-4B Base | [X] | [Y] | [Z] | [A] | 0.0 | [B] |
| + Core SFT | [X+Δ1] | [Y+Δ1] | [Z+Δ1] | [A+Δ1] | 0.0 | [B-Δ1] |
| + Memory SFT | [X+Δ2] | [Y+Δ2] | [Z+Δ2] | [A+Δ2] | [R1] | [B-Δ2] |
| + TU-GRPO (ours) | **[X+Δ3]** | **[Y+Δ3]** | **[Z+Δ3]** | **[A+Δ3]** | **[R2<R1]** | **[B-Δ3]** |
| Fixed threshold | [X+Δ2'] | [Y+Δ2'] | [Z+Δ2'] | [A+Δ2'] | [R1'>R1] | [B-Δ2'] |

**Expected findings** (待验证):
- Core SFT: large gain over Base (learns presence + bbox)
- Memory SFT: further gain + non-zero update rate
- TU-GRPO: best AUC + fewer updates than fixed threshold + lowest drift
- Fixed threshold: comparable AUC but higher update rate (over-updates)

**Table 2: Comparison with Published Methods**

| Method | AUC (LaSOT) | AUC (TNL2K) | OP50 | Notes |
|--------|-------------|-------------|------|-------|
| DTLLM-VLT [cite] | [X] | [Y] | [Z] | Static multi-granularity captions |
| DUTrack [cite] | [X'] | [Y'] | [Z'] | Fixed-threshold dynamic update |
| CognitiveTrack (ours) | **[X'']** | **[Y'']** | **[Z'']** | Learned update via TU-GRPO |

### 4.3 Ablation Studies

**Table 3: Core Design Ablations**

| Ablation | AUC | Presence F1 | Update F1 | Identity Drift |
|----------|-----|-------------|-----------|----------------|
| Full model | **[X]** | **[Y]** | **[Z]** | **[D]** |
| w/o identity anchor constraint | [X-Δ1] | [Y-Δ1] | [Z] | [D+Δ1] ↑ |
| w/o identity-state split | [X-Δ2] | [Y-Δ2] | [Z] | [D+Δ2] ↑ |
| Incremental state (not complete) | [X-Δ3] | [Y-Δ3] | [Z-Δ3] | [D+Δ3] ↑ |
| Fixed history size=1 | [X-Δ4] | [Y-Δ4] | [Z] | [D] |

**Table 4: Memory Annotation Ablations**

| Annotation Method | Memory Label Quality | Downstream AUC | Over-update |
|-------------------|---------------------|----------------|-------------|
| Event-driven dual-teacher (ours) | **[Score1]** | **[AUC1]** | **[OU1]** |
| Per-frame single-teacher | [Score2] | [AUC2] | [OU2] ↑ |
| Generic image caption | [Score3] | [AUC3] | [OU3] ↑↑ |
| No region verification | [Score4] | [AUC4] | [OU4] ↑ |

**Table 5: GRPO Reward Ablations**

| Reward Configuration | AUC | Delta-AUC@H | Update F1 | Over-update |
|---------------------|-----|-------------|-----------|-------------|
| Full TU-GRPO (counterfactual) | **[X]** | **[Δ1]** | **[F1]** | **[OU1]** |
| w/o trajectory utility (current IoU only) | [X-Δ2] | [Δ2] | [F2] | [OU2] ↑ |
| w/o event consistency | [X-Δ3] | [Δ3] | [F3] ↓ | [OU3] |
| w/o update penalty | [X-Δ4] | [Δ4] | [F4] | [OU4] ↑↑ |
| w/o identity drift penalty | [X-Δ5] | [Δ5] | [F5] | [OU5] (drift ↑) |

### 4.4 Qualitative Analysis

**Figure 3: Successful State Update Examples**
- Re-appearance after long occlusion
- Viewpoint change (front → side → rear)
- State change with identity preserved

**Figure 4: Failure Case Analysis**
- False update on transient lighting change
- Miss update on subtle pose shift
- Identity confusion with same-class distractor

**Figure 5: Counterfactual Trajectory Comparison**
- Case where accept > keep: viewpoint change → better future IoU
- Case where keep > accept: transient occlusion → accepting hurts future

### 4.5 Efficiency Analysis

**Table 6: Computational Cost**

| Method | Updates/100 frames | Tokens/frame | Latency (ms) | Total Time (s/seq) |
|--------|-------------------|--------------|--------------|-------------------|
| Qwen3-VL Base | 0 | ~150 | [L1] | [T1] |
| + Memory SFT (fixed threshold) | [U1] | ~160 | [L2] | [T2] |
| + TU-GRPO (ours) | **[U2<U1]** | ~155 | [L3] | [T3] |

---

## 5. Discussion & Limitations

### 5.1 Key Findings

1. **Disentangled identity-state memory** significantly reduces identity drift (Table 3)
2. **Event-driven annotation** produces higher-quality labels than per-frame captioning (Table 4)
3. **Trajectory-utility GRPO** achieves better AUC with fewer updates than fixed thresholds (Table 1, 5)
4. **Counterfactual comparison** reveals that some current-frame "good" updates hurt future tracking (Figure 5)

### 5.2 Limitations

1. **Computational cost**: Counterfactual replay adds overhead (can be mitigated with caching)
2. **Teacher dependency**: Memory annotation requires stronger teacher model
3. **Support frame assumption**: Stability verification assumes 2-3 future frames are available (offline only)
4. **Single-object scope**: Extends to multi-object tracking non-trivially
5. **Benchmark coverage**: CognitiveBench focuses on existing datasets; new domains may differ

### 5.3 Future Work

- Extend to multi-object tracking with relational memory
- Online state annotation (without future support frames)
- Hierarchical memory (short-term events + long-term identity)
- Cross-modal fusion with audio, text, or depth

---

## 6. Conclusion

We presented CognitiveTrack, a memory-augmented VLM tracker that learns adaptive state updates through trajectory-utility optimization. Our disentangled identity-state architecture prevents drift while enabling adaptation; event-driven annotation produces high-quality labels without per-frame captioning; and Trajectory-Utility GRPO learns when updates truly help by comparing counterfactual future trajectories. Experiments on CognitiveBench demonstrate that CognitiveTrack achieves [X%] AUC improvement with [Y%] fewer updates and stronger identity consistency. We release code, models, and benchmark annotations to facilitate future research.

---

## Appendix

### A. Full System Prompt

[完整 Prompt 文本]

### B. Data Statistics

[数据分布表格、事件类型统计、源数据集覆盖]

### C. Annotation Quality Examples

[人工审核一致性、教师 agreement 示例、验证边界 case]

### D. Additional Ablations

[Prompt 详细度、视觉 reference mode、历史布局等]

### E. Hyperparameters

[完整训练超参数表]

### F. Reproducibility Checklist

[代码、数据、模型、配置的发布计划]

---

## TODO 与优先级

### 🔥 Critical Path (必须有)

1. **完成 Core SFT 训练与 Base/Core 对比**
   - [ ] 生成全量 VLT-v6.3 core 数据
   - [ ] 训练 Qwen3-VL-4B Core checkpoint
   - [ ] Tiny 评测 → 得到 Table 1 前两行数据

2. **实现 Memory 标签流水线**
   - [ ] Candidate mining 工具
   - [ ] Dual-teacher annotation 工具
   - [ ] Verification pipeline
   - [ ] 人工审核 500-1000 事件
   - [ ] 导出 ms-swift JSONL

3. **Memory SFT 训练与因果评测**
   - [ ] 训练 Core+Memory checkpoint
   - [ ] Memory-on vs forced-null 对比
   - [ ] 得到 Table 1 第三行和 Table 3 部分数据

4. **TU-GRPO 实现与训练**
   - [ ] Reward replay 验证（不训练，先看 Delta-U 是否合理）
   - [ ] 如果可行，实现缓存代理 + 真实回放
   - [ ] 训练完整 TU-GRPO
   - [ ] 得到 Table 1 第四行和 Table 5

5. **Full 评测与核心消融**
   - [ ] 在 CognitiveBench Full 995 序列运行所有 checkpoints
   - [ ] Identity anchor ablation
   - [ ] Identity-state split ablation
   - [ ] 至少完成 Table 3 关键行

### ⚡ Strongly Recommended (强烈建议)

6. **Baseline 对比**
   - [ ] 调研 DTLLM-VLT/DUTrack 是否有公开代码/checkpoint
   - [ ] 如果有，在 CognitiveBench 上运行 → Table 2
   - [ ] 如果没有，至少实现 fixed-threshold baseline

7. **定性分析**
   - [ ] 挑选成功 case（Figure 3）
   - [ ] 挑选失败 case（Figure 4）
   - [ ] 反事实可视化（Figure 5）

8. **补充消融**
   - [ ] Annotation method ablation (Table 4)
   - [ ] Reward ablation 完整矩阵 (Table 5)
   - [ ] Efficiency analysis (Table 6)

### ✅ Nice to Have (加分项)

9. **Prompt ablation**
   - [ ] Minimal vs detailed prompt
   - [ ] Visual vs textual reference
   - [ ] History layout variants

10. **Extended analysis**
    - [ ] Per-category performance breakdown
    - [ ] Cross-dataset generalization (LaSOT train → TNL2K test)
    - [ ] Occlusion length vs re-identification success

---

## 时间规划（假设 CVPR 2027 deadline ~2026-11-15）

| 周次 | 任务 | 交付物 |
|-----|------|--------|
| Week 1-2 (Aug 14-27) | Core SFT 数据 + 训练 + Tiny 评测 | Base/Core 对比数据 |
| Week 3-4 (Aug 28 - Sep 10) | Memory 标签工具 + 人工审核 + Memory SFT | Core+Memory checkpoint |
| Week 5-6 (Sep 11-24) | TU-GRPO reward replay + 实现 + 训练 | 完整 TU-GRPO checkpoint |
| Week 7-8 (Sep 25 - Oct 8) | Full 评测 + 核心消融 | Table 1, 3, 5 主数据 |
| Week 9-10 (Oct 9-22) | Baseline 对比 + 补充消融 + 定性分析 | Table 2, 4, 6 + Figures |
| Week 11 (Oct 23-29) | 论文初稿 + Internal review | Draft v1 |
| Week 12 (Oct 30 - Nov 5) | 论文打磨 + 补充实验 | Draft v2 |
| Week 13 (Nov 6-12) | Final polish + Submission prep | Camera-ready draft |
| Nov 15 | **Submission** | 🎉 |

**风险缓冲**：
- 如果 TU-GRPO 不 work，Week 5-6 可以降级为 current-frame GRPO + event reward
- 如果 baselines 无法复现，强化 fixed-threshold 对比
- 预留 Week 12-13 处理 reviewer 可能要求的补充实验（虽然是 submission 前）

---

**下一步**：等待调研 agent 完成，补充最新文献综述和社区 SOTA 数据，更新 Related Work 和 Table 2 baseline 目标。
