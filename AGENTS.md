# CognitiveTrack Agent Instructions

**Last Updated**: 2026-08-14  
**Project**: CognitiveTrack - VLM-based Long-term Object Tracking

---

## 🎯 Current Status (2026-08-14)

### ✅ Phase 1: VLT-v6.3.1 Core SFT Framework - COMPLETED

**Achievement**: Complete data generation framework with research-validated innovations

**Key Deliverables**:
1. ✅ Prompt v6.3.1 optimization (simplified 3-image protocol)
2. ✅ MGIT action-layer text extraction (official approach)
3. ✅ Complete data generation framework (6 modules)
4. ✅ VLM tracking research report (8 major works analyzed)
5. ✅ Local smoke test (all passed)

**Target**: 50K+ Core SFT samples, 80:20 present/absent ratio

**Next Step**: Deploy to training server → full data generation (10-20 hours)

---

## 🔥 Core Innovations (Research-Validated)

### 1. **TU-GRPO (Trajectory-Utility GRPO)** 🔥🔥🔥
- **Innovation**: Counterfactual future trajectory utility
- **vs ReasoningTrack**: Current-frame IoU gain
- **Formula**: `Delta-U-H = U(future | accept) - U(future | keep)`

### 2. **Identity-State Disentanglement** 🔥🔥🔥
- **Permanent identity anchor**: First frame, never overwritten
- **Dynamic state complete replacement**: Not incremental, avoid contradiction accumulation
- **Anti-drift mechanism**: Image 1 always present as identity reference

### 3. **Event-Driven State Annotation** 🔥🔥
- **Not per-frame captioning**: Event candidate mining
- **Dual-teacher generation**: Qwen3-VL-32B × 2, consistency acceptance
- **Multi-dimensional verification**: Region-text alignment, distractor margin, stability
- **Quality over quantity**: 3-5K high-quality events vs tens of thousands per-frame

### 4. **Presence-Aware Protocol** 🔥🔥
- **Explicit supervision**: `target_status: present/absent`
- **20% absent samples**: Gap in existing works
- **State retention on absent**: Support reappearance recovery

---

## 📂 Project Structure

### Core Modules
```
cogtrack/
├── prompts/
│   └── vlt_tracking.py          # Prompt v6.3.1
├── training/
│   ├── sample_builder.py        # Sampling strategy, positive/negative samples
│   ├── state_generator.py       # VLM-based state description generation
│   └── ...

tracking/
├── synthesize_vlt_v631_core_data.py  # Main generation script
└── ...

scripts/
├── generate_vlt_v631_core_data.sh    # One-click launcher
├── visualize_training_samples.py     # Quality check
└── test_data_generation.py           # Quick validation

pytracking/datasets/
├── mgit.py                      # MGIT action-layer text extraction
├── lasot.py
└── tnl2k.py
```

### Documentation
```
docs/
├── executive_summary.md                    # 📖 START HERE
├── vlt_v631_data_generation_optimized.md  # Research-informed design
├── smoke_test_report.md                   # Local validation results
└── implementation_progress.md             # Progress tracking
```

---

## 🚀 Quick Start: Training Server Deployment

### Step 1: Deploy vLLM (Qwen2.5-VL-32B)
```bash
cd /data2/wyp/VLMTrack/CognitiveTrack
bash scripts/start_vllm_qwen25_vl_32b.sh

# Verify service
curl http://127.0.0.1:8000/v1/models
```

### Step 2: Quick Test (Single Sequence, 3 Samples)
```bash
LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
LOCAL_VLLM_API_KEY=local-test-key \
python scripts/test_data_generation.py
```

### Step 3: Visualize & Validate
```bash
python scripts/visualize_training_samples.py \
  --data_dir /tmp/vlt_v631_test \
  --num_samples 3
```

### Step 4: Full Generation (10-20 hours, ~50K samples)
```bash
bash scripts/generate_vlt_v631_core_data.sh
```

### Step 5: Quality Check
```bash
# Statistics
cat /data2/wyp/VLMTrack/CognitiveTrack/data/vlt_v631_core_sft/generation_stats.json

# Visualize 50 samples
python scripts/visualize_training_samples.py --num_samples 50
```

---

## 📊 Data Generation Specs

### Sampling Strategy
- **Max frame span**: 200 frames
- **History buffer**: 3 frames, interval 10
- **Samples per sequence**: 8/15/30 (short/medium/long)

### Sample Type Distribution
| Type | Ratio | Description |
|------|-------|-------------|
| Pure Positive | 70% | Present, all history correct |
| Current Absent | 15% | Target absent (occluded/out-of-view) |
| History Noisy | 10% | 1-2 history frames with errors |
| Mixed Hard | 5% | Current absent + noisy history |

**Result**: Present:Absent = 80:20

### Expected Data Scale
| Dataset | Sequences | Samples/Seq | Total |
|---------|-----------|-------------|-------|
| LaSOT train | ~1,120 | ~18 | ~20,000 |
| TNL2K train | ~1,300 | ~12 | ~15,600 |
| MGIT train | ~150 | ~25 | ~3,750 |
| **Total** | **~2,570** | - | **~39,350** |

Target: **40K-50K Core SFT samples**

---

## 🔬 Research Context

### Differentiation from Related Works

| Dimension | DTLLM-VLT | DUTrack | ReasoningTrack | **VLT-v6.3.1** |
|-----------|-----------|---------|----------------|----------------|
| Data Generation | SAM + caption | Fixed threshold | Fixed interval | **Event-driven + dual-teacher** |
| Text Structure | Single description | Dynamic update | CoT + bbox | **Identity-state disentanglement** |
| Absent Supervision | Not explicit | Not explicit | Not explicit | **Explicit 20% negative samples** |
| GRPO Reward | N/A | N/A | Current-frame IoU | **Future trajectory Delta-U** |
| Data Scale | 26K + 214K | Not specified | Not specified | **50K Core + 3-5K Memory** |

### Key Research Findings
- ❌ **Gap**: Existing works rarely handle absent frames explicitly
- ❌ **Limitation**: Fixed thresholds/intervals (DUTrack IoU<0.7, ReasoningTrack fixed interval)
- ❌ **Myopic reward**: Current-frame IoU insufficient (ReasoningTrack)
- ✅ **Consensus**: Region-based caption preferred
- ✅ **Discovery**: Concise better than detailed (DUTrack finding)

---

## 📅 12-Week Timeline

### Week 1-2: Core SFT Data Generation ⏳ **← CURRENT**
- Deploy vLLM (Qwen2.5-VL-32B)
- Quick test & validation
- Full generation (50K+ samples, 10-20 hours)
- Quality check (visualization + statistics)

### Week 3-4: Core SFT Training
- Train Qwen3-VL-4B
- CognitiveBench-Tiny evaluation
- Verify presence + bbox baseline capabilities

### Week 5-6: Memory Label Generation
- Event candidate mining (DINOv2 features)
- Dual-teacher generation (Qwen3-VL-32B × 2)
- Human audit 500-1000 for calibration
- Output 3-5K high-quality events

### Week 7-8: Memory SFT
- Mixed training: 70% core + 30% memory
- Causal evaluation: memory-on vs forced-null

### Week 9-10: TU-GRPO Training
- Reward replay: validate Delta-U distribution
- GRPO training
- Full benchmark evaluation

### Week 11-12: Ablation & Paper
- Complete ablation studies
- Error analysis
- Paper writing

---

## 📖 Key Documentation

### Must-Read (Priority Order)
1. **[Executive Summary](docs/executive_summary.md)** - Complete work summary, start here
2. **[Optimized Data Generation](docs/vlt_v631_data_generation_optimized.md)** - Research-informed design
3. **[Smoke Test Report](docs/smoke_test_report.md)** - Local validation results
4. **Research Report**: `/tmp/vlm_tracking_data_generation_research.md` - 8 major works analyzed

### Implementation Details
- [Implementation Progress](docs/implementation_progress.md)
- [Data Generation Design](docs/vlt_v631_data_generation.md)
- [Project Status](docs/project_status.md)

---

## 🎯 Working Principles for Agents

### 1. **Understand Current Phase**
- We are at **Phase 1 completion**: Core SFT framework ready
- Next action: **Deploy to training server**
- Do NOT jump to Memory SFT or GRPO yet

### 2. **Respect Research Foundations**
- All design decisions are research-validated
- See `/tmp/vlm_tracking_data_generation_research.md` for references
- Major changes should cite related work comparisons

### 3. **Quality over Speed**
- Event-driven annotation (3-5K) beats per-frame dense (tens of thousands)
- Dual-teacher + verification prevents bias
- Human audit calibrates thresholds

### 4. **Incremental Validation**
- Quick test first (single sequence)
- Visualize before full generation
- Statistics + manual review after generation

### 5. **Documentation Culture**
- Update [implementation_progress.md](docs/implementation_progress.md) after major milestones
- Keep [executive_summary.md](docs/executive_summary.md) as single source of truth
- Archive outdated docs to `docs/archive/`

---

## 🔧 Common Tasks

### Add New Dataset
1. Create loader in `pytracking/datasets/`
2. Follow MGIT pattern: action-layer text extraction
3. Update `synthesize_vlt_v631_core_data.py`
4. Update expected sample counts in docs

### Modify Sampling Strategy
1. Edit `cogtrack/training/sample_builder.py`
2. Run smoke test: `python -c "from cogtrack.training.sample_builder import SampleBuilder; ..."`
3. Update `sample_type_ratios` in main script
4. Document changes in `vlt_v631_data_generation_optimized.md`

### Change State Description Prompt
1. Edit `cogtrack/training/state_generator.py`
2. Test with quick generation script
3. Visualize 10-20 samples
4. Document prompt version in commit message

### Debug Data Generation Issues
1. Check vLLM service: `curl http://127.0.0.1:8000/v1/models`
2. Run test script: `python scripts/test_data_generation.py`
3. Check logs in `/workspace/tmp/vllm_*.log`
4. Visualize failed samples

---

## 🚫 What NOT to Do

1. ❌ **Do NOT** modify Prompt v6.3.1 without research justification
2. ❌ **Do NOT** change 80:20 present/absent ratio arbitrarily
3. ❌ **Do NOT** skip visualization validation
4. ❌ **Do NOT** start Memory SFT before Core SFT data is ready
5. ❌ **Do NOT** implement GRPO before Memory SFT warm-up

---

## 📞 Getting Help

### If Data Generation Fails
1. Check smoke test report: [docs/smoke_test_report.md](docs/smoke_test_report.md)
2. Verify vLLM deployment
3. Check dataset paths in environment config
4. Review error logs

### If Quality Issues Arise
1. Visualize samples: `python scripts/visualize_training_samples.py`
2. Check statistics: `generation_stats.json`
3. Review state description prompt
4. Compare with research report findings

### For Conceptual Questions
1. Read executive summary: [docs/executive_summary.md](docs/executive_summary.md)
2. Check research report: `/tmp/vlm_tracking_data_generation_research.md`
3. Review differentiation table in this file

---

**Last Git Commit**: feat: implement VLT-v6.3.1 Core SFT data generation framework (30e52e5)  
**Status**: ✅ Local smoke test passed, ready for training server deployment  
**Next Milestone**: Complete Core SFT data generation (Week 1-2)
