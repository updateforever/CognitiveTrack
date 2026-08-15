# VLT-v6.3.1 本地烟测报告

**测试时间**：2026-08-14  
**测试目的**：验证数据生成框架的核心逻辑

---

## ✅ 测试结果总结

### 1. 核心模块初始化 ✅
```
✅ Sample Builder initialized successfully
   Max frame span: 200
   History buffer: 3
   Sample ratios: correct

✅ State Generator initialized successfully
   API: http://127.0.0.1:8000/v1
   Model: Qwen2.5-VL-32B-Instruct
```

### 2. 样本类型分配 ✅
测试 100/200/1000 样本的分配比例：

| Sample Type | Expected | Actual | Status |
|-------------|----------|--------|--------|
| Pure Positive | 70% | 70.0% | ✅ |
| Current Absent | 15% | 15.0% | ✅ |
| History Noisy | 10% | 10.0% | ✅ |
| Mixed Hard | 5% | 5.0% | ✅ |
| **Present Total** | **80%** | **80.0%** | ✅ |
| **Absent Total** | **20%** | **20.0%** | ✅ |

**结论**：present:absent = 8:2，符合调研建议的 7:3~8:2 范围。

### 3. 数据集加载 ✅
```
✅ MGIT test tiny: 30 sequences
   Sequence: 001
   Frames: 9033
   Language query: A male secret agent wearing a black suit walks in the washroom...
   GT shape: (9033, 4)
   Visible: False
```

**验证**：
- ✅ MGIT action 层文本提取正确
- ✅ 序列加载成功
- ✅ GT bbox 格式正确

---

## 📋 代码修正记录

### 修正 1：Sample Builder 默认比例
**文件**：`cogtrack/training/sample_builder.py`
```python
# 修正前：60/15/15/10
# 修正后：70/15/10/5
self.sample_type_ratios = {
    SampleType.PURE_POSITIVE: 0.70,   # ✅
    SampleType.CURRENT_ABSENT: 0.15,
    SampleType.HISTORY_NOISY: 0.10,   # ✅
    SampleType.MIXED_HARD: 0.05,      # ✅
}
```

### 修正 2：状态描述 Prompt（简洁版）
**文件**：`cogtrack/training/state_generator.py`
```python
# 添加要求：
# - <30 words
# - Forbidden: coordinates, frame numbers, background, reasoning
```

### 修正 3：采样密度（目标 50K+）
**文件**：`tracking/synthesize_vlt_v631_core_data.py`
```python
# 修正前：5/10/20
# 修正后：8/15/30
SAMPLES_PER_SHORT_SEQ = 8
SAMPLES_PER_MEDIUM_SEQ = 15
SAMPLES_PER_LONG_SEQ = 30
```

### 修正 4：主脚本样本比例
**文件**：`tracking/synthesize_vlt_v631_core_data.py`
```python
sample_type_ratios={
    SampleType.PURE_POSITIVE: 0.70,   # ✅
    SampleType.CURRENT_ABSENT: 0.15,
    SampleType.HISTORY_NOISY: 0.10,   # ✅
    SampleType.MIXED_HARD: 0.05,      # ✅
}
```

---

## 🎯 核心功能验证

### ✅ 已验证
1. ✅ Sample Builder 初始化
2. ✅ State Generator 初始化
3. ✅ 样本类型分配逻辑
4. ✅ Present/absent 比例（8:2）
5. ✅ 数据集加载（MGIT）
6. ✅ MGIT action 层文本提取

### ⏳ 未验证（需要 vLLM 服务）
1. ⏳ 状态描述生成（需要调用 API）
2. ⏳ History mosaic 拼接
3. ⏳ 完整样本构造流程
4. ⏳ 可视化渲染

---

## 📊 预期数据规模

基于调研优化后的采样密度：

| 数据集 | 序列数 | 每序列样本 | 总样本 |
|--------|--------|-----------|--------|
| LaSOT train | ~1,120 | ~18 | ~20,000 |
| TNL2K train | ~1,300 | ~12 | ~15,600 |
| MGIT train | ~150 | ~25 | ~3,750 |
| **总计** | **~2,570** | **-** | **~39,350** |

**调整后预期**：**40K-50K 样本**（接近 DTLLM-VLT 的 26K 初始规模）

---

## ✅ 烟测结论

**所有本地可测试的功能均通过** ✅

**下一步**：
1. 在训练服务器部署 vLLM（Qwen2.5-VL-32B）
2. 运行快速测试（单序列，3个样本）
3. 可视化验证质量
4. 启动完整数据生成（预估 10-20 小时）

---

## 🚀 训练服务器命令

### 1. 部署 vLLM
```bash
bash scripts/start_vllm_qwen25_vl_32b.sh
```

### 2. 快速测试
```bash
LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
LOCAL_VLLM_API_KEY=local-test-key \
python scripts/test_data_generation.py
```

### 3. 完整生成
```bash
bash scripts/generate_vlt_v631_core_data.sh
```

### 4. 可视化验证
```bash
python scripts/visualize_training_samples.py --num_samples 50
```

---

**测试完成时间**：2026-08-14 23:55  
**测试结论**：✅ 本地烟测通过，可以部署到训练服务器
