# VLT-v6.3.1 Core SFT 数据生成框架

本文档描述 VLT-v6.3.1 训练数据的生成流程、设计决策和使用方法。

---

## 📋 概述

### 目标
生成约 **30,000** 个训练样本，用于 **Core SFT** 阶段训练，重点学习：
1. ✅ **Presence-aware tracking**：判断目标是否存在（present/absent）
2. ✅ **Precise localization**：精准定位目标边界框
3. ✅ **State awareness**：理解目标状态变化的概念（为 Memory SFT 和 GRPO 做准备）

### 数据来源
| 数据集 | Split | 序列数 | 平均长度 | 占比 |
|--------|-------|--------|----------|------|
| LaSOT | train | ~1,120 | ~2,500帧 | 56% |
| TNL2K | train | ~1,300 | ~500帧 | 35% |
| MGIT | train | ~150 | ~3,000帧 | 9% |

---

## 🎯 核心设计

### 1. 输入格式（VLT-v6.3.1 Protocol）

**三图输入**：
- **Image 1**：初始帧 + 红框（永久身份锚点）
- **Image 2**：历史预测轨迹 mosaic（3帧，间隔10帧）
- **Image 3**：当前搜索帧

**文本输入**：
```
Initial target identity: a male secret agent wearing a black suit
Current maintained target state: walking in the hallway
Track output:
```

### 2. 输出格式

```json
{
  "target_status": "present",  // or "absent"
  "bbox": [x, y, w, h],        // norm1000 格式
  "memory_update": "hiding behind the wall"
}
```

### 3. 采样策略

**帧间隔控制**：
- 初始化帧到当前帧：≤ 200 帧
- History 采样：3 帧，间隔 10 帧（覆盖最近 30 帧）

**每序列采样密度**：
- 短序列（< 300帧）：5 个样本
- 中等序列（300-1000帧）：10 个样本
- 长序列（> 1000帧）：20 个样本

### 4. 正负样本配比

| 类型 | 占比 | 描述 |
|------|------|------|
| **Pure Positive** | 60% | 当前帧存在，历史全部正确 |
| **Current Absent** | 15% | 当前帧目标缺失（遮挡/出界） |
| **History Noisy** | 15% | 历史帧有预测错误（1-2帧） |
| **Mixed Hard** | 10% | 当前缺失 + 历史噪声 |

**负样本构造细节**：
- **Current Absent**：优先采样 GT 标注为 absent 的帧
- **History Noisy**：对 1-2 个历史帧进行 bbox 扰动（±20% 偏移，0.7-1.3x 缩放）或设为 absent
- **Mixed Hard**：组合上述两种情况

---

## 🏗️ 代码结构

### 核心模块

**1. Sample Builder** ([cogtrack/training/sample_builder.py](cogtrack/training/sample_builder.py))
- 负责从序列中采样帧
- 构造正负样本
- 拼接 history mosaic
- 添加噪声（bbox 扰动、absent 标记）

**2. State Generator** ([cogtrack/training/state_generator.py](cogtrack/training/state_generator.py))
- 调用 vLLM API（Qwen2.5-VL-32B）
- 生成初始身份描述
- 生成状态更新描述
- 处理 present/absent 两种情况

**3. 主生成脚本** ([tracking/synthesize_vlt_v631_core_data.py](tracking/synthesize_vlt_v631_core_data.py))
- 加载 LaSOT/TNL2K/MGIT 训练集
- 整合 Sample Builder 和 State Generator
- 保存标注文件（JSON 格式）
- 生成统计信息

**4. 可视化脚本** ([scripts/visualize_training_samples.py](scripts/visualize_training_samples.py))
- 随机抽取样本
- 渲染三图 + 标注信息
- 质量检查

**5. 启动脚本** ([scripts/generate_vlt_v631_core_data.sh](scripts/generate_vlt_v631_core_data.sh))
- 一键启动数据生成
- 检查 vLLM 服务
- 设置环境变量

---

## 🚀 使用方法

### 前置条件

1. **部署 vLLM 服务**（Qwen2.5-VL-32B）：
```bash
bash scripts/start_vllm_qwen25_vl_32b.sh
```

2. **检查服务状态**：
```bash
curl http://127.0.0.1:8000/v1/models
```

### 快速测试（单序列）

```bash
# 测试数据生成流程
LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
LOCAL_VLLM_API_KEY=local-test-key \
python scripts/test_data_generation.py
```

### 完整数据生成

```bash
# 生成全部训练数据（~30k 样本）
bash scripts/generate_vlt_v631_core_data.sh
```

**预估耗时**：
- LaSOT: ~1,120 序列 × 15 样本 ≈ 16,800 样本
- TNL2K: ~1,300 序列 × 8 样本 ≈ 10,400 样本
- MGIT: ~150 序列 × 20 样本 ≈ 3,000 样本
- **总计：~30,000 样本**
- **耗时**：取决于 vLLM 推理速度，预估 10-20 小时（按每样本 2 次 VLM 调用，每次 1-2 秒）

### 可视化验证

```bash
# 随机可视化 50 个样本
python scripts/visualize_training_samples.py \
  --data_dir /data2/wyp/VLMTrack/CognitiveTrack/data/vlt_v631_core_sft \
  --output_dir /data2/wyp/VLMTrack/CognitiveTrack/data/vlt_v631_core_sft_vis \
  --num_samples 50
```

### 检查统计信息

```bash
cat /data2/wyp/VLMTrack/CognitiveTrack/data/vlt_v631_core_sft/generation_stats.json
```

---

## 📊 标注文件格式

每个样本保存为独立的 JSON 文件：

```json
{
  "sample_id": 1,
  "sequence_name": "001",
  "dataset_name": "mgit",
  "sample_type": "pure_positive",
  
  "init_frame": {
    "frame_id": 0,
    "frame_path": "/path/to/frame_0000.jpg",
    "bbox": [100.0, 150.0, 80.0, 120.0],
    "visible": true
  },
  
  "history_frames": [
    {"frame_id": 10, "frame_path": "...", "bbox": [...], "visible": true},
    {"frame_id": 20, "frame_path": "...", "bbox": [...], "visible": true},
    {"frame_id": 30, "frame_path": "...", "bbox": [...], "visible": true}
  ],
  
  "current_frame": {
    "frame_id": 40,
    "frame_path": "/path/to/frame_0040.jpg",
    "bbox": [105.0, 160.0, 85.0, 125.0],
    "visible": true
  },
  
  "initial_identity": "a male secret agent wearing a black suit",
  "previous_state": "walking in the hallway",
  "target_status": "present",
  "memory_update": "hiding behind the wall"
}
```

---

## 🎓 状态描述生成 Prompt

### 初始身份描述

```
Describe the target object in the bounding box [x, y, w, h].
Provide a concise identity description focusing on:
- Object category (e.g., person, vehicle, animal)
- Key appearance attributes (e.g., color, clothing, distinctive features)

Format: "a [category] [appearance]"
Example: "a male athlete wearing red jersey"

Identity:
```

### 状态更新描述

```
You are tracking: {initial_identity}

Previous target state: {previous_state}

Current frame analysis:
{status_description}  // present at [x,y,w,h] 或 absent

Generate the updated target state description:
- If present: describe current action, position, or significant state change
- If absent: describe the absence reason (e.g., "occluded behind wall", "out of view")
- If no significant change: keep the same description as previous state

Updated state:
```

---

## 🔍 质量保证

### 数据验证检查项

1. **样本总数**：~30,000
2. **正负样本比例**：60/15/15/10
3. **帧间隔分布**：平均 50-150 帧，最大 200
4. **数据集占比**：LaSOT 56%, TNL2K 35%, MGIT 9%
5. **文本质量**：
   - 初始身份描述与图像一致
   - 状态更新合理（present 描述动作，absent 描述原因）
   - 无重复或空白描述

### 可视化检查

随机抽取 50-100 个样本，检查：
- ✅ 三图拼接正确
- ✅ bbox 标注准确
- ✅ absent 帧正确标记
- ✅ 文本描述与视觉一致
- ✅ 状态更新合理

---

## 🆚 与相关工作的差异

### 我们的独特性

1. **Presence-aware 负样本设计**：
   - 15% Current Absent + 10% Mixed Hard = **25% 包含 absent 样本**
   - 传统跟踪器训练数据很少包含 absent 帧
   - 这是 presence-aware tracking 的核心能力来源

2. **History Noisy 样本**：
   - 15% 历史帧有预测错误
   - 训练模型不盲目信任历史预测
   - 提高对漂移的鲁棒性

3. **三阶段训练设计**：
   - **Core SFT**（本阶段）：学习 presence + bbox
   - **Memory SFT**：深度优化状态更新策略
   - **GRPO**：通过轨迹效用反馈进一步优化

4. **状态描述生成**：
   - 用大模型自动生成状态描述
   - 避免人工标注成本
   - 保持描述质量和一致性

---

## 📚 相关工作参考

**等待调研 agent 结果**，届时补充：
- DTLLM-VLT, DUTrack, R1-Track 等的数据生成策略
- 我们方案的理论支撑和差异化优势

---

## 🔧 故障排查

### vLLM 服务连接失败
```bash
# 检查服务是否运行
curl http://127.0.0.1:8000/v1/models

# 查看日志
tail -f /workspace/tmp/vllm_qwen25_vl_32b.log

# 重启服务
bash scripts/start_vllm_qwen25_vl_32b.sh
```

### 数据集路径错误
检查 `pytracking/local.py` 中的数据集路径配置。

### 内存不足
- 减小 `history_buffer_size`（从 3 降到 1）
- 减小 `max_frame_span`（从 200 降到 100）
- 减少每序列采样数量

---

## 📝 后续工作

1. ✅ **完成 Core SFT 数据生成**
2. ⏳ **Memory SFT 数据设计**（混合监督：70% masked + 30% full）
3. ⏳ **GRPO 训练流程**（轨迹效用反馈）
4. ⏳ **Presence-aware 评测指标实现**

---

**生成时间**：2026-08-14  
**Prompt 版本**：VLT-v6.3.1  
**作者**：CognitiveTrack Team
