# VLT-v6.3.1 实现进度总结

## ✅ 已完成的工作

### 1. Prompt 优化（v6.3.0 → v6.3.1）

**System Prompt 改进**：
- ✅ 简化身份锚点描述
- ✅ 明确图2是"历史预测轨迹"
- ✅ 图3单独说明：当前搜索帧
- ✅ 简化记忆更新条件

**User Prompt 改进**：
- ✅ 触发语改为 "Track output:"（更简洁）

**文件更新**：
- [cogtrack/prompts/vlt_tracking.py](../cogtrack/prompts/vlt_tracking.py)

---

### 2. MGIT 文本提取（官方方案）

**改动**：
- ✅ 从 `story` 层改为 `action` 层
- ✅ 提取第一个 action 的 `description` 字段
- ✅ 包含 `object_class + appearance + 初始动作`
- ✅ 验证脚本确认提取正确

**文件更新**：
- [pytracking/datasets/mgit.py](../pytracking/datasets/mgit.py)
- [test_mgit_action_extraction.py](../test_mgit_action_extraction.py)

**验证结果**：
```
001: male secret agent | black suit
002: garfield | orange fur
003: dog | brown fur
...
```

---

### 3. 数据生成框架实现

#### 核心模块

**A. Sample Builder** ✅
- [cogtrack/training/sample_builder.py](../cogtrack/training/sample_builder.py)
- 功能：
  - 采样策略（max_span=200, history=3, interval=10）
  - 正负样本配比（60/15/15/10）
  - 负样本构造（absent、noisy history）
  - History mosaic 拼接

**B. State Generator** ✅
- [cogtrack/training/state_generator.py](../cogtrack/training/state_generator.py)
- 功能：
  - 调用 vLLM API（Qwen2.5-VL-32B）
  - 生成初始身份描述
  - 生成状态更新描述
  - 处理 present/absent 两种情况

**C. 主生成脚本** ✅
- [tracking/synthesize_vlt_v631_core_data.py](../tracking/synthesize_vlt_v631_core_data.py)
- 功能：
  - 整合 Sample Builder + State Generator
  - 加载 LaSOT/TNL2K/MGIT 训练集
  - 保存 JSON 标注文件
  - 生成统计信息

**D. 可视化脚本** ✅
- [scripts/visualize_training_samples.py](../scripts/visualize_training_samples.py)
- 功能：
  - 随机抽取样本
  - 渲染三图 + bbox + 文本
  - 质量检查

**E. 启动脚本** ✅
- [scripts/generate_vlt_v631_core_data.sh](../scripts/generate_vlt_v631_core_data.sh)
- [scripts/test_data_generation.py](../scripts/test_data_generation.py)
- 功能：
  - 一键启动完整生成
  - 快速测试（单序列）

**F. 综合文档** ✅
- [docs/vlt_v631_data_generation.md](../docs/vlt_v631_data_generation.md)
- 内容：
  - 完整设计文档
  - 使用方法
  - 质量保证
  - 故障排查

---

## 🔄 进行中的工作

### 调研工作（Background Agent）

**任务**：调研 VLM tracking 相关工作的数据生成方法
- DTLLM-VLT, DUTrack, R1-Track, ReasoningTrack
- 状态记忆标注方法
- 负样本设计策略
- 训练数据关键设计

**目的**：
- ✅ 确保我们的方案有理论支撑
- ✅ 明确差异化优势
- ✅ 避免重复工作
- ✅ 学习最佳实践

**状态**：⏳ 运行中

---

## 📋 待办事项

### 短期（本周）

1. ⏳ **等待调研结果**
   - 根据调研结果调整数据生成方案
   - 补充理论支撑到文档

2. ⏳ **快速测试数据生成**
   ```bash
   # 测试单序列
   python scripts/test_data_generation.py
   ```

3. ⏳ **可视化验证**
   ```bash
   # 检查生成质量
   python scripts/visualize_training_samples.py --num_samples 10
   ```

4. ⏳ **修复潜在 bug**
   - 根据测试结果调整代码
   - 完善错误处理

### 中期（数据生成阶段）

5. ⏳ **完整数据生成**
   ```bash
   # 在训练服务器上运行
   bash scripts/generate_vlt_v631_core_data.sh
   ```
   - 预估耗时：10-20 小时
   - 预期输出：~30,000 样本

6. ⏳ **质量检查**
   - 统计分析（sample_type 分布、帧间隔分布）
   - 可视化检查（50-100 个样本）
   - 文本质量审核

7. ⏳ **数据集整理**
   - 转换为 SFT 训练格式
   - 划分 train/val split
   - 上传到训练服务器

### 长期（训练与评测）

8. ⏳ **Core SFT 训练**
   - 设计训练配置
   - 启动训练
   - 监控收敛

9. ⏳ **Presence-aware 评测**
   - 实现评测指标
   - 在 CognitiveBench 上测试
   - 对比传统 tracker

10. ⏳ **Memory SFT + GRPO**
    - 设计 Memory SFT 数据
    - 实现 GRPO 训练流程
    - 轨迹效用反馈

---

## 📊 关键数字

| 指标 | 目标值 | 当前状态 |
|------|--------|----------|
| Prompt 版本 | v6.3.1 | ✅ 已完成 |
| MGIT 文本提取 | action 层 | ✅ 已完成 |
| 代码模块 | 6 个 | ✅ 已完成 |
| 文档 | 完整 | ✅ 已完成 |
| 调研报告 | 1 份 | ⏳ 进行中 |
| 测试样本 | 3 个 | ⏳ 待运行 |
| 训练样本 | ~30,000 | ⏳ 待生成 |

---

## 🎯 下一步行动

**立即行动**（等待调研结果时）：
1. ✅ 快速测试 `scripts/test_data_generation.py`
2. ✅ 可视化检查生成质量
3. ✅ 修复潜在 bug

**调研完成后**：
1. 根据调研结果调整方案
2. 更新文档补充理论支撑
3. 在训练服务器启动完整数据生成

---

**更新时间**：2026-08-14 23:45  
**下次更新**：调研结果出来后
