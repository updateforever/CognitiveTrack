# 架构说明

## 1. 依赖边界

```text
tracking CLI
    └── pytracking（数据、运行、落盘）
          └── tracker adapter
                └── cogtrack（协议、VLM、认知、记忆）
```

约束如下：

1. `pytracking` 不导入具体 Qwen 类，也不判断 `present/absent`。
2. `cogtrack.vlm` 不读取数据集，不保存实验结果。
3. tracker 只组合组件，不直接实现指标。
4. observation policy 只决定本帧是否提供昂贵观测，不跳过 runner。
5. evaluator 只读取已经落盘的结果，不重新执行模型。
6. 训练与在线推理共享协议、bbox 转换和 prompt 版本。

## 2. 标准推理时序

```text
dataset.get_sequence_list()
        ↓
读取首帧和 init_info
        ↓
tracker.initialize(image, init_info)
        ↓
逐帧读取 image 和 frame_info
        ↓
ObservationPolicy.should_observe(...)
        ↓
tracker.track(image, frame_info)
        ↓
ResultWriter 写传统 TXT 与逐帧 JSONL
        ↓
Evaluator 离线计算指标
```

除了初始化帧，tracker 不允许读取 `ground_truth_rect`、`target_visible` 或未来帧。

## 3. Cognitive VLM tracker

```text
ContextBuilder
    ↓
PromptBuilder
    ↓
VLMBackend.generate
    ↓
StrictOutputParser
    ↓
CognitiveStateMachine
    ↓
MemoryUpdatePolicy（visual / semantic 双通道）
    ↓
标准 tracker output
```

初始 GT 目标锚点永久保存。预测历史只有在状态为 `present`、bbox 合法且满足
连续确认（可选几何一致性）时，才允许进入长期正记忆。
VLT-v6.3 的 Image 2 固定为 `recent_strip_3_v2`：最近三次可信观测按时间从左到右排列，
相邻 panel 由高度约 3% 的白色竖向分隔带隔开；少于三次时在右侧复制最近可用观测，
尚无动态历史时复制初始化观测。visual-v5 仍使用冻结的 `compact_grid_v1`，旧的无分隔
条带保留为 `recent_strip_3_v1`，三种布局不会静默共用实验结果。
模型把 `memory_update` 放在输出末尾：`null` 直接走快路径，非空状态快照走慢路径；
后者还必须经过 present、合法 bbox、去重、最小帧间隔和容量门控，才进入语义记忆。
初始身份文本和模板图永久不覆盖；已接受状态只替换可变的 `current_target_state`，并在
下一次 pair/mosaic 推理中作为辅助证据回读。初始化身份始终具有最高优先级。
模型输出的候选框和 benchmark 的最终跟踪框是两个概念：只有
`target_status=present` 且 bbox 合法的候选才能写入 `target_bbox`。

Qwen backend 按完整模型配置做进程级共享，序列切换只重置锚点、状态机和记忆，
不重复加载大模型权重。

## 4. SUTrack 边界

`pytracking/trackers/sutrack_adapter.py` 通过配置的 Python 模块和工厂延迟加载
SUTrack runtime。内置实现位于 `cogtrack/models/sutrack/runtime.py`，包含
Fast-iTPN、CENTER decoder、精简 CLIP 文本塔、裁剪预处理和严格 checkpoint
加载，不依赖目录外源码。网络按完整配置和 checkpoint 路径做进程级共享，序列
状态与在线模板仍由各 runtime 实例隔离。详细契约见
`cogtrack/models/sutrack/README.md`。

`HybridCognitiveTracker` 让 SUTrack 每帧运行，VLM 仅在 observation policy 选中的帧执行。
已提交的 VLM 全局框可通过可选 `runtime.correct()` 回灌运动分支；高置信
absent/different 默认抑制连续跟踪框，模型/解析等工程错误则可审计地回退 SUTrack。

## 5. 结果兼容性

每个序列同时保存：

- `<sequence>.txt`：传统像素级 `xywh`，兼容常规跟踪评测。
- `<sequence>_time.txt`：逐帧耗时。
- `<sequence>_frames.jsonl`：完整认知、错误和上下文信息。
- `<sequence>_manifest.json`：该序列的配置摘要、代码版本、运行环境和完整性。

JSON 中不可定位 bbox 使用 `null`；传统 TXT 中使用四个 `NaN` 保持帧数一致。
