# CognitiveTrack 执行摘要

> 更新日期：2026-08-17。当前主线是 Prompt 6.4.0、统一三图协议和一套混合 LoRA。

CognitiveTrack 用纯 VLM 在完整视频帧中完成同实例存在性判断、定位与状态维护。学生输入
固定为带框身份 reference、三次带框可信历史组成的条带和无框 current；输出固定为：

```json
{"bbox_2d":[100,120,400,520],"status":"present","memory_update":null}
```

训练数据分成两个职责明确的桶：

- `tracking_sft`：LaSOT/TNL2K/MGIT 大规模 presence、bbox、27 种历史组合；未知的
  present/absent `memory_update:null` 值都被 mask；
- `state_update_sft`：MGIT 官方 action 分段尽量全量利用，再补约 1,500 条
  闭源 Qwen3.6 API 标签，所有进入 release 的 update/hard-null 都全监督。

MGIT 正式 release 已生成：91 条可用序列上有 350 个 update 和 384 个 hard-null，共
734 条，零 masked-unknown。联合状态数据预计约 2,234 条，正式 API bundle 和约 1,500
条标签尚未完成。两类 release 的
生成、审核和严格合并代码已实现，最终目标是混入同一次 Qwen3-VL-4B LoRA，而不是串联
三套 adapter。

历史 v6.3.1 release 的 50,220 cases / 57,426 views 仍保留作旧协议基线，不代表当前
v6.4 数据已经生成。当前 tracking 采样计划包含 2,511 个序列和 66,600 个 unique cases，
正式渲染正在 tmux 中恢复运行。主 release 完成后可按成本选择性补标其中明确的消失/
重现样本，但补标 overlay 不覆盖原始 tracking 数据，也不与独立状态数据生成耦合。随后
打包混合数据、跑两步 LoRA smoke，再做同协议 CognitiveBench-Tiny 对照。

数据生成、processor replay 和有限 loss 只证明工程链路，不证明跟踪精度提升。TU-GRPO
必须在混合 SFT 和状态更新评测之后。
