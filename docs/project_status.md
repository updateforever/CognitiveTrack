# CognitiveTrack 当前状态

> 更新日期：2026-08-14。本文只记录当前 VLT-v6.3 主线；历史记录见
> [`archive/`](archive/README.md)。

## 已完成

- 独立 pytracking 生命周期、LaSOT/TNL2K/MGIT loader、结果写入和评测闭环；
- SUTrack 官方权重迁移与数值保真验证；
- Qwen2.5-VL/Qwen3-VL 本地及 vLLM 多图推理、严格解析和两代官方坐标视图；
- CognitiveBench v1 与 Tiny v1 的冻结标注和校验工具；
- 旧 pair Stage-1、mosaic Stage-2 LoRA 训练及同协议 Tiny 初步评测；
- VLT-v6 core 的固定三图、视觉画框、初始化文本安全路由、训练导出、字段级 loss mask、
  ms-swift 插件、训练配置和真实 processor 回放；
- VLT-v6.3 将文本明确拆为不可变 initial identity 与可替换 current target state，将
  native System Prompt 冻结为只描述身份维护、当前状态分析和按需记忆更新的 `6.3.0`，
  不再包含双图、padding、JSON、坐标或格式限制，并将
  历史布局冻结为按时间从左到右、带白色 panel 分隔的 `recent_strip_3_v2`；
- 状态标签冷启动和 TU-GRPO 的研究/验收方案已形成文档。

旧 Tiny 的 hold-last AUC：Base 21.81、Stage-1 52.64、Stage-2 55.05；observation-only
AUC：21.61、56.45、58.62。它们只证明旧跟踪 SFT 有潜力，完整口径见
[`archive/legacy_staged_tiny_results_20260813.md`](archive/legacy_staged_tiny_results_20260813.md)。

## 尚未完成

| 里程碑 | 状态 | 完成判据 |
| --- | --- | --- |
| VLT-v6.3 core 全量数据 | 待训练服务器执行 | 固定 plan、全量导出、checksum、processor audit |
| VLT-v6.3 core LoRA | 待执行 | smoke/过拟合通过，保存训练与验证记录 |
| Base/Core Tiny | 待执行 | 同一 v6.3 配置的完整 24 序列指标 |
| 通用 VLM strict comparison Prompt | 待冻结 | 独立 profile、manifest 可追踪、不得污染 native Prompt |
| `memory_labels.v1` | 方案已定，代码待实现 | 人工审核集、银标 manifest、可重放导出 |
| Memory SFT | 待执行 | update/null 与轨迹收益均通过验证 |
| TU-GRPO | 设计完成，代码/训练待实现 | reward replay、消融和 Tiny 增益 |
| CognitiveBench Full | 待执行 | 配方冻结后 995 序列主表 |

因此不能把“旧数据已做、旧 Stage-2 已训、GRPO 已有设想”写成完整方法已经完成。当前
完整的是研究闭环与工程底座，新的主线实验还需要依次落地。

## 下一步顺序

1. 在训练服务器冻结并渲染 VLT-v6.3 core 数据；
2. 从 Qwen3-VL-4B Base 完成 core LoRA；
3. 固定 VLT-v6.3 在 CognitiveBench-Tiny 比较 Base/Core；
4. 实现 `mine → annotate → verify → export` 状态标签工具，先做 500–1000 事件审核集；
5. Memory SFT 后做 memory-on/forced-null 因果回放；
6. 实现缓存版 TU-GRPO，再决定是否扩大真实短轨迹双分支回放；
7. 配方冻结后运行 Full 和全部消融。

每次训练或评测完成后只更新本页的状态与证据链接，不把阶段日志重新堆入 README。
