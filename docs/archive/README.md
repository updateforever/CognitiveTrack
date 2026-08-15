# 历史实验归档

这里保存已经冻结、但不再代表当前论文主线的实验与设计。归档文件用于复现和解释历史
结果，不能作为新实验的默认入口。

| 文件 | 归档原因 |
| --- | --- |
| `legacy_staged_tiny_results_20260813.md` | 旧二字段 Stage-1/2 的 Tiny 结果 |
| `l40_stage1_reproduction.md` | 旧 pair64 Stage-1 的服务器复现配方 |
| `visual_v5_iteration.md` | 不含初始化文本的 visual-v5 工程协议 |
| `stage2_stage3_data.md` | 早期统一混合训练草案 |
| `qwen_v4_probe.md` | Qwen2.5-VL v4.1 单点能力探针 |
| `project_status_20260813.md` | 迁移到 VLT-v6 初期的时点快照 |

当前入口依次为：

1. [`../research_plan.md`](../research_plan.md)：论文主线与实验闭环；
2. [`../state_annotation.md`](../state_annotation.md)：状态记忆标签冷启动；
3. [`../grpo.md`](../grpo.md)：轨迹效用 GRPO；
4. [`../training.md`](../training.md)：分阶段训练；
5. [`../project_status.md`](../project_status.md)：真实完成度与下一步。

归档只移动文档，不删除旧代码、配置、checkpoint 记录或结果。复现实验时必须使用归档
文档中对应的 prompt/config，不能用当前 VLT-v6.3 协议解释旧指标。
