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
| `vlt_v631_teacher_prototype/` | Qwen2.5-VL-32B 逐样本生成原型、Prompt 6.3.0 分析和旧实施草案 |
| `paper_outline_v1_v631.md` | 仍使用 Core/Memory 分阶段名称与 Prompt 6.3.1 的论文草案 |
| `superseded_v640_20260817/` | 合并为 `data.md`/`setup.md` 前的拆分数据、环境、摘要、进度和迁移文档 |

当前入口依次为：

1. [`../project_status.md`](../project_status.md)：真实完成度与下一步；
2. [`../data.md`](../data.md)：当前两类 SFT 数据与状态标签；
3. [`../training.md`](../training.md)：当前冻结 ViT 的混合全参 SFT 训练边界；
4. [`../research_plan.md`](../research_plan.md)：论文主线与实验闭环；
5. [`../grpo.md`](../grpo.md)：轨迹效用 GRPO。

归档只移动文档，不删除旧代码、配置、checkpoint 记录或结果。复现实验时必须使用归档
文档中对应的 prompt/config，不能用当前 VLT-v6.4 协议解释旧指标。
