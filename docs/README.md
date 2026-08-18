# CognitiveTrack 文档索引

> 更新日期：2026-08-17。顶层文档只描述当前 VLT-v6.4 主线；旧协议、旧 Prompt、
> 原型生成器和历史结果统一放在 [`archive/`](archive/README.md)。

## 当前事实来源

1. [`project_status.md`](project_status.md)：真实完成度、正式数据和下一步；
2. [`data.md`](data.md)：两类 SFT 数据、27 种 tracking 组合、Aliyun Qwen3-VL-Plus 标注和可选 overlay；
3. [`training.md`](training.md)：逐样本监督边界、混合全参 SFT 与训练命令；
4. [`protocol.md`](protocol.md)：三字段 JSON、状态、bbox 和 memory 协议；
5. [`architecture.md`](architecture.md)：运行时边界和在线时序；
6. [`setup.md`](setup.md)：环境、服务器路径、tmux、预检和多卡入口；
7. [`research_plan.md`](research_plan.md)：论文主线与实验矩阵；
8. [`grpo.md`](grpo.md)：TU-GRPO 候选设计，尚未实现。

当前可直接训练和发布的自包含 mixed release 是：

```text
data/releases/cogtrack_v640_mixed_sft_full_v1
```

## 历史归档

- [`archive/paper_outline_v1_v631.md`](archive/paper_outline_v1_v631.md)：历史 v6.3.1
  论文结构草案，不代表当前协议或实验已完成。
- [`archive/superseded_v640_20260817/`](archive/superseded_v640_20260817/)：本次文档收敛前
  的拆分数据、环境、摘要、进度和迁移文档。

归档文件允许保留当时的版本和命令，但不能作为启动新实验的入口。当前代码的唯一
native Prompt 版本是 6.4.0，Qwen3-VL 输出字段是 `bbox_2d/status/memory_update`。
