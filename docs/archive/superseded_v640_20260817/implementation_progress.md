# CognitiveTrack 实施进度

> 更新日期：2026-08-17。勾选只代表已有可复核代码或产物，不以计划代替实现。

## tracking_sft

- [x] Prompt 6.4.0 与 `bbox_2d/status/memory_update` 协议；
- [x] 带框 reference、三格历史、无框 current；
- [x] 80:20 同序列真实 present/absent 确定性采样；
- [x] continuous/absent/reappearance × 九种合法历史形式；
- [x] jitter/stale 单 panel 污染约束；
- [x] Qwen3 grounding、完整序列 train/val 和 taxonomy preflight；
- [x] `tracking_sft` 字段级 loss mask 与训练前检查；
- [x] 36 序列/112 行真实三图生成 smoke 与逐行预检；
- [x] 2,511 序列/66,600 unique cases 正式采样计划；
- [ ] v6.4 正式全量 release（tmux 渲染中）；
- [ ] 混合 LoRA 两步 smoke；
- [ ] 正式 LoRA 与 CognitiveBench-Tiny 对照。

## state_update_sft

- [x] 永久 identity / 完整替换 state 语义；
- [x] MGIT 数字字符串规范化与 action 分段解析；
- [x] 全量 MGIT 正式 release：734 条（350 update / 384 hard-null）；
- [x] 闭源 Qwen3.6 OpenAI-compatible API 单次生成入口；
- [x] ModelScope 便携 bundle、断点续跑 journal 与确定性质量门；
- [x] 约 1,500 条输出预算与 update/null 分层裁剪；
- [x] 两来源严格 plan/label 合并器；
- [x] 联合 release 构建与零 masked-unknown preflight；
- [x] MGIT 正式 release：645/89 train/val、零 masked-unknown；
- [ ] 额外约 1,500 条正式 Qwen3.6 API 运行；
- [ ] 预计约 2,234 条联合 release；
- [ ] 可选 tracking 明确转折样本 memory overlay（不阻塞首轮训练）；
- [ ] 与 tracking_sft 的统一数据根和单次 LoRA 混合入口。

## TU-GRPO

- [x] 反事实未来轨迹效用设计；
- [ ] 缓存代理 reward；
- [ ] accept/keep 短轨迹回放；
- [ ] reward replay、训练和消融。

TU-GRPO 必须晚于混合 SFT 与状态更新评测。
