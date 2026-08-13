# CognitiveTrack 讨论交接摘要（2026-08-13）

本文供研究讨论和其他 AI 快速接手。完整恢复约束见根目录 `AGENTS.md`，统一数据设计
见 `docs/stage2_stage3_data.md`，本机/训练服务器分工见 `docs/visual_v5_iteration.md`。
本页严格区分“旧实验”“visual-v5 工程可行性”和“尚未完成的正式训练/评测”。

## 1. 最新研究决策

下一版不再把 tracking、temporal context、memory 拆成三次 SFT，而是构造一份统一混合
数据，只训练一次三字段 Qwen3-VL-4B LoRA：

```json
{"target_status":"present","bbox_norm1000_xyxy":[100,120,400,520],"memory_update":null}
```

输入范式改为视觉指代：

- 完整首帧身份锚点直接画目标框；
- 过去可信历史帧或 mosaic 的每个 panel 直接画对应历史框；
- 最近可信 VLM 观测必须进入历史，稀疏推理时不等于字面上一帧；
- 当前完整搜索图绝不画框；
- Prompt 不再提供 reference bbox 坐标文本；
- 输出 bbox 仍是 Qwen3 官方 norm1000 xyxy，绑定最后一张当前图。

永久首帧锚点与动态最近历史同时保留。只用首帧难以覆盖长时外观变化，只滚动上一
预测又容易漂移自强化。

## 2. 已完成操作

代码状态：

- 本文随代码维护；恢复时以 `git rev-parse HEAD` 为准，不再引用讨论开始时的旧 HEAD；
- CognitiveBench v1 冻结标注已纳入 Git：995 序列、1,408,438 帧、343,616 关键帧；
- 最近一次代码验证：Ruff 通过、143 tests passed、CognitiveBench 冻结标注校验通过、
  `git diff --check` 通过。
- visual-v5 的共享绘框、统一 Prompt、在线 context、三字段 canonical/ms-swift 导出、
  tracker 配置与跨帧 semantic gate 已实现；旧 `bbox_text` 路径显式保留用于复现。
- 本机 TNL2K 小数据已完成真实图片生成和 Qwen3 processor 回放；Qwen3-VL-4B 基座已在
  CognitiveBench-Tiny 单 case 上完成真实推理、严格解析和坐标转换。

旧范式训练：

| 实验 | 最终 checkpoint | 步数 | 用时 | train loss | token acc |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage-1 pair64 | `checkpoint-19005` | 19,005 | 4h45m41s | 0.29283377 | 0.882494 |
| Stage-2 mosaic robust v2 | `checkpoint-28819` | 28,819 | 7h15m14s | 0.25926472 | 0.89407277 |

Stage-2 在 2×L40 上从 Stage-1 最终 checkpoint 继续同一个 LoRA，未叠加 adapter；峰值
显存 38.13GiB/卡。Stage-2 数据共 181,969 cases：train 172,915、val 9,054，其中
21,920 条为 `jitter_box`/`stale_box` corrupted-history case。

ModelScope：

- 私有模型仓库：`updateforever/CognitiveTrack-Qwen3VL-4B-Stage1-LoRA`；
- Stage-1 adapter 位于仓库根目录；
- Stage-2 位于 `stage2-mosaic-robust-v2/`；
- Stage-2 权重 SHA-256：
  `7437dd2be3bae21070059ef5ce704da7bbd5008607f7e04eb18b3638f04930a1`；
- 已从远端重新下载权重并完成 SHA-256 校验；
- 只上传推理 adapter、训练参数/状态、日志、曲线和 checksum，未上传 optimizer/RNG。

Base、Stage-1 与 Stage-2 已在冻结的 CognitiveBench-Tiny v1 上使用相同稀疏推理协议
完成一次初步聚合评测：

| 实验 | Hold-last AUC | Observation-only AUC | Presence F1 | Absent FPR | Reappearance recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 21.81 | 21.61 | 93.41 | 33.11 | 30.67 |
| Stage-1 | 52.64 | 56.45 | 96.05 | 45.14 | 88.96 |
| Stage-2 | 55.05 | 58.62 | 96.16 | 15.24 | 89.57 |

这组结果说明旧跟踪 SFT 有明显初步潜力，Stage-2 也改善了 Stage-1 的定位、absent FPR
和结构化输出稳定性。但它只覆盖 24 条 Tiny 完整序列，不是 Full 主表，也没有使用本页
规划的新视觉画框输入。完整协议、全部指标与限制见
[`legacy_staged_tiny_results_20260813.md`](legacy_staged_tiny_results_20260813.md)。

## 3. visual-v5 当前实现

新协议以独立配置启用，不会改变旧实验：

- `reference_mode: visual_box` 时，首帧和历史 mosaic 调用同一个 `red_box_v1` 渲染器；
- current 始终复制为无框完整图；没有可用历史时从三图自动退化为两图；
- visual-v5 配置显式关闭 `init_nlp`，与不使用语言描述的训练数据保持一致；
- Prompt v5 使用 `accepted past observations whose boxes may be imperfect`，不把动态历史
  描述成绝对真值；
- canonical 三字段样本要求明确的 memory 监督模式。`feasibility_null` 只允许链路检查，
  正式构建必须提供含来源的逐帧 explicit label；
- visual-v5 的 ms-swift 输入不含 reference `<bbox>`；present 的 assistant `<bbox>` 只绑定
  最后一张 current，absent 不创建空 `objects`；
- visual-v5 的 semantic proposal 需要两个相近的跨帧提议才落库；单次非空输出只留审计
  记录，语义确认窗口独立为 300 帧以适配稀疏观测；
- `qwen3vl_4b_visual_v5_base[_vllm].yaml`、`visual_v5_stage2_vllm.yaml` 和
  `visual_v5_sft_vllm.yaml` 分别用于基座、旧 Stage-2 兼容性诊断和新 adapter。

旧 Stage-1/2 没有接受 visual-v5 或 memory 监督，因此旧 Tiny 指标仍只是历史基线。新
基座单 case 成功也只证明模型能理解任务和输出协议，不能宣称聚合性能提升。

## 4. 下一步执行边界

1. 本机保持轻量：维护代码、生成 1–2 条序列、做像素审计/processor 回放/真实模型 smoke。
2. 训练服务器先运行 `synthesize_visual_v5_dataset.py --plan-only`，冻结 7:3 同序列
   present/absent sampling plan；新 plan 强制 `fixed_identity_anchor` 且 current 不重复。
3. 对 plan 中 current frame 构造带 provenance 的 memory label manifest；不得机械地把
   所有普通 bbox 样本补成 `null` 后声称已完成 memory 监督。
4. 训练服务器重放 plan，生成 visual-v5 probe，先做两步 smoke 和小样本过拟合，再跑
   一轮 LoRA。
5. 对同一 probe 比较从基座初始化与从旧 Stage-2 初始化，随后在 CognitiveBench-Tiny
   使用固定 visual-v5 推理配置聚合评测。
6. probe 有明确增益后，再实现论文版 case-bucket planner、rollout history 与正式大包；
   最终才运行 Full 和 GRPO。

完整命令、标签格式与验收边界见
[`visual_v5_iteration.md`](visual_v5_iteration.md)。

## 5. 已冻结与仍待实验决定

已冻结：首版固定红框和自适应线宽；最近历史保留在 mosaic；无历史使用两图退化；
Prompt 明确历史框可能有误；semantic proposal 跨帧确认后才写入。

仍待指标决定：统一 SFT 从基座还是旧 Stage-2 初始化；正式 memory 正例/难负例的审核
规模；GT/jitter/rollout/stale history 的最终配额；红框样式增强是否能提升跨模型泛化。
