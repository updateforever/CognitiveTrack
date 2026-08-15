# 旧分阶段范式 CognitiveBench-Tiny 结果归档（2026-08-13）

> 本文只用于复现旧二字段/阶段化训练，不代表当前 VLT-v6.3 论文主线。

本文归档 Base、Stage-1 与 Stage-2 在旧训练范式下的一次完整初步实验。它用于保留
研究证据和后续对照，不代表下一版统一视觉画框训练方案已经实现或验证。

## 实验边界

- 数据集：冻结的 CognitiveBench-Tiny v1，24 条完整序列，共 39,251 帧；
- 数据来源：MGIT / LaSOT / TNL2K = 1 / 7 / 16 条序列；
- 稀疏观察：9,892 个关键帧，除初始化外共 9,868 次 VLM 请求；
- Observation rate：25.20%；
- 三组实验使用同一套在线推理和评测代码，只替换模型权重；
- 推理上下文为首帧完整图、可信历史预测 mosaic 和当前完整关键帧；历史 panel 画框，
  当前图不画框，但首帧仍未画框并通过 Prompt 传入 bbox 坐标文本；
- 推理请求使用 `target_status + bbox + memory_update` 三字段协议。旧 Stage-1/2 只接受
  过二字段监督，因此这里同时检验旧 adapter 在三字段运行时中的兼容性；
- 非关键帧不执行 VLM。`hold_last` 衡量任意时刻的最近预测状态，
  `observation_only` 只衡量模型实际观察当前图时的能力；
- 本结果只覆盖 Tiny，不替代 CognitiveBench Full 的最终主表。

## 权重与训练范式

| 实验 | 权重 | 旧训练范式 |
| --- | --- | --- |
| Base | `Qwen3-VL-4B-Instruct` | 未进行跟踪 SFT |
| Stage-1 | `checkpoint-19005` | pair、二字段、reference 坐标文本 |
| Stage-2 | `stage2-mosaic-robust-v2/checkpoint-28819` | 从 Stage-1 同一 LoRA 继续训练；mosaic、二字段、含扰动历史 |

Stage-2 adapter SHA-256：
`7437dd2be3bae21070059ef5ce704da7bbd5008607f7e04eb18b3638f04930a1`。

## 聚合结果

所有数值均为同一冻结评测器的输出，百分数按 0--100 展示。

| 实验 | Dense-zero AUC | Hold-last AUC | Observation-only AUC | Presence F1 | Absent FPR | Present miss | Reappearance recovery | 错误帧 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 5.63 | 21.81 | 21.61 | 93.41 | 33.11 | 8.43 | 30.67 | 0 |
| Stage-1 | 14.45 | 52.64 | 56.45 | 96.05 | 45.14 | 1.88 | 88.96 | 163 |
| Stage-2 | 15.04 | 55.05 | 58.62 | 96.16 | 15.24 | 5.48 | 89.57 | 5 |

Stage-2 的完整定位指标为：

| 口径 | AUC | OP50 | OP75 | P@20 | Pnorm@0.2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense_zero` | 15.04 | 17.51 | 14.56 | 15.38 | 17.00 |
| `hold_last` | 55.05 | 64.35 | 46.03 | 52.47 | 60.44 |
| `observation_only` | 58.62 | 68.28 | 56.72 | 60.26 | 66.20 |

## 初步结论

Stage-1 已显著改善定位和重捕获能力，但 absent FPR 上升到 45.14%，且出现 163 个
结构化输出错误。Stage-2 在 Stage-1 基础上进一步带来：

- hold-last AUC `+2.41` 个百分点；
- observation-only AUC `+2.17` 个百分点；
- absent FPR `-29.90` 个百分点；
- reappearance recovery `+0.61` 个百分点；
- 错误帧从 163 降至 5；
- present miss 增加 `3.60` 个百分点，说明存在性判别仍有 precision/recall 权衡。

因此，旧实验支持“跟踪 SFT 与带扰动时序上下文具有潜力”这一初步判断，但不能回答
新的视觉指代输入、统一三字段监督和语义记忆学习是否有效。下一轮应以统一视觉画框
数据从头建立可复现 baseline，并将本文结果仅作为历史对照。

## 本地原始报告

若本机仍保留未纳入 Git 的运行产物，完整报告位于：

- `outputs/cognitive_vlm/qwen3vl_4b_cognitive_v1_base/cognitivebench/evaluation_tiny/`；
- `outputs/cognitive_vlm/qwen3vl_4b_cognitive_v1_stage1/cognitivebench/evaluation_tiny/`；
- `outputs/cognitive_vlm/qwen3vl_4b_cognitive_v1_stage2/cognitivebench/evaluation_tiny/`。

大规模预测、逐帧 JSONL 和模型权重不提交 Git；对外复现应固定代码 commit、数据子集
清单、adapter revision/SHA-256、tracker 配置和 evaluator 版本。
