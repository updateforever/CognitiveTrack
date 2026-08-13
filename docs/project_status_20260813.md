# CognitiveTrack 讨论交接摘要（2026-08-13）

本文供研究讨论和其他 AI 快速接手。完整恢复约束见根目录 `AGENTS.md`，统一数据与
Prompt 草案见 `docs/stage2_stage3_data.md`。本页严格区分“已经完成的旧实验”和
“尚待实现的新方案”。

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

- Git HEAD：`70b9dd1ea0721abf6c5d6b7275f96a27163f27c5`；
- CognitiveBench v1 冻结标注已纳入 Git：995 序列、1,408,438 帧、343,616 关键帧；
- 最近一次代码验证：Ruff 通过、126 tests passed、`git diff --check` 通过。

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

目前没有正式 CognitiveBench 证据证明 Stage-1/2 提升了跟踪指标。

## 3. 当前代码仍是什么范式

新方案尚未实现。当前 `TrackingContextBuilder`：

- 首帧完整图不画框；
- 首帧 bbox 作为 Prompt 坐标文本传入；
- history mosaic panel 会画框；
- 当前搜索图无框；
- 正式 Qwen3 tracker 配置仍是 pair、二字段、memory disabled。

已有但尚未正式启用的能力包括三字段严格 parser、semantic memory 回灌、视觉 history
bank、memory gate 和逐帧审计 JSONL。不能把这些代码存在等同于新范式已验证。

## 4. 下一步实现边界

需要同时修改，不能只改 Prompt：

1. `cogtrack/context/builder.py`：首帧和所有历史统一使用同一个绘框函数；当前图无框。
2. `cogtrack/prompts/pair.py`、`mosaic.py`、`common.py`：删除输入坐标文字，冻结统一
   三字段 Prompt 和新版本号。
3. 训练数据构造与 ms-swift 导出：输入不再使用 reference `<bbox>` object；assistant
   bbox 仍通过官方 object 绑定最后一张图。
4. tracker 配置：mosaic、memory output/semantic enabled、Stage-2 或新 adapter 路径。
5. 测试：像素级确认参考/历史有框、current 无框；两图/三图的图片计数与 bbox image_id；
   无 current/future GT；parser 和 memory gate。
6. 先生成小数据做 processor 回放、两步 smoke、小样本过拟合，再生成正式混合数据。

## 5. 需要讨论并冻结的问题

- 框颜色、线宽是否固定，是否加入少量样式增强；
- 历史 mosaic 是否足够突出最后一个 panel，还是给最近可信观测单独一张图；
- Prompt 使用 `trusted history` 会不会让模型过度相信扰动框，是否改成
  `accepted past observations whose boxes may be imperfect`；
- 无动态历史时采用两图退化，还是始终保留一个空 history 占位；
- 统一训练从基座开始还是从旧 Stage-2 adapter 继续，必须做同数据验证集消融；
- 普通 `memory_update=null` 的自动标签边界，以及 MGIT appearance/action/activity/story
  的审核策略；
- semantic proposal 是否必须跨帧确认后才写入，避免单帧错误文本污染后续 Prompt。

推荐先冻结这些问题，再改代码和重建数据；否则 Prompt、渲染方式和数据会反复重做。
