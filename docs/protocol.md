# CognitiveTrack v4 Presence + Memory 协议

## 1. 标注状态

模型输出和真值只使用有监督依据的二值状态：

- `present`：同一目标在当前帧可观测且可定位，并存在有效 GT bbox。
- `absent`：当前帧没有可用于定位的目标。

旧实现中的遮挡、出视野、消失、重现等六分类没有统一真值，因此不进入 v4 训练和主评测。

## 2. 模型输出

在线推理严格输出三个 JSON 字段。bbox 字段名随模型族变化，以下是 Qwen3-VL：

```json
{"target_status":"present","bbox_norm1000_xyxy":[x1,y1,x2,y2],"memory_update":null}
```

`present` 已定义为“初始化指定的那个目标可见且可定位”，因此不再额外输出
`identity_match` 或 `localizability`。`absent` 时 bbox 必须为 `null`。模型也不
输出 `uncertain`、reasoning、target_text 或数值置信度。

Qwen2.5-VL 的等价字段为 `bbox_pixel_xyxy`；同一响应只能出现其中一种 bbox 字段。

`memory_update` 是更新开关与内容的统一表示：

- `null`：普通位移、尺度变化、短时模糊或重复信息，不更新语义记忆；
- 非空短字符串：仅描述新出现、视觉可核验且有助于后续判别的外观、视角、构型或稳定区分线索；
- `absent` 时必须为 `null`，字符串不得写推理过程或完整目标复述。

该字段固定放在 JSON 最后。`null` 路径生成更少 token，是当前的快路径；非空
记忆自然延长生成，是慢路径。逐帧结果记录 `generated_tokens`，可直接量化二者开销。
本地门控还会把 `"no change"`/`"无需更新"` 等误写字符串按 `null` 处理，并拒绝
执行失败、重复文本、过密更新和容量溢出；初始化锚点永不覆盖。

## 3. 执行状态

- `ok`
- `skipped`
- `image_error`
- `model_error`
- `api_error`
- `parse_error`
- `internal_error`

任何执行错误都不得转换为 `absent`。

## 4. bbox

- 框架内部和传统结果：像素级 `[x, y, width, height]`。
- Qwen2.5-VL（零样本与微调）：processor 实际图像网格中的绝对像素 `[x1, y1, x2, y2]`。
- Qwen3-VL（零样本与微调）：当前图上的 `[x1, y1, x2, y2]`，相对到 `[0,1000]`。
- JSON 中无框为 `null`。
- TXT 中无框为 `NaN NaN NaN NaN`。

模型原始候选框不等于最终跟踪框。只有 `target_status=present` 且 bbox 合法时，
才将候选发布为 `target_bbox`。视觉正记忆还必须通过连续观测和可选几何一致性门控；
语义记忆必须同时满足模型非空提议和本地安全门控。

## 5. 评测口径

- `benchmark_standard`：每个序列先计算 Success/Precision/Normalized Precision 曲线，再对序列等权宏平均。
- `cognitive_visible_only`：仅在 GT present 且 bbox 有效的帧上做帧级微平均，用于诊断可见帧定位能力。
- `presence`：评估目标是否存在的二分类结果。
- `reappearance`：统计 absent 后重现的恢复率与恢复延迟。

两组定位指标不可混用。论文 benchmark 主表应报告 `benchmark_standard`，
`cognitive_visible_only` 只作模块诊断和消融。
