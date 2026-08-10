# SUTrack 内置运行时与插件边界

本目录同时提供稳定插件契约和可直接使用的 SUTrack 最小推理实现。内置 runtime
包含 Fast-iTPN、CENTER decoder、CLIP 文本塔、预处理和在线模板更新，不包含训练
代码或 checkpoint，也不会回退导入父仓库。核心网络算子来自 MIT 许可的 SUTrack，
新增运行时与 CognitiveTrack 其余代码使用相同许可。

## tracker 配置

```yaml
implementation:
  module: cogtrack.models.sutrack.runtime
  factory: build_sutrack_runtime
  kwargs:
    model_config: configs/models/sutrack_b384.yaml
    checkpoint: ${COGTRACK_SUTRACK_CHECKPOINT}
    device: auto

output:
  bbox_key: target_bbox
  bbox_format: xywh
  execution_key: execution
  score_key: best_score
```

模块只在 tracker 第一次 `initialize()` 时导入。工厂签名固定为：

```python
def build_sutrack_runtime(*, params, checkpoint):
    return MySUTrackRuntime(params=params, checkpoint=checkpoint)
```

运行时对象必须实现 `initialize(image, info)` 和 `track(image, info)`。图像是 RGB
`numpy.ndarray`，内部 bbox 始终为像素 `xywh`。如果原实现输出 `xyxy`，必须在
`output.bbox_format` 中明确声明，adapter 不会根据坐标数值猜测格式。

`initialize()` 可以返回 `None`，adapter 会使用 `info['init_bbox']` 记录首帧结果；
后续 `track()` 成功时则必须返回有效 bbox。执行状态采用 CognitiveTrack 的工程状态：
`ok`、`skipped`、`image_error`、`model_error`、`api_error`、`parse_error` 或
`internal_error`。这些状态只描述代码是否成功执行，不代表目标是否存在，也不会被
解释成身份判断。

工厂返回的对象若提供 `close()`，adapter 会在序列结束时调用它。模型加载、设备选择、
checkpoint 校验和显存释放均属于插件实现的职责。内置 runtime 会过滤 checkpoint
中未参与跟踪推理的 CLIP 视觉塔，并要求其余权重零 missing、零 unexpected；不同
序列共享只读网络，但不共享框、模板或认知状态。

Hybrid 重定位可选实现：

```python
def correct(self, image, bbox_xywh, info):
    # 用已经 VLM 身份门控的全局框重置运动状态/模板。
    return {"applied": True, "reason": "state and template updated"}
```

运行时没有 `correct()` 时 adapter 会明确返回 `supported=false`，不伪装回灌已成功。
可运行配置见 `configs/trackers/sutrack_b384.yaml` 与
`configs/trackers/hybrid_sutrack_b384_qwen25vl.yaml`；通用外部插件模板仍保留在
`configs/trackers/hybrid_sutrack_qwen_template.yaml`。
