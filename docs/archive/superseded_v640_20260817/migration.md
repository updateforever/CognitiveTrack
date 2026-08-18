# 能力范围与迁入清单

## 迁移原则

只迁入已经验证且属于认知跟踪主线的能力，不复制历史文件、实验分支或
机器配置。本项目的可提交代码必须能够独立安装，运行时不得导入目录外的研究模块。

## 纳入 v1

- pytracking 的 `Sequence`、`SequenceList`、`BaseDataset` 和 `BaseTracker` 范式；
- CognitiveBench、LaSOT、TNL2K、MGIT；
- 标准单序列/数据集运行与传统结果；
- Dense/Keyframe observation policy；
- SUTrack-B384 推理 baseline、公开 checkpoint runtime 与可替换插件边界；
- 每帧 SUTrack 连续定位 + 关键帧 VLM 身份裁决的 hybrid 控制器；
- Qwen2.5-VL/Qwen3-VL 本地多图推理；
- pair/mosaic 上下文；
- 严格输出解析和 bbox 转换；
- 可追溯的时序记忆；
- 传统、存在性、身份和恢复评测；
- ms-swift SFT/GRPO 数据与奖励。

## 明确排除

- 其他研究子系统的代码、配置、数据路径和实验设计；
- 无真值支持的旧六分类状态；
- 旧 VLM tracker 的重复版本；
- 本机绝对路径文件；
- 旧实验输出和可视化；
- 第一版暂不需要的 VOT/RGBD/RGBT 入口；
- 第一版暂不迁移传统 SUTrack 训练框架。

## v1.1 已完成的 SUTrack 最小迁入

- Fast-iTPN tiny/base/large 构造器，其中 B384 已完成真实 checkpoint 验证；
- CENTER decoder、任务 decoder、裁剪/坐标/Hann window 预处理；
- 只重建文本侧的 CLIP ViT-L/14，跳过跟踪推理不使用的 CLIP 视觉塔；
- 全零 token 模式与原 SUTrack `USE_NLP=False` 行为一致；
- checkpoint 严格键匹配、进程级模型缓存、在线模板更新和 Hybrid `correct()` 回灌。

当前未迁入 SUTrack 训练器、优化器和数据采样器。

### 移植保真度：逐帧数值核对

推理路径被重新组织过，所以"重构有没有引入误差"必须用证据回答，而不是靠代码
review。[tools/verify_sutrack_parity.py](../tools/verify_sutrack_parity.py) 在同一
份帧清单、同一份 checkpoint、同一个 torch 上分别跑原版 SUTrack 仓库和本项目的
内置 runtime，然后逐帧比较 bbox：

```bash
export COGTRACK_SUTRACK_CHECKPOINT=/path/to/SUTRACK_ep0180.pth.tar
python tools/verify_sutrack_parity.py all --dataset videocube --frames 400
python tools/verify_sutrack_parity.py all --dataset tnl2k    --frames 200
```

已核对结果（sutrack_b384，官方 `SUTRACK_ep0180.pth.tar`，torch 2.9.0+cu128）：

| 语言分支 | 数据集 | `USE_NLP` | 帧数 | 最大坐标差 | 最小 IoU |
| --- | --- | --- | --- | --- | --- |
| 全零 token | videocube 029 | False | 400 | 0.000000 px | 1.000000 |
| CLIP tokenize | TNL2K | True | 200 | 0.000000 px | 1.000000 |

两条语言分支都逐帧 bit 级一致，其中 videocube 覆盖了 16 次在线模板更新。

逐帧一致只证明"预测相同"，不证明"指标口径相同"，后者由
[tools/verify_metrics_parity.py](../tools/verify_metrics_parity.py) 单独核对：同一份
GT、同一份预测，一边用 AST 从原版 `extract_results.py` 抽出的函数聚合，一边走本项目
真实评测路径。

```bash
python tools/verify_metrics_parity.py \
    --result-dir outputs/sutrack_adapter/sutrack_b384_builtin_v1/mgit \
    --dataset mgit
```

在 videocube_val_tiny 全部 15 条序列（207,040 帧）上，五个指标最大绝对差
`0.00e+00`：

| 指标 | 原版 | 本项目 |
| --- | ---: | ---: |
| Success AUC | 0.5316542387 | 0.5316542387 |
| Success OP50 | 0.5917064548 | 0.5917064548 |
| Success OP75 | 0.5482248068 | 0.5482248068 |
| Precision @ 20px | 0.5256611109 | 0.5256611109 |
| Norm. Precision @ 0.2 | 0.8596459031 | 0.8596459031 |

原版那个文件不能直接 import：其 import 链会拖进训练代码，而 torch 2.x 删掉了
`torch._six`，环境里也没有 `jpeg4py`。所以工具用 AST 抽出三个纯 torch 函数并原样
`exec`，跑的是原版源码文本而不是手抄版本。

两个容易踩的坑：

- **两边必须用同一个 torch。** torch 1.11 与 2.9 的 cuBLAS kernel 不同，会带来
  约 1e-2 px 的差异，并在递归跟踪里累积到 1 px 以上。那不是移植错误。
- **`multi_modal_language` 必须为 `true`。** 原版在 `USE_NLP=False` 时仍会把全零
  token 送进 CLIP 文本塔，`text_src` 是编码结果而不是 `None`。置 `false` 会走完全
  不同的 encoder 分支，逐帧预测必然漂移。

### 指标口径与 pytracking 对齐

主指标走 [cogtrack/evaluation/pytracking_metrics.py](../cogtrack/evaluation/pytracking_metrics.py)，
它是 `lib/test/analysis/extract_results.py` 的移植，因此刻意保留了两个官方约定：

- success 曲线用**严格大于**比较 21 个阈值（含 1.0），所以完美预测的 AUC 是
  20/21 ≈ 0.9524 而不是 1.0；
- 每条序列第 0 帧的预测被强制替换成 GT（`pred_bb[0, :] = anno_bb[0, :]`）。

这两条都由 [tests/test_evaluation_metrics.py](../tests/test_evaluation_metrics.py)
锁住。改成 `>=` 或去掉首帧替换都会偏离 SUTrack 已发表数字。

第三条约定关于 `dataset` 标签。原版 `calc_seq_err_robust(pred_bb, anno_bb, dataset,
target_visible)` 用 `seq.dataset` 选分支，其中 lasot 专有一条：

```python
if dataset == 'lasot':
    err_center_normalized[~target_visible] = float("Inf")
    err_center[~target_visible] = float("Inf")
```

通用分支只做 `err_center_normalized[~valid] = -1.0`，而 `-1.0 <= threshold` 对
0~0.5 的全部阈值都成立，于是不可见帧会被算成 Pnorm **命中**。也就是说这个字符串
直接决定 Pnorm 数值，不是元数据。

移植过程中这里曾经从序列名去猜 dataset（`sequence.split("_")[0]`）。lasot 的
序列名形如 `airplane-1`，不含下划线，猜出来是 `"default"`，lasot 分支永远不执行，
lasot 的 Pnorm 会被系统性抬高。现在 `dataset` 由 runner 写进每条 JSONL 帧记录，
经 `CanonicalFrame.dataset` 原样传到误差计算，由
[tests/test_dataset_tag_metrics.py](../tests/test_dataset_tag_metrics.py) 锁住
两条分支必须给出不同数字。

在本项目当前四个数据集里，只有 lasot 走特殊分支；tnl2k / mgit / cognitivebench
都等价于通用分支，所以 mgit 上的历史数字不受影响。

CognitiveTrack 自定义的 presence / identity / reappearance 指标统一下沉到
`summary["cognitive_diagnostics"]`，只作诊断，不参与对外比较。

## VLM bbox 坐标协议

这一节记录一个已经真实发生过、且会把零样本基线整体压成 0 的错误。

Qwen2-VL 的 grounding 约定是 `[0,1000]` 相对坐标；Qwen2.5-VL 改成了
**模型输入图自身像素网格里的绝对坐标**，而 Qwen3-VL 又使用 `[0,1000]` 相对
坐标。三个名称相近，但不能据此推断坐标协议。本仓库的 Prompt 原先无条件要求
`bbox_norm1000_xyxy`，等于让 Qwen2.5-VL 输出它没被训练过的格式；解析端又把它
返回的绝对像素当成归一化值除以 1000，于是每个框都被压向左上角。

实测后果（`videoPlayer_video_09_done`，347 帧 / 69 个观测帧，Qwen2.5-VL-7B）：

| bbox 解释方式 | 观测帧 mean IoU | IoU>0.5 | IoU==0 |
| --- | --- | --- | --- |
| 按 norm1000（修复前的线上行为） | 0.0000 | 0/68 | **68/68** |
| 按模型空间绝对像素（Qwen 原生） | 0.2136 | 10/68 | 17/68 |
| 按原图绝对像素（对照） | 0.0638 | 0/68 | 35/68 |

模型的文字推理当时是对的（"video player ... in the bottom right corner"），只有
数字不对，所以这个 bug 不会以“模型看不懂”的形式暴露，只会表现为 AUC 接近 0。
判定它是坐标系错误而不是模型能力问题的依据是残差过于整齐：y1 比值中位 0.393
而标准差只有 0.013，中心 dy 中位 −434.1 而标准差只有 14.3。

参考系是**模型真正看到的那张图**，不是原始视频帧，也不是我们传给后端的图。
processor 会做两级缩放，两级都不能省：

```text
原图 930x510 --max_image_side=648--> 648x355 --smart_resize(28 的整数倍)--> 644x364
```

所以 `max_image_side` 的值不能用来反算模型空间。真实尺寸由
`HuggingFaceQwenVLBackend._model_image_sizes` 从 processor 自己产出的
`image_grid_thw * patch_size` 回推，经 `VLMResponse.image_sizes` 传到解析端。
不在本仓库重写一份 `smart_resize`，是为了避免 transformers 改变缩放规则时产生
静默偏移。

当前实现有两条协议，由 tracker 配置的 `bbox_protocol` 显式选择，并且训练/推理
保持同代际一致：

- `qwen_abs_pixel`：历史模型 JSON 字段名为 `bbox_pixel_xyxy`，用于 Qwen2.5-VL；
- `norm1000`：当前 Qwen3-VL 模型可见字段名为 `bbox_2d`，值为 `[0,1000] xyxy`；
  旧 Prompt 6.3.1 曾使用 `bbox_norm1000_xyxy`。

字段名与模型代际/Prompt 版本共同冻结，旧字段不能跨版本混用。协议名和模型空间尺寸
都逐帧写入
`ContextInfo.bbox_protocol` / `ContextInfo.model_image_size`，否则事后无法判断
某个 run 的框该怎么解释。

`qwen_abs_pixel` 下缺少 `model_image_size` 时解析器直接报 `parse_error`，不用原图
尺寸兜底——两者通常不等，兜底会引入与目标位置相关的系统性偏移，并被记成正常
预测，正是这次的故障形态。行为由
[tests/test_bbox_protocol.py](../tests/test_bbox_protocol.py) 锁住，其中包含
frame 3 的真实数字：同一组模型输出在 norm1000 下 IoU 恰为 0，在原生协议下为 0.228。

训练侧已改成 ms-swift 官方 `<bbox> + objects.bbox + image_id` 表示：canonical 标注
只负责审计，Qwen2.5-VL 模板生成 resize 后绝对像素，Qwen3-VL 模板生成相对
0-to-1000 坐标。两代模型使用独立 JSONL 目录，启动脚本在训练前检查模型族，旧的
通用 `ms_swift/train.jsonl` 不再作为有效训练入口。
