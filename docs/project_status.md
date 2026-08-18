# CognitiveTrack 当前状态

> 更新日期：2026-08-18。当前代码主线是 VLT-v6.4 / Prompt 6.4.0；v6.3.1 Core
> release 与旧 Stage-1/2 adapter 只作为历史基线保留。

## 当前冻结协议

- 主模型：`Qwen3-VL-4B-Instruct`；
- 三图输入：带框 reference、三个带框可信历史观测、无框 current；
- reference/history 严格早于 current，current 永远不画框；
- Qwen3-VL 输出：

```json
{"bbox_2d":[100,120,400,520],"status":"present","memory_update":null}
```

`bbox_2d` 是 Image 3 上 `[0,1000]` 归一化 `xyxy`。非空 `memory_update` 是完整替换快照，
永久 identity anchor 不被覆盖。

## 已完成的当前代码

- Prompt 6.4.0、`bbox_2d/status/memory_update` parser 与训练导出；
- 大规模 `tracking_sft` 单一正式入口；
- 3 个时间事件、3 个历史质量等级、H0-H3 完整度组成的 27 种合法视觉组合；
- H0/H1/stale 等污染约束及逐行 taxonomy 预检；
- `tracking_sft` 字段级 mask：present/absent 的未知状态值都 mask；
- `state_update_sft` 全监督档位；
- MGIT action 分段挖掘与固定 anchor 状态链；
- OpenAI-compatible frontier API、便携 bundle、断点续跑 journal 和最多约 1,500 条
  额外标签入口；
- 两类状态标签的严格 plan/label 合并器和统一 release 脚本；
- API 标注/合并相关 11 项测试、Ruff、真实 MGIT 回放和 tracking 三图生成 smoke。

## 当前数据事实

历史 v6.3.1 release 保留在：

```text
data/releases/cogtrack_vlt_v631_core_r80_20_case20_robust15_v1
```

其 50,220 unique cases、57,426 views 和 checksum 仅描述旧 Prompt 6.3.1 release，不能
改名冒充当前 v6.4 数据，也不能与新字段协议交叉训练。

当前 v6.4 `tracking_sft` 正式全量 release 已完成：

```text
data/releases/cogtrack_vlt_v640_tracking_sft_r80_20_case20_mgit200_robust15_v1
```

它覆盖 2,511 个序列、66,600 个 unique cases；加入 8,433 个历史污染视图后共有
75,033 行，train/val 为 71,199/3,834，present/absent 为 59,787/15,246。27 种合法视觉
组合全部覆盖，监督审计为 75,033 个 `masked_unknown`、零错误。2026-08-18 已补齐真实
Qwen3-VL train/val processor replay：`<bbox>` 均绑定 Image 3，解码坐标与 norm1000
期望一致。

MGIT tiny/train 的正式状态 release 已完成：

```text
data/releases/cogtrack_vlt_v640_state_update_mgit_segments_v1
```

它包含 91 个可用序列、350 个 verified update、384 个 present hard-null，共 734 条；
train/val 为 645/89，大小约 81 MiB，`masked_unknown=0`。Qwen3 模型族与训练视图预检通过。

额外 API 标注已经完成并通过本地兼容修复与质量门。输入 bundle 曾上传至 ModelScope
private dataset repo：

```text
updateforever/CognitiveTrack-sft-memory
inputs/cogtrack_vlt_v640_state_update_api_qwen36_1500_v1
```

历史 bundle 为 405 MiB，含 228 条序列、4,107 个决策点和 8,442 张图片；全量为 3,024 个
present 与 1,083 个 absent。默认按因果前缀裁剪到 3,000 个输入，即 2,264 次 API
调用和 736 条 GT 零成本 absent。上传 8,449 个文件全部成功，并回读关键文件逐字节验证。
最终保留 2,329 条 API 标签，包含 1,903 个 update 和 426 个 verified hard-null；实际
teacher 模型名为 `qwen3-vl-plus`。不得把同一 teacher 的 QC 描述成 independent
verification。

联合 `state_update_sft` 已生成：

```text
data/releases/cogtrack_vlt_v640_state_update_sft_combined_3063_v1
```

它包含 315 个序列、3,063 条全监督样本，train/val 为 2,861/202；2,253 条 update、
810 条 verified hard-null，`masked_unknown=0`。6,441 张被引用图片全部可解码，train/val
序列零重叠，Qwen3-VL train/val processor replay 均通过。

当前生产 bundle 使用 Prompt 2.2.1：三图角色在消息中显式标注，Image 3 是唯一当前帧，
允许 dynamic memory 大幅纠正粗糙初始文本，并抑制普通方向/背景变化造成的更新。正式
bundle 为 `data/annotation_bundles/cogtrack_vlt_v640_state_update_api_qwen36_1500_v2`，已
上传至：

```text
updateforever/CognitiveTrack-sft-memory
inputs/cogtrack_vlt_v640_state_update_api_qwen36_1500_v2
```

上传 8,449/8,449 文件成功，远端关键文件逐字节回读一致。本机 Aliyun
`qwen3-vl-plus` 的 Prompt 2.2.0 回归中，之前错误的 frame 2540 已修正为正确的空中状态。
API key 不写入命令、bundle 或 Git。

## 训练与评测状态

| 里程碑 | 当前状态 | 完成判据 |
| --- | --- | --- |
| v6.4 tracking SFT 正式数据 | 已完成 | 75,033 行、27 类齐全、监督审计和 train/val processor replay 通过 |
| MGIT 状态数据正式产物 | 已完成 | 734 行、零 masked_unknown、Qwen3 预检通过 |
| 额外状态标签 | 已完成 | 2,329 行、224 序列、质量门报告通过 |
| 联合 state_update SFT release | 已完成 | 3,063 行、零 masked_unknown、图片与 processor replay 通过 |
| 自包含 mixed SFT release | 已完成 | 79,110 条 90/4/6 train；完整图片、README、split 与 checksum 审计通过 |
| tracking 选择性 memory overlay | 可选、尚未实现 | 主数据完成后仅补明确转折，不覆盖原 release |
| 混合全参 SFT smoke | 已完成 | 2×H100、ZeRO-2、冻结 ViT；有限 loss 与 optimizer step 通过 |
| 正式 Qwen3-VL-4B 全参 SFT | 进行中 | 2×H100、每卡 batch 16、全局 batch 32、共 2,473 steps |
| CognitiveBench-Tiny 对照 | 待执行 | Base、旧 Stage-2、新模型同协议比较 |
| TU-GRPO | 待执行 | 晚于 SFT 与状态评测 |

数据生成、processor 回放和两步有限 loss 都不能证明跟踪指标提升。只有冻结
CognitiveBench 的同协议对照可以支持性能结论。

## 立即下一步

1. 完成当前 mixed 全参 SFT，并保留训练、验证和 checkpoint 报告；
2. 完成 v6.4 在线推理全参 checkpoint 加载与无 GT 泄漏 smoke；
3. 在同一 CognitiveBench-Tiny v6.4 配置比较 Base、旧 Stage-2 和新模型；
4. 视成本和收益决定是否制作选择性 memory overlay；
5. SFT 与状态更新评测完成后再进入 TU-GRPO。
