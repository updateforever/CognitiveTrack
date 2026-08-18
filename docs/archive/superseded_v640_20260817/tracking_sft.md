# VLT-v6.4 tracking SFT

> 当前正式数据入口是 `scripts/generate_tracking_sft_data.sh`。这里的
> `tracking_sft` 指大规模跟踪监督，不是模型输出协议的字段名。

## 固定三图输入

1. Image 1：同序列更早的任意有效 present reference，红框是永久身份锚点；它不必是
   视频原始第 0 帧。
2. Image 2：reference 与 current 之间严格更早的三次可信 present 观测，按时间从左到右
   排列。历史不足时复制最近观测或 reference 进行 padding；padding 不代表额外预测。
3. Image 3：当前完整搜索图，永远不画框。

历史条带不含帧号、箭头或文本；panel 之间只有白色分隔带。图片导出默认长边 648，
与当前 Qwen3-VL 推理视觉 token 预算对齐。

## 模型输出

Qwen3-VL 固定输出：

```json
{"bbox_2d":[100,120,400,520],"status":"present","memory_update":null}
```

目标缺失时：

```json
{"bbox_2d":null,"status":"absent","memory_update":null}
```

`bbox_2d` 是当前图上的 `[0,1000]` 归一化 `xyxy`。`memory_update` 仍是最终推理协议
的一部分：非空值必须是完整、自包含的替换状态快照；tracking SFT 的未知占位 `null`
不提供该值的 loss 监督。

## 9 个主场景与 27 种视觉组合

时间事件分为：

- `continuous_present`：相对 Image 2 最近可信观测，中间没有 absent；
- `absent`：current 是真实同序列缺失帧；另记 `start/middle/end/single` 供审计；
- `reappearance`：最近可信历史之后出现过 absent，current 又恢复 present。

历史框质量分为：

- `clean`：所有历史框正确；
- `jitter_box`：恰好一个 panel 的框发生平移/缩放扰动；
- `stale_box`：恰好一个 panel 使用另一个不同历史观测的旧框。

历史完整度由不同帧号而不是 panel 数量判定：

- H0 `h0_reference_only`：`[reference, reference, reference]`；
- H1 `h1_one_observation`：一个动态观测，其余复制；
- H2 `h2_two_observations`；
- H3 `h3_three_observations`。

合法组合是：clean×H0/H1/H2/H3，jitter×H1/H2/H3，stale×H2/H3。三个时间事件与这
九种形式笛卡尔积得到 27 种组合。H0 不允许任何扰动，H1 不生成 stale；stale 必须
确实改变一个 panel 的 bbox。

生成器将以下字段写入 source metadata，并在写训练 JSONL 前由
`tracking/validate_tracking_sft_taxonomy.py` 逐行复算：
`temporal_event`、`absent_phase`、`history_quality`、`history_completeness`、
`tracking_scenario`、`visual_combination`、`reference_source`。

## tracking_sft 监督

大规模数据不读取 MGIT 状态文本，也不调用状态教师。它的职责是先把定位、存在性和输出
结构训练起来；`memory_update` 只保留 schema 占位，不在这一阶段猜测状态文本：

- present：`memory_update:null`，状态为 `masked_unknown`，只 mask `null` 值；
- absent：同样写 `memory_update:null`，也标记为 `masked_unknown`，不把“缺失后是否应
  更新记忆”伪装成确定答案；
- bbox、status、字段名、JSON 闭合和 EOS 始终监督。

因此 `tracking_sft` 不会强迫模型在推理时永远输出 `null`，也不会覆盖模型自身的状态
分析能力。明确的消失、持续缺失、重现和显著变化标签属于独立的 `state_update_sft`。

正式 `tracking_sft` release 完成后，可以可选地筛选其中时间转折明确但仍为
`masked_unknown` 的样本（例如可证实的消失或重现）交给强教师补标。补标结果必须作为
按 `sample_id` 对齐的 overlay/派生 release 保存，不能回写覆盖原始 release；训练打包时
由补标行替换原 masked 行，避免同一个 case 同时出现冲突的两个版本。无法从时序真值
确定状态转折的样本继续保持 mask。

新监督档位为 `tracking_sft`，插件名为 `cogtrack_tracking_sft`。旧
`tracking_core` / `cogtrack_tracking_core` 只作为读取历史 release 的兼容别名，不会
写入新产物。

## 生成与预检

```bash
bash scripts/generate_tracking_sft_data.sh \
  cogtrack_vlt_v640_tracking_sft_r80_20_case20_mgit200_robust15_v1
```

小规模检查：

```bash
DRY_RUN=1 DRY_RUN_SEQS=12 MAX_CASES_PER_SEQ=3 MGIT_CAP=3 \
  bash scripts/generate_tracking_sft_data.sh smoke_tracking_sft
```

训练前至少执行：

```bash
python tracking/validate_qwen_training_view.py \
  --model /root/public/models/Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/releases/<release>/ms_swift/qwen3_vl/train.jsonl \
  --dataset data/releases/<release>/ms_swift/qwen3_vl/val.jsonl \
  --expected-family qwen3_vl
python tracking/validate_sft_supervision.py \
  --profile tracking_sft \
  --dataset data/releases/<release>/ms_swift/qwen3_vl/train.jsonl \
  --dataset data/releases/<release>/ms_swift/qwen3_vl/val.jsonl
```

协议从旧 v6.3.1 的 `target_status/bbox_norm1000_xyxy` 改为当前冻结的
`status/bbox_2d`，因此必须使用新的 release 名和 Prompt 6.4.0；旧 release 不原地修改。
