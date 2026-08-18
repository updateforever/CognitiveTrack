# state_update_sft 数据方案

> 更新日期：2026-08-17。当前方案不是“只做 1,500 条”，而是两部分相加：MGIT
> 官方 action 分段尽量全量利用，再额外生成约 1,500 条 LaSOT/TNL2K
> Qwen3.6 API 动态指代表达标签。

## 1. 监督目标

模型输出保持与在线跟踪完全一致：

```json
{"bbox_2d":[100,120,400,520],"status":"present","memory_update":null}
```

`memory_update` 非空时必须是完整、自包含的动态指代表达，不是增量日志。初始化身份永久
不覆盖；动态 memory 可以明确记录消失，并在重现时替换为新的当前指代表达。

`state_update_sft` 只收两类有证据的标签：

- `verified_update`：`memory_update` 是可靠的非空替换快照；
- `verified_hard_null`：明确确认当前不应更新，`memory_update:null` 全监督。

不确定的 present 帧不进入这个数据集，也不会被伪造成 hard-null。它们只在大规模
`tracking_sft` 中以 `masked_unknown` 存在，其 `null` 值不参与 loss。

## 2. 两个数据来源

### 2.1 MGIT 官方 action 分段：尽量全量使用

MGIT `attribute/description/<sequence>.json` 的 `action` 层提供带起止帧的分段文本。第一段
作为初始化身份/状态，之后真正发生文本变化的可靠分段在变化后的第一个 present 帧形成
`verified_update`。每个可靠分段再取一个距边界至少 30 帧的稳定 present 帧作为
`verified_hard_null`。

当前 tiny/train 镜像的事实：官方名单 105 条，10 条缺帧目录、4 条目录为空，实际使用
91 条。2026-08-17 全量只读规划结果为：

| 项目 | 数量 |
| --- | ---: |
| 原始 action 分段 | 513 |
| 丢弃的损坏分段 | 3 |
| 可靠 `verified_update` | 350 |
| 稳定 `verified_hard_null` | 384 |
| MGIT 合计 | 734 |
| 因边界/超长等原因降级后未收录的 probe | 29 |

这里的“全量”是全量使用可形成可靠状态监督的分段，不是把一个分段内所有帧重复成相同
标签。一段状态只需要一次 update；重复逐帧会人为放大 MGIT 权重和 null 比例。

正式产物已生成：

```text
data/releases/cogtrack_vlt_v640_state_update_mgit_segments_v1
```

train/val 为 645/89 行，约 81 MiB，零 `masked_unknown`。关键 checksum：

```text
sampling_plan.json                 f811eef498942f140c19e13b4c73baebf5b9e919f3c1b8ea329196cd64c449c5
source_samples.jsonl               4a5ec339a8e84dd6894a9db78689d8cdf5b05c82320b371a8e6c6d38462f9bc9
ms_swift/qwen3_vl/train.jsonl      905f431576f02683d3aceb640479287366feba514faf6a270c9b53e6feaa3e00
ms_swift/qwen3_vl/val.jsonl        b83e9435037b4a16c90282c4863cb6396c1b384e97ecc42ccb994a9e2181e6fd
mgit_state_update_labels.jsonl     0e9d207cf5dc8ea28ca7b4581e535f3a3676acacf272d150e5ac0c0d86c1d91d
```

重建入口：

```bash
bash scripts/generate_state_update_sft_data.sh <mgit-release-name>
```

该脚本会生成 MGIT 计划、标签和独立训练视图；最终联合 release 仍需与第二部分合并。

### 2.2 额外约 1,500 条 Qwen3.6 API 标签

LaSOT/TNL2K 没有 MGIT 这种逐段人工文本，因此本机从 train split 构造独立固定锚点
采样计划，并打包为可通过 ModelScope 搬运的三图标注数据。远端调用更强的闭源 Qwen3.6
OpenAI-compatible API，每个 present 决策点只生成一次；Prompt 要求模型输出
`update/keep/uncertain`、变化要素、置信度、证据充分性和完整指代表达。

规则层只做结构与可信度质量门，不用视角/姿态/尺度黑名单替模型作语义判断。低置信度、
`uncertain`、格式错误、身份漂移和与输入状态等价的空更新全部丢弃。消失转折不调用 API：
数据集 GT 直接产生 `The target has disappeared ...`；持续 absent 保持 null。后续 present
case 若输入状态是“已消失”，必须生成带重现语义的当前指代表达，否则该行丢弃。

默认最终标签上限是 1,500；跨序列 round-robin 只截断各序列的时间后缀，不能随机删除
状态链中间节点。
这 1,500 是 MGIT 734 条之外的额外数据，因此首版联合数据预计约 2,234 条，实际数以
质量门产出为准。

```bash
bash scripts/generate_state_update_api_bundle.sh <bundle-release-name>
bash scripts/modelscope_state_update_transfer.sh upload \
  <owner/dataset-repo> data/annotation_bundles/<bundle-release-name> \
  inputs/<bundle-release-name>
```

远端下载 bundle 后设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`API_MODEL`，运行包内
`tools/annotate_state_update_openai_api.py`。脚本逐行写 journal、支持 `--resume`，结果目录
带完整原始响应、标签、报告和 `SHA256SUMS`，可传回同一个 ModelScope dataset repo。
正式报告要求 `annotation_policy=single_pass_frontier_api_v1`、`quality_gate_applied=true`、
`dry_run=false`、`minimum_output_reached=true`，默认最低接受 1,200 条。该策略不是独立
verifier，不得在论文或报告中写成 independently verified。

## 3. 固定锚点与因果状态链

状态文本必须相对于同一个永久 identity anchor 解释，因此两类状态数据都固定使用序列
首个有效 present 帧作为 Image 1。每条序列按时间顺序维护 `current_target_state`：只有
已接受的非空 update 才推进快照；hard-null 和被拒候选不推进。消失转折是非空 update，
因此会把动态 memory 推进为“已消失”；持续 absent 不重复更新，重现后再替换为当前描述。

学生仍只看到正式三图输入：

1. 带红框的固定 identity reference；
2. 三个更早可信 present 观测组成的带框历史条带；
3. 当前无框搜索图。

API 教师看到与学生相同的 identity/history；present current 额外带 GT 红框以免教师承担
定位任务。GT current 框、原始响应和证据解释只用于离线标注，不进入学生输入。

## 4. 合并为一个 state_update_sft release

不能直接用 shell 拼接两个标签 JSONL。teacher 计划还包含未产出可靠标签的决策点；若不
过滤，显式监督会缺帧或偷偷混入 `masked_unknown`。合并器会只保留真正有标签的帧，重算
统一 sampling plan，并检查固定锚点、因果状态链、重复键和 API 质量报告：

```bash
bash scripts/build_state_update_sft_release.sh \
  <final-release-name> \
  <mgit-plan> <mgit-labels> \
  <teacher-plan> <teacher-labels> <teacher-report>
```

最终 release 使用 `sft_supervision_profile=state_update_sft` 和
`memory_supervision=explicit`。逐行必须全监督，以下预检不能出现 `masked_unknown`：

```bash
python tracking/validate_sft_supervision.py \
  --profile state_update_sft \
  --dataset data/releases/<release>/ms_swift/qwen3_vl/train.jsonl \
  --dataset data/releases/<release>/ms_swift/qwen3_vl/val.jsonl
```

## 5. 与大规模 tracking_sft 的关系

两套数据承担不同信号：

| 数据 | 规模 | bbox/status | `memory_update` 值 |
| --- | ---: | ---: | ---: |
| `tracking_sft` | 66,600 unique cases，另含鲁棒视图 | 全监督 | present/absent 的 null 都是占位并 mask |
| `state_update_sft` | 约 2.2K，待正式生成 | 全监督 | 指代表达、消失、重现与 hard-null 全监督 |

最终目标是把两类数据混合到同一次 Qwen3-VL-4B LoRA SFT 中，让同一套 LoRA 同时学习
跟踪和状态更新；不会串联三套 LoRA。正式混合比例和统一数据根打包仍需在两份 release
生成后冻结，并通过两步 smoke 验证逐样本 loss mask。

## 6. 可选的 tracking 样本选择性补标

第一版流程不会把 `state_update_sft` 标签回填到 `tracking_sft`，也不要求两类 case 一一
对应。两份 release 必须先独立完成和审计。

主数据完成后，可以另加一次低优先级的选择性补标：从 `tracking_sft` 中筛出 GT 能明确
证明状态转折、但 `memory_update:null` 仍为 `masked_unknown` 的 case，例如消失转折、
重现或少量显著外观变化候选，再交给 Qwen3.6 标注。持续缺失只有在完整时序链能确认
“无需再次更新”时才可成为 hard-null；状态转折不明确的行继续 mask。

补标必须保存为按 `sample_id` 对齐的 overlay 或新的派生 release：

- 不覆盖、重写或改名原始 `tracking_sft`；
- 训练打包时用补标行替换对应 masked 行，不能把两个冲突版本同时采样；
- 保存教师、Prompt、原始响应、质量门和 provenance；
- 它仍属于 `tracking_sft` 的可选监督增强，不引入第三个数据桶，也不与首轮约 1,500 条
  独立 `state_update_sft` 生成耦合。

是否执行这一步应在两份主 release 完成后、冻结最终混合配方前，根据 API 成本和标签
覆盖率决定；它不阻塞当前 tracking release、约 1,500 条独立状态标签或两类数据的基础
训练 smoke。

## 7. 质量与泄漏约束

- 只使用 LaSOT/TNL2K/MGIT 的 train split，按完整序列切分；
- reference/history 必须严格早于 current，current 永远不画框；
- 只有消失转折产生 absent 非空更新；持续 absent 不重复改写；
- API 教师的 GT current 框、原始响应和理由不进入学生 JSONL；
- 数据生成、processor 回放或有限 loss 不代表跟踪提升；只有冻结 CognitiveBench 对照能
  支持性能结论；
- TU-GRPO 必须晚于混合 SFT 和状态更新评测。
