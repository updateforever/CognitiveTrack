# CognitiveBench v1

CognitiveBench v1 是 CognitiveTrack 的冻结长时认知跟踪评测标注。该目录随 Git
同步，只保存 bbox、逐帧目标存在状态、稀疏观察关键帧和来源映射；**不复制任何原始
视频帧**。

## 规模

| 来源 | split | 序列 | 帧 | 关键帧 | absent 帧 |
| --- | --- | ---: | ---: | ---: | ---: |
| LaSOT | test | 280 | 685,360 | 172,371 | 18,347 |
| TNL2K | test | 700 | 516,038 | 124,151 | 65,265 |
| MGIT | val | 15 | 207,040 | 47,094 | 50,641 |
| 合计 | test | 995 | 1,408,438 | 343,616 | 134,253 |

总体 absent 比例为 9.5320%；关键帧中的 absent 比例为 9.6634%。MGIT 使用 val 是
因为其 test GT 不公开。

## 目录格式

```text
benchmarks/cognitivebench/v1/
├── benchmark_meta.json
├── README.md
└── test/<sequence>/
    ├── groundtruth.txt    # 每帧 xywh
    ├── target_status.txt  # 1=present, 0=absent
    ├── keyframes.txt      # 0-based 稀疏观察索引
    └── meta.json          # 原始数据集、split、序列名和标注来源
```

`groundtruth.txt` 是稠密评测真值；`keyframes.txt` 只决定昂贵 VLM 在哪些帧执行，不会
缩短序列，也不会允许 tracker 读取未来 GT。非关键帧仍进入标准 runner：纯 VLM 可以
输出 skipped/NaN，hybrid 则可继续运行轻量跟踪器。

## 原始图像依赖

本目录不能脱离以下原始数据集图像单独运行：

- LaSOT Protocol-II test；
- TNL2K test subset；
- MGIT val（15 条有公开 GT 的序列）。

机器路径写入被 Git 忽略的 `configs/env.local.yaml`：

```yaml
datasets:
  cognitivebench: ./benchmarks/cognitivebench/v1
  lasot: /datasets/LaSOT
  tnl2k: /datasets/TNL2K
  mgit: /datasets/MGIT
```

检查标注包，不读取原始图片：

```bash
python tools/verify_cognitivebench.py \
  --root benchmarks/cognitivebench/v1
```

完整标注指纹应为
`fc10d30be2042b9e227c608c19b77772de375b14eddc1134a50e5acfb7fa5a0e`。

检查 loader 与某条真实序列：

```bash
python tracking/inspect_dataset.py \
  --dataset-config configs/datasets/cognitivebench.yaml \
  --env-config configs/env.local.yaml \
  --sequence 005 --check-files
```

## 评测约束

- 标注版本固定为 `v1`，bbox 为像素级 `xywh`，关键帧为 0-based。
- 首帧必须 present，满足标准 SOT 初始化条件。
- `target_status=present` 但 bbox 无效时保留 presence 标签，定位指标跳过无效框。
- v1 已知有 2 个上述无效 present 框（`lion-5` 第 552 帧、`tiger-6` 第 117 帧，
  均按 0-based 索引）；这是上游冻结标注，不在 v1 中静默修正。
- `--debug-frames` 截断结果默认不计入正式指标。
- 主指标沿用 pytracking AUC/OP50/OP75/P/Pnorm；presence、重现等指标仅作认知诊断。
- benchmark 快照是 CognitiveTrack 的评测资产；其上游视频和标注权利仍属于各原始
  数据集发布方。对外再分发前必须遵守 LaSOT、TNL2K、MGIT 的许可条款。

关键帧生成上游不属于本独立仓库，未迁入任何 SOIBench 代码或数据。本目录保存的
冻结索引就是 v1 的评测定义；如需改变关键帧策略，应发布新 benchmark 版本，不能
原地覆盖 v1。
