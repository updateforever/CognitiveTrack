#!/usr/bin/env python3
"""从 MGIT action 分段生成逐帧 state-update 监督标签 JSONL。

用法是三步流水线的第二步，因为标签依赖**实际被采样的当前帧**：

1. ``synthesize_vlt_v6_dataset.py --plan-only`` 产出 ``sampling_plan.json``；
2. 本脚本读取该 plan，只对 MGIT 序列生成 ``state_update_labels.jsonl``；
3. 正式构建时同时传 ``--sampling-plan`` 与标签文件，
   并使用 ``--memory-supervision three_state``。

把标签生成独立成一步的理由：文本标签是全流程里唯一"可能错标"的环节，独立产物才能
被离线复核、diff 和版本化，而不是埋在一次几小时的图片构建里。

只输出 MGIT。LaSOT/TNL2K 没有逐帧状态标注，在 ``three_state`` 模式下会自动落到
``masked_unknown``（占位 ``null`` 且其值不参与 loss），不需要也不应该伪造标签。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# 历史工具位于 tools/archive/state_teacher_v1/，向上三级才是仓库根。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.loss_mask import (  # noqa: E402
    MEMORY_STATE_MASKED_UNKNOWN,
    MEMORY_STATE_VERIFIED_UPDATE,
)
from cogtrack.training.mgit_state_labels import (  # noqa: E402
    MGIT_HARD_NULL_SOURCE,
    MGIT_LABEL_SOURCE,
    SegmentParseReport,
    build_frame_memory_labels,
    load_action_segments,
)
from pytracking.utils.io import load_numeric_table  # noqa: E402

MGIT_DATASET = "mgit"


def _plan_frames(plan_path: Path) -> dict[str, list[int]]:
    """从 sampling plan 提取每个 MGIT 序列的当前帧计划（升序去重）。"""

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = payload.get("sequences")
    if not isinstance(entries, list):
        raise ValueError(f"sampling plan 缺少 sequences 列表：{plan_path}")
    frames: dict[str, set[int]] = {}
    for entry in entries:
        if entry.get("dataset") != MGIT_DATASET:
            continue
        name = str(entry["sequence"])
        for frame_id in entry.get("frame_ids") or ():
            frames.setdefault(name, set()).add(int(frame_id))
    return {name: sorted(values) for name, values in sorted(frames.items())}


def _initial_state(description_path: Path) -> str | None:
    """Prompt 侧展示的初始身份描述：第一段 description。

    与 :mod:`pytracking.datasets.mgit` 的 ``language_scope=first_action_description``
    保持一致。若两边不一致，模型会看到一个记忆、被监督另一个记忆。
    """

    segments = load_action_segments(description_path)
    return segments[0].description if segments else None


def _absent_frames(mgit_root: Path, sequence: str) -> frozenset[int]:
    """读取真实 absent 帧，防止不可见边界文本被误写成状态更新。"""

    path = mgit_root / "attribute" / "absent" / f"{sequence}.txt"
    if not path.is_file():
        raise ValueError(f"MGIT 状态更新标签需要真实 absent 标注：{path}")
    values = load_numeric_table(path).reshape(-1)
    return frozenset(int(index) for index, value in enumerate(values) if float(value) != 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-plan", required=True, help="第一步产出的 sampling_plan.json")
    parser.add_argument(
        "--mgit-root",
        default="/root/nas/user-mhf/user-data/PUBLIC_DATASETS/MGIT",
        help="MGIT 数据根目录；需包含 attribute/description/<seq>.json。",
    )
    parser.add_argument("--output", required=True, help="输出 state_update_labels.jsonl")
    parser.add_argument("--report", help="可选：写出逐序列解析报告 JSON。")
    parser.add_argument(
        "--within-segment-hard-null",
        action="store_true",
        help=(
            "把深入段内部的 present 帧也标为 verified_hard_null。默认关闭：action "
            "标签不变并不严格等价于外观不变，开启会显著增加 null 方向的监督量。"
        ),
    )
    parser.add_argument(
        "--boundary-margin",
        type=int,
        default=30,
        help="距最近变化点小于该帧数的稳定帧一律 masked_unknown。",
    )
    args = parser.parse_args()

    plan_path = Path(args.sampling_plan).expanduser().resolve()
    mgit_root = Path(args.mgit_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    frames_by_sequence = _plan_frames(plan_path)
    if not frames_by_sequence:
        raise SystemExit(f"sampling plan 中没有 MGIT 序列：{plan_path}")

    states = Counter()
    reasons = Counter()
    reports: list[dict[str, Any]] = []
    missing: list[str] = []
    rows: list[dict[str, Any]] = []

    for sequence, frame_plan in frames_by_sequence.items():
        description_path = mgit_root / "attribute" / "description" / f"{sequence}.json"
        if not description_path.exists():
            missing.append(sequence)
            continue
        report = SegmentParseReport(sequence=sequence)
        segments = load_action_segments(description_path, report=report)
        sequence_absent_frames = _absent_frames(mgit_root, sequence)
        labels = build_frame_memory_labels(
            segments,
            frame_plan,
            absent_frames=sequence_absent_frames,
            initial_state=_initial_state(description_path),
            boundary_margin=args.boundary_margin,
            within_segment_hard_null=args.within_segment_hard_null,
            report=report,
        )
        for frame_id, label in sorted(labels.items()):
            states[label.state] += 1
            reasons[label.reason] += 1
            # masked_unknown 永不落盘：three_state 模式下缺标签的帧自动就是
            # masked_unknown，少写一份冗余标签就少一处可能不一致的真源。
            #
            # verified_hard_null 必须落盘（只在 --within-segment-hard-null 下产生）。
            # 它是"标注已证明这一帧不需要更新"的正面声明，不落盘就退化成缺标签，
            # 于是整个开关静默失效。absent 帧由采样器判定并无条件压过标签文件，
            # 因此这里写出的 hard-null 只对 present 帧起作用。
            if label.state == MEMORY_STATE_MASKED_UNKNOWN:
                continue
            is_update = label.state == MEMORY_STATE_VERIFIED_UPDATE
            rows.append(
                {
                    "dataset": MGIT_DATASET,
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "target_status": "absent"
                    if frame_id in sequence_absent_frames
                    else "present",
                    "memory_update": label.memory_update if is_update else None,
                    # null 是真负标签还是占位符，全靠这一位区分。
                    "verified_null": not is_update,
                    "source": MGIT_LABEL_SOURCE if is_update else MGIT_HARD_NULL_SOURCE,
                    "reviewed": False,
                    # 审计用：这一帧的输入侧已存状态。绝不进入 response。
                    "input_state": label.input_state,
                    "reason": label.reason,
                }
            )
        reports.append(report.as_dict())

    # 防泄漏硬断言：更新文本绝不能等于它自己的输入侧已存状态。
    # hard-null 行的 memory_update 本就是 None，跳过。
    for row in rows:
        if row["memory_update"] is None:
            continue
        if row["memory_update"] == row["input_state"]:
            raise SystemExit(
                f"标签自相矛盾：{row['sequence']} frame={row['frame_id']} 的更新等于输入状态"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(states.values())
    summary = {
        "schema_version": "cogtrack.mgit_state_update_labels.v1",
        "sampling_plan": str(plan_path),
        "mgit_root": str(mgit_root),
        "output": str(output_path),
        "within_segment_hard_null": args.within_segment_hard_null,
        "boundary_margin": args.boundary_margin,
        "planned_sequences": len(frames_by_sequence),
        "sequences_without_description": missing,
        "planned_frames": total,
        "written_labels": len(rows),
        "written_update_labels": sum(1 for row in rows if row["memory_update"] is not None),
        "written_hard_null_labels": sum(1 for row in rows if row["verified_null"]),
        "sequences_with_updates": len(
            {row["sequence"] for row in rows if row["memory_update"] is not None}
        ),
        # 计划帧的三态分布。写盘的只有 verified_update；其余两态在构建时自动得出。
        "planned_state_distribution": dict(states.most_common()),
        "reason_distribution": dict(reasons.most_common()),
        "unique_update_texts": len(
            {row["memory_update"] for row in rows if row["memory_update"] is not None}
        ),
        "masked_unknown_share": (
            states[MEMORY_STATE_MASKED_UNKNOWN] / total if total else 0.0
        ),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {"summary": summary, "sequences": reports}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
