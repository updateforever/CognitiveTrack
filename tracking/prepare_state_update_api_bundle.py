#!/usr/bin/env python3
"""把已渲染的固定锚点三图 case 打包为可跨服务器 API 标注的数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.context.visual import draw_reference_box  # noqa: E402
from cogtrack.prompts.state_api_teacher import state_api_prompt_contract  # noqa: E402

BUNDLE_SCHEMA = "cogtrack.state_update_api_bundle.v1"
CASE_SCHEMA = "cogtrack.state_update_api_case.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return text or hashlib.sha256(value.encode()).hexdigest()[:12]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} 必须是 JSON object")
            rows.append(value)
    return rows


def _copy_once(source: Path, target: Path, seen: dict[Path, str]) -> str:
    source_hash = _sha256(source)
    previous = seen.get(target)
    if previous is not None:
        if previous != source_hash:
            raise ValueError(f"不同资产映射到同一 bundle 路径：{target}")
        return source_hash
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    seen[target] = source_hash
    return source_hash


def _boxed_current(source: Path, target: Path, bbox: list[float]) -> str:
    image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise OSError(f"当前图无法读取：{source}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    xywh = (
        x1 * width / 1000.0,
        y1 * height / 1000.0,
        max(1.0, (x2 - x1) * width / 1000.0),
        max(1.0, (y2 - y1) * height / 1000.0),
    )
    marked = draw_reference_box(image_rgb, xywh)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(target),
        cv2.cvtColor(marked, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not ok:
        raise OSError(f"当前带框图写入失败：{target}")
    return _sha256(target)


def _write_checksums(root: Path) -> Path:
    checksum_path = root / "SHA256SUMS"
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return checksum_path


def build_bundle(source_release: Path, output: Path, *, sampling_plan: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bundle 输出目录非空，请使用新的 release 名：{output}")
    output.mkdir(parents=True, exist_ok=True)
    source_jsonl = source_release / "source_samples.jsonl"
    rows = _read_jsonl(source_jsonl)
    manifest_path = output / "manifest.jsonl"
    copied: dict[Path, str] = {}
    manifest: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    sequences: set[tuple[str, str]] = set()
    for row in rows:
        status = str(row.get("target_status") or "")
        if status not in {"present", "absent"}:
            raise ValueError(f"非法 target_status：{row.get('id')}")
        metadata = row.get("metadata") or {}
        if metadata.get("source_split") != "train":
            raise ValueError(f"拒绝非 train case：{row.get('id')}")
        if metadata.get("history_quality", "clean") != "clean":
            continue
        images = row.get("images") or []
        if len(images) != 3:
            raise ValueError(f"API 标注 case 必须正好三张图：{row.get('id')}")
        dataset = str(metadata.get("source_dataset") or metadata.get("dataset") or "").lower()
        sequence = str(metadata.get("source_sequence") or metadata.get("sequence") or "")
        frame_id = int(metadata["frame_id"])
        reference_id = int(metadata["reference_frame_id"])
        case_id = f"{dataset}::{sequence}::{reference_id:08d}::{frame_id:08d}"
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        sequences.add((dataset, sequence))
        sequence_dir = Path("images") / _safe(dataset) / _safe(sequence)
        reference_rel = sequence_dir / f"reference_{reference_id:08d}.jpg"
        history_rel = sequence_dir / f"history_before_{frame_id:08d}.jpg"
        current_rel = sequence_dir / (
            f"current_boxed_{frame_id:08d}.jpg"
            if status == "present"
            else f"current_absent_{frame_id:08d}.jpg"
        )
        reference_source = source_release / images[0]
        history_source = source_release / images[1]
        current_source = source_release / images[2]
        reference_hash = _copy_once(reference_source, output / reference_rel, copied)
        history_hash = _copy_once(history_source, output / history_rel, copied)
        current_hash = (
            _boxed_current(
                current_source,
                output / current_rel,
                list(row["bbox_norm1000_xyxy"]),
            )
            if status == "present"
            else _copy_once(current_source, output / current_rel, copied)
        )
        manifest.append(
            {
                "schema_version": CASE_SCHEMA,
                "case_id": case_id,
                "dataset": dataset,
                "sequence": sequence,
                "frame_id": frame_id,
                "anchor_frame_id": reference_id,
                "frame_gap": frame_id - reference_id,
                "history_frame_ids": list(metadata.get("history_frame_ids") or []),
                "target_status": status,
                "initial_identity": str(
                    metadata.get("initial_identity_description")
                    or metadata.get("initial_target_text")
                    or "the target marked in Image 1"
                ),
                "bbox_norm1000_xyxy": (
                    list(row["bbox_norm1000_xyxy"])
                    if row.get("bbox_norm1000_xyxy") is not None
                    else None
                ),
                "temporal_event": metadata.get("temporal_event"),
                "images": [
                    reference_rel.as_posix(),
                    history_rel.as_posix(),
                    current_rel.as_posix(),
                ],
                "image_sha256": [reference_hash, history_hash, current_hash],
            }
        )
    manifest.sort(key=lambda row: (row["dataset"], row["sequence"], row["frame_id"]))
    if not manifest:
        raise ValueError("没有可打包的 present 三图 case")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    shutil.copy2(sampling_plan, output / "sampling_plan.json")
    (output / "prompt_contract.json").write_text(
        json.dumps(state_api_prompt_contract(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    portable_script = PROJECT_ROOT / "tracking" / "annotate_state_update_openai_api.py"
    if portable_script.is_file():
        (output / "tools").mkdir(exist_ok=True)
        shutil.copy2(portable_script, output / "tools" / portable_script.name)
    info = {
        "schema_version": BUNDLE_SCHEMA,
        "source_release": source_release.name,
        "sampling_plan_sha256": _sha256(sampling_plan),
        "manifest_sha256": _sha256(manifest_path),
        "cases": len(manifest),
        "sequences": len(sequences),
        "by_dataset": dict(sorted(Counter(row["dataset"] for row in manifest).items())),
        "image_count": len({path for row in manifest for path in row["images"]}),
        "current_has_gt_box": True,
        "student_current_has_gt_box": False,
        "contains_selected_rendered_frames": True,
        "contains_complete_raw_datasets": False,
        "absent_cases": sum(row["target_status"] == "absent" for row in manifest),
        "absent_supervision_policy": {
            "first_observed_absence": "dataset_gt_disappearance_transition_v1",
            "continued_absence": "dataset_gt_continued_absence_v1",
        },
    }
    (output / "bundle_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# CognitiveTrack state-update API annotation bundle\n\n"
        "This portable bundle contains only selected three-image annotation cases. Image 3 has an "
        "offline GT red box; it must never be copied into student inputs.\n\n"
        "Run `pip install openai`, set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `API_MODEL`, then:\n\n"
        "```bash\npython tools/annotate_state_update_openai_api.py --bundle . "
        "--output-dir annotation_result --resume\n```\n",
        encoding="utf-8",
    )
    checksum_path = _write_checksums(output)
    info["checksums"] = checksum_path.name
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--sampling-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build_bundle(
        Path(args.source_release).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        sampling_plan=Path(args.sampling_plan).expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
