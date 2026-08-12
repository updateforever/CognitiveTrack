#!/usr/bin/env python3
"""Run a deterministic small held-out Stage-1 probe for base/adapter comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cogtrack.vlm import parse_tracking_output
from cogtrack.vlm.qwen_vl import HuggingFaceQwenVLBackend, QwenVLConfig


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _select(rows: list[dict], state: str, count: int) -> list[dict]:
    pool = sorted(
        (row for row in rows if row["metadata"]["temporal_case"] == state),
        key=lambda row: row["metadata"]["frame_id"] - row["metadata"]["reference_frame_id"],
        reverse=True,
    )
    selected = []
    seen = set()
    for row in pool:
        metadata = row["metadata"]
        key = (metadata["source_dataset"], metadata["source_sequence"])
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == count:
            break
    return selected


def _iou(first: list[float] | None, second: list[float] | None) -> float | None:
    if first is None or second is None:
        return None
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def main() -> int:
    args = _args()
    root = Path(args.dataset_root).resolve()
    rows = [json.loads(line) for line in (root / "ms_swift/qwen3_vl/val.jsonl").open()]
    selected = _select(rows, "present", 4) + _select(rows, "absent", 4)
    backend = HuggingFaceQwenVLBackend(
        QwenVLConfig(
            model_path=args.model,
            adapter_path=args.adapter,
            model_name="qwen3vl4b-stage1-probe",
            torch_dtype="bfloat16",
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            processor_use_fast=False,
            max_pixels=200704,
            max_image_side=648,
        )
    )
    results = []
    for row in selected:
        metadata = row["metadata"]
        prompt = row["messages"][1]["content"].replace("<image><image>\n", "", 1)
        prompt = prompt.replace("<bbox>", str(metadata["reference_bbox_norm1000_xyxy"]), 1)
        response = backend.generate(
            [root / path for path in row["images"]],
            prompt,
            system_prompt=row["messages"][0]["content"],
        )
        error = None
        try:
            prediction = parse_tracking_output(
                response.text,
                image_width=1000,
                image_height=1000,
                bbox_protocol="norm1000",
                require_memory_update=False,
            )
            predicted_status = prediction.status.value
            predicted_box = list(prediction.bbox.to_xyxy()) if prediction.bbox else None
        except Exception as exc:  # noqa: BLE001 - probe records parser failures verbatim
            predicted_status, predicted_box, error = "parse_error", None, str(exc)
        ground_truth_status = metadata["temporal_case"]
        ground_truth_box = row["objects"]["bbox"][-1] if ground_truth_status == "present" else None
        results.append(
            {
                "dataset": metadata["source_dataset"],
                "sequence": metadata["source_sequence"],
                "reference_frame_id": metadata["reference_frame_id"],
                "frame_id": metadata["frame_id"],
                "gap": metadata["frame_id"] - metadata["reference_frame_id"],
                "ground_truth_status": ground_truth_status,
                "predicted_status": predicted_status,
                "ground_truth_bbox": ground_truth_box,
                "predicted_bbox": predicted_box,
                "iou": _iou(predicted_box, ground_truth_box),
                "raw": response.text,
                "latency_ms": response.latency_ms,
                "error": error,
            }
        )
    present_ious = [item["iou"] for item in results if item["iou"] is not None]
    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "summary": {
            "case_count": len(results),
            "format_ok": sum(item["predicted_status"] != "parse_error" for item in results),
            "status_correct": sum(
                item["predicted_status"] == item["ground_truth_status"] for item in results
            ),
            "present_mean_iou": sum(present_ious) / len(present_ious) if present_ious else None,
        },
        "cases": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
