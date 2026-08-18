#!/usr/bin/env python3
"""用 OpenAI Python 客户端调用兼容 API，单次生成动态指代表达标签。

本文件刻意只依赖 Python 标准库和 ``openai``，会被复制进 annotation bundle，远端无需
安装 CognitiveTrack。不同序列可并发，同一序列严格按时间顺序维护当前指代表达。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

POLICY = "single_pass_frontier_api_v1"
LABEL_SOURCE = "frontier_api_teacher_v1"
HARD_NULL_SOURCE = "frontier_api_teacher_hard_null_v1"
DISAPPEARANCE_SOURCE = "dataset_gt_disappearance_transition_v1"
CONTINUED_ABSENCE_SOURCE = "dataset_gt_continued_absence_v1"
ABSENT_DESCRIPTION = "The target has disappeared and is currently not visible in the search frame."
IMAGE_ROLE_LABELS: tuple[str, str, str] = (
    "IMAGE 1 - PERMANENT VISUAL IDENTITY ANCHOR (earlier frame):",
    "IMAGE 2 - TRUSTED HISTORY STRIP (earlier frames, chronological left to right):",
    "IMAGE 3 - CURRENT FRAME (use this image for the current decision and replacement):",
)
ALLOWED_ELEMENTS = {
    "action",
    "pose",
    "appearance",
    "viewpoint",
    "scale",
    "visibility",
    "interaction",
    "scene",
    "other",
}


def _label_source(*, target_status: str, memory_update: str | None) -> str:
    if target_status == "absent":
        return DISAPPEARANCE_SOURCE if memory_update is not None else CONTINUED_ABSENCE_SOURCE
    return LABEL_SOURCE if memory_update is not None else HARD_NULL_SOURCE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object：{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} 必须是 object")
            rows.append(value)
    return rows


def _extract_json(raw: Any) -> Mapping[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, Mapping) else None


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("`", " ")).strip(' "\'')


def _word_set(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2}


def _vacuous(candidate: str, baseline: str) -> bool:
    first, second = _normalize(candidate), _normalize(baseline)
    if not first:
        return True
    if not second:
        return False
    if first.casefold().rstrip(".") == second.casefold().rstrip("."):
        return True
    a, b = _word_set(first), _word_set(second)
    return bool(a and b and len(a & b) / len(a | b) > 0.90)


def _parse_and_gate(
    raw: str,
    *,
    input_state: str,
    confidence_threshold: float,
    require_update: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    payload = _extract_json(raw)
    required = {
        "decision",
        "changed_elements",
        "memory_update",
        "confidence",
        "evidence_sufficiency",
        "significant_change",
        "identity_consistent",
        "standalone_complete",
        "evidence",
    }
    if payload is None:
        return None, "unparseable_json"
    if set(payload) != required:
        return None, "schema_keys_mismatch"
    decision = str(payload.get("decision") or "").strip().lower()
    if decision not in {"update", "keep", "uncertain"}:
        return None, "invalid_decision"
    raw_elements = payload.get("changed_elements")
    if not isinstance(raw_elements, list) or any(not isinstance(item, str) for item in raw_elements):
        return None, "invalid_changed_elements"
    elements = list(dict.fromkeys(item.strip().lower() for item in raw_elements))
    if any(item not in ALLOWED_ELEMENTS for item in elements):
        return None, "invalid_changed_elements"
    raw_update = payload.get("memory_update")
    if raw_update is not None and not isinstance(raw_update, str):
        return None, "invalid_memory_update_type"
    update = _normalize(raw_update) or None
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return None, "invalid_confidence"
    if not 0.0 <= confidence <= 1.0:
        return None, "invalid_confidence"
    sufficiency = str(payload.get("evidence_sufficiency") or "").strip().lower()
    if sufficiency not in {"sufficient", "insufficient"}:
        return None, "invalid_evidence_sufficiency"
    bool_keys = ("significant_change", "identity_consistent", "standalone_complete")
    if any(not isinstance(payload.get(key), bool) for key in bool_keys):
        return None, "non_boolean_audit_field"
    evidence = _normalize(payload.get("evidence"))
    if not evidence:
        return None, "empty_evidence"
    if not evidence.lower().startswith("image 3 shows"):
        return None, "invalid_current_evidence"
    significant = bool(payload["significant_change"])
    identity = bool(payload["identity_consistent"])
    standalone = bool(payload["standalone_complete"])
    if decision == "uncertain":
        return None, "model_uncertain"
    if confidence < confidence_threshold:
        return None, "low_confidence"
    if not identity:
        return None, "identity_inconsistent"
    if decision == "keep":
        if require_update:
            return None, "reappearance_requires_update"
        if update is not None or elements or significant or sufficiency != "sufficient":
            return None, "inconsistent_keep"
    else:
        if (
            update is None
            or not elements
            or not significant
            or sufficiency != "sufficient"
            or not standalone
        ):
            return None, "inconsistent_update"
        if len(update.split()) < 5 or len(update.split()) > 30 or len(update) > 256:
            return None, "invalid_update_length"
        if _vacuous(update, input_state):
            return None, "vacuous_update"
    parsed = dict(payload)
    parsed["decision"] = decision
    parsed["changed_elements"] = elements
    parsed["memory_update"] = update
    parsed["confidence"] = confidence
    parsed["evidence"] = evidence
    return parsed, "accepted"


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _response_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        texts: list[str] = []
        for item in message_content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                texts.append(str(item["text"]))
            elif hasattr(item, "text"):
                texts.append(str(item.text))
        return "\n".join(texts)
    return str(message_content or "")


def _build_multimodal_content(
    user_prompt: str,
    image_paths: list[Path],
) -> list[dict[str, Any]]:
    if len(image_paths) != len(IMAGE_ROLE_LABELS):
        raise ValueError(f"API teacher expects exactly 3 images, got {len(image_paths)}")
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for label, path in zip(IMAGE_ROLE_LABELS, image_paths, strict=True):
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
    return content


def _call_api(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[Path],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    json_mode: bool,
) -> tuple[str, dict[str, int]]:
    content = _build_multimodal_content(user_prompt, image_paths)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            text = _response_text(response.choices[0].message.content)
            usage = getattr(response, "usage", None)
            tokens = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            return text, tokens
        except Exception as exc:  # API SDK/供应商异常类型不统一，保留原文并重试。
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"API 调用失败，已重试 {max_retries} 次：{last_error}") from last_error


def _diverse_cap(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    """跨序列 round-robin 裁剪，避免按序列名截断造成集中。"""

    if len(rows) <= limit:
        return rows
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["dataset"], row["sequence"])].append(row)
    rng = random.Random(seed)
    keys = sorted(buckets)
    rng.shuffle(keys)
    # 每条序列只能保留时间前缀：后续标签的 input_state 依赖之前被接受的 update，
    # 若按行随机抽样会留下“前置状态已丢、后续标签仍在”的断链数据。
    for key in keys:
        buckets[key].sort(key=lambda row: int(row["frame_id"]))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _write_checksums(root: Path) -> None:
    target = root / "SHA256SUMS"
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=".")
    parser.add_argument("--output-dir", default="annotation_result")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("API_MODEL", ""))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument(
        "--max-input-cases",
        type=int,
        default=3000,
        help="正式运行前按跨序列时间前缀裁剪候选，控制 API 成本；0 表示不裁剪",
    )
    parser.add_argument("--max-output-labels", type=int, default=1500)
    parser.add_argument("--min-output-labels", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--limit", type=int, help="仅调用前 N 个 case，用于远端 API smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-json-mode", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("请通过环境变量 OPENAI_API_KEY 提供密钥，不要把密钥写进命令或 bundle")
    if not args.base_url or not args.model:
        raise SystemExit("请设置 OPENAI_BASE_URL 和 API_MODEL，或传 --base-url/--model")
    if (
        args.workers <= 0
        or args.max_input_cases < 0
        or args.max_output_labels <= 0
        or args.min_output_labels <= 0
    ):
        raise SystemExit("workers、max/min output labels 必须为正，max input cases 不能为负")
    if args.min_output_labels > args.max_output_labels:
        raise SystemExit("min-output-labels 不能大于 max-output-labels")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("confidence-threshold 必须位于 [0,1]")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("缺少 openai：请先运行 pip install 'openai>=1.40'") from exc

    bundle = Path(args.bundle).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle / "manifest.jsonl"
    prompt_path = bundle / "prompt_contract.json"
    full_manifest = _read_jsonl(manifest_path)
    manifest = full_manifest
    if args.limit is not None:
        manifest = manifest[: args.limit]
    elif args.max_input_cases:
        manifest = _diverse_cap(manifest, limit=args.max_input_cases, seed=args.seed)
    contract = _read_json(prompt_path)
    config = {
        "schema_version": "cogtrack.state_update_api_run.v1",
        "annotation_policy": POLICY,
        "manifest_sha256": _sha256(manifest_path),
        "prompt_contract_sha256": _sha256(prompt_path),
        "prompt_version": contract.get("version"),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "confidence_threshold": args.confidence_threshold,
        "max_input_cases": args.max_input_cases,
        "seed": args.seed,
        "limit": args.limit,
        "json_mode": not args.disable_json_mode,
    }
    config_path = output / "run_config.json"
    if config_path.exists():
        if not args.resume:
            raise SystemExit(f"输出已存在；确认配置不变后加 --resume：{output}")
        if _read_json(config_path) != config:
            raise SystemExit("--resume 配置与既有 run_config.json 不一致，拒绝混写")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    journal_path = output / "raw_responses.jsonl"
    existing_rows = _read_jsonl(journal_path)
    existing = {str(row["case_id"]): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise SystemExit("raw_responses.jsonl 存在重复 case_id")
    by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in manifest:
        by_sequence[(str(case["dataset"]), str(case["sequence"]))].append(case)
    for cases in by_sequence.values():
        cases.sort(key=lambda row: int(row["frame_id"]))

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
    )
    write_lock = threading.Lock()
    result_lock = threading.Lock()
    all_results: dict[str, dict[str, Any]] = dict(existing)
    failures: list[str] = []

    def process_sequence(cases: list[dict[str, Any]]) -> None:
        current_state = str(cases[0]["initial_identity"])
        for case in cases:
            case_id = str(case["case_id"])
            old = existing.get(case_id)
            if old is not None:
                if str(old.get("input_state")) != current_state:
                    raise ValueError(f"resume 状态链不一致：{case_id}")
                if old.get("accepted") and old.get("decision") == "update":
                    current_state = str(old["memory_update"])
                continue
            if case["target_status"] == "absent":
                is_transition = current_state != ABSENT_DESCRIPTION
                parsed = {
                    "decision": "update" if is_transition else "keep",
                    "changed_elements": ["visibility"] if is_transition else [],
                    "memory_update": ABSENT_DESCRIPTION if is_transition else None,
                    "confidence": 1.0,
                    "evidence_sufficiency": "sufficient",
                    "significant_change": is_transition,
                    "identity_consistent": True,
                    "standalone_complete": True,
                    "evidence": "dataset GT marks the current frame as target absent",
                }
                record = {
                    "case_id": case_id,
                    "dataset": case["dataset"],
                    "sequence": case["sequence"],
                    "frame_id": case["frame_id"],
                    "target_status": "absent",
                    "input_state": current_state,
                    "accepted": True,
                    "outcome": (
                        "dataset_gt_disappearance_transition"
                        if is_transition
                        else "dataset_gt_continued_absence"
                    ),
                    "decision": parsed["decision"],
                    "memory_update": parsed["memory_update"],
                    "parsed": parsed,
                    "raw_response": None,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                with write_lock:
                    with journal_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                with result_lock:
                    all_results[case_id] = record
                if is_transition:
                    current_state = ABSENT_DESCRIPTION
                continue
            user_prompt = str(contract["user_prompt_template"]).format(
                initial_identity=case["initial_identity"],
                current_state=current_state,
                frame_gap=case["frame_gap"],
            )
            raw, usage = _call_api(
                client,
                model=args.model,
                system_prompt=str(contract["system_prompt"]),
                user_prompt=user_prompt,
                image_paths=[bundle / value for value in case["images"]],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                json_mode=not args.disable_json_mode,
            )
            parsed, outcome = _parse_and_gate(
                raw,
                input_state=current_state,
                confidence_threshold=args.confidence_threshold,
                require_update=current_state == ABSENT_DESCRIPTION,
            )
            accepted = parsed is not None
            record = {
                "case_id": case_id,
                "dataset": case["dataset"],
                "sequence": case["sequence"],
                "frame_id": case["frame_id"],
                "target_status": "present",
                "input_state": current_state,
                "accepted": accepted,
                "outcome": outcome,
                "decision": parsed.get("decision") if parsed else None,
                "memory_update": parsed.get("memory_update") if parsed else None,
                "parsed": parsed,
                "raw_response": raw,
                "usage": usage,
            }
            with write_lock:
                with journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            with result_lock:
                all_results[case_id] = record
            if accepted and parsed["decision"] == "update":
                current_state = str(parsed["memory_update"])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_sequence, cases): key for key, cases in by_sequence.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{key[0]}/{key[1]}: {exc}")
                print(f"[ERROR] {failures[-1]}", flush=True)
    if failures:
        (output / "errors.json").write_text(
            json.dumps({"errors": failures}, ensure_ascii=False, indent=2) + "\n"
        )
        print("存在 API 失败；修复服务后使用相同命令加 --resume", flush=True)
        return 2

    accepted_rows: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    usage_total: Counter[str] = Counter()
    for case in manifest:
        record = all_results.get(str(case["case_id"]))
        if record is None:
            continue
        outcomes[str(record["outcome"])] += 1
        usage_total.update(record.get("usage") or {})
        if not record.get("accepted"):
            continue
        parsed = record["parsed"]
        update = parsed["memory_update"]
        accepted_rows.append(
            {
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "sequence": case["sequence"],
                "frame_id": case["frame_id"],
                "target_status": case["target_status"],
                "memory_update": update,
                "verified_null": parsed["decision"] == "keep",
                "source": _label_source(
                    target_status=str(case["target_status"]),
                    memory_update=update,
                ),
                "reviewed": False,
                "quality_gate_passed": True,
                "review_method": POLICY,
                "input_state": record["input_state"],
                "confidence": parsed["confidence"],
                "changed_elements": parsed["changed_elements"],
            }
        )
    accepted_rows = _diverse_cap(
        accepted_rows, limit=args.max_output_labels, seed=args.seed
    )
    accepted_rows.sort(key=lambda row: (row["dataset"], row["sequence"], row["frame_id"]))
    labels_path = output / "state_update_api_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in accepted_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    updates = sum(row["memory_update"] is not None for row in accepted_rows)
    minimum_reached = len(accepted_rows) >= args.min_output_labels and args.limit is None
    report = {
        "schema_version": "cogtrack.state_update_api_report.v1",
        "annotation_policy": POLICY,
        "teacher_model": args.model,
        "provider_protocol": "openai_compatible_chat_completions",
        "prompt_version": contract.get("version"),
        "independently_verified": False,
        "single_pass_frontier_teacher": True,
        "quality_gate_applied": True,
        "dry_run": args.limit is not None,
        "manifest_sha256": _sha256(manifest_path),
        "bundle_manifest_cases": len(full_manifest),
        "planned_cases": len(manifest),
        "api_completed_cases": len(all_results),
        "api_requested_cases": sum(row.get("raw_response") is not None for row in all_results.values()),
        "accepted_before_cap": sum(bool(row.get("accepted")) for row in all_results.values()),
        "written_labels": len(accepted_rows),
        "written_update_labels": updates,
        "written_hard_null_labels": len(accepted_rows) - updates,
        "min_output_labels": args.min_output_labels,
        "max_output_labels": args.max_output_labels,
        "minimum_output_reached": minimum_reached,
        "confidence_threshold": args.confidence_threshold,
        "outcomes": dict(outcomes.most_common()),
        "usage": dict(usage_total),
        "per_dataset": dict(Counter(row["dataset"] for row in accepted_rows)),
        "sequences": len({(row["dataset"], row["sequence"]) for row in accepted_rows}),
    }
    (output / "annotation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_checksums(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if minimum_reached or args.limit is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
