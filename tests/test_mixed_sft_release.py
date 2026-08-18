from __future__ import annotations

import json
from pathlib import Path

from tracking.build_mixed_sft_release import (
    MASKED_UNKNOWN,
    STATE_PROFILE,
    TRACKING_PROFILE,
    VERIFIED_HARD_NULL,
    VERIFIED_UPDATE,
    build_release,
    build_weighted_train,
    harmonise_splits,
    seed_loss_scale_arrow_schema,
)


def _row(profile: str, state: str, dataset: str, sequence: str, index: int) -> dict:
    return {
        "id": f"{dataset}::{sequence}::{index}",
        "messages": [{"role": "assistant", "content": "{}"}],
        "images": ["images/a.jpg", "images/b.jpg", "images/c.jpg"],
        "metadata": {
            "source_dataset": dataset,
            "source_sequence": sequence,
            "sft_supervision_profile": profile,
            "memory_supervision_state": state,
        },
    }


def test_harmonise_splits_prefers_tracking_for_shared_sequence() -> None:
    tracking = [(_row(TRACKING_PROFILE, MASKED_UNKNOWN, "lasot", "a", 1), "train", "tracking")]
    state = [(_row(STATE_PROFILE, VERIFIED_UPDATE, "lasot", "a", 2), "val", "state_update")]
    splits, report = harmonise_splits(tracking, state)

    assert len(splits["train"]) == 2
    assert not splits["val"]
    assert report["conflicting_sequences"] == 1
    assert report["moved_state_rows"] == 1
    state_row = splits["train"][1][0]
    assert state_row["metadata"]["mixed_original_split"] == "val"
    assert state_row["metadata"]["mixed_resolved_split"] == "train"


def test_weighted_train_is_deterministic_90_4_6() -> None:
    rows = [
        _row(TRACKING_PROFILE, MASKED_UNKNOWN, "lasot", f"t{i}", i)
        for i in range(90)
    ]
    rows += [
        _row(STATE_PROFILE, VERIFIED_UPDATE, "lasot", f"u{i}", i)
        for i in range(2)
    ]
    rows += [
        _row(STATE_PROFILE, VERIFIED_HARD_NULL, "lasot", f"n{i}", i)
        for i in range(2)
    ]
    for row in rows:
        row["metadata"]["mixed_release_source"] = (
            "tracking"
            if row["metadata"]["sft_supervision_profile"] == TRACKING_PROFILE
            else "state_update"
        )

    first, report = build_weighted_train(rows, seed=7)
    second, _ = build_weighted_train(rows, seed=7)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert report["samples"] == 100
    assert report["counts"] == {
        MASKED_UNKNOWN: 90,
        VERIFIED_HARD_NULL: 6,
        VERIFIED_UPDATE: 4,
    }
    assert report["maximum_row_occurrence"] == 3


def test_release_module_has_no_embedded_absolute_dataset_path() -> None:
    source = Path("tracking/build_mixed_sft_release.py").read_text(encoding="utf-8")
    assert "/root/nas/" not in source
    # Keep this a useful JSON import smoke rather than relying only on compilation.
    assert json.loads(json.dumps({"ok": True})) == {"ok": True}


def test_loss_scale_schema_seed_preserves_rows_and_moves_explicit_first() -> None:
    masked_a = _row(TRACKING_PROFILE, MASKED_UNKNOWN, "lasot", "a", 1)
    masked_b = _row(TRACKING_PROFILE, MASKED_UNKNOWN, "lasot", "b", 2)
    explicit = _row(STATE_PROFILE, VERIFIED_UPDATE, "lasot", "c", 3)
    explicit["messages"][0]["loss_scale"] = 1.0

    ordered = seed_loss_scale_arrow_schema([masked_a, masked_b, explicit])

    assert [row["id"] for row in ordered] == [explicit["id"], masked_a["id"], masked_b["id"]]
    assert sorted(row["id"] for row in ordered) == sorted(
        row["id"] for row in [masked_a, masked_b, explicit]
    )


def test_build_self_contained_release(tmp_path: Path) -> None:
    tracking_root = tmp_path / "tracking"
    state_root = tmp_path / "state"
    preview_root = tmp_path / "preview"
    for root in (tracking_root, state_root):
        (root / "ms_swift/qwen3_vl").mkdir(parents=True)
        (root / "images").mkdir()
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (root / "images" / name).write_bytes(f"{root.name}-{name}".encode())
    (preview_root / "assets").mkdir(parents=True)
    (preview_root / "assets/example.jpg").write_bytes(b"preview")
    (preview_root / "README.md").write_text("# Preview\n\nExample.\n", encoding="utf-8")

    tracking_train = [
        _row(TRACKING_PROFILE, MASKED_UNKNOWN, "lasot", f"t{i}", i)
        for i in range(90)
    ]
    tracking_val = [_row(TRACKING_PROFILE, MASKED_UNKNOWN, "lasot", "tv", 1)]
    state_train = [
        _row(STATE_PROFILE, VERIFIED_UPDATE, "lasot", "u", 1),
        _row(STATE_PROFILE, VERIFIED_HARD_NULL, "lasot", "n", 1),
    ]
    state_val = [_row(STATE_PROFILE, VERIFIED_UPDATE, "lasot", "uv", 1)]
    for root, train, val in (
        (tracking_root, tracking_train, tracking_val),
        (state_root, state_train, state_val),
    ):
        for split, rows in (("train", train), ("val", val)):
            path = root / "ms_swift/qwen3_vl" / f"{split}.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

    output = tmp_path / "release"
    report = build_release(
        tracking_root=tracking_root,
        state_root=state_root,
        preview_root=preview_root,
        output_root=output,
        seed=3,
        checksum_workers=2,
    )

    assert report["self_contained"] is True
    assert report["weighted_training_view"]["samples"] == 100
    assert (output / "images/tracking/a.jpg").is_file()
    assert (output / "images/state_update/a.jpg").is_file()
    assert (output / "README.md").read_text(encoding="utf-8").startswith(
        "# CognitiveTrack VLT-v6.4 Mixed SFT Dataset"
    )
    checksums = (output / "checksums.sha256").read_text(encoding="utf-8")
    assert "  train.jsonl\n" in checksums
    assert "  images/tracking/a.jpg\n" in checksums
