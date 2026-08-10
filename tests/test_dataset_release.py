import hashlib
import json
import tarfile
from pathlib import Path

from cogtrack.training.release import package_dataset_release


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def test_release_uses_relative_metadata_and_extractable_sequence_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    release = tmp_path / "release"
    for sequence in ("seq-1", "seq-2"):
        _write(source / "images" / "synthetic" / sequence / "reference.jpg", b"reference")
        _write(source / "images" / "synthetic" / sequence / "current.jpg", b"current")
    _write(
        source / "dataset_info.json",
        json.dumps(
            {
                "source_datasets": ["synthetic"],
                "splits": {"case_count": 2},
                "build": {"sample_count": 2},
            }
        ),
    )
    for relative in (
        "build_report.json",
        "source_samples.jsonl",
        "ms_swift/dataset_info.json",
        "ms_swift/qwen2_5_vl/train.jsonl",
        "ms_swift/qwen2_5_vl/val.jsonl",
        "ms_swift/qwen2_5_vl/dataset_info.json",
        "ms_swift/qwen2_5_vl/validation_train.json",
        "ms_swift/qwen2_5_vl/validation_val.json",
        "ms_swift/qwen3_vl/train.jsonl",
        "ms_swift/qwen3_vl/val.jsonl",
        "ms_swift/qwen3_vl/dataset_info.json",
        "ms_swift/qwen3_vl/validation_train.json",
        "ms_swift/qwen3_vl/validation_val.json",
        "sampling_plan.json",
    ):
        _write(source / relative, "{}\n")

    report = package_dataset_release(source, release, max_shard_bytes=32)

    assert report.image_sequence_count == 2
    assert report.image_file_count == 4
    manifest_text = (release / "release_manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert json.loads(manifest_text)["release_root"] == "."

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    for shard in report.shards:
        with tarfile.open(release / shard.path) as archive:
            archive.extractall(extracted)
    assert (extracted / "images/synthetic/seq-1/reference.jpg").read_bytes() == b"reference"
    assert (extracted / "images/synthetic/seq-2/current.jpg").read_bytes() == b"current"

    for line in (release / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        actual = hashlib.sha256((release / relative).read_bytes()).hexdigest()
        assert actual == expected

    card = (release / "README.md").read_text(encoding="utf-8")
    assert "Qwen2.5-VL uses absolute coordinates" in card
    assert "Qwen3-VL uses relative" in card
    assert "ms_swift/qwen2_5_vl/{train,val}.jsonl" in card
    assert "ms_swift/qwen3_vl/{train,val}.jsonl" in card
