import hashlib
from pathlib import Path

from tools.verify_stage1_sources import compute_source_fingerprints


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_source_fingerprints_are_path_independent(tmp_path: Path) -> None:
    def build(root: Path) -> dict:
        lasot = root / "LaSOT"
        tnl2k = root / "TNL2K/TNL2K_train_subset"
        _write(lasot / "training_set.txt", "airplane-1\n")
        for name in ("groundtruth.txt", "full_occlusion.txt", "out_of_view.txt"):
            _write(lasot / "airplane/airplane-1" / name, f"{name}\n")
        _write(tnl2k / "sequence-b/groundtruth.txt", "2,2,3,3\n")
        _write(tnl2k / "sequence-a/groundtruth.txt", "1,1,2,2\n")
        return compute_source_fingerprints(lasot, tnl2k.parent)

    first = build(tmp_path / "host-a")
    second = build(tmp_path / "host-b")

    for key in (
        "lasot_training_set_sha256",
        "lasot_annotations_sha256",
        "tnl2k_groundtruth_sha256",
        "tnl2k_sequence_names_sha256",
    ):
        assert first[key] == second[key]
    assert first["tnl2k_sequence_names_sha256"] == hashlib.sha256(
        b"sequence-a\nsequence-b\n"
    ).hexdigest()
