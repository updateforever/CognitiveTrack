from pathlib import Path

from pytracking.evaluation.reproducibility import (
    collect_reproducibility_metadata,
    file_fingerprint,
    git_head,
    source_tree_fingerprint,
)


def test_fingerprint_and_referenced_model_config(tmp_path: Path):
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model_path: local-model\n", encoding="utf-8")
    tracker_config = tmp_path / "tracker.yaml"
    tracker_config.write_text("model_config: model.yaml\n", encoding="utf-8")

    fingerprint = file_fingerprint(model_config)
    assert fingerprint is not None
    assert len(fingerprint["sha256"]) == 64

    metadata = collect_reproducibility_metadata(
        project_root=tmp_path,
        tracker_config=tracker_config,
        environment_config=None,
    )
    assert metadata["tracker_config"]["path"] == str(tracker_config)
    assert metadata["referenced_configs"][0]["path"] == str(model_config)
    assert metadata["packages"]["numpy"] is not None


def test_nested_checkpoint_and_model_config_are_fingerprinted(tmp_path: Path, monkeypatch):
    model_config = tmp_path / "model.yaml"
    model_config.write_text("MODEL: {}\n", encoding="utf-8")
    checkpoint = tmp_path / "weights.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("TEST_SUTRACK_CHECKPOINT", str(checkpoint))
    tracker_config = tmp_path / "tracker.yaml"
    tracker_config.write_text(
        "implementation:\n"
        "  kwargs:\n"
        "    model_config: model.yaml\n"
        "    checkpoint: ${TEST_SUTRACK_CHECKPOINT}\n",
        encoding="utf-8",
    )

    metadata = collect_reproducibility_metadata(
        project_root=tmp_path,
        tracker_config=tracker_config,
        environment_config=None,
    )
    referenced = {item["kind"]: item for item in metadata["referenced_configs"]}
    assert referenced["model_config"]["path"] == str(model_config)
    assert referenced["checkpoint"]["path"] == str(checkpoint)


def test_git_head_without_invoking_git(tmp_path: Path):
    git_dir = tmp_path / ".git"
    ref = git_dir / "refs/heads/main"
    ref.parent.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    ref.write_text("a" * 40 + "\n", encoding="utf-8")
    assert git_head(tmp_path) == "a" * 40


def test_source_tree_hash_changes_with_code(tmp_path: Path):
    module = tmp_path / "cogtrack/example.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_fingerprint(tmp_path)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    second = source_tree_fingerprint(tmp_path)
    assert first["file_count"] == second["file_count"] == 1
    assert first["sha256"] != second["sha256"]
