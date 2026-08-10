#!/usr/bin/env python3
"""校验 LaSOT/TNL2K 原始训练集是否与 Stage-1 v1 构建源一致。

聚合摘要只包含相对路径和每个标注文件自身的 SHA-256，不把机器绝对路径写入
hash。排序显式使用文件系统字节序，避免服务器 locale 改变结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

EXPECTED = {
    "lasot_training_set_sha256": "0ae7df00644ee36794ac1d67f123612eb6341b4e94fef933169607d650ade893",
    "lasot_annotations_sha256": "2c665501c9ee752f6ab2df3d61b806c2f1dc6a04f0e0209cb4582a989b987ed4",
    "lasot_sequence_count": 1120,
    "tnl2k_groundtruth_sha256": "f551c0afaa9d9811b20dab162a535869326e2ef0576426f5109e8ca983b1db94",
    "tnl2k_sequence_names_sha256": "db25334d28f236c0cd9c40e89edf34b529d4ba0d6cd93c8936b2a9a144fe8516",
    "tnl2k_sequence_count": 1300,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_file_digests(rows: Iterable[tuple[Path, str]]) -> str:
    aggregate = hashlib.sha256()
    for path, relative in rows:
        aggregate.update(f"{_file_sha256(path)}  {relative}\n".encode("utf-8"))
    return aggregate.hexdigest()


def _resolve_tnl2k_train_root(configured_root: Path) -> Path:
    if configured_root.name.lower() == "tnl2k_train_subset":
        return configured_root
    candidate = configured_root / "TNL2K_train_subset"
    return candidate if candidate.is_dir() else configured_root


def compute_source_fingerprints(lasot_root: str | Path, tnl2k_root: str | Path) -> dict[str, Any]:
    """计算与正式 Stage-1 v1 审计相同的源数据摘要。"""

    lasot = Path(lasot_root).expanduser().resolve()
    tnl2k = _resolve_tnl2k_train_root(Path(tnl2k_root).expanduser().resolve())
    training_set = lasot / "training_set.txt"
    if not training_set.is_file():
        raise FileNotFoundError(f"缺少 LaSOT training_set.txt：{training_set}")
    if not tnl2k.is_dir():
        raise FileNotFoundError(f"TNL2K train 目录不存在：{tnl2k}")

    lasot_names = [
        line
        for line in training_set.read_text(encoding="utf-8-sig").splitlines()
        if line
    ]
    lasot_rows: list[tuple[Path, str]] = []
    for sequence in lasot_names:
        class_name = sequence.rsplit("-", 1)[0]
        for filename in ("groundtruth.txt", "full_occlusion.txt", "out_of_view.txt"):
            path = lasot / class_name / sequence / filename
            if not path.is_file():
                raise FileNotFoundError(f"缺少 LaSOT 标注：{path}")
            lasot_rows.append((path, f"{sequence}/{filename}"))

    tnl2k_groundtruth = list(tnl2k.glob("*/groundtruth.txt"))
    tnl2k_groundtruth.sort(
        key=lambda path: os.fsencode(path.relative_to(tnl2k).as_posix())
    )
    tnl2k_names = [path.name for path in tnl2k.iterdir() if path.is_dir()]
    tnl2k_names.sort(key=os.fsencode)

    return {
        "lasot_root": str(lasot),
        "tnl2k_train_root": str(tnl2k),
        "lasot_training_set_sha256": _file_sha256(training_set),
        "lasot_annotations_sha256": _aggregate_file_digests(lasot_rows),
        "lasot_sequence_count": len(lasot_names),
        "tnl2k_groundtruth_sha256": _aggregate_file_digests(
            (path, path.relative_to(tnl2k).as_posix()) for path in tnl2k_groundtruth
        ),
        "tnl2k_sequence_names_sha256": hashlib.sha256(
            "".join(f"{name}\n" for name in tnl2k_names).encode("utf-8")
        ).hexdigest(),
        "tnl2k_sequence_count": len(tnl2k_names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lasot-root", required=True)
    parser.add_argument("--tnl2k-root", required=True)
    parser.add_argument("--output", help="可选 JSON 报告路径")
    args = parser.parse_args()

    try:
        report = compute_source_fingerprints(args.lasot_root, args.tnl2k_root)
    except (OSError, UnicodeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    mismatches = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in EXPECTED.items()
        if report.get(key) != expected
    }
    result = {
        "schema_version": "cogtrack.stage1_source_verification.v1",
        **report,
        "expected": EXPECTED,
        "mismatches": mismatches,
        "ok": not mismatches,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
