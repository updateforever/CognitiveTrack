#!/usr/bin/env python3
"""运行标准单目标跟踪实验。

推荐使用结构化配置：

    python tracking/test.py \
        --config configs/env.local.yaml \
        --tracker-config configs/trackers/qwen25vl_7b_pair.yaml \
        --dataset-config configs/datasets/cognitivebench.yaml \
        --sequence 005 --debug-frames 5

同时保留 SUTrack/pytracking 熟悉的 ``tracker parameter`` 位置参数，便于快速
运行 ``python tracking/test.py dummy default --dataset lasot``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytracking.datasets.registry import list_datasets, load_dataset  # noqa: E402
from pytracking.evaluation.environment import load_environment  # noqa: E402
from pytracking.evaluation.observation_policy import (  # noqa: E402
    DenseObservationPolicy,
    KeyframeObservationPolicy,
)
from pytracking.evaluation.runner import DatasetRunner, TrackingFrameError  # noqa: E402
from pytracking.evaluation.tracker import TrackerSpec  # noqa: E402


def _read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"配置顶层必须是 mapping: {config_path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CognitiveTrack 标准逐帧推理入口")
    parser.add_argument("tracker", nargs="?", help="tracker 模块名；配置模式下可省略")
    parser.add_argument("parameter", nargs="?", default=None, help="参数名，默认 default")
    parser.add_argument("--tracker-config", help="tracker YAML；优先于位置参数中的参数名")
    parser.add_argument("--dataset", choices=list_datasets(), help="数据集注册名")
    parser.add_argument("--dataset-config", help="数据集 YAML，包含 name/split/root/source_roots")
    parser.add_argument("--split", help="覆盖数据集 split")
    parser.add_argument("--sequence", action="append", help="只运行指定序列，可重复传入")
    parser.add_argument("--config", "--env-config", dest="env_config", help="本机环境 YAML")
    parser.add_argument("--results-dir", help="临时覆盖结果根目录")
    parser.add_argument("--run-id", type=int, help="重复实验编号")
    parser.add_argument("--observation", choices=("dense", "keyframe"), help="覆盖观察策略")
    parser.add_argument("--keyframes", help="TXT/JSON 外部关键帧索引")
    parser.add_argument("--keyframe-index-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--missing-keyframes", choices=("dense", "none", "error"), default="dense")
    parser.add_argument("--debug-frames", type=int, help="仅跑前 N 帧；结果 manifest 会标记为 debug")
    parser.add_argument("--force", action="store_true", help="覆盖已完成结果")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单帧失败后继续；失败帧 execution=internal_error 且 bbox=null/NaN",
    )
    return parser


def _resolve_configs(args: argparse.Namespace) -> tuple[TrackerSpec, str, str | None, dict[str, str], str]:
    tracker_payload: dict[str, Any] = {}
    if args.tracker_config:
        tracker_payload = _read_yaml(args.tracker_config)
    tracker_name = args.tracker or tracker_payload.get("tracker_name")
    if not tracker_name:
        raise ValueError("请提供位置参数 tracker 或 --tracker-config 中的 tracker_name")
    parameter_name = args.parameter or tracker_payload.get("experiment_name") or "default"
    tracker_spec = TrackerSpec(
        name=str(tracker_name),
        parameter_name=str(parameter_name),
        run_id=args.run_id,
        config_path=args.tracker_config,
    )

    dataset_payload: dict[str, Any] = {}
    if args.dataset_config:
        dataset_payload = _read_yaml(args.dataset_config)
    dataset_name = args.dataset or dataset_payload.get("name")
    if not dataset_name:
        raise ValueError("请提供 --dataset 或 --dataset-config 中的 name")
    # 未显式指定时交给 registry 的 DatasetSpec 决定默认 split；这样 mgit_val
    # 之类的别名不会被 CLI 无意覆盖成 test。
    split = args.split or dataset_payload.get("split")

    dataset_overrides: dict[str, str] = {}
    if dataset_payload.get("root"):
        dataset_overrides[str(dataset_name).lower()] = str(dataset_payload["root"])
    source_roots = dataset_payload.get("source_roots", {}) or {}
    if not isinstance(source_roots, dict):
        raise TypeError("dataset config 的 source_roots 必须是 mapping")
    dataset_overrides.update({str(name).lower(): str(path) for name, path in source_roots.items()})

    policy_kind = (
        args.observation or str((tracker_payload.get("observation_policy") or {}).get("type", "dense")).lower()
    )
    return tracker_spec, str(dataset_name), str(split) if split is not None else None, dataset_overrides, policy_kind


def main() -> int:
    args = _parser().parse_args()
    try:
        tracker_spec, dataset_name, split, dataset_overrides, policy_kind = _resolve_configs(args)
        environment = load_environment(args.env_config, overrides=dataset_overrides)
        if args.results_dir:
            environment = environment.with_results_path(args.results_dir)
        dataset_kwargs = {"split": split} if split is not None else {}
        sequences = load_dataset(
            dataset_name,
            environment=environment,
            sequence_names=args.sequence,
            **dataset_kwargs,
        )

        if policy_kind == "dense":
            policy = DenseObservationPolicy()
        elif policy_kind == "keyframe":
            policy = KeyframeObservationPolicy(
                index_file=args.keyframes,
                index_base=args.keyframe_index_base,
                missing=args.missing_keyframes,
            )
        else:
            raise ValueError(f"未知 observation policy: {policy_kind}")

        runner = DatasetRunner(
            tracker_spec,
            environment,
            observation_policy=policy,
            fail_fast=not args.continue_on_error,
            max_frames=args.debug_frames,
            overwrite=args.force,
        )
        results = runner.run(sequences)
    except (OSError, KeyError, TypeError, ValueError, TrackingFrameError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    for result in results:
        if result.skipped_existing:
            print(f"[跳过] {result.dataset}/{result.sequence}: 完整结果已存在")
        else:
            fps = len(result.records) / result.total_time if result.total_time > 0 else float("inf")
            print(
                f"[完成] {result.dataset}/{result.sequence}: frames={len(result.records)}, "
                f"errors={result.errors}, tracker_fps={fps:.3f}"
            )
    print(f"结果根目录: {environment.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
