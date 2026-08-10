#!/usr/bin/env python3
"""使用本地 Qwen-VL 完成一次真实的双图跟踪推理。

该脚本从标准数据集 loader 读取首帧和一个后续帧，只把首帧 GT 传给 tracker；
后续帧 GT 仅用于脚本结束后的人工核对，不进入模型输入或 ``frame_info``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytracking.datasets.registry import load_dataset  # noqa: E402
from pytracking.evaluation.environment import load_environment  # noqa: E402
from pytracking.evaluation.runner import read_image_rgb  # noqa: E402
from pytracking.trackers.base import TrackerParams  # noqa: E402
from pytracking.trackers.cognitive_vlm import CognitiveVLMTracker  # noqa: E402


def _yaml(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"配置顶层必须是 mapping: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--tracker-config",
        help="完整 cognitive_vlm tracker YAML（推荐，可保留模型族对应的 bbox 协议）",
    )
    config_group.add_argument(
        "--model-config",
        help="兼容旧命令：仅提供本地 Qwen 模型 YAML，脚本按模型名称推断 bbox 协议",
    )
    parser.add_argument("--dataset-config", required=True, help="数据集 YAML")
    parser.add_argument("--env-config", help="可选本机环境 YAML")
    parser.add_argument("--sequence", help="序列名；默认取配置数据集第一个序列")
    parser.add_argument("--frame-id", type=int, help="搜索帧；默认选首个非首帧 present 关键帧")
    parser.add_argument("--output", help="可选：把结构化输出写入 JSON")
    return parser.parse_args()


def _legacy_bbox_protocol(model_config: str | Path) -> str:
    """为旧版 ``--model-config`` 命令选择模型原生坐标协议。

    新实验应直接传 ``--tracker-config``。保留此分支只是为了让已有命令仍可运行，
    但不再允许 Qwen3 静默落入 Qwen2.5 的绝对像素默认值。
    """

    payload = _yaml(model_config)
    identity = " ".join(
        str(payload.get(key, "")) for key in ("model_name", "model_path", "backend")
    ).lower()
    compact = identity.replace("-", "").replace("_", "").replace(".", "")
    if "qwen3vl" in compact:
        return "norm1000"
    if "qwen25vl" in compact:
        return "qwen_abs_pixel"
    raise ValueError(
        "仅凭 model YAML 无法可靠判断 bbox 协议；请改用 --tracker-config 指定完整实验配置"
    )


def _build_tracker_params(args: argparse.Namespace, environment) -> TrackerParams:
    """构造与标准 runner 相同语义的 tracker 参数，并注入本机路径。"""

    if args.tracker_config:
        payload = _yaml(args.tracker_config)
        if payload.get("tracker_name") not in (None, "cognitive_vlm"):
            raise ValueError("Qwen smoke test 只支持 tracker_name=cognitive_vlm")
    else:
        model_config = str(Path(args.model_config).expanduser().resolve())
        payload = {
            "tracker_name": "cognitive_vlm",
            "experiment_name": "legacy_model_config_smoke",
            "context_mode": "pair",
            "bbox_protocol": _legacy_bbox_protocol(model_config),
            "memory": {"enabled": False},
            "model_config": model_config,
            "_config_path": str(PROJECT_ROOT / "configs/trackers/smoke_test_legacy.yaml"),
        }

    # 机器相关路径始终来自环境配置，不写入可提交的 tracker YAML。
    payload["runtime"] = {
        "model_root": str(environment.model_root) if environment.model_root is not None else None,
        "project_root": str(environment.project_root),
    }
    payload["save_raw_response"] = True
    return TrackerParams(payload)


def _choose_frame(sequence, requested: int | None) -> int:
    if requested is not None:
        if requested <= 0 or requested >= len(sequence):
            raise ValueError(f"frame-id 必须位于 [1, {len(sequence) - 1}]")
        return requested
    candidates = sorted(index for index in sequence.keyframe_indices if index > 0)
    if sequence.target_visible is not None:
        visible = [index for index in candidates if bool(sequence.target_visible[index])]
        if visible:
            return visible[0]
    if candidates:
        return candidates[0]
    return 1


def main() -> int:
    args = _args()
    dataset_config = _yaml(args.dataset_config)
    dataset_name = str(dataset_config["name"])
    overrides = {}
    if dataset_config.get("root"):
        overrides[dataset_name] = str(dataset_config["root"])
    overrides.update({str(key): str(value) for key, value in (dataset_config.get("source_roots") or {}).items()})
    environment = load_environment(args.env_config, overrides=overrides)
    sequences = load_dataset(
        dataset_name,
        environment=environment,
        sequence_names=[args.sequence] if args.sequence else None,
        limit=None if args.sequence else 1,
        split=dataset_config.get("split", "test"),
    )
    sequence = sequences[0]
    frame_id = _choose_frame(sequence, args.frame_id)

    tracker_params = _build_tracker_params(args, environment)
    tracker = CognitiveVLMTracker(tracker_params)
    init_image = read_image_rgb(sequence.frames[0])
    current_image = read_image_rgb(sequence.frames[frame_id])
    init_info = sequence.init_info()
    init_info.update(frame_num=0, frame_path=sequence.frames[0], is_observation_frame=True)
    tracker.initialize(init_image, init_info)
    frame_info = sequence.frame_info(frame_id)
    frame_info.update(
        frame_num=frame_id,
        frame_path=sequence.frames[frame_id],
        is_observation_frame=True,
        observation_reason="smoke_test",
    )
    try:
        output = tracker.track(current_image, frame_info)
    finally:
        tracker.close()

    report = {
        "sequence": sequence.name,
        "frame_id": frame_id,
        "tracker_config": (
            str(Path(args.tracker_config).expanduser().resolve()) if args.tracker_config else None
        ),
        "model_config": str(tracker_params["model_config"]),
        "bbox_protocol": str(tracker_params["bbox_protocol"]),
        "tracker_output": output,
        # 只在推理完成后附加用于人工检查的 GT，未传入 tracker。
        "ground_truth_for_manual_check": {
            "target_presence": (
                "present" if sequence.target_visible is None or bool(sequence.target_visible[frame_id]) else "absent"
            ),
            "bbox_xywh": sequence.ground_truth_rect[frame_id].tolist(),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    status = output.get("execution", {}).get("status")
    if status != "ok":
        print(f"[失败] 本地 Qwen 推理未得到合法结果，execution={status}", file=sys.stderr)
        return 2
    print("[成功] 本地 Qwen-VL 加载、双图生成、严格解析和坐标转换均已通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
