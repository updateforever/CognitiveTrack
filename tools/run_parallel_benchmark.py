#!/usr/bin/env python3
"""按关键帧工作量均衡切分 CognitiveBench，并发调用 vLLM 集群。

每个 worker 仍运行标准 ``tracking/test.py``，因此结果格式、断点续跑和评测规范
不发生变化；本工具只负责序列级调度，不另造一套推理循环。
"""

from __future__ import annotations

import argparse
import heapq
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytracking.datasets.registry import load_dataset  # noqa: E402
from pytracking.datasets.subset import sequence_subset_from_config  # noqa: E402
from pytracking.evaluation.environment import load_environment  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="并行运行标准 CognitiveBench")
    parser.add_argument("--config", required=True, help="环境配置 YAML")
    parser.add_argument("--tracker-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--log-dir", default="outputs/logs/benchmark_workers")
    parser.add_argument("--results-dir", help="覆盖结果根目录；默认使用环境配置")
    parser.add_argument("--debug-frames", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _yaml(path: str) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML 顶层必须是 mapping: {path}")
    return value


def _lpt_assign(weighted, bins: int):
    """把 ``(weight, name)`` 用 LPT 分配到固定数量的箱。"""

    heap = [(0, index, []) for index in range(bins)]
    heapq.heapify(heap)
    for weight, name in weighted:
        load, index, items = heapq.heappop(heap)
        items.append((weight, name))
        heapq.heappush(heap, (load + weight, index, items))
    return sorted(heap, key=lambda item: item[1])


def _balanced_shards(sequences, gpu_count: int, workers_per_gpu: int):
    """两级 LPT：先均衡 GPU 总请求量，再均衡卡内 worker。

    vLLM 是一卡一个服务，同卡 worker 共享相同的批处理队列。直接在所有 worker 上
    一次分箱后再连续映射 GPU，会在 Tiny 序列较少时把最长序列集中到第一张卡。
    """

    weighted = []
    for sequence in sequences:
        keyframes = set(int(item) for item in (sequence.keyframe_indices or []))
        observations = len(keyframes - {0})
        if not keyframes:
            observations = max(0, len(sequence) - 1)
        weighted.append((observations, sequence.name))
    weighted.sort(key=lambda item: (-item[0], item[1]))

    gpu_bins = _lpt_assign(weighted, gpu_count)
    shards = []
    for _gpu_load, gpu_rank, gpu_items in gpu_bins:
        worker_bins = _lpt_assign(gpu_items, workers_per_gpu)
        for worker_in_gpu, (load, _index, items) in enumerate(worker_bins):
            worker = gpu_rank * workers_per_gpu + worker_in_gpu
            shards.append((load, worker, gpu_rank, [name for _weight, name in items]))
    return sorted(shards, key=lambda item: item[1])


def main() -> int:
    args = _parser().parse_args()
    if args.gpu_count <= 0 or args.workers_per_gpu <= 0:
        raise ValueError("gpu-count / workers-per-gpu 必须为正整数")
    dataset_payload = _yaml(args.dataset_config)
    dataset_name = str(dataset_payload["name"])
    split = dataset_payload.get("split")
    source_roots = dataset_payload.get("source_roots", {}) or {}
    overrides = {str(key): str(value) for key, value in source_roots.items()}
    if dataset_payload.get("root"):
        overrides[dataset_name.lower()] = str(dataset_payload["root"])
    environment = load_environment(args.config, overrides=overrides)
    selected_names = sequence_subset_from_config(dataset_payload, args.dataset_config)
    sequences = load_dataset(
        dataset_name,
        environment=environment,
        split=split,
        sequence_names=selected_names,
    )

    worker_count = args.gpu_count * args.workers_per_gpu
    shards = _balanced_shards(sequences, args.gpu_count, args.workers_per_gpu)
    # Tiny 的序列数可能小于 gpu_count * workers_per_gpu。空 shard 不能启动
    # tracking/test.py：CLI 没有 --sequence 时会按 dataset config 加载整个 Tiny，
    # 从而意外重复跑全量子集。
    active_shards = [shard for shard in shards if shard[3]]
    total_observations = sum(load for load, _worker, _gpu, _names in shards)
    gpu_loads = [
        sum(load for load, _worker, gpu, _names in shards if gpu == gpu_rank)
        for gpu_rank in range(args.gpu_count)
    ]
    print(
        f"[切分] sequences={len(sequences)} observations={total_observations} "
        f"workers={len(active_shards)}/{worker_count} "
        f"worker_load=[{min(x[0] for x in active_shards)}, {max(x[0] for x in active_shards)}] "
        f"gpu_load=[{min(gpu_loads)}, {max(gpu_loads)}]",
        flush=True,
    )
    if args.dry_run:
        for load, worker, gpu_rank, names in active_shards:
            print(f"worker={worker:02d} gpu={gpu_rank} load={load} seqs={len(names)}")
        return 0

    tracker_payload = _yaml(args.tracker_config)
    experiment = str(tracker_payload.get("experiment_name", "default"))
    log_dir = (PROJECT_ROOT / args.log_dir / experiment).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for load, worker, gpu_rank, names in active_shards:
        port = args.base_port + gpu_rank
        log_path = log_dir / f"worker{worker:02d}_gpu{gpu_rank}.log"
        handle = log_path.open("a", encoding="utf-8")
        handles.append(handle)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tracking" / "test.py"),
            "--config",
            str(Path(args.config).resolve()),
            "--tracker-config",
            str(Path(args.tracker_config).resolve()),
            "--dataset-config",
            str(Path(args.dataset_config).resolve()),
            "--continue-on-error",
        ]
        if args.results_dir:
            command.extend(["--results-dir", str(Path(args.results_dir).resolve())])
        if args.debug_frames:
            command.extend(["--debug-frames", str(args.debug_frames)])
        for name in names:
            command.extend(["--sequence", name])
        env = os.environ.copy()
        env["COGTRACK_VLLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker, gpu_rank, load, process, log_path))
        print(f"[worker {worker:02d}] gpu={gpu_rank} load={load} seqs={len(names)} pid={process.pid}", flush=True)

    try:
        pending = {worker for worker, _gpu, _load, _process, _log in processes}
        failures = []
        while pending:
            for worker, gpu, _load, process, log_path in processes:
                if worker not in pending:
                    continue
                code = process.poll()
                if code is None:
                    continue
                pending.remove(worker)
                print(f"[退出] worker={worker:02d} gpu={gpu} code={code} log={log_path}", flush=True)
                if code:
                    failures.append((worker, code, log_path))
            if pending:
                print(f"[运行中] remaining_workers={len(pending)}", flush=True)
                time.sleep(30)
        if failures:
            print(f"[失败] {failures}", file=sys.stderr)
            return 2
        print("[完成] 全部 worker 正常退出", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[中断] 正在终止 worker；已完成序列可直接断点续跑", flush=True)
        for _worker, _gpu, _load, process, _log in processes:
            if process.poll() is None:
                process.terminate()
        return 130
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
