#!/usr/bin/env python3
"""在多张 GPU 上启动一模型一进程的 vLLM 服务集群。

4B 模型能完整放进单张 4090。相比 TP=8，一卡一副本能并行处理八倍独立视频
序列，也不需要跨卡通信，更符合 CognitiveBench 的序列级并行范式。
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 vLLM 单卡副本集群")
    parser.add_argument("--model", required=True, help="基础或已合并 LoRA 的模型目录")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-images-per-prompt",
        type=int,
        default=3,
        help="正式 mosaic 协议需要 anchor/history/current 三张图；pair 消融可设为 2",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--log-dir", default="outputs/logs/vllm_fleet")
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    return parser


def _ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> int:
    args = _parser().parse_args()
    model = Path(args.model).expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus 不能为空")
    if args.max_images_per_prompt <= 0:
        raise ValueError("--max-images-per-prompt 必须为正整数")
    log_dir = (PROJECT_ROOT / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[str, int, subprocess.Popen, object]] = []
    stopping = False

    def stop_all(*_unused) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for _gpu, _port, process, _handle in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 15
        for _gpu, _port, process, _handle in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    process.kill()
        for _gpu, _port, _process, handle in processes:
            handle.close()

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    vllm = Path(sys.executable).with_name("vllm")
    compat_path = PROJECT_ROOT / "tools" / "vllm_compat"
    for rank, gpu in enumerate(gpus):
        port = args.base_port + rank
        log_path = log_dir / f"gpu{gpu}_port{port}.log"
        handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(compat_path), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        command = [
            str(vllm),
            "serve",
            str(model),
            "--served-model-name",
            args.served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--limit-mm-per-prompt",
            f'{{"image":{args.max_images_per_prompt},"video":0}}',
            "--mm-processor-kwargs",
            '{"use_fast":false}',
            "--generation-config",
            "vllm",
            "--disable-uvicorn-access-log",
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=False,
        )
        processes.append((gpu, port, process, handle))
        print(f"[启动] GPU {gpu} -> 127.0.0.1:{port}，日志 {log_path}", flush=True)

    deadline = time.time() + args.startup_timeout
    pending = {port for _gpu, port, _process, _handle in processes}
    while pending and time.time() < deadline and not stopping:
        for gpu, port, process, _handle in processes:
            if port not in pending:
                continue
            code = process.poll()
            if code is not None:
                stop_all()
                raise RuntimeError(f"GPU {gpu} 的 vLLM 服务提前退出，code={code}")
            if _ready(port):
                pending.remove(port)
                print(f"[就绪] GPU {gpu} / port {port}", flush=True)
        time.sleep(1)
    if pending:
        stop_all()
        raise TimeoutError(f"vLLM 服务启动超时，未就绪端口: {sorted(pending)}")

    print(f"[集群就绪] replicas={len(processes)}，按 Ctrl-C 安全停止", flush=True)
    try:
        while not stopping:
            for gpu, port, process, _handle in processes:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"GPU {gpu} / port {port} 意外退出，code={code}")
            time.sleep(5)
    finally:
        stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
