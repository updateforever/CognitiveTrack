#!/usr/bin/env python3
"""用本地 VLM 教师为 LaSOT/TNL2K 生成 state-update 监督标签 JSONL。

补的是 ``build_mgit_state_update_labels.py`` 明确留下的空白：LaSOT/TNL2K 没有逐帧状态标注，
在 ``three_state`` 模式下全部落到 ``masked_unknown``，于是 ``verified_update`` 只能来自
MGIT 一个源。本脚本产出的标签与 MGIT 标签同 schema，可直接拼接后一起喂给第 3 步。

为什么不做候选召回
------------------
归档设计 ``docs/archive/superseded_v640_20260817/state_annotation.md`` 原设想先用启发式
召回可能变化的帧再送教师。实测否掉了这条：
LaSOT 复现帧只占 present 的 0.1%、TNL2K 0.3%，而 bbox 派生信号（尺度比 p99=1.12、
长宽比 p99=1.24）在"语义状态是否变化"上没有区分度——极值 48× 是标注跳变不是状态变化。
用它筛帧等于把"尺度变了 12%"当成"状态变了"，并且会让训练出的更新决策边界依赖一个推理
时根本不存在的信号。改为**固定步长均匀游走**，让教师自己决定 update / keep：这既无偏，
也让 keep 判定成为真标签（``verified_hard_null``）而不是被丢掉的负样本。

流水线
------
1. ``synthesize_vlt_v6_dataset.py --plan-only`` 产出 ``sampling_plan.json``；
2. 本脚本读 plan，按 ``--stride`` 沿时间顺序游走每条序列的计划帧，逐步维护状态文本，
   对每个决策点跑两次不同 seed 的教师生成 + 一次跨族 verifier 裁决；
3. 与 MGIT 标签拼接后传给第 3 步 ``--memory-labels``。

顺序游走而非独立采样是必须的：``current_target_state`` 是整体替换快照，第 t 帧的输入侧
状态取决于此前哪些更新被接受。独立采样每帧都会拿序列初始身份当输入侧，产出的更新链在
训练回放时根本不自洽。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# 必须在导入 vLLM 之前设置。vLLM V1 engine 默认用 fork 起 EngineCore 子进程，而本脚本在
# 建模型之前就要 load_environment() 并遍历数据集，那条导入链会初始化 CUDA，随后 fork 出的
# 子进程再碰 CUDA 就会 "Cannot re-initialize CUDA in forked subprocess" 直接崩掉。
# 在模块级设而不是让调用方 export，是因为忘记设的表现是加载几分钟后才失败。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# 历史工具位于 tools/archive/state_teacher_v1/，向上三级才是仓库根。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.prompts.state_teacher import (  # noqa: E402
    STATE_TEACHER_PROMPT_VERSION,
    build_state_keep_verifier_prompt,
    build_state_teacher_prompt,
    build_state_verifier_prompt,
)
from cogtrack.training.state_teacher_labels import (  # noqa: E402
    RejectionLog,
    is_vacuous_update,
    parse_keep_verifier_response,
    parse_teacher_response,
    parse_verifier_response,
    reconcile_dual_pass,
)

#: 教师标签的来源标记。与 MGIT 的 ``mgit_action_segment_v1`` 并列，下游可按来源分层评估。
TEACHER_LABEL_SOURCE = "vlm_teacher_v1"
TEACHER_HARD_NULL_SOURCE = "vlm_teacher_hard_null_v1"

#: 只对这两个源跑教师。MGIT 有人工 action 分段，用教师覆盖它是拿弱标签换强标签。
TEACHER_DATASETS = ("lasot", "tnl2k")

PLACEHOLDER = "<|image_pad|>"


def _plan_frames(plan_path: Path, datasets: tuple[str, ...]) -> dict[tuple[str, str], list[int]]:
    """从 sampling plan 提取 ``(dataset, sequence) -> 升序计划帧``。"""

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = payload.get("sequences")
    if not isinstance(entries, list):
        raise ValueError(f"sampling plan 缺少 sequences 列表：{plan_path}")
    wanted = {d.strip().lower() for d in datasets}
    frames: dict[tuple[str, str], set[int]] = {}
    for entry in entries:
        dataset = str(entry.get("dataset") or "").strip().lower()
        if dataset not in wanted:
            continue
        key = (dataset, str(entry["sequence"]))
        for frame_id in entry.get("frame_ids") or ():
            frames.setdefault(key, set()).add(int(frame_id))
    return {key: sorted(values) for key, values in sorted(frames.items())}


def _select_sequences(
    frames_by_key: dict[tuple[str, str], list[int]],
    *,
    dataset: str,
    limit: int,
    seed: int,
    min_frames: int,
) -> list[tuple[str, list[int]]]:
    """确定性地挑选本数据源要标注的序列。

    按序列名哈希而不是 plan 顺序抽样，这样加大 ``--limit`` 时旧选择是新选择的子集，
    已经花掉的教师推理不会因为扩量而作废。
    """

    pool = [
        (name, plan)
        for (ds, name), plan in frames_by_key.items()
        if ds == dataset and len(plan) >= min_frames
    ]
    pool.sort(key=lambda item: item[0])
    rng = random.Random(f"teacher-select-{dataset}-{seed}")
    rng.shuffle(pool)
    return pool[:limit]


def _decision_points(
    frame_plan: list[int], *, stride: int, max_steps: int, min_frame_id: int = 0
) -> list[int]:
    """从计划帧里按步长取决策点。

    只从 plan 内取帧：标签落在没被采样的帧上不会进入任何一行训练数据，纯属浪费推理。

    ``min_frame_id`` 挡掉紧贴锚点的帧。实测教师在 gap=1 上一律判 keep——这是对的，
    相邻两帧本来就不可能有持久状态变化——但那次推理的结论是预先就知道的，等于白烧。
    """

    if stride <= 0:
        raise ValueError("stride 必须为正")
    eligible = [f for f in frame_plan if f >= min_frame_id]
    picked = eligible[::stride]
    return picked[:max_steps]


class TeacherRunner:
    """封装 vLLM 批式推理。``dry_run`` 下不加载模型，只走通流水线结构。"""

    def __init__(
        self,
        model_path: str,
        *,
        dry_run: bool = False,
        max_pixels: int = 1024 * 28 * 28,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.model_path = model_path
        self.dry_run = dry_run
        self._llm = None
        self._processor = None
        self._max_pixels = max_pixels
        self._gpu_util = gpu_memory_utilization
        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size 必须为正数")
        self._tensor_parallel_size = tensor_parallel_size

    def _ensure_loaded(self) -> None:
        if self._llm is not None or self.dry_run:
            return
        from transformers import AutoProcessor
        from vllm import LLM

        self._processor = AutoProcessor.from_pretrained(
            self.model_path, max_pixels=self._max_pixels
        )
        self._llm = LLM(
            model=self.model_path,
            # 两图输入，必须显式放开否则 vLLM 默认只接受一张。
            limit_mm_per_prompt={"image": 2},
            gpu_memory_utilization=self._gpu_util,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=False,
            tensor_parallel_size=self._tensor_parallel_size,
        )

    def generate(
        self,
        requests: list[tuple[str, str, list[Any]]],
        *,
        seed: int,
        temperature: float,
        max_tokens: int = 320,
    ) -> list[str]:
        """批量生成。``requests`` 是 ``(system, user, images)`` 三元组列表。"""

        if self.dry_run:
            rows: list[str] = []
            for system, _user, _images in requests:
                if "proposed no-update decision" in system:
                    rows.append(
                        '{"accept_keep": true, "failure_mode": "none", '
                        '"justification": "dry-run"}'
                    )
                elif "audit proposed state-memory updates" in system:
                    rows.append(
                        '{"accept": true, "failure_mode": "none", '
                        '"justification": "dry-run"}'
                    )
                else:
                    rows.append(
                        '{"state_changed": false, "reason_code": "no_significant_change", '
                        '"new_state": "", "evidence": "dry-run"}'
                    )
            return rows
        if not requests:
            return []
        self._ensure_loaded()
        from vllm import SamplingParams

        prompts = []
        for system, user, images in requests:
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": user}],
                },
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append({"prompt": text, "multi_modal_data": {"image": images}})
        params = SamplingParams(
            temperature=temperature,
            top_p=0.9 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
            seed=seed,
        )
        outputs = self._llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    def shutdown(self) -> None:
        """释放显存。教师和 verifier 在单卡上必须顺序驻留，不能同时占用。"""

        if self._llm is None:
            return
        import gc

        import torch

        del self._llm
        self._llm = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _marked_frame(sequence: Any, frame_id: int, *, max_side: int) -> Any:
    """读一帧并画上 GT 框，返回 PIL Image。

    画框是这条流水线成立的前提：教师因此不需要跟踪或定位，只需判断框内目标的状态。
    降采样长边是为了控制 visual token 数——两图输入下原始 1080p 会把上下文吃满。
    """

    import cv2
    import numpy as np
    from PIL import Image

    from cogtrack.context.visual import draw_reference_box

    bgr = cv2.imread(str(sequence.frames[frame_id]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"无法读取帧：{sequence.frames[frame_id]}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # 转 list：draw_reference_box 走 collections.abc.Sequence 校验，ndarray 不是其实例。
    box = [float(v) for v in np.asarray(sequence.ground_truth_rect[frame_id], dtype=float)]
    marked = draw_reference_box(rgb, box)
    height, width = marked.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1.0:
        marked = cv2.resize(
            marked, (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return Image.fromarray(marked)


def _is_present(sequence: Any, frame_id: int) -> bool:
    """与采样层一致的存在性判据：正面积框且未被标为不可见。"""

    import numpy as np

    box = np.asarray(sequence.ground_truth_rect[frame_id], dtype=float)
    if not (box[2] > 0 and box[3] > 0):
        return False
    if sequence.target_visible is not None:
        return bool(sequence.target_visible[frame_id])
    return True


class SequenceWalk:
    """一条序列的顺序游走状态。

    每条序列独立维护"当前已接受状态"，但所有序列在同一 step index 上可以并入一个 batch。
    顺序依赖只存在于序列内部，跨序列并行不会破坏替换快照语义——这是本流水线能把
    batch size 提到序列数量级的原因。
    """

    def __init__(
        self,
        dataset: str,
        sequence: Any,
        frame_plan: list[int],
        identity: str,
        *,
        stride: int,
        max_steps: int,
        min_anchor_gap: int,
    ):
        self.dataset = dataset
        self.sequence = sequence
        self.name = sequence.name
        self.identity = identity
        self.current_state = identity
        self.anchor_frame = next(
            (f for f in range(len(sequence.frames)) if _is_present(sequence, f)), 0
        )
        # 只在 present 帧上做判定。absent 帧的记忆由采样器无条件压成占位 null，
        # 在这里问教师"目标状态变了吗"没有意义——它连目标都看不到。
        present_plan = [f for f in frame_plan if _is_present(sequence, f)]
        self.points = _decision_points(
            present_plan,
            stride=stride,
            max_steps=max_steps,
            min_frame_id=self.anchor_frame + min_anchor_gap,
        )
        self.step = 0
        self.rows: list[dict[str, Any]] = []

    def has_next(self) -> bool:
        return self.step < len(self.points)

    def current_frame(self) -> int:
        return self.points[self.step]

    def record(
        self,
        *,
        new_state: str | None,
        reason: str,
        verified: bool,
    ) -> None:
        """落一条标签。``new_state is None`` 表示 keep。"""

        frame_id = self.current_frame()
        if new_state:
            self.rows.append(
                {
                    "dataset": self.dataset,
                    "sequence": self.name,
                    "frame_id": frame_id,
                    "target_status": "present",
                    "memory_update": new_state,
                    "verified_null": False,
                    "source": TEACHER_LABEL_SOURCE,
                    "reviewed": False,
                    "input_state": self.current_state,
                    "reason": reason,
                }
            )
            # 接受后才推进状态：下一个决策点的输入侧必须是这一条，否则更新链不自洽。
            self.current_state = new_state
        elif verified:
            self.rows.append(
                {
                    "dataset": self.dataset,
                    "sequence": self.name,
                    "frame_id": frame_id,
                    "target_status": "present",
                    "memory_update": None,
                    # 教师明确判定"无需更新"是正面声明，不是缺标签。
                    "verified_null": True,
                    "source": TEACHER_HARD_NULL_SOURCE,
                    "reviewed": False,
                    "input_state": self.current_state,
                    "reason": reason,
                }
            )
        self.step += 1

    def skip(self) -> None:
        """判定不可用时跳过该帧，既不落 update 也不落 hard-null。

        落 hard-null 等于把"教师没说清"伪造成"已证明无需更新"，会污染 null 方向的监督。
        什么都不落则自动退回 masked_unknown，是唯一诚实的处理。
        """

        self.step += 1


def _run_teacher_stage(
    walks: list[SequenceWalk],
    runner: TeacherRunner,
    *,
    max_side: int,
    seeds: tuple[int, int],
    temperature: float,
    rejections: RejectionLog,
    reasons: Counter,
    log_every: int,
    raw_sink: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """按 step index 对齐地并行推进所有序列，直到全部走完。

    这里**不做** verify。单卡 46 GB 装不下教师和 verifier 同时驻留（vLLM 按
    ``gpu_memory_utilization`` 预占显存），所以教师先独立走完、建出一条自洽的更新链，
    卸载后再由 verifier 批量裁剪。代价是教师链里可能含有 verifier 会否掉的链节，
    ``_apply_verifier_verdicts`` 通过重放修正下游 ``input_state``。
    """

    step_index = 0
    while any(walk.has_next() for walk in walks):
        active = [w for w in walks if w.has_next()]
        requests: list[tuple[str, str, list[Any]]] = []
        prepared: list[tuple[SequenceWalk, list[Any], int]] = []
        for walk in active:
            frame_id = walk.current_frame()
            try:
                images = [
                    _marked_frame(walk.sequence, walk.anchor_frame, max_side=max_side),
                    _marked_frame(walk.sequence, frame_id, max_side=max_side),
                ]
            except OSError:
                rejections.add("frame_read_failure")
                walk.skip()
                continue
            gap = frame_id - walk.anchor_frame
            spec = build_state_teacher_prompt(
                initial_identity=walk.identity,
                current_state=walk.current_state,
                frame_gap=gap,
            )
            requests.append((spec.system_prompt, spec.user_prompt, images))
            prepared.append((walk, images, gap))
        if not prepared:
            step_index += 1
            continue

        first_raw = runner.generate(requests, seed=seeds[0], temperature=temperature)
        second_raw = runner.generate(requests, seed=seeds[1], temperature=temperature)

        # strict=True：输出数与请求数不等说明推理端错配，静默截断会把标签贴到错误的帧上。
        for (walk, _images, _gap), raw_a, raw_b in zip(
            prepared, first_raw, second_raw, strict=True
        ):
            first, reason_a = parse_teacher_response(raw_a)
            second, reason_b = parse_teacher_response(raw_b)
            if first is None:
                rejections.add(reason_a)
            if second is None:
                rejections.add(reason_b)
            decision, reason = reconcile_dual_pass(first, second)
            reasons[reason] += 1
            if raw_sink is not None:
                raw_sink.append(
                    {
                        "dataset": walk.dataset,
                        "sequence": walk.name,
                        "frame_id": walk.current_frame(),
                        "anchor_frame": walk.anchor_frame,
                        "input_state": walk.current_state,
                        "outcome": reason,
                        "pass_1_rejection": reason_a,
                        "pass_2_rejection": reason_b,
                        "pass_1_raw": raw_a,
                        "pass_2_raw": raw_b,
                    }
                )
            if decision is None:
                walk.skip()
                continue
            if not decision.state_changed:
                walk.record(new_state=None, reason=decision.reason_code, verified=True)
            elif is_vacuous_update(decision.new_state, walk.current_state):
                # 声称变了却把输入状态换个说法抄回。不落 hard-null：教师的判断是"变了"，
                # 把它反转成"已证明没变"是替它下一个它没做过的结论。
                rejections.add("vacuous_update")
                walk.skip()
            else:
                walk.record(
                    new_state=decision.new_state, reason=decision.reason_code, verified=True
                )

        step_index += 1
        if log_every and step_index % log_every == 0:
            done = sum(len(w.rows) for w in walks)
            remaining = sum(1 for w in walks if w.has_next())
            print(
                f"  step {step_index}: 已落标签 {done}，仍在游走的序列 {remaining}",
                file=sys.stderr,
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for walk in walks:
        rows.extend(walk.rows)
    return rows


def _apply_verifier_verdicts(
    rows: list[dict[str, Any]],
    walks: list[SequenceWalk],
    verifier: TeacherRunner,
    *,
    max_side: int,
    batch_size: int,
    rejections: RejectionLog,
) -> list[dict[str, Any]]:
    """用独立模型裁决所有 update 与 hard-null，然后重放状态链。

    重放是必须的：教师链假设它自己每条更新都被接受，一旦 verifier 否掉第 t 条，第 t+1
    条记录的 ``input_state`` 就指向一个从未成立的状态。``input_state`` 只做审计不进
    response，但它同时是防泄漏断言的依据，留着错值等于让那道断言失去意义。
    """

    if batch_size <= 0:
        raise ValueError("verifier batch_size 必须为正数")
    walk_by_key = {(w.dataset, w.name): w for w in walks}
    by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault((row["dataset"], row["sequence"]), []).append(row)
    for seq_rows in by_sequence.values():
        seq_rows.sort(key=lambda row: int(row["frame_id"]))

    positions = {key: 0 for key in by_sequence}
    states = {
        key: walk_by_key[key].identity
        for key in by_sequence
        if key in walk_by_key
    }
    kept: list[dict[str, Any]] = []
    processed = 0
    total = len(rows)

    # 同一序列必须按时间顺序裁决，因为第 t 条是否被接受决定第 t+1 条看到的 previous_state。
    # 每轮每条序列最多取一行，不同序列仍可组成大 batch。
    while True:
        active = [
            (key, seq_rows[positions[key]])
            for key, seq_rows in sorted(by_sequence.items())
            if positions[key] < len(seq_rows)
        ]
        if not active:
            break
        for start in range(0, len(active), batch_size):
            chunk = active[start : start + batch_size]
            requests: list[tuple[str, str, list[Any]]] = []
            prepared: list[
                tuple[tuple[str, str], dict[str, Any], SequenceWalk, str]
            ] = []
            for key, row in chunk:
                walk = walk_by_key.get(key)
                if walk is None:
                    rejections.add("missing_sequence_walk")
                    positions[key] += 1
                    processed += 1
                    continue
                current_state = states[key]
                try:
                    images = [
                        _marked_frame(walk.sequence, walk.anchor_frame, max_side=max_side),
                        _marked_frame(
                            walk.sequence, int(row["frame_id"]), max_side=max_side
                        ),
                    ]
                except OSError:
                    rejections.add("frame_read_failure")
                    positions[key] += 1
                    processed += 1
                    continue
                if row["memory_update"] is None:
                    spec = build_state_keep_verifier_prompt(
                        initial_identity=walk.identity,
                        previous_state=current_state,
                        frame_gap=int(row["frame_id"]) - walk.anchor_frame,
                    )
                else:
                    spec = build_state_verifier_prompt(
                        initial_identity=walk.identity,
                        previous_state=current_state,
                        candidate_state=str(row["memory_update"]),
                        frame_gap=int(row["frame_id"]) - walk.anchor_frame,
                    )
                requests.append((spec.system_prompt, spec.user_prompt, images))
                prepared.append((key, row, walk, current_state))

            if requests:
                # verifier 用贪心解码：二分裁决不应引入采样随机性。
                verdicts = verifier.generate(requests, seed=0, temperature=0.0)
                for (key, row, _walk, current_state), raw in zip(
                    prepared, verdicts, strict=True
                ):
                    accept, mode = (
                        parse_keep_verifier_response(raw)
                        if row["memory_update"] is None
                        else parse_verifier_response(raw)
                    )
                    if accept is None:
                        rejections.add(mode)
                    elif not accept:
                        rejections.add(mode)
                    elif row["memory_update"] is None:
                        kept.append(
                            {**row, "input_state": current_state, "reviewed": True}
                        )
                    elif is_vacuous_update(str(row["memory_update"]), current_state):
                        rejections.add("vacuous_after_pruning")
                    else:
                        kept.append(
                            {**row, "input_state": current_state, "reviewed": True}
                        )
                        states[key] = str(row["memory_update"])
                    positions[key] += 1
                    processed += 1
            print(
                f"  verify {processed}/{total}",
                file=sys.stderr,
                flush=True,
            )
    return kept


def _sequence_identity(sequence: Any) -> str:
    """取序列的不可变初始身份文本，与采样层 ``initial_target`` 语言域一致。

    取不到就返回空串让调用方跳过整条序列：没有身份锚点，教师无从判断"是否改写了身份"，
    产出的描述可能整体漂移到另一个物体上。
    """

    for attr in ("language_query", "target_language", "nlp"):
        value = getattr(sequence, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = getattr(sequence, "metadata", None) or {}
    for key in ("language_query", "initial_language", "nlp"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-plan", required=True, help="第一步产出的 sampling_plan.json")
    parser.add_argument("--output", required=True, help="输出 state_update_teacher_labels.jsonl")
    parser.add_argument("--report", help="可选：写出教师运行报告 JSON。")
    parser.add_argument(
        "--teacher-model",
        default="/root/public/models/Qwen/Qwen3-VL-32B-Instruct",
        help="教师权重路径；当前正式候选流程使用 Qwen3-VL-32B。",
    )
    parser.add_argument(
        "--verifier-model",
        default="",
        help=(
            "独立 verifier 权重路径，建议用不同模型族（如 OpenGVLab/InternVL2_5-26B）"
            "以避免同源偏差。留空则跳过裁决，标签全部 reviewed=False。"
        ),
    )
    parser.add_argument(
        "--require-independent-verifier",
        action="store_true",
        help="没有 verifier 时直接失败；正式 state_update_sft 必须打开。",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(TEACHER_DATASETS),
        help="逗号分隔的数据源。MGIT 有人工分段标注，默认不含。",
    )
    parser.add_argument(
        "--sequences-per-dataset",
        type=int,
        default=60,
        help="每个数据源标注的序列数。按序列名哈希确定性选取，扩量时旧选择是新选择的子集。",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help=(
            "在计划帧列表上的取样步长。正式 40-case plan 默认 2，约提供 3,000 个"
            "决策点，给双次一致性和 verifier 淘汰留出余量。"
        ),
    )
    parser.add_argument(
        "--max-steps-per-sequence",
        type=int,
        default=30,
        help="单序列最多决策点数，用来给总推理量设上限。",
    )
    parser.add_argument(
        "--max-output-labels",
        type=int,
        help="确定性裁剪最终标签数量；约 1,500 条额外 teacher 数据可设为 1500。",
    )
    parser.add_argument(
        "--min-output-labels",
        type=int,
        default=0,
        help=(
            "正式运行期望的最低 verifier 通过数；低于该值仍会写出标签和报告，但命令"
            "返回失败，防止数量不足的产物被误认为约 1,500 条正式数据。"
        ),
    )
    parser.add_argument(
        "--min-planned-frames",
        type=int,
        default=8,
        help="计划帧少于该数的序列不进教师流程：太短的链产不出有意义的状态演化。",
    )
    parser.add_argument(
        "--min-anchor-gap",
        type=int,
        default=120,
        help=(
            "决策点距锚点至少这么多帧。默认 120（30fps 下约 4 秒）：更近的帧教师一律判 "
            "keep，那次推理的结论预先就知道，纯属浪费。"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260816, help="序列选取与生成的基准 seed。")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="教师采样温度。双次生成必须有随机性才能测出自一致性，0 会让两次输出完全相同。",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=896,
        help="送入教师前把长边降到该像素数，控制两图输入的 visual token 预算。",
    )
    parser.add_argument("--verify-batch-size", type=int, default=64, help="verifier 批大小。")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90, help="vLLM 显存占用比例。"
    )
    parser.add_argument(
        "--teacher-tensor-parallel-size",
        type=int,
        default=1,
        help="Qwen3-VL teacher 使用的 vLLM tensor-parallel GPU 数。",
    )
    parser.add_argument(
        "--verifier-tensor-parallel-size",
        type=int,
        default=1,
        help="独立 verifier 使用的 vLLM tensor-parallel GPU 数；两模型按阶段顺序驻留。",
    )
    parser.add_argument("--log-every", type=int, default=5, help="每多少 step 打一次进度。")
    parser.add_argument(
        "--raw-output",
        help=(
            "可选：把每次教师原始输出写成 JSONL。被拒样本只有计数无法判断是 prompt 不清"
            "还是模型能力问题，留下原文才能定位。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不加载模型，只走通选序列、读帧、拼 prompt、落盘的结构。",
    )
    args = parser.parse_args()
    if args.require_independent_verifier and not args.verifier_model and not args.dry_run:
        raise SystemExit("正式 teacher 标签必须提供 --verifier-model")
    if args.max_output_labels is not None and args.max_output_labels <= 0:
        raise SystemExit("--max-output-labels 必须为正数")
    if args.min_output_labels < 0:
        raise SystemExit("--min-output-labels 不能为负数")
    if args.teacher_tensor_parallel_size <= 0 or args.verifier_tensor_parallel_size <= 0:
        raise SystemExit("teacher/verifier tensor parallel size 必须为正数")
    if (
        args.max_output_labels is not None
        and args.min_output_labels > args.max_output_labels
    ):
        raise SystemExit("--min-output-labels 不能大于 --max-output-labels")

    plan_path = Path(args.sampling_plan).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    datasets = tuple(d.strip().lower() for d in args.datasets.split(",") if d.strip())
    if not datasets:
        raise SystemExit("--datasets 不能为空")
    if args.temperature <= 0 and not args.dry_run:
        raise SystemExit("--temperature 必须为正：双次生成需要随机性才能测自一致性")

    frames_by_key = _plan_frames(plan_path, datasets)
    if not frames_by_key:
        raise SystemExit(f"sampling plan 中没有 {datasets} 的序列：{plan_path}")

    from pytracking.datasets import iter_dataset
    from pytracking.evaluation.environment import load_environment

    environment = load_environment()

    walks: list[SequenceWalk] = []
    skipped_no_identity: list[str] = []
    for dataset in datasets:
        chosen = _select_sequences(
            frames_by_key,
            dataset=dataset,
            limit=args.sequences_per_dataset,
            seed=args.seed,
            min_frames=args.min_planned_frames,
        )
        wanted = {name for name, _ in chosen}
        if not wanted:
            continue
        plan_by_name = dict(chosen)
        # 必须显式传 split="train"：iter_dataset 默认给 test split，而 sampling plan 建在
        # train 上，两者序列名零交集。更要紧的是把教师标签建到测试序列上会污染评测。
        for sequence in iter_dataset(dataset, environment=environment, split="train"):
            if sequence.name not in wanted:
                continue
            source_split = str(sequence.metadata.get("split", "")).lower()
            if source_split != "train":
                raise SystemExit(
                    f"拒绝非训练划分：{dataset}/{sequence.name} split={source_split!r}"
                )
            identity = _sequence_identity(sequence)
            if not identity:
                skipped_no_identity.append(f"{dataset}/{sequence.name}")
                continue
            walk = SequenceWalk(
                dataset,
                sequence,
                plan_by_name[sequence.name],
                identity,
                stride=args.stride,
                max_steps=args.max_steps_per_sequence,
                min_anchor_gap=args.min_anchor_gap,
            )
            if walk.points:
                walks.append(walk)

    if not walks:
        raise SystemExit("没有可标注的序列：检查 --sequences-per-dataset 与 --min-planned-frames")

    planned_decisions = sum(len(w.points) for w in walks)
    if args.max_output_labels is not None and planned_decisions < args.max_output_labels:
        raise SystemExit(
            "教师候选计划不足以产出目标标签量："
            f"decision_points={planned_decisions} < max_output_labels={args.max_output_labels}。"
            "请增大 plan 的每序列 case、序列数，或减小 stride。"
        )
    print(
        f"教师阶段：{len(walks)} 条序列，{planned_decisions} 个决策点，"
        f"双次生成共 {planned_decisions * 2} 次推理",
        file=sys.stderr,
        flush=True,
    )

    rejections = RejectionLog()
    verifier_rejections = RejectionLog()
    reasons: Counter = Counter()

    raw_sink: list[dict[str, Any]] | None = [] if args.raw_output else None
    teacher = TeacherRunner(
        args.teacher_model,
        dry_run=args.dry_run,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.teacher_tensor_parallel_size,
    )
    rows = _run_teacher_stage(
        walks,
        teacher,
        max_side=args.max_image_side,
        seeds=(args.seed, args.seed + 1),
        temperature=args.temperature,
        rejections=rejections,
        reasons=reasons,
        log_every=args.log_every,
        raw_sink=raw_sink,
    )
    teacher.shutdown()

    if raw_sink is not None:
        raw_path = Path(args.raw_output).expanduser().resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            for entry in raw_sink:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    verified = False
    if args.verifier_model:
        update_candidates = sum(1 for row in rows if row["memory_update"] is not None)
        print(
            "verifier 阶段："
            f"{len(rows)} 个候选（update={update_candidates}, "
            f"hard-null={len(rows) - update_candidates}）",
            file=sys.stderr,
            flush=True,
        )
        verifier = TeacherRunner(
            args.verifier_model,
            dry_run=args.dry_run,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.verifier_tensor_parallel_size,
        )
        rows = _apply_verifier_verdicts(
            rows,
            walks,
            verifier,
            max_side=args.max_image_side,
            batch_size=args.verify_batch_size,
            rejections=verifier_rejections,
        )
        verifier.shutdown()
        verified = True

    # 与 MGIT builder 同一条防泄漏硬断言：更新文本不能等于它自己的输入侧状态。
    for row in rows:
        if row["memory_update"] is None:
            continue
        if row["memory_update"] == row["input_state"]:
            raise SystemExit(
                f"标签自相矛盾：{row['sequence']} frame={row['frame_id']} 的更新等于输入状态"
            )

    if args.max_output_labels is not None and len(rows) > args.max_output_labels:
        # 更新与 hard-null 各自稳定排序后再裁剪，避免数量限制把一类监督信号全部吞掉。
        updates = [row for row in rows if row["memory_update"] is not None]
        hard_nulls = [row for row in rows if row["memory_update"] is None]
        update_budget = min(len(updates), max(1, args.max_output_labels // 2))
        null_budget = args.max_output_labels - update_budget
        if null_budget > len(hard_nulls):
            update_budget = min(len(updates), args.max_output_labels - len(hard_nulls))
            null_budget = args.max_output_labels - update_budget
        updates = sorted(
            updates,
            key=lambda row: f"{row['dataset']}::{row['sequence']}::{row['frame_id']}",
        )[:update_budget]
        hard_nulls = sorted(
            hard_nulls,
            key=lambda row: f"{row['dataset']}::{row['sequence']}::{row['frame_id']}",
        )[:null_budget]
        rows = updates + hard_nulls

    rows.sort(key=lambda r: (r["dataset"], r["sequence"], int(r["frame_id"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    updates = [r for r in rows if r["memory_update"] is not None]
    per_dataset: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = per_dataset.setdefault(row["dataset"], {"update": 0, "hard_null": 0})
        bucket["update" if row["memory_update"] is not None else "hard_null"] += 1

    summary = {
        "schema_version": "cogtrack.state_update_teacher_labels.v1",
        "prompt_version": STATE_TEACHER_PROMPT_VERSION,
        "sampling_plan": str(plan_path),
        "output": str(output_path),
        "teacher_model": args.teacher_model,
        "verifier_model": args.verifier_model or None,
        # 未经独立裁决的标签不能当"高质量"用，这一位必须显式落进产物。
        "independently_verified": verified,
        "dry_run": args.dry_run,
        "datasets": list(datasets),
        "sequences_per_dataset": args.sequences_per_dataset,
        "stride": args.stride,
        "max_steps_per_sequence": args.max_steps_per_sequence,
        "temperature": args.temperature,
        "seeds": [args.seed, args.seed + 1],
        "max_image_side": args.max_image_side,
        "teacher_tensor_parallel_size": args.teacher_tensor_parallel_size,
        "verifier_tensor_parallel_size": args.verifier_tensor_parallel_size,
        "walked_sequences": len(walks),
        "sequences_without_identity": skipped_no_identity,
        "planned_decision_points": planned_decisions,
        "written_labels": len(rows),
        "written_update_labels": len(updates),
        "written_hard_null_labels": len(rows) - len(updates),
        "max_output_labels": args.max_output_labels,
        "min_output_labels": args.min_output_labels,
        "minimum_output_reached": len(rows) >= args.min_output_labels,
        "per_dataset": dict(sorted(per_dataset.items())),
        "unique_update_texts": len({r["memory_update"] for r in updates}),
        "update_yield": len(updates) / planned_decisions if planned_decisions else 0.0,
        "dual_pass_outcomes": dict(reasons.most_common()),
        "teacher_rejections": rejections.to_dict(),
        "verifier_rejections": verifier_rejections.to_dict(),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
    print(text, end="")
    if len(rows) < args.min_output_labels:
        print(
            "[错误] verifier 通过标签不足："
            f"written={len(rows)} < min_output_labels={args.min_output_labels}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
