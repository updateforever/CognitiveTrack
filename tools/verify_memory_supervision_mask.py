#!/usr/bin/env python3
"""用真实 Qwen3-VL processor + ms-swift 模板回放记忆监督三态，逐 token 断言 mask。

本工具不加载模型权重，只加载本地 processor/tokenizer，因此可以在数据生成之前先
锁定监督语义。它在内存里构造四条样本（共用同一张真实图片），分别断言：

* ``masked_unknown``：present 且记忆为占位 ``null``。**只有** ``null`` 这几个
  token 被置为 ``-100``；bbox 坐标、``status`` 值、``"memory_update":`` 键名、
  JSON 闭合 ``}`` 和 EOS 全部保持监督。
* ``verified_hard_null``：absent。``null`` **参与** loss，被 mask 的 token 数为 0。
* ``verified_update``：present 且有可靠状态文本。文本参与 loss，被 mask 的 token
  数为 0。

四条样本用同一个 template 实例、同一个 ``--loss_scale cogtrack_tracking_core``
连续编码，因此同时证明了三态可以共存于同一数据集与同一 batch：per-message
``loss_scale`` 命中 ms-swift ``LossScale._inner_call`` 的旁路分支，不需要修改
site-packages。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.protocol import (  # noqa: E402
    BBOX_PROTOCOL_NORM1000,
    MEMORY_UPDATE_JSON_KEY,
    TARGET_STATUS_JSON_KEY,
    bbox_protocol_json_key,
)
from cogtrack.training.loss_mask import (  # noqa: E402
    MEMORY_STATE_MASKED_UNKNOWN,
    MEMORY_STATE_VERIFIED_HARD_NULL,
    MEMORY_STATE_VERIFIED_UPDATE,
    assistant_loss_scale_for_state,
    decide_memory_supervision_state,
)

BBOX_KEY = bbox_protocol_json_key(BBOX_PROTOCOL_NORM1000)

SYSTEM_PROMPT = "You are a visual object tracker."
USER_PROMPT = "<image><image><image>\nLocate the target in the final frame."


def _answer(*, status: str, memory_update: str | None) -> str:
    """复刻导出层的紧凑 JSON，字段顺序 bbox -> status -> memory_update。"""

    bbox: Any = "<bbox>" if status == "present" else None
    parts = [
        f'"{BBOX_KEY}":' + ("<bbox>" if status == "present" else "null"),
        f'"{TARGET_STATUS_JSON_KEY}":{json.dumps(status)}',
        f'"{MEMORY_UPDATE_JSON_KEY}":{json.dumps(memory_update)}',
    ]
    del bbox
    return "{" + ",".join(parts) + "}"


def _case(
    *, status: str, memory_update: str | None, verified_null: bool = False
) -> dict[str, Any]:
    state = decide_memory_supervision_state(
        status=status, memory_update=memory_update, verified_null=verified_null
    )
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": _answer(status=status, memory_update=memory_update),
    }
    scale = assistant_loss_scale_for_state(state)
    if scale is not None:
        assistant["loss_scale"] = scale
    return {
        "state": state,
        "status": status,
        "memory_update": memory_update,
        "verified_null": verified_null,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            assistant,
        ],
    }


def _verify(
    *,
    model_path: Path,
    image_path: Path,
    max_pixels: int,
) -> dict[str, Any]:
    from swift.model import get_processor
    from swift.template import StdTemplateInputs, TemplateInputs, get_template
    from swift.utils import import_external_file

    # 复用 swift sft 的真实外部插件导入方式；不修改 ms-swift 安装目录。
    import_external_file(str(PROJECT_ROOT / "cogtrack" / "training" / "ms_swift_plugin.py"))

    processor = get_processor(
        str(model_path), model_type="qwen3_vl", download_model=False, use_fast=False
    )
    tokenizer = processor.tokenizer
    # 四条样本共用同一个 template 实例，证明三态在同一数据集/batch 内共存。
    template = get_template(
        processor, max_pixels=max_pixels, loss_scale="cogtrack_tracking_core"
    )
    template.set_mode("train")
    baseline = get_template(processor, max_pixels=max_pixels, loss_scale="default")
    baseline.set_mode("train")

    images = [str(image_path)] * 3
    real_box = [10.0, 20.0, 110.0, 140.0]

    def encode(tpl: Any, case: dict[str, Any]) -> dict[str, Any]:
        std = StdTemplateInputs(
            system=case["messages"][0]["content"],
            messages=json.loads(json.dumps(case["messages"][1:])),
            images=list(images),
        )
        if case["status"] == "present":
            # 只把 assistant 框绑定到最后一张 current 图。
            std.objects = {"bbox": [real_box], "bbox_type": "real", "image_id": [2]}
        return tpl.encode(TemplateInputs(chosen=std))

    cases = [
        _case(status="present", memory_update=None),
        _case(status="absent", memory_update=None),
        _case(status="present", memory_update="the skateboarder is now airborne"),
        # present + 已证明无需更新：null 必须参与 loss，masked_token_count 应为 0。
        # 这一条与第一条的 response 逐字节相同，唯一区别是 loss_scale，因此它直接
        # 证明"同一段文本能否被监督"完全由 loss_scale 决定。
        _case(status="present", memory_update=None, verified_null=True),
    ]
    expected_states = [
        MEMORY_STATE_MASKED_UNKNOWN,
        MEMORY_STATE_VERIFIED_HARD_NULL,
        MEMORY_STATE_VERIFIED_UPDATE,
        MEMORY_STATE_VERIFIED_HARD_NULL,
    ]

    reports = []
    for case, expected_state in zip(cases, expected_states, strict=True):
        if case["state"] != expected_state:
            raise AssertionError(f"状态判定错误：{case['state']} != {expected_state}")
        masked = encode(template, case)
        default = encode(baseline, case)
        if masked["input_ids"] != default["input_ids"]:
            raise AssertionError(f"{case['state']}：启用 loss mask 后 input_ids 发生变化")

        masked_indices = [
            index
            for index, (d, m) in enumerate(
                zip(default["labels"], masked["labels"], strict=True)
            )
            if d != -100 and m == -100
        ]
        masked_text = tokenizer.decode(
            [masked["input_ids"][i] for i in masked_indices], skip_special_tokens=False
        )
        supervised_text = tokenizer.decode(
            [
                token
                for token, label in zip(masked["input_ids"], masked["labels"], strict=True)
                if label != -100
            ],
            skip_special_tokens=False,
        )

        if expected_state == MEMORY_STATE_MASKED_UNKNOWN:
            if masked_text != "null":
                raise AssertionError(f"masked_unknown 的 mask 范围错误：{masked_text!r}")
        else:
            if masked_indices:
                raise AssertionError(
                    f"{expected_state} 必须全量监督，实际被 mask：{masked_text!r}"
                )

        # 三态共同的硬约束：bbox、status、JSON 闭合与 EOS 始终参与 loss。
        if case["status"] == "present":
            if f'"{BBOX_KEY}":[' not in supervised_text:
                raise AssertionError(f"{expected_state}：bbox 坐标 token 被错误屏蔽")
        elif f'"{BBOX_KEY}":null' not in supervised_text:
            raise AssertionError(f"{expected_state}：absent 的 bbox null 被错误屏蔽")
        status_text = f'"{TARGET_STATUS_JSON_KEY}":"{case["status"]}"'
        if status_text not in supervised_text:
            raise AssertionError(f"{expected_state}：status 值被错误屏蔽")
        if f'"{MEMORY_UPDATE_JSON_KEY}":' not in supervised_text:
            raise AssertionError(f"{expected_state}：memory_update 键名被错误屏蔽")
        if not supervised_text.rstrip().endswith("<|im_end|>"):
            raise AssertionError(f"{expected_state}：EOS 未参与 loss")
        if "}" not in supervised_text:
            raise AssertionError(f"{expected_state}：JSON 闭合符未参与 loss")
        # 记忆值本身：masked 行不得出现在监督文本里，其余两态必须出现。
        rendered_memory = json.dumps(case["memory_update"])
        memory_supervised = f'"{MEMORY_UPDATE_JSON_KEY}":{rendered_memory}' in supervised_text
        if expected_state == MEMORY_STATE_MASKED_UNKNOWN:
            if memory_supervised:
                raise AssertionError("masked_unknown 的记忆值仍在监督文本中")
        elif not memory_supervised:
            raise AssertionError(f"{expected_state} 的记忆值未参与 loss")

        reports.append(
            {
                "memory_supervision_state": expected_state,
                "status": case["status"],
                "assistant_loss_scale": case["messages"][2].get("loss_scale"),
                "masked_text": masked_text,
                "masked_token_count": len(masked_indices),
                "memory_value_supervised": memory_supervised,
                "bbox_supervised": True,
                "status_supervised": True,
                "memory_key_supervised": True,
                "json_close_and_eos_supervised": True,
                "ok": True,
            }
        )
    return {
        "schema_version": "cogtrack.memory_supervision_mask.v1",
        "model_path": str(model_path),
        "image": str(image_path),
        "max_pixels": max_pixels,
        "loss_scale_plugin": "cogtrack_tracking_core",
        "shared_template_instance": True,
        "reports": reports,
        "ok": all(report["ok"] for report in reports),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen3-model", required=True)
    parser.add_argument("--image", required=True, help="任意真实图片；只用于占位三图输入。")
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--output")
    args = parser.parse_args()

    os.environ["QWENVL_BBOX_FORMAT"] = "new"
    result = _verify(
        model_path=Path(args.qwen3_model).expanduser().resolve(),
        image_path=Path(args.image).expanduser().resolve(),
        max_pixels=args.max_pixels,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
