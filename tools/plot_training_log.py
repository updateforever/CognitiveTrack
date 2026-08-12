#!/usr/bin/env python3
"""Render ms-swift logging.jsonl training curves to a PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="CognitiveTrack training")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.input).open(encoding="utf-8")
        if line.strip()
    ]
    records = [record for record in records if "loss" in record and "global_step/max_steps" in record]
    if not records:
        raise ValueError("logging.jsonl does not contain step-level loss records")

    steps = [int(record["global_step/max_steps"].split("/", 1)[0]) for record in records]
    fields = [
        ("loss", "Loss"),
        ("token_acc", "Token accuracy"),
        ("learning_rate", "Learning rate"),
        ("grad_norm", "Gradient norm"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, fields, strict=True):
        values = [record.get(field) for record in records]
        valid = [(step, value) for step, value in zip(steps, values, strict=True) if value is not None]
        axis.plot([item[0] for item in valid], [item[1] for item in valid], linewidth=1.2)
        axis.set_title(label)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    figure.suptitle(args.title)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
