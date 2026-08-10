#!/usr/bin/env python3
"""比较原版与 CognitiveTrack 的逐帧 bbox。"""

import json
import sys
from pathlib import Path

A = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/parity/out_original.json")
B = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/parity/out_ours.json")


def iou(p, q):
    ax1, ay1, aw, ah = p
    bx1, by1, bw, bh = q
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    a = json.loads(A.read_text(encoding="utf-8"))
    b = json.loads(B.read_text(encoding="utf-8"))

    print("=== 开关对齐 ===")
    for key in ("multi_modal_vision", "multi_modal_language", "update_intervals", "update_threshold"):
        va, vb = a.get(key), b.get(key)
        flag = "OK " if va == vb else "!!!"
        print(f"  {flag} {key}: original={va} ours={vb}")
    print(f"  --  torch: original={a.get('torch')} ours={b.get('torch')}")
    print(f"  --  original use_nlp={a.get('use_nlp')}")

    ra, rb = a["records"], b["records"]
    if len(ra) != len(rb):
        print(f"!!! 帧数不同: {len(ra)} vs {len(rb)}")
        return 1

    max_abs = 0.0
    min_iou = 1.0
    worst_frame = -1
    rows = []
    for x, y in zip(ra, rb, strict=True):
        d = max(abs(u - v) for u, v in zip(x["bbox"], y["bbox"], strict=True))
        j = iou(x["bbox"], y["bbox"])
        if d > max_abs:
            max_abs, worst_frame = d, x["frame"]
        min_iou = min(min_iou, j)
        rows.append((x["frame"], d, j, x["bbox"], y["bbox"]))

    print(f"\n=== 逐帧对比 ({len(ra)} 帧) ===")
    print(f"  最大坐标绝对差: {max_abs:.6f} px  (frame {worst_frame})")
    print(f"  最小 IoU:       {min_iou:.6f}")

    print("\n  前 5 帧:")
    for frame, d, j, ba, bb in rows[:5]:
        print(f"    f{frame:<4d} dmax={d:.6f} iou={j:.6f}")
        print(f"          orig={[round(v, 4) for v in ba]}")
        print(f"          ours={[round(v, 4) for v in bb]}")

    drift = [r for r in rows if r[1] > 0.5]
    if drift:
        print(f"\n  坐标差 > 0.5px 的帧: {len(drift)} / {len(rows)}，最早 frame {drift[0][0]}")
        for frame, d, j, ba, bb in drift[:5]:
            print(f"    f{frame:<4d} dmax={d:.6f} iou={j:.6f}")
            print(f"          orig={[round(v, 4) for v in ba]}")
            print(f"          ours={[round(v, 4) for v in bb]}")

    print()
    if max_abs < 1e-3:
        print("结论: 数值完全一致（< 1e-3 px）")
    elif min_iou > 0.99:
        print("结论: 存在浮点级差异，但轨迹一致（IoU > 0.99）")
    elif min_iou > 0.9:
        print("结论: 轨迹基本一致但有可见漂移，需要排查")
    else:
        print("结论: 轨迹显著分叉，存在移植错误")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
