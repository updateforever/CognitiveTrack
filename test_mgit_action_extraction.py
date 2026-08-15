#!/usr/bin/env python3
"""验证 MGIT action 层文本提取"""

import json
from pathlib import Path

desc_dir = Path("/data2/DATASETS_PUBLIC/MGIT/attribute/description")

print("=" * 80)
print("MGIT Action Layer Text Extraction Test")
print("=" * 80)

# 测试前 10 个序列
for i, json_file in enumerate(sorted(desc_dir.glob("*.json"))[:10], 1):
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    # 官方方案：从 action 层读取第一个 action 的 description
    actions = data.get("action", {})
    if not actions:
        print(f"\n{i}. {json_file.stem}: [NO ACTION LAYER]")
        continue

    first_action = sorted(actions.values(), key=lambda x: x.get("start_frame", 0))[0]

    # 提取关键字段
    obj_class = first_action.get("object_class", "N/A")
    appearance = first_action.get("appearance", "N/A")
    desc = first_action.get("description", "N/A")
    start_frame = first_action.get("start_frame", 0)
    end_frame = first_action.get("end_frame", 0)

    print(f"\n{i}. {json_file.stem}")
    print(f"   Object Class: {obj_class}")
    print(f"   Appearance:   {appearance}")
    print(f"   Frame Range:  [{start_frame}, {end_frame}]")
    print(f"   Description:  {desc[:100]}..." if len(desc) > 100 else f"   Description:  {desc}")

print("\n" + "=" * 80)
