# MGIT 文本提取最优方案

> 发现：MGIT 有 action/activity/story 三层标注，每层有帧范围
> 最佳：使用 action 层的 object_class + appearance

---

## 一、MGIT 完整结构

```json
{
  "action": {
    "action_1": {
      "start_frame": 0,
      "end_frame": 1246,
      "object_class": "male secret agent",      // ✅ 目标类别
      "appearance": "black suit",               // ✅ 外观描述
      "action_1": "walk",                       // ❌ 动作（未来事件）
      "scene": "washroom",                      // ⚠️ 场景（可选）
      "description": "A male secret agent wearing a black suit walks..."
    },
    "action_2": {...},
    ...
  },
  "activity": {
    "activity_1": {
      "start_frame": 0,
      "end_frame": 2400,
      "description": "A male secret agent... walks... stands... fights..."
    },
    ...
  },
  "story": {
    "story_1": {
      "start_frame": 0,
      "end_frame": 9032,
      "description": "A male secret agent... (整段故事)"
    }
  }
}
```

---

## 二、最佳方案：Object Class + Appearance

### **为什么最好？**

1. ✅ **不含未来事件**：`object_class` 和 `appearance` 是静态属性
2. ✅ **与 LaSOT/TNL2K 一致**：都是描述目标本身，不是动作
3. ✅ **简洁且信息充足**："a male secret agent wearing a black suit"
4. ✅ **覆盖初始化帧**：action_1 通常从 frame 0 开始

### **提取逻辑**

```python
def extract_mgit_identity(json_data: dict, init_frame: int = 0) -> str | None:
    """
    从 MGIT JSON 提取安全的初始化身份描述
    
    策略：
    1. 优先：找到覆盖 init_frame 的 action，提取 object_class + appearance
    2. Fallback：使用 action_1 的 object_class + appearance
    3. 最后：从 story description 提取主语
    """
    
    # 策略 1 & 2: 从 action 层提取
    actions = json_data.get("action", {})
    if isinstance(actions, dict):
        # 找到覆盖 init_frame 的 action
        for action_key in sorted(actions.keys()):
            action = actions[action_key]
            if not isinstance(action, dict):
                continue
            
            start = action.get("start_frame", -1)
            end = action.get("end_frame", -1)
            
            # 检查是否覆盖初始化帧
            if start <= init_frame <= end:
                object_class = action.get("object_class", "").strip()
                appearance = action.get("appearance", "").strip()
                
                if object_class:
                    # 组合 object_class + appearance
                    if appearance and appearance.lower() not in ['nan', 'none', '']:
                        identity = f"{object_class} wearing {appearance}"
                    else:
                        identity = object_class
                    
                    # 添加冠词并小写
                    identity = add_article(identity)
                    return identity
    
    # 策略 3: Fallback 到 story 提取（原方案）
    story = json_data.get("story", {})
    if isinstance(story, dict):
        for key in ["story_1"] + sorted(k for k in story if k != "story_1"):
            item = story.get(key)
            if isinstance(item, dict) and item.get("description"):
                desc = str(item["description"]).strip()
                if desc:
                    return extract_subject_from_story(desc)
    
    return None


def add_article(text: str) -> str:
    """添加冠词 a/an"""
    text = text.strip()
    if not text:
        return text
    
    # 已有冠词
    if text.lower().startswith(('a ', 'an ', 'the ')):
        return text.lower()
    
    # 添加 a/an
    first_word = text.split()[0].lower()
    if first_word[0] in 'aeiou':
        return f"an {text.lower()}"
    else:
        return f"a {text.lower()}"
```

---

## 三、实际提取示例

### **示例 1: 001.json**

```python
action_1:
  object_class: "male secret agent"
  appearance: "black suit"

提取结果: "a male secret agent wearing black suit"
```

### **示例 2: 002.json**

检查实际文件：

```bash
python -c "
import json
with open('/data2/DATASETS_PUBLIC/MGIT/attribute/description/002.json') as f:
    data = json.load(f)
action_1 = data['action']['action_1']
print('Object class:', action_1.get('object_class'))
print('Appearance:', action_1.get('appearance'))
"
```

### **示例 3: 003.json**

检查实际文件：

```bash
python -c "
import json
with open('/data2/DATASETS_PUBLIC/MGIT/attribute/description/003.json') as f:
    data = json.load(f)
action_1 = data['action']['action_1']
print('Object class:', action_1.get('object_class'))
print('Appearance:', action_1.get('appearance'))
"
```

---

## 四、完整实现代码

```python
# pytracking/datasets/mgit.py

import re
from typing import Any

def _extract_safe_identity(self, json_data: dict, init_frame: int = 0) -> str | None:
    """
    从 MGIT JSON 提取安全的初始化身份描述
    
    优先级：
    1. 覆盖 init_frame 的 action 的 object_class + appearance
    2. action_1 的 object_class + appearance
    3. 从 story description 提取主语（fallback）
    """
    
    # 尝试从 action 层提取
    identity = self._extract_from_action_layer(json_data, init_frame)
    if identity:
        return identity
    
    # Fallback: 从 story 提取主语
    identity = self._extract_from_story_layer(json_data)
    if identity:
        return identity
    
    return None


def _extract_from_action_layer(self, json_data: dict, init_frame: int) -> str | None:
    """从 action 层提取 object_class + appearance"""
    
    actions = json_data.get("action", {})
    if not isinstance(actions, dict):
        return None
    
    # 找到覆盖 init_frame 的 action
    target_action = None
    
    for action_key in sorted(actions.keys()):
        action = actions[action_key]
        if not isinstance(action, dict):
            continue
        
        start = action.get("start_frame", -1)
        end = action.get("end_frame", -1)
        
        if start <= init_frame <= end:
            target_action = action
            break
    
    # Fallback: 使用 action_1
    if target_action is None and "action_1" in actions:
        target_action = actions["action_1"]
    
    if target_action is None:
        return None
    
    # 提取 object_class + appearance
    object_class = str(target_action.get("object_class", "")).strip()
    appearance = str(target_action.get("appearance", "")).strip()
    
    if not object_class or object_class.lower() in ['nan', 'none']:
        return None
    
    # 组合
    if appearance and appearance.lower() not in ['nan', 'none', '']:
        # 处理 appearance 格式
        if 'wearing' in appearance.lower() or 'with' in appearance.lower():
            identity = f"{object_class} {appearance}"
        else:
            identity = f"{object_class} wearing {appearance}"
    else:
        identity = object_class
    
    # 添加冠词并小写
    return self._add_article(identity.lower())


def _add_article(self, text: str) -> str:
    """添加冠词 a/an"""
    text = text.strip()
    if not text:
        return text
    
    # 已有冠词
    if text.lower().startswith(('a ', 'an ', 'the ')):
        return text.lower()
    
    # 添加 a/an
    first_word = text.split()[0].lower()
    if first_word and first_word[0] in 'aeiou':
        return f"an {text}"
    else:
        return f"a {text}"


def _extract_from_story_layer(self, json_data: dict) -> str | None:
    """从 story 层提取主语（fallback）"""
    
    story = json_data.get("story", {})
    if not isinstance(story, dict):
        return None
    
    for key in ["story_1"] + sorted(k for k in story if k != "story_1"):
        item = story.get(key)
        if isinstance(item, dict) and item.get("description"):
            desc = str(item["description"]).strip()
            if desc:
                # 提取第一句的主语部分
                subject = self._extract_subject_from_story(desc)
                if subject:
                    return subject
    
    return None


def _extract_subject_from_story(self, story_text: str) -> str | None:
    """从 story 文本提取主语（去掉动作）"""
    
    first_sentence = story_text.split('.')[0].strip()
    
    # 匹配模式：A/An/The + 名词短语 + 动词
    match = re.match(
        r'^(An?\s+[^.]+?)\s+(walks|waits|plays|is|are|stands|runs|goes|comes|wakes|slides|crouches|fights|lifts|talks)',
        first_sentence,
        re.IGNORECASE
    )
    
    if match:
        subject = match.group(1).strip().rstrip(',')
        return subject[0].lower() + subject[1:] if subject else None
    
    # Fallback: 取前 8 个词
    words = first_sentence.split()[:8]
    result = ' '.join(words)
    return result[0].lower() + result[1:] if result else None


def _load_description(self, name: str) -> str | None:
    """加载并提取安全的初始化描述"""
    path = self.base_path / "attribute" / "description" / f"{name}.json"
    if not path.is_file():
        return None
    
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"MGIT 描述文件损坏: {path}") from exc
    
    # 提取安全的初始化描述（init_frame=0）
    return self._extract_safe_identity(payload, init_frame=0)
```

---

## 五、验证脚本

```python
#!/usr/bin/env python3
"""验证 MGIT 文本提取"""

import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from pytracking.evaluation.environment import load_environment

env = load_environment()
mgit_root = env.dataset_root('mgit')
desc_dir = mgit_root / 'attribute' / 'description'

# 测试前 10 个序列
json_files = sorted(desc_dir.glob('*.json'))[:10]

print('=== MGIT 文本提取验证 ===\n')

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
    
    # 提取 object_class + appearance
    action_1 = data.get('action', {}).get('action_1', {})
    object_class = action_1.get('object_class', '')
    appearance = action_1.get('appearance', '')
    
    # 组合
    if object_class:
        if appearance and str(appearance).lower() not in ['nan', 'none', '']:
            identity = f"a {object_class} wearing {appearance}"
        else:
            identity = f"a {object_class}"
    else:
        identity = "(empty)"
    
    # 对比 story
    story_desc = data.get('story', {}).get('story_1', {}).get('description', '')
    story_first = story_desc.split('.')[0] if story_desc else ''
    
    print(f'{json_file.stem}:')
    print(f'  提取: {identity}')
    print(f'  Story 首句: {story_first[:60]}...')
    print()
```

运行：
```bash
python scripts/verify_mgit_text_extraction.py
```

---

## 六、预期输出示例

```
001:
  提取: a male secret agent wearing black suit
  Story 首句: A male secret agent wearing a black suit walks in the...

002:
  提取: a gardield (或 cat)
  Story 首句: A day in the life of Garfield the cat...

003:
  提取: a brown fur dog
  Story 首句: A brown fur dog waits his owner in the station...
```

---

## 七、优势对比

| 方案 | 示例 | 问题 | 推荐 |
|------|------|------|------|
| **Object + Appearance** | "a male secret agent wearing black suit" | ✅ 无未来事件 | ✅✅✅ |
| Story 提取主语 | "a male secret agent wearing a black suit" | ⚠️ 需要正则提取 | ✅ |
| Story 完整 | "A male... walks... stands... fights..." | ❌ 包含未来动作 | ❌ |
| Activity | "A male... walks... stands... fights..." | ❌ 包含多个动作 | ❌ |

---

## 八、下一步

1. ✅ 实现上述代码到 `pytracking/datasets/mgit.py`
2. ✅ 运行验证脚本检查提取结果
3. ✅ 确认所有 MGIT 序列都能提取到合理的文本
4. ✅ 集成到数据生成流程

**需要我现在生成完整的代码修改吗？**
