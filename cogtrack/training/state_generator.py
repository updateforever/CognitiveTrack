"""VLT-v6.3.1 状态描述生成器。

使用大模型（Qwen2.5-VL 等）为训练样本生成目标状态描述。
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


class StateGenerator:
    """目标状态描述生成器"""

    def __init__(
        self,
        *,
        api_base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "local-test-key",
        model_name: str = "Qwen2.5-VL-32B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        """
        Args:
            api_base_url: vLLM API 地址
            api_key: API 密钥
            model_name: 模型名称
            temperature: 采样温度
            max_tokens: 最大生成长度
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_initial_identity(
        self,
        init_frame_path: Path,
        bbox: tuple[float, float, float, float],
        language_query: str | None = None,
    ) -> str:
        """生成初始目标身份描述。

        Args:
            init_frame_path: 初始帧图像路径
            bbox: 目标边界框 (x, y, w, h)
            language_query: 数据集提供的文本描述（LaSOT/TNL2K/MGIT）

        Returns:
            初始身份描述，例如 "a male secret agent wearing a black suit"
        """
        # 如果数据集已提供描述，直接使用
        if language_query:
            return language_query.strip()

        # 否则用大模型生成
        prompt = self._build_initial_identity_prompt(bbox)
        image = Image.open(init_frame_path).convert("RGB")

        response = self._call_vlm(prompt, [image])
        return self._parse_initial_identity(response)

    def generate_state_update(
        self,
        initial_identity: str,
        previous_state: str,
        current_frame_path: Path,
        target_status: str,
        bbox: tuple[float, float, float, float] | None,
    ) -> str:
        """生成当前帧的目标状态描述。

        Args:
            initial_identity: 初始目标身份
            previous_state: 前一帧的目标状态
            current_frame_path: 当前帧图像路径
            target_status: 目标状态 "present" 或 "absent"
            bbox: 当前帧的边界框（absent 时为 None）

        Returns:
            更新后的状态描述
        """
        prompt = self._build_state_update_prompt(
            initial_identity=initial_identity,
            previous_state=previous_state,
            target_status=target_status,
            bbox=bbox,
        )

        image = Image.open(current_frame_path).convert("RGB")
        response = self._call_vlm(prompt, [image])
        return self._parse_state_update(response)

    def _build_initial_identity_prompt(self, bbox: tuple[float, float, float, float]) -> str:
        """构造初始身份生成的 prompt"""
        x, y, w, h = bbox
        return f"""Describe the target object in the bounding box [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}].
Provide a concise identity description focusing on:
- Object category (e.g., person, vehicle, animal)
- Key appearance attributes (e.g., color, clothing, distinctive features)

Format: "a [category] [appearance]"
Example: "a male athlete wearing red jersey"

Identity:"""

    def _build_state_update_prompt(
        self,
        initial_identity: str,
        previous_state: str,
        target_status: str,
        bbox: tuple[float, float, float, float] | None,
    ) -> str:
        """构造状态更新生成的 prompt（调研优化：简洁版，< 30 词）"""
        if target_status == "present" and bbox:
            status_desc = "The target is present."
        else:
            status_desc = "The target is absent."

        return f"""You are tracking: {initial_identity}

Previous state: {previous_state}

Current frame: {status_desc}

Generate a concise state update (<30 words):
- If present: describe action and position
- If absent: describe absence reason (e.g., "occluded", "out of view")
- If no significant change: keep previous state
- Forbidden: coordinates, frame numbers, background details, reasoning

State:"""

    def _call_vlm(self, prompt: str, images: list[Image.Image]) -> str:
        """调用 vLLM API 生成响应"""
        # 编码图像为 base64
        image_urls = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_urls.append(f"data:image/jpeg;base64,{img_base64}")

        # 构造 OpenAI-compatible 请求
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"VLM API 调用失败: {e}") from e

    def _parse_initial_identity(self, response: str) -> str:
        """解析初始身份描述"""
        # 简单清理
        text = response.strip()
        # 移除可能的前缀
        if text.lower().startswith("identity:"):
            text = text[9:].strip()
        return text

    def _parse_state_update(self, response: str) -> str:
        """解析状态更新描述"""
        # 简单清理
        text = response.strip()
        # 移除可能的前缀
        if text.lower().startswith("updated state:"):
            text = text[14:].strip()
        return text


__all__ = ["StateGenerator"]
