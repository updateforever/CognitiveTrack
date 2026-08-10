"""CognitiveTrack 内置的 SUTrack checkpoint 推理运行时。"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cogtrack.protocol import validate_xywh

from .network import build_sutrack
from .network.clip_text import convert_clip_text_weights_to_fp16
from .preprocessing import ImagePreprocessor, clip_box, hann2d, sample_target, transform_image_to_crop


class ConfigNode(dict[str, Any]):
    """递归属性访问配置；只用于网络构造，不包含全局可变单例。"""

    def __init__(self, values: Mapping[str, Any]):
        super().__init__({key: self._convert(value) for key, value in values.items()})

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


_MODEL_CACHE: dict[tuple[Any, ...], tuple[torch.nn.Module, ConfigNode]] = {}
_MODEL_CACHE_LOCK = threading.RLock()


def _expand_path(value: str | os.PathLike[str]) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in text:
        raise ValueError(f"路径包含未设置的环境变量: {value}")
    return Path(text)


def _resolve_file(
    value: str | os.PathLike[str],
    *,
    params: Mapping[str, Any],
    prefer_model_root: bool,
) -> Path:
    path = _expand_path(value)
    if path.is_absolute():
        candidates = [path]
    else:
        runtime = params.get("runtime") if isinstance(params.get("runtime"), Mapping) else {}
        config_path = params.get("_config_path")
        candidates = []
        if prefer_model_root and runtime.get("model_root"):
            candidates.append(Path(str(runtime["model_root"])) / path)
        if config_path:
            candidates.append(Path(str(config_path)).resolve().parent / path)
        if runtime.get("project_root"):
            candidates.append(Path(str(runtime["project_root"])) / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_file():
            return resolved
    checked = ", ".join(str(candidate.resolve(strict=False)) for candidate in candidates)
    raise FileNotFoundError(f"文件不存在: {value!s}；检查过: {checked}")


def _read_config(path: Path) -> ConfigNode:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"SUTrack model config 顶层必须是 mapping: {path}")
    cfg = ConfigNode(payload)
    for dotted in (
        "MODEL.ENCODER.TYPE",
        "MODEL.DECODER.TYPE",
        "DATA.SEARCH.SIZE",
        "DATA.TEMPLATE.SIZE",
        "TEST.SEARCH_SIZE",
        "TEST.TEMPLATE_SIZE",
    ):
        current: Any = cfg
        try:
            for part in dotted.split("."):
                current = current[part]
        except (KeyError, TypeError) as error:
            raise ValueError(f"SUTrack model config 缺少 {dotted}: {path}") from error
    return cfg


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 兼容
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and isinstance(payload.get("net"), Mapping):
        payload = payload["net"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"SUTrack checkpoint 必须是 state_dict 或含 net 的 mapping: {path}")
    state = {str(key): value for key, value in payload.items() if isinstance(value, torch.Tensor)}
    if not state:
        raise ValueError(f"SUTrack checkpoint 中没有 Tensor 权重: {path}")
    return state


def _build_cached_model(
    model_config: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ConfigNode]:
    config_stat = model_config.stat()
    checkpoint_stat = checkpoint.stat()
    # 同一路径被原地替换时也必须重建模型，不能错误复用旧权重。
    key = (
        str(model_config),
        config_stat.st_size,
        config_stat.st_mtime_ns,
        str(checkpoint),
        checkpoint_stat.st_size,
        checkpoint_stat.st_mtime_ns,
        str(device),
    )
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        cfg = _read_config(model_config)
        model = build_sutrack(cfg)
        if device.type == "cuda":
            # 官方 OpenAI CLIP 在 CUDA 上采用特定的 fp16/fp32 混合参数布局。
            # 必须在 load_state_dict 前转换模块，否则 fp16 checkpoint 会被
            # 静默转换成模型初始化时的 fp32，造成逐帧预测漂移。
            convert_clip_text_weights_to_fp16(model.text_encoder.clip)
        state = _checkpoint_state(checkpoint)
        # 内置运行时重建 CLIP 文本塔但不保留未参与跟踪的 CLIP 视觉塔。
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("text_encoder.clip.visual.")
            and key != "text_encoder.clip.logit_scale"
        }
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "SUTrack checkpoint 与模型配置不兼容："
                f"missing={incompatible.missing_keys[:12]} "
                f"unexpected={incompatible.unexpected_keys[:12]}"
            )
        model = model.to(device).eval()
        _MODEL_CACHE[key] = (model, cfg)
        return model, cfg


def clear_sutrack_model_cache() -> None:
    """显式释放进程级模型缓存，主要供测试和长驻服务维护使用。"""

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


class BuiltinSUTrackRuntime:
    """逐序列状态与进程级共享网络分离的 SUTrack 推理实现。"""

    def __init__(
        self,
        *,
        params: Mapping[str, Any],
        model_config: str,
        checkpoint: str,
        device: str = "auto",
        amp: bool = False,
        language_mode: str = "auto",
        multi_modal_vision: bool = True,
        multi_modal_language: bool = True,
        use_nlp_datasets: Sequence[str] = (),
    ) -> None:
        self.params = dict(params)
        self.device = self._resolve_device(device)
        self.use_amp = bool(amp) and self.device.type == "cuda"
        if language_mode not in {"auto", "zero", "pretokenized"}:
            raise ValueError("language_mode 只允许 auto、zero 或 pretokenized")
        self.language_mode = language_mode
        self.multi_modal_vision = bool(multi_modal_vision)
        self.multi_modal_language = bool(multi_modal_language)
        self.use_nlp_datasets = frozenset(name.strip().lower() for name in use_nlp_datasets)
        config_path = _resolve_file(model_config, params=params, prefer_model_root=False)
        checkpoint_path = _resolve_file(checkpoint, params=params, prefer_model_root=True)
        self.network, self.cfg = _build_cached_model(config_path, checkpoint_path, self.device)
        self.preprocessor = ImagePreprocessor(self.device)
        self.search_size = int(self.cfg.TEST.SEARCH_SIZE)
        self.template_size = int(self.cfg.TEST.TEMPLATE_SIZE)
        self.search_factor = float(self.cfg.TEST.SEARCH_FACTOR)
        self.template_factor = float(self.cfg.TEST.TEMPLATE_FACTOR)
        self.num_templates = int(self.cfg.TEST.NUM_TEMPLATES)
        self.output_window = hann2d(
            self.search_size // int(self.cfg.MODEL.ENCODER.STRIDE), self.device
        ) if bool(self.cfg.TEST.WINDOW) else None
        dataset = str(self.params.get("runtime", {}).get("dataset_name", "default")).upper()
        self.update_interval = int(self._dataset_setting(self.cfg.TEST.UPDATE_INTERVALS, dataset))
        self.update_threshold = float(self._dataset_setting(self.cfg.TEST.UPDATE_THRESHOLD, dataset))
        self.state: list[float] | None = None
        self.template_list: list[torch.Tensor] = []
        self.template_anno_list: list[torch.Tensor] = []
        self.text_src: torch.Tensor | None = None
        self.frame_id = 0

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前进程 torch.cuda.is_available() 为 False")
        return device

    @staticmethod
    def _dataset_setting(values: Mapping[str, Any], dataset: str) -> Any:
        mapped = "GOT10K" if "GOT10K" in dataset else "LASOT" if "LASOT" in dataset else dataset
        if "OTB" in mapped:
            mapped = "TNL2K"
        return values.get(mapped, values["DEFAULT"])

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """根据 MULTI_MODAL_VISION 决定是否复制到 6 通道。"""
        if image.ndim != 3 or image.shape[2] not in {3, 6}:
            raise ValueError(f"SUTrack 输入必须为 HxWx3/6 RGB array，收到 {image.shape}")
        if self.multi_modal_vision and image.shape[2] == 3:
            return np.concatenate((image, image), axis=2)
        return image

    def _template(self, image: np.ndarray, bbox: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
        patch, resize_factor = sample_target(
            self._prepare_image(image), bbox, self.template_factor, self.template_size
        )
        tensor = self.preprocessor.process(patch)
        annotation = transform_image_to_crop(
            bbox, bbox, resize_factor, self.template_size
        ).to(self.device).unsqueeze(0)
        return tensor, annotation

    def _should_use_nlp(self, info: Mapping[str, Any]) -> bool:
        """判断当前序列是否应该使用真实 NLP token。"""
        if not self.multi_modal_language:
            return False
        dataset = str(info.get("dataset_name", "")).strip().lower()
        return dataset in self.use_nlp_datasets

    def _text_tokens(self, info: Mapping[str, Any]) -> torch.Tensor:
        """根据数据集和 language_mode 决定输入 CLIP 的 token。

        与原版 ``extract_token_from_nlp_clip`` 对齐：``use_nlp`` 为假时输入全零
        token（注意仍然会过文本塔），为真时对 ``init_nlp`` 做 CLIP tokenize。
        """
        zeros = torch.zeros((1, 77), dtype=torch.long, device=self.device)
        if not self._should_use_nlp(info):
            return zeros
        if self.language_mode == "zero":
            return zeros

        tokens = info.get("init_nlp_tokens")
        if tokens is None and self.language_mode == "auto":
            text = info.get("init_nlp")
            if text is None:
                # 原版同样把缺失的 nlp 当成 None 处理，落回全零 token。
                return zeros
            from .clip_tokenizer import tokenize_text

            tokens = tokenize_text(str(text))
        if tokens is None:
            raise ValueError(
                f"数据集 {info.get('dataset_name')} 需要 NLP token，但 info 中缺少 init_nlp_tokens。"
                "请在 runner 中调用 CLIP tokenizer 并传入，改用 language_mode: auto，"
                "或将该数据集从 use_nlp_datasets 移除。"
            )
        tensor = torch.as_tensor(tokens, dtype=torch.long, device=self.device).reshape(1, -1)
        if tensor.shape[1] != 77:
            raise ValueError(f"init_nlp_tokens 必须有 77 个 token，收到 {tensor.shape[1]}")
        return tensor

    def initialize(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        bbox = list(validate_xywh(info["init_bbox"]))
        template, annotation = self._template(image, bbox)
        self.template_list = [template for _ in range(self.num_templates)]
        self.template_anno_list = [annotation for _ in range(self.num_templates)]
        self.state = bbox
        self.frame_id = 0
        if self.multi_modal_language:
            with torch.inference_mode(), self._autocast():
                self.text_src = self.network.forward_textencoder(self._text_tokens(info))
        else:
            self.text_src = None
        return {
            "target_bbox": bbox,
            "best_score": 1.0,
            "execution": {"status": "ok", "latency_ms": (time.perf_counter() - started) * 1000},
        }

    def track(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("BuiltinSUTrackRuntime 必须先 initialize()")
        started = time.perf_counter()
        image_height, image_width = image.shape[:2]
        self.frame_id += 1
        patch, resize_factor = sample_target(
            self._prepare_image(image), self.state, self.search_factor, self.search_size
        )
        search = self.preprocessor.process(patch)
        with torch.inference_mode(), self._autocast():
            encoded = self.network.forward_encoder(
                self.template_list,
                [search],
                self.template_anno_list,
                self.text_src,
                None,
            )
            output = self.network.forward_decoder(feature=encoded)
            response = output["score_map"]
            if self.output_window is not None:
                response = response * self.output_window
            predicted, score = self.network.decoder.cal_bbox(
                response,
                output["size_map"],
                output["offset_map"],
                return_score=True,
            )
        box = predicted.reshape(-1, 4).mean(dim=0)
        box = (box * self.search_size / resize_factor).detach().float().cpu().tolist()
        self.state = clip_box(
            self._map_box_back(box, resize_factor), image_height, image_width, margin=10
        )
        score_value = float(torch.as_tensor(score).detach().float().cpu().reshape(-1).mean().item())
        if (
            self.num_templates > 1
            and self.frame_id % self.update_interval == 0
            and score_value > self.update_threshold
        ):
            self._replace_online_template(image, self.state)
        return {
            "target_bbox": self.state,
            "best_score": score_value,
            "execution": {"status": "ok", "latency_ms": (time.perf_counter() - started) * 1000},
        }

    def _map_box_back(self, box_cxcywh: Sequence[float], resize_factor: float) -> list[float]:
        if self.state is None:
            raise RuntimeError("SUTrack state 尚未初始化")
        previous_cx = self.state[0] + 0.5 * self.state[2]
        previous_cy = self.state[1] + 0.5 * self.state[3]
        cx, cy, width, height = (float(value) for value in box_cxcywh)
        half_side = 0.5 * self.search_size / resize_factor
        real_cx = cx + previous_cx - half_side
        real_cy = cy + previous_cy - half_side
        return [real_cx - 0.5 * width, real_cy - 0.5 * height, width, height]

    def _replace_online_template(self, image: np.ndarray, bbox: Sequence[float]) -> None:
        template, annotation = self._template(image, bbox)
        index = 1 if self.num_templates > 1 else 0
        self.template_list[index] = template
        self.template_anno_list[index] = annotation

    def correct(
        self,
        image: np.ndarray,
        bbox_xywh: Sequence[float],
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """接受身份门控后的 VLM 全局框，并重置在线模板槽。"""

        del info
        self.state = list(validate_xywh(bbox_xywh))
        self._replace_online_template(image, self.state)
        return {"applied": True, "reason": "已重置 SUTrack 状态和在线模板"}

    def close(self) -> None:
        """释放序列状态；共享网络继续留在进程缓存中供下一序列使用。"""

        self.state = None
        self.template_list.clear()
        self.template_anno_list.clear()
        self.text_src = None


def build_sutrack_runtime(*, params: Mapping[str, Any], **kwargs: Any) -> BuiltinSUTrackRuntime:
    """供 ``SUTrackAdapter`` 延迟加载的内置工厂。"""

    return BuiltinSUTrackRuntime(params=params, **kwargs)


__all__ = [
    "BuiltinSUTrackRuntime",
    "build_sutrack_runtime",
    "clear_sutrack_model_cache",
]
