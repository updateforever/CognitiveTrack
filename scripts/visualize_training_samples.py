#!/usr/bin/env python3
"""可视化 VLT-v6.3.1 训练样本。

随机抽取样本，渲染三图 + 标注，检查数据质量。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bbox_on_image(image: np.ndarray, bbox: tuple[float, ...], color: tuple[int, int, int], thickness: int = 3):
    """在图像上绘制边界框"""
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def add_text_to_image(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]):
    """在图像上添加文本"""
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def visualize_sample(annotation_path: Path, output_path: Path):
    """可视化单个训练样本"""
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)

    # 加载图像
    init_img = cv2.imread(data["init_frame"]["frame_path"])
    current_img = cv2.imread(data["current_frame"]["frame_path"])

    if init_img is None or current_img is None:
        print(f"⚠️  Failed to load images for {annotation_path.name}")
        return

    # 1. 初始帧 + 红框
    init_vis = init_img.copy()
    draw_bbox_on_image(init_vis, data["init_frame"]["bbox"], (0, 0, 255), thickness=3)
    add_text_to_image(init_vis, "Image 1: Init", (10, 30), (0, 0, 255))

    # 2. 历史帧 mosaic
    history_images = []
    for i, hist in enumerate(data["history_frames"]):
        hist_img = cv2.imread(hist["frame_path"])
        if hist_img is None:
            continue
        # 绘制 bbox
        if hist["visible"]:
            draw_bbox_on_image(hist_img, hist["bbox"], (0, 255, 0), thickness=2)
        else:
            # absent 帧用红色叉号标记
            h, w = hist_img.shape[:2]
            cv2.line(hist_img, (0, 0), (w, h), (0, 0, 255), 3)
            cv2.line(hist_img, (w, 0), (0, h), (0, 0, 255), 3)
        add_text_to_image(hist_img, f"H{i+1}", (10, 30), (255, 255, 255))
        history_images.append(hist_img)

    # 拼接历史帧（水平）
    if history_images:
        # 调整高度一致
        target_height = history_images[0].shape[0]
        resized = []
        for img in history_images:
            h, w = img.shape[:2]
            new_w = int(w * target_height / h)
            resized.append(cv2.resize(img, (new_w, target_height)))
        history_mosaic = np.hstack(resized)
    else:
        history_mosaic = np.zeros_like(init_img)

    add_text_to_image(history_mosaic, "Image 2: History", (10, 60), (0, 255, 0))

    # 3. 当前帧
    current_vis = current_img.copy()
    target_status = data["target_status"]
    if target_status == "present":
        draw_bbox_on_image(current_vis, data["current_frame"]["bbox"], (0, 255, 0), thickness=3)
        status_text = "Present"
        status_color = (0, 255, 0)
    else:
        # absent 用红色叉号
        h, w = current_vis.shape[:2]
        cv2.line(current_vis, (0, 0), (w, h), (0, 0, 255), 5)
        cv2.line(current_vis, (w, 0), (0, h), (0, 0, 255), 5)
        status_text = "Absent"
        status_color = (0, 0, 255)

    add_text_to_image(current_vis, f"Image 3: Current ({status_text})", (10, 30), status_color)

    # 调整三张图到相同高度
    target_h = 400
    init_vis = cv2.resize(init_vis, (int(init_vis.shape[1] * target_h / init_vis.shape[0]), target_h))
    history_mosaic = cv2.resize(
        history_mosaic, (int(history_mosaic.shape[1] * target_h / history_mosaic.shape[0]), target_h)
    )
    current_vis = cv2.resize(current_vis, (int(current_vis.shape[1] * target_h / current_vis.shape[0]), target_h))

    # 水平拼接三图
    combined = np.hstack([init_vis, history_mosaic, current_vis])

    # 添加文本标注信息
    text_panel_height = 200
    text_panel = np.ones((text_panel_height, combined.shape[1], 3), dtype=np.uint8) * 255

    # 使用 PIL 绘制多行文本
    text_pil = Image.fromarray(text_panel)
    draw = ImageDraw.Draw(text_pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    text_lines = [
        f"Sample ID: {data['sample_id']} | Dataset: {data['dataset_name']} | Sequence: {data['sequence_name']}",
        f"Sample Type: {data['sample_type']}",
        f"",
        f"Initial Identity: {data['initial_identity']}",
        f"Previous State: {data['previous_state']}",
        f"Target Status: {data['target_status']}",
        f"Memory Update: {data['memory_update']}",
    ]

    y_offset = 10
    for line in text_lines:
        # 截断过长的文本
        if len(line) > 120:
            line = line[:117] + "..."
        draw.text((10, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += 25

    text_panel = np.array(text_pil)

    # 垂直拼接
    final = np.vstack([combined, text_panel])

    # 保存
    cv2.imwrite(str(output_path), final)
    print(f"✅ Saved: {output_path.name}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize VLT-v6.3.1 training samples")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data2/wyp/VLMTrack/CognitiveTrack/data/vlt_v631_core_sft",
        help="Training data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data2/wyp/VLMTrack/CognitiveTrack/data/vlt_v631_core_sft_vis",
        help="Output directory for visualizations",
    )
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有标注文件
    annotation_files = sorted(data_dir.glob("sample_*.json"))
    if not annotation_files:
        print(f"❌ No annotation files found in {data_dir}")
        return

    print(f"Found {len(annotation_files)} annotation files")
    print(f"Visualizing {args.num_samples} random samples...")
    print()

    # 随机采样
    random.seed(args.seed)
    sampled_files = random.sample(annotation_files, min(args.num_samples, len(annotation_files)))

    # 可视化
    for anno_file in sampled_files:
        output_path = output_dir / f"{anno_file.stem}_vis.jpg"
        try:
            visualize_sample(anno_file, output_path)
        except Exception as e:
            print(f"❌ Error visualizing {anno_file.name}: {e}")

    print()
    print(f"✅ Visualization complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
