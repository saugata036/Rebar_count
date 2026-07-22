from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image

GEMINI_MODEL = "gemini-3.5-flash"
MASK_GENERATION_PROMPT = """
Analyze the provided construction rebar column image carefully.
Your task is to identify and isolate ALL front-facing vertical and horizontal rebar elements that form the outermost front grid layer closest to the camera (including all main vertical bars, horizontal ties, intersection wire ties, and corner concrete spacer blocks on this front plane).

CRITICAL INSTRUCTION: Do NOT assume a fixed count of bars. Count and extract ALL visible front-layer vertical bars (e.g., 5, 8, 9, or more) and ALL front-layer horizontal ties present in the image.
IMPORTANT FOR EDGE BARS: Ensure you capture the extreme outermost left-side and right-side vertical bars completely from top to bottom. Do NOT omit or truncate the first vertical bar on the far left.

Generate a valid JSON object containing:
1. "vertical_bars": An array of centerlines representing ALL vertical bars on the outermost front plane. Each bar is represented by its start (x1, y1) and end (x2, y2) point center coordinates.
2. "horizontal_bars": An array of centerlines representing ALL horizontal ties on the outermost front plane. Each tie is represented by its start (x1, y1) and end (x2, y2) point center coordinates.
3. "spacer_blocks": An array of bounding boxes tracing any corner concrete spacer blocks on this front plane.

Do not include any background elements: erase or omit deep interior bars inside the cage, secondary rear layers, ground, blue pipes, scaffolding, and sky.
Exclude all diagonal cross-ties and depth-defining side columns that recede into the distance.

Return your response strictly in this JSON format:
{
  "vertical_bars": [
    {"x1": int, "y1": int, "x2": int, "y2": int}
  ],
  "horizontal_bars": [
    {"x1": int, "y1": int, "x2": int, "y2": int}
  ],
  "spacer_blocks": [
    {"ymin": int, "xmin": int, "ymax": int, "xmax": int}
  ]
}
Note: Scale all coordinate values on a normalized 0 to 1000 system relative to the image bounds. Return ONLY the raw JSON block.
"""

REBAR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "vertical_bars": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "x1": {"type": "INTEGER"},
                    "y1": {"type": "INTEGER"},
                    "x2": {"type": "INTEGER"},
                    "y2": {"type": "INTEGER"},
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        },
        "horizontal_bars": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "x1": {"type": "INTEGER"},
                    "y1": {"type": "INTEGER"},
                    "x2": {"type": "INTEGER"},
                    "y2": {"type": "INTEGER"},
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        },
        "spacer_blocks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "ymin": {"type": "INTEGER"},
                    "xmin": {"type": "INTEGER"},
                    "ymax": {"type": "INTEGER"},
                    "xmax": {"type": "INTEGER"},
                },
                "required": ["ymin", "xmin", "ymax", "xmax"],
            },
        },
    },
    "required": ["vertical_bars", "horizontal_bars", "spacer_blocks"],
}


def _safe_extract_box(box_item) -> tuple[int, int, int, int]:
    if isinstance(box_item, (list, tuple)) and len(box_item) >= 4:
        return int(box_item[0]), int(box_item[1]), int(box_item[2]), int(box_item[3])
    if isinstance(box_item, dict):
        clean = {str(k).lower().replace("_", "").replace("-", ""): v for k, v in box_item.items()}
        ymin = clean.get("ymin", clean.get("y1", clean.get("top", 0)))
        xmin = clean.get("xmin", clean.get("x1", clean.get("left", 0)))
        ymax = clean.get("ymax", clean.get("y2", clean.get("bottom", 1000)))
        xmax = clean.get("xmax", clean.get("x2", clean.get("right", 1000)))
        return int(ymin), int(xmin), int(ymax), int(xmax)
    return 0, 0, 1000, 1000


def _safe_extract_line(line_item) -> tuple[int, int, int, int]:
    if isinstance(line_item, (list, tuple)) and len(line_item) >= 4:
        return int(line_item[0]), int(line_item[1]), int(line_item[2]), int(line_item[3])
    if isinstance(line_item, dict):
        clean = {str(k).lower().replace("_", "").replace("-", ""): v for k, v in line_item.items()}
        x1 = clean.get("x1", clean.get("startx", 0))
        y1 = clean.get("y1", clean.get("starty", 0))
        x2 = clean.get("x2", clean.get("endx", 1000))
        y2 = clean.get("y2", clean.get("endy", 1000))
        return int(x1), int(y1), int(x2), int(y2)
    return 0, 0, 1000, 1000


def _draw_perspective_line(mask: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], max_dim: int, val: int, bottom_ratio: float, top_ratio: float) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    h_img = mask.shape[0]
    length = np.hypot(x2 - x1, y2 - y1)
    if length < 1e-5:
        return
    num_steps = int(max(15, min(150, length / 8.0)))
    for i in range(num_steps):
        t = i / float(num_steps)
        t_next = (i + 1) / float(num_steps)
        curr_x = int(x1 + t * (x2 - x1))
        curr_y = int(y1 + t * (y2 - y1))
        next_x = int(x1 + t_next * (x2 - x1))
        next_y = int(y1 + t_next * (y2 - y1))
        y_norm = curr_y / float(h_img)
        y_norm = max(0.0, min(1.0, y_norm))
        ratio = top_ratio + (bottom_ratio - top_ratio) * y_norm
        thickness = max(2, int(max_dim * ratio))
        cv2.line(mask, (curr_x, curr_y), (next_x, next_y), val, thickness)


def clean_rebar_structure(image_path: Path, output_path: Path) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env")

    client = genai.Client(api_key=api_key)
    pil_img = Image.open(image_path).convert("RGB")

    config = types.GenerateContentConfig.model_validate(
        {
            "response_mime_type": "application/json",
            "response_schema": REBAR_SCHEMA,
            "temperature": 0.1,
        }
    )

    text_data = ""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[MASK_GENERATION_PROMPT, pil_img],
        config=config,
    )
    text_data = response.text or ""
    json_match = re.search(r"\{[\s\S]*\}", text_data)
    mask_data = json.loads(json_match.group(0) if json_match else text_data.strip())

    src_img = cv2.imread(str(image_path))
    if src_img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w, _ = src_img.shape
    max_dim = max(h, w)
    gc_mask = np.zeros((h, w), dtype=np.uint8)

    # Extend vertical lines that span most of the frame towards top/bottom margins
    # so edge bars aren't truncated by the model's detected endpoints.
    processed_verticals = []
    for bar in mask_data.get("vertical_bars", []):
        x1, y1, x2, y2 = _safe_extract_line(bar)
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if abs(y2 - y1) > 400:
            y1 = max(0, y1 - 40)
            y2 = min(1000, y2 + 40)
        processed_verticals.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    all_target_lines = processed_verticals + mask_data.get("horizontal_bars", [])

    # Step A: Wide corridors (Probable Background)
    for bar in all_target_lines:
        x1, y1, x2, y2 = _safe_extract_line(bar)
        pt1 = (int(x1 / 1000.0 * w), int(y1 / 1000.0 * h))
        pt2 = (int(x2 / 1000.0 * w), int(y2 / 1000.0 * h))
        is_edge_bar = x1 < 120 or x2 < 120 or x1 > 880 or x2 > 880
        bg_ratio = 0.065 if is_edge_bar else 0.048
        _draw_perspective_line(gc_mask, pt1, pt2, max_dim, cv2.GC_PR_BGD, bg_ratio, 0.018)

    # Step B: Rebar cylinder bodies (Probable Foreground)
    for bar in all_target_lines:
        x1, y1, x2, y2 = _safe_extract_line(bar)
        pt1 = (int(x1 / 1000.0 * w), int(y1 / 1000.0 * h))
        pt2 = (int(x2 / 1000.0 * w), int(y2 / 1000.0 * h))
        is_edge_bar = x1 < 120 or x2 < 120 or x1 > 880 or x2 > 880
        fg_ratio = 0.038 if is_edge_bar else 0.028
        _draw_perspective_line(gc_mask, pt1, pt2, max_dim, cv2.GC_PR_FGD, fg_ratio, 0.012)

    # Step C: Core seeds (Definite Foreground)
    for bar in all_target_lines:
        x1, y1, x2, y2 = _safe_extract_line(bar)
        pt1 = (int(x1 / 1000.0 * w), int(y1 / 1000.0 * h))
        pt2 = (int(x2 / 1000.0 * w), int(y2 / 1000.0 * h))
        _draw_perspective_line(gc_mask, pt1, pt2, max_dim, cv2.GC_FGD, 0.016, 0.006)

    for block in mask_data.get("spacer_blocks", []):
        ymin, xmin, ymax, xmax = _safe_extract_box(block)
        p_ymin = int((ymin / 1000.0) * h)
        p_xmin = int((xmin / 1000.0) * w)
        p_ymax = int((ymax / 1000.0) * h)
        p_xmax = int((xmax / 1000.0) * w)
        cv2.rectangle(gc_mask, (p_xmin, p_ymin), (p_xmax, p_ymax), cv2.GC_FGD, -1)

    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    sky_pixels = gray > 225
    shadow_pixels = gray < 25
    prune_mask = sky_pixels | shadow_pixels
    gc_mask[(prune_mask) & (gc_mask != cv2.GC_FGD)] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(src_img, gc_mask, None, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask)
    min_area = int(h * w * 0.0005)
    cleaned_mask = np.zeros_like(final_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255

    cleaned_mask = cv2.GaussianBlur(cleaned_mask, (3, 3), 0)
    studio_canvas = np.full_like(src_img, 235)
    mask_3ch = cv2.merge([cleaned_mask, cleaned_mask, cleaned_mask])
    np.copyto(studio_canvas, src_img, where=(mask_3ch > 127))
    cv2.imwrite(str(output_path), studio_canvas)
