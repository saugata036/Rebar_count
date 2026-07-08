"""Draw counted bar lines on the front-face image."""

from __future__ import annotations

import cv2
import numpy as np

from rebar_xray.config import RenderConfig


def _text_size(text: str, font_scale: float, thickness: int) -> tuple[int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return width, height, baseline


def _draw_label(
    image: np.ndarray,
    text: str,
    center_x: int,
    center_y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    width, height, baseline = _text_size(text, font_scale, thickness)
    x = int(center_x - width / 2)
    y = int(center_y + height / 2)
    pad = max(4, thickness)
    cv2.rectangle(
        image,
        (x - pad, y - height - pad),
        (x + width + pad, y + baseline + pad),
        (255, 255, 255),
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (x - pad, y - height - pad),
        (x + width + pad, y + baseline + pad),
        color,
        max(1, thickness // 2),
    )
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)



def draw_counts(
    image_bgr: np.ndarray,
    vertical_positions: list[int],
    horizontal_positions: list[int],
    region: tuple[int, int, int, int],
    config: RenderConfig | None = None,
) -> np.ndarray:
    cfg = config or RenderConfig()
    output = image_bgr.copy()
    y0, y1, x0, x1 = region
    thickness = max(cfg.line_thickness, min(output.shape[0], output.shape[1]) // 200)
    font_scale = max(0.9, min(2.0, output.shape[0] / 1000.0))
    text_thickness = max(2, thickness // 2)

    vertical_sorted = sorted(vertical_positions)
    horizontal_sorted = sorted(horizontal_positions)

    for x in vertical_sorted:
        cv2.line(output, (x, y0), (x, y1), cfg.vertical_color, thickness, cv2.LINE_AA)
    for y in horizontal_sorted:
        cv2.line(output, (x0, y), (x1, y), cfg.horizontal_color, thickness, cv2.LINE_AA)

    # Vertical numbers: in the gap between bars, near the top (not on the bar line).
    label_y = y0 + int((y1 - y0) * 0.07)
    for index, x in enumerate(vertical_sorted):
        if index == 0:
            center_x = (x0 + x) // 2
        else:
            center_x = (vertical_sorted[index - 1] + x) // 2
        _draw_label(
            output,
            str(index + 1),
            center_x,
            label_y,
            cfg.label_vertical_color,
            font_scale,
            text_thickness,
        )

    # Horizontal numbers: left side of cage, vertically between bars (not on the bar line).
    label_x = x0 + int((x1 - x0) * 0.03)
    for index, y in enumerate(horizontal_sorted):
        if index == 0:
            center_y = (y0 + y) // 2
        else:
            center_y = (horizontal_sorted[index - 1] + y) // 2
        _draw_label(
            output,
            str(index + 1),
            label_x,
            center_y,
            cfg.label_horizontal_color,
            font_scale,
            text_thickness,
        )

    summary = f"{len(vertical_sorted)}V / {len(horizontal_sorted)}H"
    banner_h = max(48, int(52 * font_scale))
    cv2.rectangle(output, (0, 0), (output.shape[1], banner_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        summary,
        (12, int(34 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    return output
