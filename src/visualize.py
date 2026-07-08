"""Draw numbered arrows on detected vertical and horizontal rods."""

from __future__ import annotations

import cv2
import numpy as np


def _scale_from_image(height: int) -> float:
    return max(0.5, min(1.2, height / 3500.0))


def annotate_rods(
    column_bgr: np.ndarray,
    vertical_positions: list[int],
    horizontal_positions: list[int],
    column_region: dict[str, int],
) -> np.ndarray:
    """
    Draw numbered arrows on each vertical and horizontal rod inside the column view.

    Vertical rods: red arrows pointing to each rod (numbered 1, 2, 3 ... left to right).
    Horizontal rods: blue arrows pointing to each rod (numbered 1, 2, 3 ... top to bottom).
    """
    output = column_bgr.copy()
    y0 = column_region["top"]
    y1 = column_region["bottom"]
    x0 = column_region["left"]
    x1 = column_region["right"]
    col_h = max(1, y1 - y0)
    col_w = max(1, x1 - x0)
    scale = _scale_from_image(output.shape[0])
    thickness = max(2, int(3 * scale))
    font_scale = max(0.7, 1.0 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX

    vertical_sorted = sorted(vertical_positions)
    horizontal_sorted = sorted(horizontal_positions)

    for index, x in enumerate(vertical_sorted, start=1):
        x = int(np.clip(x, x0, x1 - 1))
        cv2.line(output, (x, y0), (x, y1 - 1), (0, 0, 255), thickness)

        label_y = y0 + int(col_h * 0.08) + (index - 1) * max(28, col_h // 25)
        label_y = min(label_y, y1 - 20)
        arrow_start = (x0 + max(12, col_w // 30), label_y)
        arrow_end = (x - max(18, col_w // 40), label_y)
        if arrow_end[0] <= arrow_start[0]:
            arrow_end = (x + max(18, col_w // 40), label_y)
            arrow_start = (x0 + max(12, col_w // 30), label_y)

        cv2.arrowedLine(output, arrow_start, arrow_end, (0, 0, 255), thickness, tipLength=0.2)
        cv2.putText(
            output,
            str(index),
            (arrow_end[0] + 6, arrow_end[1] + 6),
            font,
            font_scale,
            (0, 0, 255),
            max(2, thickness),
            cv2.LINE_AA,
        )

    for index, y in enumerate(horizontal_sorted, start=1):
        y = int(np.clip(y, y0, y1 - 1))
        cv2.line(output, (x0, y), (x1 - 1, y), (255, 0, 0), thickness)

        label_x = x1 - max(60, col_w // 8) - (index - 1) * max(24, col_w // 30)
        label_x = max(x0 + 20, label_x)
        arrow_start = (label_x, y0 + max(12, col_h // 30))
        arrow_end = (label_x, y - max(18, col_h // 40))
        if arrow_end[1] <= arrow_start[1]:
            arrow_start = (label_x, y + max(18, col_h // 40))
            arrow_end = (label_x, y1 - max(12, col_h // 30))

        cv2.arrowedLine(output, arrow_start, arrow_end, (255, 0, 0), thickness, tipLength=0.2)
        cv2.putText(
            output,
            str(index),
            (label_x + 8, y + 6),
            font,
            font_scale,
            (255, 0, 0),
            max(2, thickness),
            cv2.LINE_AA,
        )

    summary_h = max(44, int(50 * scale))
    cv2.rectangle(output, (0, 0), (output.shape[1], summary_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        f"Vertical rods: {len(vertical_sorted)}   Horizontal rods: {len(horizontal_sorted)}",
        (12, int(32 * scale)),
        font,
        font_scale,
        (255, 255, 255),
        max(2, thickness),
        cv2.LINE_AA,
    )
    return output
