"""Count vertical and horizontal rebar rods inside a building column."""

from __future__ import annotations

import cv2
import numpy as np


def _smooth_1d(profile: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    if profile.size == 0:
        return profile
    k = max(3, kernel_size | 1)
    return cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (k, 1), 0).flatten()


def _column_bounds(column_mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(column_mask > 0)
    if ys.size == 0:
        h, w = column_mask.shape[:2]
        return 0, 0, h, w
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _find_peaks(profile: np.ndarray, min_distance: int, min_height: float) -> list[int]:
    if profile.size < 3:
        return []

    peaks: list[int] = []
    for idx in range(1, len(profile) - 1):
        if profile[idx] >= min_height and profile[idx] >= profile[idx - 1] and profile[idx] >= profile[idx + 1]:
            peaks.append(idx)

    if not peaks:
        return []

    peaks.sort(key=lambda i: profile[i], reverse=True)
    kept: list[int] = []
    for peak in peaks:
        if all(abs(peak - chosen) >= min_distance for chosen in kept):
            kept.append(peak)
    return sorted(kept)


def _count_vertical_rods(cleaned_gray: np.ndarray, column_mask: np.ndarray) -> tuple[int, list[int]]:
    """Count main vertical iron bars (long rods running up the column)."""
    y0, y1, x0, x1 = _column_bounds(column_mask)
    roi = cleaned_gray[y0:y1, x0:x1]
    roi_mask = column_mask[y0:y1, x0:x1]
    height, width = roi.shape[:2]

    vert_len = max(15, height // 25)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    enhanced = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, vert_kernel)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask)

    grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    profile = np.sum(np.abs(grad_x), axis=0)
    profile = _smooth_1d(profile, kernel_size=max(7, width // 80))

    min_distance = max(12, width // 18)
    min_height = float(np.percentile(profile, 82))
    peaks = _find_peaks(profile, min_distance=min_distance, min_height=min_height)
    return len(peaks), [x0 + p for p in peaks]


def _count_horizontal_rods(cleaned_gray: np.ndarray, column_mask: np.ndarray) -> tuple[int, list[int]]:
    """Count horizontal ring/stirrup rods (crinkle iron) across the column."""
    y0, y1, x0, x1 = _column_bounds(column_mask)
    roi = cleaned_gray[y0:y1, x0:x1]
    roi_mask = column_mask[y0:y1, x0:x1]
    height, width = roi.shape[:2]

    horiz_len = max(15, width // 25)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    enhanced = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, horiz_kernel)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask)

    grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    profile = np.sum(np.abs(grad_y), axis=1)
    profile = _smooth_1d(profile, kernel_size=max(7, height // 80))

    min_distance = max(12, height // 22)
    min_height = float(np.percentile(profile, 82))
    peaks = _find_peaks(profile, min_distance=min_distance, min_height=min_height)
    return len(peaks), [y0 + p for p in peaks]


def count_column_rebars(cleaned_gray: np.ndarray, column_mask: np.ndarray) -> dict:
    """
    Count vertical main rods and horizontal stirrup/ring rods visible on the column face.
    """
    vertical_count, vertical_positions = _count_vertical_rods(cleaned_gray, column_mask)
    horizontal_count, horizontal_positions = _count_horizontal_rods(cleaned_gray, column_mask)

    y0, y1, x0, x1 = _column_bounds(column_mask)
    return {
        "vertical_rods": vertical_count,
        "horizontal_rods": horizontal_count,
        "total_rods": vertical_count + horizontal_count,
        "column_region": {
            "top": y0,
            "bottom": y1,
            "left": x0,
            "right": x1,
        },
        "vertical_positions_px": vertical_positions,
        "horizontal_positions_px": horizontal_positions,
    }
