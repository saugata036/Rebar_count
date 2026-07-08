"""Image preprocessing for rebar counting."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk in BGR format."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def denoise_image(
    image_bgr: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    denoise_strength: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return denoised BGR and grayscale images."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or could not be loaded.")

    if len(image_bgr.shape) == 2:
        gray = image_bgr.copy()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image_bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))
    enhanced = clahe.apply(gray)
    denoised_gray = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    denoised_bgr = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)
    return denoised_bgr, denoised_gray


def estimate_column_mask(denoised_gray: np.ndarray) -> np.ndarray:
    """Find the building column face in the photo."""
    h, w = denoised_gray.shape[:2]
    blurred = cv2.GaussianBlur(denoised_gray, (5, 5), 0)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    mag_min = float(magnitude.min())
    mag_max = float(magnitude.max())
    if mag_max - mag_min < 1e-6:
        magnitude = np.zeros_like(magnitude, dtype=np.uint8)
    else:
        magnitude = np.clip((magnitude - mag_min) / (mag_max - mag_min) * 255.0, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 0.08 * h * w:
            cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    if np.count_nonzero(mask) == 0:
        margin_x = int(w * 0.12)
        margin_y = int(h * 0.12)
        mask[margin_y : h - margin_y, margin_x : w - margin_x] = 255

    shrink_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    return cv2.erode(mask, shrink_kernel, iterations=1)


def clean_column_interior(denoised_gray: np.ndarray, column_mask: np.ndarray) -> np.ndarray:
    """Remove inner surface speckle while keeping rod edges."""
    cleaned = denoised_gray.copy()
    interior = column_mask > 0
    if not np.any(interior):
        return cleaned

    smoothed = cv2.bilateralFilter(denoised_gray, d=11, sigmaColor=75, sigmaSpace=75)
    cleaned[interior] = smoothed[interior]

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
    cleaned[interior] = opened[interior]

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    cleaned[interior] = closed[interior]
    return cleaned


def _column_bounds(column_mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(column_mask > 0)
    if ys.size == 0:
        h, w = column_mask.shape[:2]
        return 0, 0, h, w
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def build_column_only_view(cleaned_gray: np.ndarray, column_mask: np.ndarray, padding: int = 12) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Keep only the building column section and erase side/background noise.

    Returns cropped column BGR image, cropped mask, and region offsets in the original frame.
    """
    y0, y1, x0, x1 = _column_bounds(column_mask)
    h, w = cleaned_gray.shape[:2]

    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(h, y1 + padding)
    x1 = min(w, x1 + padding)

    crop_gray = cleaned_gray[y0:y1, x0:x1].copy()
    crop_mask = column_mask[y0:y1, x0:x1].copy()

    column_bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    background = np.full_like(column_bgr, 40)
    column_bgr = np.where(crop_mask[:, :, None] > 0, column_bgr, background)

    region = {
        "top": y0,
        "bottom": y1,
        "left": x0,
        "right": x1,
        "crop_top": 0,
        "crop_bottom": y1 - y0,
        "crop_left": 0,
        "crop_right": x1 - x0,
    }
    return column_bgr, crop_mask, region


def shift_positions_to_crop(positions: list[int], axis_offset: int) -> list[int]:
    return [pos - axis_offset for pos in positions]


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Denoise, isolate column, and clean interior noise.

    Returns original_bgr, denoised_bgr, cleaned_gray, column_mask.
    """
    original_bgr = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    denoised_bgr, denoised_gray = denoise_image(original_bgr)
    column_mask = estimate_column_mask(denoised_gray)
    cleaned_gray = clean_column_interior(denoised_gray, column_mask)
    return original_bgr, denoised_bgr, cleaned_gray, column_mask
