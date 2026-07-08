"""Count vertical and horizontal rebar from a front-face image."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_prominences

from rebar_xray.config import CountConfig
from rebar_xray.utils import mask_bbox


@dataclass
class BarCountResult:
    vertical_positions: list[int]
    horizontal_positions: list[int]
    region: tuple[int, int, int, int]
    vertical_count: int
    horizontal_count: int
    debug: dict = field(default_factory=dict)


def _foreground_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    mask = (gray < threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _smooth(profile: np.ndarray, size: int) -> np.ndarray:
    k = max(5, size | 1)
    return cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (k, 1), 0).flatten()


def _column_profile(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dark = (255 - gray).astype(np.float32)
    dark = cv2.bitwise_and(dark.astype(np.uint8), dark.astype(np.uint8), mask=mask).astype(np.float32)
    return _smooth(np.sum(dark, axis=0), max(9, gray.shape[1] // 60))


def _row_profile(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dark = (255 - gray).astype(np.float32)
    dark = cv2.bitwise_and(dark.astype(np.uint8), dark.astype(np.uint8), mask=mask).astype(np.float32)
    return _smooth(np.sum(dark, axis=1), max(9, gray.shape[0] // 60))


def _estimate_peak_distance(
    profile: np.ndarray,
    prominence: float,
    min_distance: int,
    rough_distance_divisor: int,
    spacing_fraction: float,
    fallback_divisor: int,
) -> int:
    """Estimate minimum peak spacing from a coarse first pass."""
    rough_distance = max(min_distance, profile.size // rough_distance_divisor)
    rough_peaks, _ = find_peaks(profile, distance=rough_distance, prominence=prominence)
    if len(rough_peaks) >= 3:
        spacing = float(np.percentile(np.diff(rough_peaks), 25))
        return max(min_distance, int(spacing * spacing_fraction))
    if len(rough_peaks) >= 2:
        spacing = float(np.median(np.diff(rough_peaks)))
        return max(min_distance, int(spacing * spacing_fraction))
    return max(min_distance, profile.size // fallback_divisor)


def _recover_edge_peaks(
    profile: np.ndarray,
    peaks: np.ndarray,
    distance: int,
    prominence_frac: float,
    *,
    edge_prominence_fraction: float,
    edge_gap_fraction: float,
    edge_peak_height_fraction: float,
    edge_margin_px: int,
) -> np.ndarray:
    """Recover edge bars whose prominence is reduced by the image boundary."""
    if peaks.size == 0:
        return peaks

    peak_list = [int(p) for p in peaks]
    median_gap = float(np.median(np.diff(peak_list))) if len(peak_list) >= 2 else float(distance)
    median_height = float(np.median([profile[p] for p in peak_list]))
    min_height = edge_peak_height_fraction * median_height
    edge_prominence = max(edge_prominence_fraction * prominence_frac * float(profile.max()), 1.0)
    min_gap = int(edge_gap_fraction * median_gap)

    def _edge_candidate_prominence(candidate: int) -> float:
        window = profile[max(0, candidate - 3) : min(profile.size, candidate + 4)]
        if profile[candidate] < float(window.max()) * 0.98:
            return 0.0
        prominence, _, _ = peak_prominences(profile, [candidate])
        return float(prominence[0])

    tail_start = peak_list[-1] + min_gap
    if tail_start < profile.size - edge_margin_px:
        candidate = tail_start + int(np.argmax(profile[tail_start:]))
        if candidate - peak_list[-1] >= min_gap and profile[candidate] >= min_height:
            if _edge_candidate_prominence(candidate) >= edge_prominence:
                peak_list.append(candidate)

    head_end = peak_list[0] - min_gap
    if head_end > edge_margin_px:
        candidate = int(np.argmax(profile[:head_end]))
        if peak_list[0] - candidate >= min_gap and profile[candidate] >= min_height:
            if _edge_candidate_prominence(candidate) >= edge_prominence:
                peak_list.insert(0, candidate)

    return np.array(sorted(peak_list), dtype=int)


def _find_bar_peaks(
    profile: np.ndarray,
    distance_divisor: int,
    prominence_frac: float,
    min_distance: int,
    *,
    adaptive: bool,
    rough_distance_divisor: int,
    spacing_fraction: float,
    recover_edge_peaks: bool,
    edge_prominence_fraction: float,
    edge_gap_fraction: float,
    edge_peak_height_fraction: float,
    edge_margin_px: int,
) -> np.ndarray:
    if profile.size == 0 or float(profile.max()) <= 0:
        return np.array([], dtype=int)
    prominence = max(prominence_frac * float(profile.max()), 1.0)
    if adaptive:
        distance = _estimate_peak_distance(
            profile,
            prominence,
            min_distance,
            rough_distance_divisor,
            spacing_fraction,
            distance_divisor,
        )
    else:
        distance = max(min_distance, profile.size // distance_divisor)
    peaks, _ = find_peaks(profile, distance=distance, prominence=prominence)
    peaks = peaks.astype(int)
    if recover_edge_peaks:
        peaks = _recover_edge_peaks(
            profile,
            peaks,
            distance,
            prominence_frac,
            edge_prominence_fraction=edge_prominence_fraction,
            edge_gap_fraction=edge_gap_fraction,
            edge_peak_height_fraction=edge_peak_height_fraction,
            edge_margin_px=edge_margin_px,
        )
    return peaks


def count_bars(image_bgr: np.ndarray, config: CountConfig | None = None) -> BarCountResult:
    """Count vertical and horizontal bars on a front-face rebar image."""
    cfg = config or CountConfig()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = _foreground_mask(gray, cfg.background_threshold)

    x0, y0, x1, y1 = mask_bbox(mask)
    roi_gray = gray[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]

    v_profile = _column_profile(roi_gray, roi_mask)
    h_profile = _row_profile(roi_gray, roi_mask)

    peak_kwargs = {
        "adaptive": cfg.adaptive_peak_distance,
        "rough_distance_divisor": cfg.rough_peak_distance_divisor,
        "spacing_fraction": cfg.adaptive_spacing_fraction,
        "recover_edge_peaks": cfg.recover_edge_peaks,
        "edge_prominence_fraction": cfg.edge_prominence_fraction,
        "edge_gap_fraction": cfg.edge_gap_fraction,
        "edge_peak_height_fraction": cfg.edge_peak_height_fraction,
        "edge_margin_px": cfg.edge_margin_px,
    }
    v_peaks = _find_bar_peaks(
        v_profile,
        cfg.vertical_distance_divisor,
        cfg.prominence_fraction,
        cfg.min_peak_distance_px,
        **peak_kwargs,
    )
    h_peaks = _find_bar_peaks(
        h_profile,
        cfg.horizontal_distance_divisor,
        cfg.prominence_fraction,
        cfg.min_peak_distance_px,
        **peak_kwargs,
    )

    vertical_positions = [x0 + int(p) for p in v_peaks]
    horizontal_positions = [y0 + int(p) for p in h_peaks]

    return BarCountResult(
        vertical_positions=vertical_positions,
        horizontal_positions=horizontal_positions,
        region=(y0, y1, x0, x1),
        vertical_count=len(vertical_positions),
        horizontal_count=len(horizontal_positions),
        debug={
            "region": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
            "vertical_peaks_px": vertical_positions,
            "horizontal_peaks_px": horizontal_positions,
        },
    )
