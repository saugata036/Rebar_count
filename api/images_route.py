"""RESTful images API for the Rebar Analyzer.

Endpoints:
- POST /api/images              -> upload & denoise image (returns image_id and metadata)
- GET  /api/images              -> list all images and metadata
- GET  /api/images/{id}         -> metadata for single image
- GET  /api/images/{id}/clean   -> cleaned (denoised) image file
- POST /api/images/{id}/analyze -> run analysis on cleaned image
- GET  /api/images/{id}/analysis -> analysis metadata
- GET  /api/images/{id}/analysis/image -> analyzed overlay image
- DELETE /api/images/{id}       -> delete image + metadata + analysis

This file is intentionally standalone and replaces the previous denoise/analyze route files.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types
from PIL import Image

from rebar_xray.config import PipelineConfig
from rebar_xray.pipeline import RebarXRayPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "outputs" / "images"
UPLOAD_DIR = IMAGES_DIR / "uploads"
CLEAN_DIR = IMAGES_DIR / "clean"
ANALYZE_DIR = IMAGES_DIR / "analysis"
METADATA_FILE = IMAGES_DIR / "metadata.json"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

# Model & prompt used by the denoising step (keeps original behavior)
GEMINI_MODEL = "gemini-3.5-flash"
MASK_GENERATION_PROMPT = """
Analyze the provided construction rebar column image carefully.
Your task is to identify and isolate ONLY the absolute front-facing grid layer (the 5 vertical bars and 9 horizontal bars closest to the camera) and the 4 corner concrete spacer blocks.

Generate a valid JSON object containing:
1. "vertical_bars": An array of centerlines representing the 5 vertical front-facing bars. Each bar is represented by its start and end point center coordinates.
2. "horizontal_bars": An array of centerlines representing the 9 horizontal front-facing ties. Each tie is represented by its start and end point center coordinates.
3. "spacer_blocks": An array of bounding boxes tracing the 4 corner concrete spacer blocks.

Do not include any background elements.
Return your response strictly in this JSON format and scale coordinates 0..1000.
"""

REBAR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "vertical_bars": {"type": "ARRAY"},
        "horizontal_bars": {"type": "ARRAY"},
        "spacer_blocks": {"type": "ARRAY"},
    },
    "required": ["vertical_bars", "horizontal_bars", "spacer_blocks"],
}

router = APIRouter(prefix="/api", tags=["images"])


# --- metadata helpers ---
def _load_metadata() -> Dict[str, Dict[str, Any]]:
    if not METADATA_FILE.exists():
        return {}
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load metadata: {e}") from e


def _save_metadata(data: Dict[str, Dict[str, Any]]) -> None:
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _add_metadata(entry: Dict[str, Any]) -> None:
    data = _load_metadata()
    data[entry["image_id"]] = entry
    _save_metadata(data)


def _update_metadata(image_id: str, update: Dict[str, Any]) -> None:
    data = _load_metadata()
    if image_id not in data:
        raise HTTPException(status_code=404, detail="Image ID not found")
    data[image_id].update(update)
    _save_metadata(data)


def _remove_metadata(image_id: str) -> Dict[str, Any]:
    data = _load_metadata()
    if image_id not in data:
        raise HTTPException(status_code=404, detail="Image ID not found")
    entry = data.pop(image_id)
    _save_metadata(data)
    return entry


# --- small helpers ported from previous file ---

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


def _draw_perspective_line(mask, pt1, pt2, max_dim, val, bottom_ratio, top_ratio) -> None:
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


def _clean_rebar_structure(image_path: Path, output_path: Path) -> None:
    """Run the Gemini mask generation + OpenCV masking to produce a clean front-face image.

    Raises HTTPException with code 500 on failure so the router can return appropriate errors.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing. Set it in .env")

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
    try:
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
            raise ValueError(f"Could not read image: {image_path}")
        h, w, _ = src_img.shape
        max_dim = max(h, w)

        gc_mask = np.zeros((h, w), dtype=np.uint8)
        target_lines = mask_data.get("vertical_bars", []) + mask_data.get("horizontal_bars", [])

        for bar in target_lines:
            x1, y1, x2, y2 = _safe_extract_line(bar)
            pt1 = (int(x1 / 1000.0 * w), int(y1 / 1000.0 * h))
            pt2 = (int(x2 / 1000.0 * w), int(y2 / 1000.0 * h))
            _draw_perspective_line(gc_mask, pt1, pt2, max_dim, cv2.GC_PR_BGD, 0.048, 0.015)

        for bar in target_lines:
            x1, y1, x2, y2 = _safe_extract_line(bar)
            pt1 = (int(x1 / 1000.0 * w), int(y1 / 1000.0 * h))
            pt2 = (int(x2 / 1000.0 * w), int(y2 / 1000.0 * h))
            _draw_perspective_line(gc_mask, pt1, pt2, max_dim, cv2.GC_PR_FGD, 0.028, 0.009)

        for bar in target_lines:
            x1, y1, x2, y2 = _safe_extract_line(bar)
            pt1 = (int(x1 / 1000.0 * w), int(y1 / 1000.0 * h))
            pt2 = (int(x2 / 1000.0 * w), int(y2 / 1000.0 * h))
            _draw_perspective_line(gc_mask, pt1, pt2, max_dim, cv2.GC_FGD, 0.012, 0.004)

        for block in mask_data.get("spacer_blocks", []):
            ymin, xmin, ymax, xmax = _safe_extract_box(block)
            p_ymin, p_xmin = int((ymin / 1000.0) * h), int((xmin / 1000.0) * w)
            p_ymax, p_xmax = int((ymax / 1000.0) * h), int((xmax / 1000.0) * w)
            cv2.rectangle(gc_mask, (p_xmin, p_ymin), (p_xmax, p_ymax), cv2.GC_FGD, -1)

        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        sky_pixels = gray > 210
        shadow_pixels = gray < 40
        prune_mask = sky_pixels | shadow_pixels
        gc_mask[(prune_mask) & (gc_mask != cv2.GC_FGD)] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        grabcut_rect = (0, 0, w, h)
        cv2.grabCut(src_img, gc_mask, grabcut_rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask)
        min_area = int(h * w * 0.0015)

        cleaned_mask = np.zeros_like(final_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 255

        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (3, 3), 0)
        studio_canvas = np.full_like(src_img, 235)
        mask_3ch = cv2.merge([cleaned_mask, cleaned_mask, cleaned_mask])
        np.copyto(studio_canvas, src_img, where=(mask_3ch > 127))
        cv2.imwrite(str(output_path), studio_canvas)

    except json.JSONDecodeError as je:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON formatting. Raw response preview: {text_data[:500]}") from je
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Masking pipeline execution failed: {e}") from e


# --- API endpoints ---
@router.post("/images")
async def create_image(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_id = uuid.uuid4().hex
    uploaded_path = UPLOAD_DIR / f"{image_id}_{file.filename}"
    clean_path = CLEAN_DIR / f"{image_id}.png"

    with uploaded_path.open("wb") as f:
        f.write(await file.read())

    # synchronous denoise (keeps behavior familiar). Could be async background job later.
    _clean_rebar_structure(uploaded_path, clean_path)
    if not clean_path.exists():
        raise HTTPException(status_code=500, detail="Denoising failed to produce a clean image.")

    entry = {
        "image_id": image_id,
        "original_filename": file.filename,
        "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
        "uploaded_path": str(uploaded_path.resolve()),
        "clean_path": str(clean_path.resolve()),
        "status": "ready",
        "analysis": None,
    }
    _add_metadata(entry)

    return JSONResponse(status_code=201, content={
        "image_id": image_id,
        "original_filename": file.filename,
        "uploaded_at": entry["uploaded_at"],
        "status": "ready",
        "clean_image_url": f"/api/images/{image_id}/clean",
        "metadata_url": f"/api/images/{image_id}",
    })


@router.get("/images")
def list_images() -> JSONResponse:
    data = _load_metadata()
    items = []
    for entry in data.values():
        item = dict(entry)
        item["clean_image_url"] = f"/api/images/{entry['image_id']}/clean"
        if entry.get("analysis"):
            item["analysis_url"] = f"/api/images/{entry['image_id']}/analysis"
            item["analysis_image_url"] = f"/api/images/{entry['image_id']}/analysis/image"
        items.append(item)
    return JSONResponse(content={"items": items})


@router.get("/images/{image_id}")
def get_image(image_id: str) -> JSONResponse:
    data = _load_metadata()
    entry = data.get(image_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")
    item = dict(entry)
    item["clean_image_url"] = f"/api/images/{image_id}/clean"
    if entry.get("analysis"):
        item["analysis_url"] = f"/api/images/{image_id}/analysis"
        item["analysis_image_url"] = f"/api/images/{image_id}/analysis/image"
    return JSONResponse(content=item)


@router.get("/images/{image_id}/clean")
def get_clean_image(image_id: str) -> FileResponse:
    data = _load_metadata()
    entry = data.get(image_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(entry["clean_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clean image not found")
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.post("/images/{image_id}/analyze")
def analyze_image(image_id: str) -> JSONResponse:
    data = _load_metadata()
    entry = data.get(image_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")
    clean_path = Path(entry["clean_path"])
    if not clean_path.exists():
        raise HTTPException(status_code=404, detail="Clean image not found")

    try:
        result = RebarXRayPipeline(PipelineConfig()).run(clean_path, output_dir=ANALYZE_DIR / image_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {e}") from e

    analysis_entry = {
        "analyzed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "output_dir": str((ANALYZE_DIR / image_id).resolve()),
        "analysis_image_path": str((ANALYZE_DIR / image_id / "counted.png").resolve()),
        "bar_counts": result.payload["bar_counts"],
        "vertical_positions_px": result.payload.get("vertical_positions_px"),
        "horizontal_positions_px": result.payload.get("horizontal_positions_px"),
        "debug": result.payload.get("debug"),
        "message": result.payload.get("message"),
    }
    _update_metadata(image_id, {"analysis": analysis_entry})

    return JSONResponse(content={
        "image_id": image_id,
        "analysis": analysis_entry,
        "analysis_image_url": f"/api/images/{image_id}/analysis/image",
    })


@router.get("/images/{image_id}/analysis")
def get_analysis(image_id: str) -> JSONResponse:
    data = _load_metadata()
    entry = data.get(image_id)
    if not entry or not entry.get("analysis"):
        raise HTTPException(status_code=404, detail="Analysis not found")
    return JSONResponse(content=entry["analysis"])


@router.get("/images/{image_id}/analysis/image")
def get_analysis_image(image_id: str) -> FileResponse:
    path = ANALYZE_DIR / image_id / "counted.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis image not found")
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.delete("/images/{image_id}")
def delete_image(image_id: str) -> JSONResponse:
    entry = _remove_metadata(image_id)
    for p in (entry.get("uploaded_path"), entry.get("clean_path")):
        if p:
            pth = Path(p)
            if pth.exists():
                pth.unlink()
    analyse_dir = ANALYZE_DIR / image_id
    if analyse_dir.exists():
        shutil.rmtree(analyse_dir)
    return JSONResponse(content={"status": "ok", "image_id": image_id})
