from __future__ import annotations

import base64
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import sqlite3
import uuid
import datetime
import shutil
from pathlib import Path

load_dotenv()

from api.analysis import run_analysis
from api.denoise import clean_rebar_structure

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = PROJECT_ROOT / "outputs" / "images"
UPLOADS_DIR = STORAGE_DIR / "uploads"
CLEAN_DIR = STORAGE_DIR / "clean"
ANALYSIS_DIR = STORAGE_DIR / "analysis"
DB_PATH = STORAGE_DIR / "images.db"

for d in (UPLOADS_DIR, CLEAN_DIR, ANALYSIS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Initialize DB
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            original_filename TEXT,
            uploaded_at TEXT,
            analyzed_at TEXT,
            noisy_path TEXT,
            denoised_path TEXT,
            overlay_path TEXT,
            vertical_rods INTEGER,
            horizontal_rods INTEGER,
            status TEXT,
            message TEXT
        )
        """
    )

    # ensure analyzed_at column exists for older DBs
    cur.execute("PRAGMA table_info(images)")
    cols = [r[1] for r in cur.fetchall()]
    if 'analyzed_at' not in cols:
        try:
            cur.execute("ALTER TABLE images ADD COLUMN analyzed_at TEXT")
        except Exception:
            pass

    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="Rebar Analysis Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _format_iso(iso_str: str | None) -> str | None:
    if not iso_str:
        return None
    try:
        # Accept 'Z' as UTC
        dt = datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        return iso_str


@app.get("/images")
async def list_images() -> JSONResponse:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, original_filename, uploaded_at, analyzed_at, vertical_rods, horizontal_rods, status FROM images ORDER BY uploaded_at DESC")
    rows = cur.fetchall()
    conn.close()
    items = []
    for r in rows:
        uploaded_iso = r[2]
        analyzed_iso = r[3]
        items.append(
            {
                "id": r[0],
                "original_filename": r[1],
                "uploaded_at": uploaded_iso,
                "uploaded_at_human": _format_iso(uploaded_iso),
                "analyzed_at": analyzed_iso,
                "analyzed_at_human": _format_iso(analyzed_iso),
                "vertical_rods": r[4],
                "horizontal_rods": r[5],
                "status": r[6],
            }
        )
    return JSONResponse(content={"items": items})


@app.get("/images/{image_id}")
async def get_image_metadata(image_id: str) -> JSONResponse:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, original_filename, uploaded_at, analyzed_at, noisy_path, denoised_path, overlay_path, vertical_rods, horizontal_rods, status, message FROM images WHERE id = ?", (image_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    keys = ["id","original_filename","uploaded_at","analyzed_at","noisy_path","denoised_path","overlay_path","vertical_rods","horizontal_rods","status","message"]
    item = dict(zip(keys, row))
    # add human-friendly timestamps
    item["uploaded_at_human"] = _format_iso(item.get("uploaded_at"))
    item["analyzed_at_human"] = _format_iso(item.get("analyzed_at"))
    return JSONResponse(content=item)


@app.get("/images/{image_id}/noisy")
async def get_noisy_image(image_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT noisy_path FROM images WHERE id = ?", (image_id,))
    row = cur.fetchone()
    conn.close()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Noisy image not found")
    path = Path(row[0])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Noisy image file missing")
    return FileResponse(path, media_type="image/png")


@app.get("/images/{image_id}/denoised")
async def get_denoised_image(image_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT denoised_path FROM images WHERE id = ?", (image_id,))
    row = cur.fetchone()
    conn.close()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Denoised image not found")
    path = Path(row[0])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Denoised image file missing")
    return FileResponse(path, media_type="image/png")


@app.get("/images/{image_id}/overlay")
async def get_overlay_image(image_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT overlay_path FROM images WHERE id = ?", (image_id,))
    row = cur.fetchone()
    conn.close()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Overlay image not found")
    path = Path(row[0])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Overlay image file missing")
    return FileResponse(path, media_type="image/png")


@app.delete("/images/{image_id}")
async def delete_image(image_id: str) -> JSONResponse:
    """Delete image record and associated files (noisy, denoised, overlay, analysis dir)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT noisy_path, denoised_path, overlay_path FROM images WHERE id = ?", (image_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    noisy_path, denoised_path, overlay_path = row
    try:
        for p in (noisy_path, denoised_path, overlay_path):
            if p:
                pth = Path(p)
                if pth.exists():
                    if pth.is_file():
                        pth.unlink()
                    elif pth.is_dir():
                        shutil.rmtree(pth)
        # remove analysis directory if present
        analysis_dir = ANALYSIS_DIR / image_id
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)

        # delete DB row
        cur.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {exc}") from exc

    conn.close()
    return JSONResponse(content={"status": "ok", "id": image_id})


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # generate persistent IDs and paths
    image_id = uuid.uuid4().hex
    suffix = Path(file.filename).suffix or ".png"

    noisy_path = UPLOADS_DIR / f"{image_id}_{file.filename}"
    denoised_path = CLEAN_DIR / f"{image_id}.png"
    analysis_output_dir = ANALYSIS_DIR / image_id

    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        contents = await file.read()
        with noisy_path.open("wb") as handle:
            handle.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}") from exc

    # Run denoise -> clean image saved to denoised_path
    try:
        clean_rebar_structure(noisy_path, denoised_path)
    except Exception as exc:
        # record failure in DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO images (id, original_filename, uploaded_at, analyzed_at, noisy_path, denoised_path, overlay_path, status, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, file.filename, datetime.datetime.utcnow().isoformat() + "Z", None, str(noisy_path), None, None, "failed_denoise", str(exc)),
        )
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Denoising failed: {exc}") from exc

    if not denoised_path.exists():
        raise HTTPException(status_code=500, detail="Denoised image was not created.")

    # Run analysis - results stored under analysis_output_dir
    try:
        analysis_result = run_analysis(denoised_path, analysis_output_dir)
    except Exception as exc:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO images (id, original_filename, uploaded_at, analyzed_at, noisy_path, denoised_path, overlay_path, status, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, file.filename, datetime.datetime.utcnow().isoformat() + "Z", None, str(noisy_path), str(denoised_path), None, "failed_analysis", str(exc)),
        )
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    output_image_path = Path(analysis_result.get("output_image", ""))
    if not output_image_path.exists():
        raise HTTPException(status_code=500, detail="Analysis overlay image not found.")

    # Move or copy produced files into the persistent analysis folder if they are not already there
    try:
        # If pipeline saved into a different directory, copy to our persistent analysis dir
        persistent_overlay = analysis_output_dir / Path(output_image_path).name
        if str(output_image_path.resolve()) != str(persistent_overlay.resolve()):
            shutil.copy(str(output_image_path), str(persistent_overlay))
        else:
            persistent_overlay = output_image_path

        # Ensure denoised is in CLEAN_DIR (it already is)
        persistent_denoised = denoised_path

        # Save DB record with counts
        bar_counts = analysis_result.get("bar_counts") or {}
        vertical = bar_counts.get("vertical_rods") if isinstance(bar_counts, dict) else None
        horizontal = bar_counts.get("horizontal_rods") if isinstance(bar_counts, dict) else None

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO images (id, original_filename, uploaded_at, analyzed_at, noisy_path, denoised_path, overlay_path, vertical_rods, horizontal_rods, status, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                file.filename,
                datetime.datetime.utcnow().isoformat() + "Z",
                datetime.datetime.utcnow().isoformat() + "Z",
                str(noisy_path),
                str(persistent_denoised),
                str(persistent_overlay),
                vertical,
                horizontal,
                "ok",
                analysis_result.get("message"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed saving results: {exc}") from exc

    try:
        # encode denoised image
        denoised_bytes = persistent_denoised.read_bytes()
        denoised_image = "data:image/png;base64," + base64.b64encode(denoised_bytes).decode("ascii")

        # encode overlay image
        overlay_bytes = persistent_overlay.read_bytes()
        overlay_image = "data:image/png;base64," + base64.b64encode(overlay_bytes).decode("ascii")

        # encode noisy image
        noisy_bytes = noisy_path.read_bytes()
        noisy_image = "data:image/png;base64," + base64.b64encode(noisy_bytes).decode("ascii")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not encode result images: {exc}") from exc

    return JSONResponse(
        content={
            "status": "ok",
            "counts": {
                "vertical_rods": vertical,
                "horizontal_rods": horizontal,
            },
            "noisy_image": noisy_image,
            "denoised_image": denoised_image,
            "overlay_image": overlay_image,
            "id": image_id,
        }
    )
