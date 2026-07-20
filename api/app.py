"""FastAPI entrypoint. Run with: uvicorn api.app:app --reload"""

from __future__ import annotations

from fastapi import FastAPI

from api.images_route import router as images_router

app = FastAPI(title="Rebar Analyzer API")

app.include_router(images_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
