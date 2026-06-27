"""FastAPI backend for Local AI Research Paper Summarizer."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import (  # noqa: E402
    documents,
    metrics,
    summarize,
    system,
    test_cards,
    training,
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.device_utils import device_status_message, get_device

    device = get_device()
    os.environ.setdefault("TORCH_DEVICE", device)
    logger.info(device_status_message())

    from src.marker_converter import marker_profile

    logger.info("Marker profile: %s", marker_profile())

    from backend.model_cache import get_marker_converter

    try:
        get_marker_converter()
    except Exception as exc:
        logger.warning("Marker preload skipped or failed: %s", exc)
    yield


app = FastAPI(
    title="PDF to Markdown Converter API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(documents.router)
app.include_router(summarize.router)
app.include_router(test_cards.router)
app.include_router(metrics.router)
app.include_router(training.router)


@app.get("/")
def root() -> dict:
    return {"message": "PDF to Markdown Converter API", "docs": "/docs"}
