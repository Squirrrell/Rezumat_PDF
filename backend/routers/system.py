"""System health and model info."""

import os

from fastapi import APIRouter

from backend.schemas import HealthResponse
from src.device_utils import cuda_device_info, get_device
from src.instruct_generator import FLAN_MODEL, QWEN_MODEL
from src.marker_converter import MARKER_MODEL_NAME
from src.vector_store import EMBEDDING_MODEL_NAME

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    device = get_device()
    cuda_info = cuda_device_info()
    marker_device = os.environ.get("TORCH_DEVICE", device)

    return HealthResponse(
        status="ok",
        device=device,
        cuda_available=cuda_info["cuda_available"],
        gpu_name=cuda_info.get("gpu_name"),
        marker_device=marker_device,
        converter_model=MARKER_MODEL_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
        qwen_model=QWEN_MODEL,
        flan_model=FLAN_MODEL,
    )
