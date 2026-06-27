"""Torch device helpers shared across ML modules."""

from __future__ import annotations

import os

import torch

_CUDA_INSTALL_HINT = (
    "CUDA is not available. Install GPU-enabled PyTorch (see README) and NVIDIA drivers, "
    "or set ML_ALLOW_CPU=1 to run on CPU for development."
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def resolve_device() -> str:
    """
    Resolve the ML device for all inference workloads.

    Priority:
    1. TORCH_DEVICE environment variable
    2. CUDA if available
    3. CPU when ML_ALLOW_CPU=1
    4. RuntimeError with install instructions
    """
    env_device = os.environ.get("TORCH_DEVICE", "").strip().lower()
    if env_device:
        return env_device

    if torch.cuda.is_available():
        return "cuda"

    if _env_flag("ML_ALLOW_CPU"):
        return "cpu"

    raise RuntimeError(_CUDA_INSTALL_HINT)


def get_device() -> str:
    """Alias for resolve_device() used across the codebase."""
    return resolve_device()


def get_torch_dtype(device: str | None = None) -> torch.dtype:
    """Return float16 on CUDA, float32 on CPU."""
    device = device or get_device()
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def cuda_device_info() -> dict:
    """Return CUDA availability and GPU metadata for health checks."""
    available = torch.cuda.is_available()
    info: dict = {
        "cuda_available": available,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    if available:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
        except Exception:
            pass
    return info


def device_status_message() -> str:
    """Human-readable device line for startup logs."""
    device = get_device()
    if device.startswith("cuda"):
        info = cuda_device_info()
        name = info.get("gpu_name") or "CUDA"
        mem = info.get("gpu_memory_gb")
        if mem is not None:
            return f"Using ML device: cuda ({name}, {mem} GB VRAM)"
        return f"Using ML device: cuda ({name})"
    if device == "cpu" and _env_flag("ML_ALLOW_CPU"):
        return "Using ML device: cpu (ML_ALLOW_CPU=1 fallback; GPU recommended for Marker)"
    return f"Using ML device: {device}"
