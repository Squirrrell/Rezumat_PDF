"""PDF to Markdown conversion using Marker (marker-pdf)."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Literal

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings as surya_settings

from src.device_utils import cuda_device_info, get_device, get_torch_dtype

if TYPE_CHECKING:
    from marker.converters.pdf import PdfConverter as PdfConverterType

MARKER_MODEL_NAME = "marker-pdf"
MarkerProfile = Literal["fast", "quality"]

FAST_PROCESSOR_PATHS = [
    "marker.processors.order.OrderProcessor",
    "marker.processors.block_relabel.BlockRelabelProcessor",
    "marker.processors.line_merge.LineMergeProcessor",
    "marker.processors.blockquote.BlockquoteProcessor",
    "marker.processors.code.CodeProcessor",
    "marker.processors.footnote.FootnoteProcessor",
    "marker.processors.ignoretext.IgnoreTextProcessor",
    "marker.processors.list.ListProcessor",
    "marker.processors.page_header.PageHeaderProcessor",
    "marker.processors.sectionheader.SectionHeaderProcessor",
    "marker.processors.text.TextProcessor",
    "marker.processors.reference.ReferenceProcessor",
    "marker.processors.blank_page.BlankPageProcessor",
]

_LOW_VRAM_GB = 6.0
_BATCH_DEFAULTS_LOW_VRAM = {
    "layout_batch_size": 8,
    "detection_batch_size": 4,
    "ocr_error_batch_size": 6,
}
_BATCH_DEFAULTS_HIGH_VRAM = {
    "layout_batch_size": 12,
    "detection_batch_size": 8,
    "ocr_error_batch_size": 12,
}


def marker_profile() -> MarkerProfile:
    """Return active Marker pipeline profile (fast by default)."""
    val = os.environ.get("MARKER_PROFILE", "fast").strip().lower()
    return "quality" if val == "quality" else "fast"


def _marker_disable_ocr() -> bool:
    """OCR is off by default for faster conversion on text-based PDFs."""
    val = os.environ.get("MARKER_DISABLE_OCR", "1").strip().lower()
    return val not in {"0", "false", "no"}


def _marker_extract_images() -> bool:
    """Image extraction is off by default (text-only markdown output)."""
    val = os.environ.get("MARKER_EXTRACT_IMAGES", "0").strip().lower()
    return val in {"1", "true", "yes"}


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _vram_batch_defaults() -> dict[str, int]:
    info = cuda_device_info()
    mem = info.get("gpu_memory_gb")
    if mem is not None and mem <= _LOW_VRAM_GB:
        return dict(_BATCH_DEFAULTS_LOW_VRAM)
    return dict(_BATCH_DEFAULTS_HIGH_VRAM)


def _apply_batch_env_overrides(config: dict) -> None:
    for env_name, key in (
        ("MARKER_LAYOUT_BATCH_SIZE", "layout_batch_size"),
        ("MARKER_DETECTION_BATCH_SIZE", "detection_batch_size"),
        ("MARKER_OCR_ERROR_BATCH_SIZE", "ocr_error_batch_size"),
    ):
        value = _env_int(env_name)
        if value is not None:
            config[key] = value


def _marker_config(profile: MarkerProfile, device: str | None = None) -> dict:
    device = device or get_device()
    config: dict = {
        "disable_ocr": _marker_disable_ocr(),
        "extract_images": _marker_extract_images(),
        "disable_tqdm": True,
    }

    if profile == "fast":
        config.update(
            {
                "lowres_image_dpi": 72,
                "highres_image_dpi": 120,
                "disable_ocr_math": True,
            }
        )

    if device.startswith("cuda"):
        if profile == "fast":
            config.update(_vram_batch_defaults())
        _apply_batch_env_overrides(config)

    return config


def _create_artifact_dict(
    device: str,
    dtype,
    profile: MarkerProfile,
    attention_implementation: str | None = None,
) -> dict:
    if profile == "quality":
        return create_model_dict(
            device=device,
            dtype=dtype,
            attention_implementation=attention_implementation,
        )

    foundation_kwargs = {
        "attention_implementation": attention_implementation,
        "device": device,
        "dtype": dtype,
    }
    return {
        "layout_model": LayoutPredictor(
            FoundationPredictor(
                checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT,
                **foundation_kwargs,
            )
        ),
        "recognition_model": RecognitionPredictor(
            FoundationPredictor(
                checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT,
                **foundation_kwargs,
            )
        ),
        "detection_model": DetectionPredictor(device=device, dtype=dtype),
        "ocr_error_model": OCRErrorPredictor(device=device, dtype=dtype),
    }


def create_pdf_converter() -> PdfConverterType:
    """Build a Marker PdfConverter for the active profile."""
    device = get_device()
    dtype = get_torch_dtype(device)
    profile = marker_profile()
    kwargs: dict = {
        "artifact_dict": _create_artifact_dict(device, dtype, profile),
        "config": _marker_config(profile, device),
    }
    if profile == "fast":
        kwargs["processor_list"] = list(FAST_PROCESSOR_PATHS)
    return PdfConverter(**kwargs)


def convert_pdf_to_markdown(
    file_bytes: bytes,
    converter: PdfConverterType | None = None,
) -> tuple[str, int]:
    """
    Convert PDF bytes to Markdown using Marker.

    Returns:
        Tuple of (markdown text, page count).

    Raises:
        ValueError: If input is empty or conversion yields no text.
        RuntimeError: On OOM or other conversion failures.
    """
    if not file_bytes:
        raise ValueError("No PDF data provided.")

    if converter is None:
        converter = create_pdf_converter()

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        try:
            rendered = converter(temp_path)
        except MemoryError as exc:
            raise RuntimeError(
                "Marker ran out of memory while converting the PDF. "
                "Try a shorter document or set TORCH_DEVICE=cpu."
            ) from exc
        except Exception as exc:
            if _is_oom(exc):
                raise RuntimeError(
                    "Marker ran out of memory while converting the PDF. "
                    "Try a shorter document or set TORCH_DEVICE=cpu."
                ) from exc
            raise

        markdown, _, _ = text_from_rendered(rendered)
        markdown = (markdown or "").strip()
        if not markdown:
            raise ValueError(
                "Marker produced no Markdown output. The PDF may be empty or unreadable."
            )

        page_count = _page_count_from_rendered(rendered)
        return markdown, page_count
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _page_count_from_rendered(rendered: object) -> int:
    """Best-effort page count from Marker render output."""
    metadata = getattr(rendered, "metadata", None)
    if metadata is not None:
        pages = getattr(metadata, "page_count", None)
        if pages is not None:
            return int(pages)
        if isinstance(metadata, dict):
            pages = metadata.get("page_count")
            if pages is not None:
                return int(pages)

    markdown = getattr(rendered, "markdown", None) or ""
    if isinstance(markdown, str) and markdown.strip():
        page_markers = markdown.count("-" * 48)
        if page_markers > 0:
            return page_markers + 1
        return max(1, markdown.count("\n\n") // 20 or 1)
    return 1


def _is_oom(exc: BaseException) -> bool:
    """Return True if the exception looks like an out-of-memory error."""
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg and "memory" in msg
