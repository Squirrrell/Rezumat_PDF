"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str
    answer_length: str = "medium"
    qa_top_k: int = Field(default=5, ge=1, le=10)
    use_instruct_qa: bool = True
    instruct_model_key: str = "qwen"


class SummarizeRequest(BaseModel):
    source: Literal["pdf", "markdown"] = "pdf"
    length: Literal["short", "medium", "long"] = "medium"
    max_chunks: int = Field(default=12, ge=3, le=20)


class ComprehensiveSummarizeRequest(BaseModel):
    source: Literal["pdf", "markdown"] = "pdf"
    instruct_model_key: str = "qwen"
    qa_top_k: int = Field(default=5, ge=1, le=10)


class TestCardsGenerateRequest(BaseModel):
    num_cards: int = Field(default=5, ge=3, le=8)
    instruct_model_key: str = "qwen"


class TestCardAnswer(BaseModel):
    card_id: str
    answer: str


class TestCardAnswerRequest(BaseModel):
    card_id: str
    instruct_model_key: str = "qwen"
    qa_top_k: int = Field(default=5, ge=1, le=10)
    answer_length: str = "medium"


class TestCardsVerifyRequest(BaseModel):
    answers: list[TestCardAnswer]
    instruct_model_key: str = "qwen"


class DocumentMetadataResponse(BaseModel):
    document_id: str
    paper_name: str
    page_count: int
    chunk_count: int
    character_count: int
    sections_detected: list[str]
    sections_coverage_ratio: float
    sections_fallback_body: bool
    has_markdown: bool = False
    has_summary: bool = False
    last_runtime: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    gpu_name: str | None = None
    marker_device: str
    converter_model: str
    embedding_model: str
    qwen_model: str
    flan_model: str


class MetricsResponse(BaseModel):
    document_metrics: dict[str, Any] | None = None
    summary_metrics: dict[str, Any] | None = None
    qa_metrics: dict[str, Any] | None = None
    sections: dict[str, Any] | None = None
    last_runtime: dict[str, float] = Field(default_factory=dict)


class TrainingStartRequest(BaseModel):
    model_name: str = "google-t5/t5-small"
    dataset_config: Literal["pubmed", "arxiv"] = "arxiv"
    train_size: int = Field(default=1000, ge=10, le=20000)
    val_size: int = Field(default=100, ge=5, le=2000)
    epochs: int = Field(default=5, ge=1, le=20)
    learning_rate: float = Field(default=5e-5, gt=0, le=1e-2)


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    config: dict[str, Any]
    total_epochs: int
    current_epoch: float
    current_step: int
    log_history: list[dict[str, Any]] = Field(default_factory=list)
    checkpoint_path: str | None = None
    error: str | None = None


class TrainingEvaluateRequest(BaseModel):
    dataset_config: Literal["pubmed", "arxiv"] = "arxiv"
    checkpoint: str = ""
    train_size: int = Field(default=1000, ge=10, le=20000)
    eval_size: int = Field(default=100, ge=5, le=2000)
    run_baseline: bool = True
    run_finetuned: bool = True


class TrainingInfoResponse(BaseModel):
    device: str
    cuda_available: bool
    gpu_name: str | None = None
    dependencies_installed: bool
    dependencies_message: str | None = None
    default_dataset_config: str
    default_epochs: int
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)
    final_checkpoint: str | None = None
    last_results: dict[str, Any] | None = None
    active_job: TrainingJobResponse | None = None
