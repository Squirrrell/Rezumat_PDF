"""Streamlit app: Local AI Research Paper Summarizer."""

import hashlib
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.comparison import run_comparison
from src.evaluation import (
    calculate_rouge,
    evaluate_qa,
    evaluate_summary,
    format_summary_export,
    metrics_to_dict,
    word_count,
)
from src.model_manager import (
    DEFAULT_PRETRAINED_MODEL,
    hierarchical_summarize_for_model,
    load_finetuned_model,
    load_pretrained_model,
)
from src.pdf_utils import extract_text_from_pdf
from src.instruct_generator import FLAN_MODEL, QWEN_MODEL, load_instruct_model_with_fallback
from src.qa_system import answer_question, get_answer_preset
from src.section_detector import PaperSections, detect_sections
from src.structured_summary import StructuredSummaryResult
from src.summarizer import (
    MODEL_NAME as SUMMARIZER_MODEL_NAME,
    SUMMARY_PRESETS,
    get_device,
    get_summary_preset,
    hierarchical_summarize,
    load_summarizer,
)
from src.text_utils import chunk_text, clean_text
from src.vector_store import (
    EMBEDDING_MODEL_NAME,
    VectorIndex,
    build_index,
    load_embedding_model,
)

st.set_page_config(
    page_title="Local AI Research Paper Summarizer",
    layout="wide",
)

st.title("Local AI Research Paper Summarizer")

# --- Sidebar settings ---
st.sidebar.header("Settings")

with st.sidebar.expander("Summarization", expanded=True):
    summary_type = st.selectbox(
        "Summary type",
        options=["short", "detailed", "bullet points"],
        index=0,
    )
    _preset = SUMMARY_PRESETS.get(summary_type, SUMMARY_PRESETS["short"])
    st.caption(f"Basic summarizer: **{SUMMARIZER_MODEL_NAME}**")
    structured_model_choice = st.selectbox(
        "Structured summarizer",
        options=["Qwen2.5-0.5B-Instruct", "flan-t5-small (fallback)"],
        index=0,
        help="Used for section-aware structured summaries and optional Q&A.",
    )
    detailed_structured = st.checkbox(
        "Detailed structured format",
        value=(summary_type == "detailed"),
        help=(
            "Generate sections: Title/Topic, Main idea, Method, Results, "
            "Limitations, Conclusion, Key takeaways."
        ),
    )
    section_aware_structured = st.checkbox(
        "Use section-aware structured summary",
        value=True,
        disabled=not detailed_structured,
        help=(
            "Retrieve different PDF chunks per section (recommended). "
            "Reduces repeated generic sentences."
        ),
    )
    show_field_sources = st.checkbox(
        "Show sources for each structured section",
        value=False,
        disabled=not (detailed_structured and section_aware_structured),
    )
    structured_top_k = st.slider(
        "Chunks per structured section",
        min_value=3,
        max_value=8,
        value=5,
        step=1,
        disabled=not (detailed_structured and section_aware_structured),
    )
    max_summary_length = st.slider(
        "Max summary length (tokens)",
        min_value=128,
        max_value=512,
        value=min(_preset["max_length"], 512),
        step=16,
    )
    min_summary_length = st.slider(
        "Min summary length (tokens)",
        min_value=40,
        max_value=200,
        value=min(_preset["min_length"], max_summary_length - 1),
        step=10,
    )
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=800,
        max_value=1500,
        value=1200,
        step=50,
    )
    overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=50,
        max_value=300,
        value=150,
        step=10,
    )
    max_chunks = st.slider(
        "Max chunks to summarize",
        min_value=4,
        max_value=30,
        value=12,
        step=1,
    )

summary_settings = get_summary_preset(
    summary_type,
    max_override=max_summary_length,
    min_override=min_summary_length,
)

with st.sidebar.expander("Q&A", expanded=True):
    answer_length = st.selectbox(
        "Answer length",
        options=["short", "medium", "long"],
        index=1,
    )
    qa_top_k = st.slider(
        "Retrieved chunks for Q&A",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )
    use_instruct_qa = st.checkbox(
        "Use instruct model for Q&A",
        value=True,
        help=f"Uses {structured_model_choice} instead of DistilBART for answers.",
    )

qa_settings = get_answer_preset(answer_length)
instruct_model_key = (
    "flan" if "flan" in structured_model_choice.lower() else "qwen"
)


@st.cache_resource
def get_summarizer():
    return load_summarizer()


@st.cache_resource
def get_embedder():
    return load_embedding_model()


@st.cache_resource
def get_instruct_generator(model_key: str):
    return load_instruct_model_with_fallback(model_key)


@st.cache_data
def process_pdf(file_bytes: bytes, chunk_size: int, overlap: int):
    """Extract, clean, and chunk PDF text."""
    t0 = time.perf_counter()
    raw_text, page_count = extract_text_from_pdf(file_bytes)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    extract_time = time.perf_counter() - t0
    return raw_text, cleaned, chunks, page_count, extract_time


def init_session_state():
    defaults = {
        "raw_text": "",
        "cleaned_text": "",
        "chunks": [],
        "page_count": 0,
        "sections": None,
        "vector_index": None,
        "summary": "",
        "summary_metrics": None,
        "comparison_results": [],
        "model_comparison": None,
        "qa_answer": "",
        "qa_sources": [],
        "qa_metrics": None,
        "last_runtime": {},
        "file_hash": None,
        "paper_name": "",
        "structured_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_summary_metrics(metrics, prefix: str = ""):
    """Show evaluation metrics in a row of st.metric widgets."""
    cols = st.columns(4)
    cols[0].metric(
        f"{prefix}Source words",
        f"{metrics.source_words:,}",
    )
    cols[1].metric(
        f"{prefix}Summary words",
        f"{metrics.summary_words:,}",
    )
    cols[2].metric(
        f"{prefix}Compression",
        f"{metrics.compression_percent:.1f}% reduction",
    )
    cols[3].metric(
        f"{prefix}Runtime",
        f"{metrics.runtime_seconds:.2f}s",
    )


def render_runtime_sidebar():
    with st.sidebar.expander("Runtime & models", expanded=False):
        device = get_device()
        st.write(f"**Device:** {device}")
        st.write(f"**Chunk summarizer:** {SUMMARIZER_MODEL_NAME}")
        st.write(f"**Structured / Q&A:** {QWEN_MODEL if instruct_model_key == 'qwen' else FLAN_MODEL}")
        st.write(f"**Embeddings:** {EMBEDDING_MODEL_NAME}")
        rt = st.session_state.get("last_runtime", {})
        if rt:
            st.divider()
            for label, seconds in rt.items():
                st.write(f"**{label}:** {seconds:.2f}s")


init_session_state()
render_runtime_sidebar()

# --- PDF upload ---
uploaded = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded is not None:
    file_bytes = uploaded.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.file_hash:
        try:
            with st.spinner("Processing PDF..."):
                raw, cleaned, chunks, pages, extract_time = process_pdf(
                    file_bytes, chunk_size, overlap
                )
                sections = detect_sections(cleaned)

            st.session_state.raw_text = raw
            st.session_state.cleaned_text = cleaned
            st.session_state.chunks = chunks
            st.session_state.page_count = pages
            st.session_state.sections = sections
            st.session_state.file_hash = file_hash
            st.session_state.paper_name = uploaded.name
            st.session_state.summary = ""
            st.session_state.structured_result = None
            st.session_state.summary_metrics = None
            st.session_state.comparison_results = []
            st.session_state.qa_answer = ""
            st.session_state.qa_sources = []
            st.session_state.qa_metrics = None
            st.session_state.vector_index = None

            t0 = time.perf_counter()
            embed_model = get_embedder()
            st.session_state.vector_index = build_index(chunks, embed_model)
            index_time = time.perf_counter() - t0

            st.session_state.last_runtime = {
                "pdf_extraction": extract_time,
                "index_build": index_time,
            }
            section_info = (
                f"{len(sections.detected)} sections detected"
                if not sections.fallback_body
                else "section detection unavailable (using full text)"
            )
            st.success(
                f"Processed **{uploaded.name}** — {pages} pages, "
                f"{len(chunks)} chunks, {section_info}."
            )
            if sections.fallback_body:
                st.warning(
                    "No standard section headings found. Comparison "
                    "'intro + conclusion' will use the first and last "
                    "portions of the document."
                )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Failed to process PDF: {exc}")

# --- Main tabs ---
tab_summarize, tab_sections, tab_comparison, tab_model_comparison, tab_metrics = st.tabs(
    [
        "Summarize & Q&A",
        "Detected sections",
        "Comparison",
        "Model Comparison",
        "Thesis metrics",
    ]
)

# ===================== Tab: Summarize & Q&A =====================
with tab_summarize:
    if st.session_state.cleaned_text:
        with st.expander("Extracted text preview", expanded=False):
            preview = st.session_state.cleaned_text[:2000]
            if len(st.session_state.cleaned_text) > 2000:
                preview += "\n\n..."
            st.text(preview)
            st.caption(
                f"{len(st.session_state.cleaned_text):,} characters · "
                f"{len(st.session_state.chunks)} chunks · "
                f"{st.session_state.page_count} pages"
            )

    st.subheader("Summary")

    if st.button("Generate summary", type="primary", key="btn_summary"):
        if not st.session_state.chunks:
            st.warning("Upload a PDF with extractable text first.")
        else:
            try:
                use_section_aware = (
                    detailed_structured
                    and section_aware_structured
                )
                spinner_msg = (
                    "Generating section-aware structured summary "
                    "(retrieval + instruct model per field; may take several minutes)..."
                    if use_section_aware
                    else (
                        "Generating structured summary (multiple passes)..."
                        if detailed_structured
                        else "Generating summary (this may take a while on CPU)..."
                    )
                )
                with st.spinner(spinner_msg):
                    t0 = time.perf_counter()
                    tokenizer, model, device = get_summarizer()
                    embed_model = get_embedder()
                    instruct_gen = None
                    instruct_warning = None

                    if use_section_aware or (
                        use_instruct_qa and detailed_structured
                    ):
                        instruct_gen, instruct_warning = get_instruct_generator(
                            instruct_model_key
                        )
                        if instruct_warning:
                            st.warning(instruct_warning)

                    if st.session_state.vector_index is None:
                        st.session_state.vector_index = build_index(
                            st.session_state.chunks, embed_model
                        )

                    result = hierarchical_summarize(
                        st.session_state.chunks,
                        tokenizer,
                        model,
                        device,
                        summary_type=summary_type,
                        max_chunks=max_chunks,
                        settings=summary_settings,
                        detailed_structured=detailed_structured,
                        section_aware_structured=use_section_aware,
                        vector_index=st.session_state.vector_index,
                        embed_model=embed_model,
                        instruct_generator=instruct_gen,
                        structured_top_k=structured_top_k,
                    )
                    summarize_time = time.perf_counter() - t0

                if isinstance(result, tuple):
                    summary, structured = result
                    st.session_state.structured_result = structured
                else:
                    summary = result
                    st.session_state.structured_result = None

                metrics = evaluate_summary(
                    st.session_state.cleaned_text,
                    summary,
                    summarize_time,
                    strategy="full",
                )
                st.session_state.summary = summary
                st.session_state.summary_metrics = metrics
                rt = dict(st.session_state.get("last_runtime", {}))
                rt["summarization"] = summarize_time
                st.session_state.last_runtime = rt
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Summarization failed: {exc}")

    if st.session_state.summary:
        st.markdown(st.session_state.summary)

    structured_result: StructuredSummaryResult | None = (
        st.session_state.structured_result
    )
    if (
        show_field_sources
        and structured_result
        and structured_result.fields
    ):
        st.subheader("Sources by section")
        for field in structured_result.fields:
            label = f"Sources used for {field.field_name}"
            with st.expander(label, expanded=False):
                if not field.sources:
                    st.caption("No chunks retrieved for this field.")
                for i, src in enumerate(field.sources, start=1):
                    st.markdown(f"**Passage {i}** (distance: {src['score']:.4f})")
                    st.text(
                        src["text"][:1200]
                        + ("..." if len(src["text"]) > 1200 else "")
                    )

    if st.session_state.summary_metrics:
        with st.expander("Evaluation metrics", expanded=True):
            display_summary_metrics(st.session_state.summary_metrics)
            st.json(metrics_to_dict(st.session_state.summary_metrics))

        export_text = format_summary_export(
            st.session_state.summary,
            st.session_state.summary_metrics,
            paper_name=st.session_state.paper_name,
            extra_sections=(
                st.session_state.sections.sections
                if st.session_state.sections
                else None
            ),
        )
        st.download_button(
            label="Export summary to TXT",
            data=export_text,
            file_name="summary_export.txt",
            mime="text/plain",
        )

    st.divider()
    st.subheader("Ask a question about the paper")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What method did the authors propose?",
        key="qa_input",
    )

    if st.button("Get answer", key="btn_qa"):
        if not st.session_state.chunks:
            st.warning("Upload a PDF with extractable text first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching and generating answer..."):
                    t0 = time.perf_counter()
                    tokenizer, model, device = get_summarizer()
                    embed_model = get_embedder()

                    if st.session_state.vector_index is None:
                        st.session_state.vector_index = build_index(
                            st.session_state.chunks, embed_model
                        )

                    instruct_gen = None
                    if use_instruct_qa:
                        instruct_gen, warn = get_instruct_generator(
                            instruct_model_key
                        )
                        if warn:
                            st.warning(warn)

                    answer, sources = answer_question(
                        question,
                        st.session_state.vector_index,
                        embed_model,
                        tokenizer,
                        model,
                        device,
                        top_k=qa_top_k,
                        settings=qa_settings,
                        answer_length=answer_length,
                        use_instruct_qa=use_instruct_qa,
                        instruct_generator=instruct_gen,
                    )
                    qa_time = time.perf_counter() - t0

                st.session_state.qa_answer = answer
                st.session_state.qa_sources = sources
                st.session_state.qa_metrics = evaluate_qa(answer, sources, qa_time)
                rt = dict(st.session_state.get("last_runtime", {}))
                rt["qa_total"] = qa_time
                st.session_state.last_runtime = rt
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Q&A failed: {exc}")

    if st.session_state.qa_answer:
        st.markdown("**Answer**")
        st.write(st.session_state.qa_answer)

    if st.session_state.qa_metrics:
        qm = st.session_state.qa_metrics
        st.caption(
            f"Q&A: {qm['answer_words']} answer words · "
            f"{qm['num_sources']} sources · "
            f"{qm['runtime_seconds']:.2f}s"
        )

    if st.session_state.qa_sources:
        with st.expander("Sources", expanded=True):
            for i, src in enumerate(st.session_state.qa_sources, start=1):
                st.markdown(f"**Source {i}** (distance: {src['score']:.4f})")
                st.text(
                    src["text"][:1500]
                    + ("..." if len(src["text"]) > 1500 else "")
                )
                if i < len(st.session_state.qa_sources):
                    st.divider()

# ===================== Tab: Detected sections =====================
with tab_sections:
    sections: PaperSections | None = st.session_state.sections
    if not sections or not st.session_state.cleaned_text:
        st.info("Upload a PDF to detect sections.")
    else:
        st.write(
            f"**Coverage:** {sections.coverage_ratio * 100:.1f}% of document text "
            f"in named sections"
        )
        if sections.fallback_body:
            st.warning("Heuristic detection found no headings; full text stored as 'body'.")
        else:
            st.write(f"**Detected:** {', '.join(sections.detected)}")

        rows = []
        for key, text in sections.sections.items():
            rows.append(
                {
                    "Section": key,
                    "Characters": len(text),
                    "Words": len(text.split()),
                }
            )
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)

        for key, text in sections.sections.items():
            with st.expander(f"{key.title()} ({len(text):,} chars)", expanded=False):
                st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

# ===================== Tab: Comparison =====================
with tab_comparison:
    st.markdown(
        "Compare three summarization strategies for thesis experiments. "
        "This runs summarization **three times** and may take several minutes on CPU."
    )

    if st.button("Run comparison", type="primary", key="btn_comparison"):
        if not st.session_state.chunks:
            st.warning("Upload a PDF with extractable text first.")
        else:
            try:
                status = st.empty()
                tokenizer, model, device = get_summarizer()
                embed_model = get_embedder()

                if st.session_state.vector_index is None:
                    st.session_state.vector_index = build_index(
                        st.session_state.chunks, embed_model
                    )

                def progress(msg: str):
                    status.info(msg)

                t_total = time.perf_counter()
                comparison_results, comp_runtimes = run_comparison(
                    cleaned_text=st.session_state.cleaned_text,
                    chunks=st.session_state.chunks,
                    sections=st.session_state.sections or PaperSections(),
                    vector_index=st.session_state.vector_index,
                    embed_model=embed_model,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    summary_type=summary_type,
                    max_chunks=max_chunks,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    retrieval_top_k=qa_top_k,
                    settings=summary_settings,
                    detailed_structured=detailed_structured,
                    on_progress=progress,
                )
                status.empty()

                st.session_state.comparison_results = comparison_results
                rt = dict(st.session_state.get("last_runtime", {}))
                rt.update(comp_runtimes)
                rt["comparison_total"] = time.perf_counter() - t_total
                st.session_state.last_runtime = rt
                st.success("Comparison complete.")
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    for result in st.session_state.comparison_results:
        st.subheader(result.strategy.replace("_", " ").title())
        if result.summary:
            st.markdown(result.summary)
        else:
            st.warning("No summary produced for this strategy.")
        display_summary_metrics(result.metrics, prefix="")
        st.divider()

# ===================== Tab: Model Comparison =====================
with tab_model_comparison:
    st.markdown(
        "**Pretrained model:** a general model used without additional training.\n\n"
        "**Fine-tuned model:** the same pretrained model family further trained on "
        "scientific paper summarization data.\n\n"
        "This tab compares **pretrained T5-small** vs your **local fine-tuned T5** "
        "checkpoint on the **same PDF text and the same chunks**."
    )

    uploaded_cmp = st.file_uploader(
        "Upload PDF for model comparison",
        type=["pdf"],
        key="cmp_pdf",
        help="Uses the same PDF extraction as the main app. Scanned PDFs without OCR may fail.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        pretrained_model_name = st.text_input(
            "Pretrained model (Hugging Face id)",
            value=DEFAULT_PRETRAINED_MODEL,
            help="Recommended baseline for fairness: google-t5/t5-small",
        )
    with col_b:
        finetuned_path = st.text_input(
            "Fine-tuned model path (local folder)",
            value="training/checkpoints/t5-small-scientific/final",
            help="Folder containing config.json + model weights (saved by training scripts).",
        )

    cmp_max_new_tokens = st.slider(
        "Max new tokens (final summary)",
        min_value=64,
        max_value=256,
        value=128,
        step=16,
        help="Higher values produce longer summaries but use more time/memory.",
    )

    st.caption(
        f"Low-resource settings: batch size 1, inference-only, CPU fallback, "
        f"max chunks = {max_chunks} (set in sidebar)."
    )

    reference_summary = st.text_area(
        "Optional reference summary / abstract (for ROUGE)",
        value="",
        height=140,
        help="Paste a gold abstract or reference summary to compute ROUGE for both models.",
    )

    if st.button("Compare Models", type="primary", key="btn_model_compare"):
        st.session_state.model_comparison = None
        if uploaded_cmp is None:
            st.warning("Upload a PDF first.")
        else:
            try:
                raw_text, pages = extract_text_from_pdf(uploaded_cmp.read())
                cleaned = clean_text(raw_text)
                if not cleaned.strip():
                    raise ValueError(
                        "No extractable text found. This may be a scanned PDF without OCR."
                    )
                chunks_cmp = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
                if not chunks_cmp:
                    raise ValueError("PDF text was extracted, but chunking produced no chunks.")

                selected_chunks = chunks_cmp[:max_chunks]

                generation_chunk = {"max_new_tokens": 80, "num_beams": 4, "early_stopping": True}
                generation_final = {"max_new_tokens": int(cmp_max_new_tokens), "num_beams": 4, "early_stopping": True}

                results: dict[str, dict] = {}

                with st.spinner("Loading pretrained model..."):
                    pretrained = load_pretrained_model(pretrained_model_name.strip() or DEFAULT_PRETRAINED_MODEL)
                with st.spinner("Generating pretrained summary..."):
                    pre_summary, pre_stats = hierarchical_summarize_for_model(
                        selected_chunks,
                        tokenizer=pretrained.tokenizer,
                        model=pretrained.model,
                        device=pretrained.device,
                        max_chunks=len(selected_chunks),
                        generation_config_chunk=generation_chunk,
                        generation_config_final=generation_final,
                    )
                results["pretrained"] = {
                    "label": pretrained.source,
                    "device": pretrained.device,
                    "summary": pre_summary,
                    "runtime_seconds": pre_stats["runtime_seconds"],
                    "generated_tokens": int(pre_stats["generated_tokens"]),
                }

                finetuned_err = None
                try:
                    with st.spinner("Loading fine-tuned model..."):
                        finetuned = load_finetuned_model(finetuned_path.strip())
                    with st.spinner("Generating fine-tuned summary..."):
                        ft_summary, ft_stats = hierarchical_summarize_for_model(
                            selected_chunks,
                            tokenizer=finetuned.tokenizer,
                            model=finetuned.model,
                            device=finetuned.device,
                            max_chunks=len(selected_chunks),
                            generation_config_chunk=generation_chunk,
                            generation_config_final=generation_final,
                        )
                    results["finetuned"] = {
                        "label": finetuned.source,
                        "device": finetuned.device,
                        "summary": ft_summary,
                        "runtime_seconds": ft_stats["runtime_seconds"],
                        "generated_tokens": int(ft_stats["generated_tokens"]),
                    }
                except FileNotFoundError as exc:
                    finetuned_err = str(exc)
                except Exception as exc:
                    finetuned_err = f"Failed to load/run fine-tuned model: {exc}"

                if finetuned_err:
                    st.error(finetuned_err)

                # Metrics (word counts, compression, time, token approximations)
                original_words = word_count(cleaned)
                table_rows = []

                def _row(metric: str, pre_val: str, ft_val: str = ""):
                    table_rows.append(
                        {"Metric": metric, "Pretrained Model": pre_val, "Fine-tuned Model": ft_val}
                    )

                pre_words = word_count(results["pretrained"]["summary"])
                _row("Original word count", str(original_words), str(original_words))
                _row("Summary word count", str(pre_words), str(word_count(results.get("finetuned", {}).get("summary", ""))))
                _row(
                    "Compression ratio (summary/original)",
                    f"{(pre_words / original_words) if original_words else 0.0:.4f}",
                    (
                        f"{(word_count(results['finetuned']['summary']) / original_words) if (original_words and 'finetuned' in results) else 0.0:.4f}"
                    ),
                )
                _row(
                    "Generation time (s)",
                    f"{float(results['pretrained']['runtime_seconds']):.2f}",
                    (
                        f"{float(results['finetuned']['runtime_seconds']):.2f}"
                        if "finetuned" in results
                        else ""
                    ),
                )
                _row(
                    "Approx tokens generated",
                    str(int(results["pretrained"]["generated_tokens"])),
                    str(int(results["finetuned"]["generated_tokens"])) if "finetuned" in results else "",
                )

                rouge_pre = rouge_ft = None
                if reference_summary.strip():
                    try:
                        rouge_pre = calculate_rouge(results["pretrained"]["summary"], reference_summary)
                        if "finetuned" in results:
                            rouge_ft = calculate_rouge(results["finetuned"]["summary"], reference_summary)
                        _row("ROUGE-1 (F1)", f"{rouge_pre['rouge1']:.4f}", f"{rouge_ft['rouge1']:.4f}" if rouge_ft else "")
                        _row("ROUGE-2 (F1)", f"{rouge_pre['rouge2']:.4f}", f"{rouge_ft['rouge2']:.4f}" if rouge_ft else "")
                        _row("ROUGE-L (F1)", f"{rouge_pre['rougeL']:.4f}", f"{rouge_ft['rougeL']:.4f}" if rouge_ft else "")
                    except Exception as exc:
                        st.warning(f"ROUGE calculation failed: {exc}")

                st.session_state.model_comparison = {
                    "pages": pages,
                    "chunks_total": len(chunks_cmp),
                    "chunks_used": len(selected_chunks),
                    "pretrained": results["pretrained"],
                    "finetuned": results.get("finetuned"),
                    "metrics_table": table_rows,
                }
                st.success("Model comparison complete.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Model comparison failed: {exc}")

    cmp = st.session_state.get("model_comparison")
    if cmp:
        st.caption(
            f"Processed {cmp['pages']} pages · {cmp['chunks_total']} chunks (used {cmp['chunks_used']})."
        )
        left, right = st.columns(2)
        with left:
            st.subheader("Pretrained Model Summary")
            st.caption(f"Model: `{cmp['pretrained']['label']}` · Device: `{cmp['pretrained']['device']}`")
            st.markdown(cmp["pretrained"]["summary"] or "_No summary produced._")
        with right:
            st.subheader("Fine-tuned Model Summary")
            if cmp.get("finetuned"):
                st.caption(f"Model: `{cmp['finetuned']['label']}` · Device: `{cmp['finetuned']['device']}`")
                st.markdown(cmp["finetuned"]["summary"] or "_No summary produced._")
            else:
                st.info("Fine-tuned model not available (check the path and retry).")

        st.subheader("Comparison table")
        st.dataframe(cmp["metrics_table"], width="stretch", hide_index=True)

# ===================== Tab: Thesis metrics =====================
with tab_metrics:
    st.subheader("Latest summary metrics")
    if st.session_state.summary_metrics:
        m = st.session_state.summary_metrics
        display_summary_metrics(m)
        metrics_rows = [
            {"Metric": "Strategy", "Value": m.strategy},
            {"Metric": "Source words", "Value": str(m.source_words)},
            {"Metric": "Summary words", "Value": str(m.summary_words)},
            {"Metric": "Compression ratio", "Value": f"{m.compression_ratio:.4f}"},
            {"Metric": "Reduction %", "Value": f"{m.compression_percent:.1f}"},
            {"Metric": "Runtime (s)", "Value": f"{m.runtime_seconds:.2f}"},
        ]
        st.dataframe(metrics_rows, width="stretch", hide_index=True)
    else:
        st.info("Generate a summary to see metrics.")

    st.subheader("Q&A metrics")
    if st.session_state.qa_metrics:
        st.json(st.session_state.qa_metrics)
    else:
        st.info("Ask a question to see Q&A metrics.")

    st.subheader("Comparison overview")
    if st.session_state.comparison_results:
        comp_table = {
            "Strategy": [],
            "Source words": [],
            "Summary words": [],
            "Reduction %": [],
            "Runtime (s)": [],
        }
        for r in st.session_state.comparison_results:
            comp_table["Strategy"].append(r.strategy)
            comp_table["Source words"].append(r.metrics.source_words)
            comp_table["Summary words"].append(r.metrics.summary_words)
            comp_table["Reduction %"].append(
                f"{r.metrics.compression_percent:.1f}"
            )
            comp_table["Runtime (s)"].append(
                f"{r.metrics.runtime_seconds:.2f}"
            )
        st.dataframe(comp_table, width="stretch", hide_index=True)
    else:
        st.info("Run comparison mode to see strategy metrics side by side.")

    st.subheader("Section detection")
    sections = st.session_state.sections
    if sections and st.session_state.cleaned_text:
        st.write(f"Coverage ratio: **{sections.coverage_ratio * 100:.1f}%**")
        st.write(f"Sections found: **{', '.join(sections.detected) or 'none'}**")
    else:
        st.info("Upload a PDF to see section detection stats.")

    st.subheader("All runtimes")
    rt = st.session_state.get("last_runtime", {})
    if rt:
        for label, seconds in rt.items():
            st.write(f"- **{label}:** {seconds:.2f}s")
    else:
        st.info("No timing data yet.")

# --- Footer ---
if st.session_state.get("last_runtime"):
    st.divider()
    cols = st.columns(4)
    device = get_device()
    cols[0].metric("Device", device.upper())
    cols[1].metric("Chunks", len(st.session_state.chunks))
    cols[2].metric("Pages", st.session_state.page_count)
    total_time = sum(st.session_state.last_runtime.values())
    cols[3].metric("Total tracked time", f"{total_time:.1f}s")
