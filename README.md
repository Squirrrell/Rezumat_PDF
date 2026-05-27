# Local AI Research Paper Summarizer

A lightweight, fully local web app for summarizing research PDFs and asking questions about their content. Designed for modest hardware and suitable as an implementation artifact for a **bachelor or license thesis** in computer science or related fields.

## Purpose

Upload a text-based PDF research paper and get:

- Short, detailed, or bullet-point summaries (hierarchical summarization)
- Semantic Q&A over the document with cited source chunks
- **Thesis-oriented evaluation:** word counts, compression ratio, runtime metrics
- **Section detection:** Abstract, Introduction, Methodology, Results, Conclusion
- **Comparison mode:** full document vs. intro+conclusion vs. retrieved chunks
- Export summaries to TXT for your thesis appendix

Everything runs on your machine using small pretrained models.

## Hardware limitations

This project targets:

- **4 GB VRAM** (optional GPU; CPU works fine)
- **24 GB RAM**
- **Local-only** inference

Models are kept small (~330M parameters total) to stay within these limits. Very long papers are capped by a configurable maximum number of chunks to summarize.

## Installation

1. Clone or copy this project folder.
2. Create a virtual environment (recommended):

```bash
python -m venv venv
```

3. Activate it:

- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

4. Install dependencies:

```bash
pip install -r requirements.txt
```

On first run, Hugging Face and sentence-transformers will download models (~1 GB total). You can set `HF_HOME` to control the cache location.

## How to run

From the project root:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### App tabs

| Tab | Description |
|-----|-------------|
| **Summarize & Q&A** | Generate summary, metrics, TXT export, ask questions |
| **Detected sections** | Heuristic section split with previews |
| **Comparison** | Run three summarization strategies for experiments |
| **Model Comparison** | Compare pretrained T5-small vs a local fine-tuned T5 checkpoint side by side |
| **Thesis metrics** | Aggregated metrics and runtime breakdown |

## Models used

| Model | Role | Approx. size |
|-------|------|----------------|
| [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) | Chunk / simple summarization | ~250M params |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Structured fields + optional Q&A | ~0.5B params |
| [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) | Fallback structured / Q&A | ~80M params |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Chunk embeddings for search | ~80M params |

Summarization uses **hierarchical** passes: each text chunk is summarized, summaries are combined, then a final summary is produced. Q&A retrieves the most relevant chunks with FAISS and summarizes them in the context of your question.

## Comparing Pretrained vs Fine-tuned Models

This app includes a **Model Comparison** tab for a thesis-style evaluation of:

- **Pretrained (baseline)**: `google-t5/t5-small`
- **Fine-tuned**: a local T5 checkpoint folder (same architecture, trained on scientific paper summarization)

### Where to put the fine-tuned model

Point the UI at any Hugging Face-style checkpoint directory that contains a `config.json`, for example:

- `training/checkpoints/t5-small-scientific/final/` (produced by the optional training scripts)
- `./models/fine_tuned_t5_small/` (your own exported folder)

### How to compare

1. Open the **Model Comparison** tab.
2. Upload the same PDF you want to evaluate.
3. Set the fine-tuned folder path.
4. Click **Compare Models**.

Both models use the **same extracted text** and the **same chunks** (limited by the sidebar “Max chunks to summarize”) so the comparison is fair.

### Metrics reported

- **Word counts**: original vs each summary
- **Compression ratio**: \(\\text{summary words} / \\text{original words}\\)
- **Generation time** per model
- **Approx. tokens generated** (from generation output length)
- **Optional ROUGE-1/2/L** if you paste a reference abstract/summary

This is useful for a bachelor/license thesis because it provides a controlled baseline-vs-adapted comparison on the same document under low-resource constraints.

## Project architecture

```
PDF upload
    → PyMuPDF extract text
    → clean & chunk text
    → detect sections (heuristic)
    → ┌─ hierarchical summarizer (distilbart)
      ├─ comparison: full / intro+conclusion / retrieved
      └─ embed chunks → FAISS index
              → user question → top-k chunks → summarizer → answer + sources
    → evaluation metrics (words, compression, runtime)
```

```
Rezumat_PDF/
├── app.py                 # Streamlit UI
├── requirements.txt
├── README.md
└── src/
    ├── pdf_utils.py       # PDF text extraction
    ├── text_utils.py      # Cleaning and chunking
    ├── summarizer.py      # Hierarchical summarization
    ├── vector_store.py    # Embeddings + FAISS
    ├── qa_system.py       # Question answering
    ├── section_detector.py # Heuristic section detection
    ├── evaluation.py      # Metrics and export
    ├── comparison.py      # Three-strategy comparison
    ├── structured_summary.py  # Section-aware structured summaries
    └── instruct_generator.py  # Qwen / FLAN for structured Q&A
```

## Sidebar settings

### Summarization

- **Summary type:** short, detailed, or bullet points
- **Basic summarizer:** `distilbart-cnn-12-6` (chunk-level and simple summaries)
- **Structured summarizer:** `Qwen2.5-0.5B-Instruct` (default) or `flan-t5-small` fallback
- **Detailed structured format:** multi-section paper layout
- **Use section-aware structured summary:** different FAISS retrieval query per field (recommended; reduces repetition)
- **Show sources for each structured section:** expanders with chunks used per field
- **Chunks per structured section:** top-k retrieval per field (default 5)
- **Max / min summary length (tokens):** override DistilBART final-pass length
- **Chunk size / overlap:** how the PDF is split
- **Max chunks to summarize:** limits work on long papers

### Q&A

- **Answer length:** short, medium, or long
- **Retrieved chunks for Q&A:** how many source passages to use (default **5**)
- **Use instruct model for Q&A:** Qwen or FLAN instead of DistilBART (grounded on retrieved chunks only)

## Section-aware structured summary

Older structured mode fed the **same combined text** into every section using DistilBART, which often repeated one generic sentence across all fields.

**Section-aware mode** (checkbox on by default when structured format is enabled):

1. Keeps **MiniLM + FAISS** for embeddings and search.
2. For each field (Title/Topic, Main idea, Method, …), runs a **different semantic query** and retrieves top-k chunks.
3. Generates that field only from its retrieved context using **Qwen2.5-0.5B-Instruct** (or **flan-t5-small** if Qwen fails or is selected).
4. Skips weak retrieval (high L2 distance) with: `Not clearly specified in the document.`
5. **Duplicate detection:** if two fields are >75% similar (`difflib`), regenerates or marks missing.

| Field | Retrieval query (abbreviated) |
|-------|----------------------------|
| Title / Topic | title, abstract, topic, subject |
| Main idea | main contribution, overview, introduction |
| Problem addressed | problem, motivation, gap, challenge |
| Proposed method | method, architecture, algorithm, framework |
| Dataset / experiments | dataset, benchmark, training, evaluation setup |
| Results | results, findings, performance, accuracy |
| Limitations | limitations, threats, future work |
| Conclusion | conclusion, implications, summary |
| Key takeaways | main findings, contributions |

**Comparison mode** still uses DistilBART hierarchical summaries (faster; no 9× instruct calls).

## Controlling output length

The app uses **sshleifer/distilbart-cnn-12-6** with explicit generation settings (`num_beams=4`, `no_repeat_ngram_size=3`, configurable `max_length` / `min_length`). No larger model is required.

| Summary type | Default max tokens | Default min tokens | Output style |
|--------------|-------------------|-------------------|--------------|
| short | 220 | 80 | Single hierarchical summary, longer than before |
| detailed | 512 | 150 | Long summary; enable **Detailed structured format** for sectioned output |
| bullet points | 550 | 120 | Up to 12 bullet points after final pass |

**Hierarchical flow:** each chunk is summarized with higher intermediate limits (max 200 tokens), combined summaries are merged less aggressively for detailed mode, then a final pass (or structured multi-section passes) produces the result.

**Legacy structured mode** (structured format without section-aware) still uses DistilBART on the same combined text per field. **Section-aware structured mode** uses per-field retrieval + Qwen/FLAN and is recommended for thesis output.

**Q&A answers** are generated only from retrieved chunks. The prompt instructs the model not to add unsupported information. If context is thin or retrieval scores are weak, a short **limitation notice** is prepended. Always verify claims against the **Sources** expander.

**Performance:** longer settings increase CPU/GPU time. Lower **Max chunks to summarize** for faster runs. Structured detailed mode is the slowest option (multiple passes).

---

## Thesis documentation

This section frames the project for academic writing (problem, method, experiments, metrics).

### Problem statement

Researchers and students face **information overload** when reading scientific papers. Long PDF documents are time-consuming to parse, and cloud-based large language models raise **privacy**, **cost**, and **hardware** concerns. This project addresses the need for a **local**, **resource-efficient** tool that can summarize research papers and support basic question answering without training custom models or requiring high-end GPUs.

### Methodology

1. **Document ingestion:** Text is extracted from PDFs with PyMuPDF (text-layer PDFs only).
2. **Preprocessing:** Whitespace normalization, artifact removal, and overlapping character-based chunking.
3. **Section detection:** Regex-based heading detection splits the paper into Abstract, Introduction, Methodology, Results, and Conclusion where headings match common academic patterns.
4. **Summarization:** `distilbart-cnn-12-6` performs **hierarchical summarization** (per-chunk → combine → final pass) with configurable output length.
5. **Retrieval:** `all-MiniLM-L6-v2` embeds chunks; FAISS `IndexFlatL2` finds the top-k passages for Q&A and for the retrieved-only comparison strategy.
6. **Q&A:** Retrieved passages are summarized in the context of the user question (retrieve-and-summarize, not generative QA).
7. **Evaluation:** Intrinsic metrics computed in `evaluation.py` without reference summaries.

### Experiments

Use the **Comparison** tab to run three strategies on the same paper:

| Strategy | Description |
|----------|-------------|
| `full` | Hierarchical summary over the first N document chunks |
| `intro_conclusion` | Summary of detected Introduction + Conclusion (fallback: first/last 20% of text) |
| `retrieved_only` | Summary of top-k chunks retrieved with a fixed semantic query |

**Suggested thesis protocol:**

1. Collect 5–10 text-based PDFs (e.g. arXiv papers with standard section headings).
2. For each paper, run Comparison mode and record metrics from the **Thesis metrics** tab.
3. Optionally export summaries via **Export summary to TXT**.
4. Compare strategies by compression ratio, runtime, and qualitative readability.

**Example results table (fill in your runs):**

| Paper | Strategy | Source words | Summary words | Reduction % | Runtime (s) |
|-------|----------|--------------|---------------|-------------|-------------|
| paper1.pdf | full | | | | |
| paper1.pdf | intro_conclusion | | | | |
| paper1.pdf | retrieved_only | | | | |

### Metrics

**Intrinsic (implemented in the app):**

| Metric | Definition |
|--------|------------|
| Source words | Word count of input text for the strategy |
| Summary words | Word count of generated summary |
| Compression ratio | `summary_words / source_words` |
| Reduction % | `(1 - compression_ratio) × 100` |
| Runtime | Wall-clock seconds per operation |
| Section coverage | Fraction of document text assigned to detected sections |
| Q&A metrics | Answer length, number of sources, average source length, Q&A runtime |

**Extrinsic (future work):**

- **ROUGE-1/2/L** against gold abstracts from the [scientific_papers](https://huggingface.co/datasets/scientific_papers) dataset
- Human evaluation (readability, factual correctness)

### Limitations

- **No OCR:** Scanned/image-only PDFs are not supported.
- **Heuristic sections:** Non-standard layouts, two-column PDFs, or missing headings reduce section detection accuracy.
- **General-domain summarizer:** DistilBART is trained on news (CNN/DailyMail), not scientific prose; terminology may be simplified or lost.
- **Q&A is not true reasoning:** Answers are abstractive summaries of retrieved chunks, not grounded generative QA.
- **Comparison cost:** Running all three strategies triples summarization time.
- **No reference-based evaluation in-app:** ROUGE requires gold summaries and is deferred to the fine-tuning phase.

### Future work: fine-tuning T5-small

An optional starter experiment lives in [`training/README_training.md`](training/README_training.md) (fine-tune **google-t5/t5-small** on a `scientific_papers` subset + ROUGE evaluation; **not required** to run the app).

Planned extensions for a follow-up thesis chapter or MSc work:

1. Fine-tune **[T5-small](https://huggingface.co/t5-small)** on the `scientific_papers` dataset (PubMed and arXiv splits).
2. Integrate the fine-tuned checkpoint as an optional summarizer in the app.
3. Evaluate with **ROUGE** against reference abstracts.
4. **Compare** pretrained distilbart vs. fine-tuned T5 on the same test set (compression, ROUGE, runtime).
5. Optional: OCR pipeline for scanned PDFs.

---

## Limitations (operational)

- **Scanned PDFs** without a text layer will fail (no OCR).
- **Q&A** uses retrieval + summarization, not a chat LLM; answers work best for factual questions about paper content.
- First summary on a long paper can take several minutes on CPU.

## License

Use and modify freely for learning and research.
