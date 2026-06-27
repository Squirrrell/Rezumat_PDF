# Local AI Research Paper Summarizer

A lightweight, fully local web app for summarizing research PDFs and asking questions about their content. Designed for modest hardware and suitable as an implementation artifact for a **bachelor or license thesis** in computer science or related fields.

## Purpose

Upload a text-based PDF research paper and get:

- Short, detailed, or bullet-point summaries (hierarchical summarization)
- Semantic Q&A over the document with cited source chunks
- **Thesis-oriented evaluation:** word counts, compression ratio, runtime metrics
- **Section detection:** Abstract, Introduction, Methodology, Results, Conclusion
- **Test cards:** auto-generated comprehension questions with keyword-based scoring
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

On first run, Marker and Hugging Face will download models (several GB total). You can set `HF_HOME` to control the Hugging Face cache location.

### GPU acceleration (recommended for Marker)

Marker PDF conversion is **much faster on an NVIDIA GPU**. The default `pip install torch` from PyPI is often **CPU-only**, which makes `/api/health` report `"device":"cpu"`.

After installing `backend/requirements.txt`, reinstall PyTorch with CUDA (example for CUDA 12.4):

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Copy [`.env.example`](.env.example) to `.env` or set environment variables before starting uvicorn:

```powershell
$env:TORCH_DEVICE = "cuda"
# Only if you must run without a GPU (slow):
# $env:ML_ALLOW_CPU = "1"
```

Restart the backend after changing PyTorch builds or Marker settings.

**Marker `fast` profile (default)** loads fewer models and skips heavy table/equation processors — best for digital PDFs on 4 GB laptop GPUs. Set `MARKER_PROFILE=quality` for full Marker table/equation formatting (slower). OCR and image extraction stay off by default (`MARKER_DISABLE_OCR=1`, `MARKER_EXTRACT_IMAGES=0`). On GPUs with ≤6 GB VRAM, fast mode uses conservative batch sizes automatically; override with `MARKER_LAYOUT_BATCH_SIZE` if needed.


## How to run

The app uses a **FastAPI backend** and a **React frontend** (Vite + Tailwind).

### Backend (API)

From the project root:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`

### Backend with OpenAI (optional)

To run all LLM tasks (brief summary, comprehensive summary, Q&A, test cards) via OpenAI instead of local models:

1. Copy `.env.example` to `.env` and set:
   - `SQUIRRELAI_LLM_BACKEND=openai`
   - `OPENAI_API_KEY=your-key-here`
   - Optional: `OPENAI_MODEL=gpt-4o-mini` or `gpt-5-mini` (API params are selected automatically)
2. Install dependencies (includes `openai`, `python-dotenv`).
3. From the project root:

```powershell
.\scripts\run-backend-openai.ps1
```

The UI is unchanged; embeddings, FAISS, and PDF ingest still run locally. Never commit `.env` or API keys to git.

### Frontend (UI)

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` — the Vite dev server proxies `/api` to the backend.

### App tabs

| Tab | Description |
|-----|-------------|
| **Summarize & Q&A** | Generate summary, metrics, TXT export, ask questions |
| **Detected sections** | Heuristic section split with previews |
| **Test cards** | Generate comprehension questions from the paper and verify your answers |
| **Thesis metrics** | Aggregated metrics and runtime breakdown |

## Models used

| Model | Role | Approx. size |
|-------|------|----------------|
| [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) | Chunk / simple summarization | ~250M params |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Structured fields + optional Q&A | ~0.5B params |
| [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) | Fallback structured / Q&A | ~80M params |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Chunk embeddings for search | ~80M params |

Summarization uses **hierarchical** passes: each text chunk is summarized, summaries are combined, then a final summary is produced. Q&A retrieves the most relevant chunks with FAISS and summarizes them in the context of your question.

## Test cards

The **Test cards** tab helps you check how well you understood a paper:

1. Upload a PDF and open **Test cards**.
2. Click **Generate question cards** (3–8 cards; default 5).
3. The instruct model (Qwen or FLAN) creates questions from retrieved PDF passages.
4. Write your answers in each card.
5. Click **Verify answers** to see a **score per card** (0–100%) based on how many key phrases your answer includes.

Reference answers and key phrases stay on the server; only matched/missed phrases are shown after verification.

Optional T5 fine-tuning experiments remain available offline under [`training/`](training/README_training.md) (not exposed in the UI).

## Project architecture

```
PDF upload
    → PyMuPDF extract text
    → clean & chunk text
    → detect sections (heuristic)
    → ┌─ hierarchical summarizer (distilbart)
      ├─ test cards: retrieve chunks → instruct model → Q&A cards → keyword scoring
      └─ embed chunks → FAISS index
              → user question → top-k chunks → summarizer → answer + sources
    → evaluation metrics (words, compression, runtime)
```

```
Rezumat_PDF/
├── backend/
│   ├── main.py            # FastAPI app + CORS
│   ├── requirements.txt
│   ├── routers/           # API endpoints
│   ├── session_store.py   # In-memory document sessions
│   └── model_cache.py     # Cached ML models
├── frontend/
│   ├── src/App.jsx        # React UI (Vite + Tailwind)
│   └── src/api/client.js  # API client (axios)
├── requirements.txt       # Points to backend/requirements.txt
├── README.md
└── src/                   # Core ML/PDF logic (unchanged)
    ├── pdf_utils.py
    ├── text_utils.py
    ├── summarizer.py
    ├── vector_store.py
    ├── qa_system.py
    ├── section_detector.py
    ├── evaluation.py
    ├── test_cards.py
    ├── structured_summary.py
    ├── instruct_generator.py
    └── model_manager.py
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

## Documentation

- **[TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)** — detailed tools, `src/` modules, API endpoints, and logic flows (with diagrams)
- **[APP_OVERVIEW.md](APP_OVERVIEW.md)** — shorter architectural overview
- **[z_documentatie/TOOLURI_ML.md](z_documentatie/TOOLURI_ML.md)** — explicații detaliate ML/AI în română (FAISS, MiniLM, DistilBART, Qwen, Marker etc.)

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
5. **Retrieval:** `all-MiniLM-L6-v2` embeds chunks; FAISS `IndexFlatL2` finds the top-k passages for Q&A and test cards.
6. **Q&A:** Retrieved passages are summarized in the context of the user question (retrieve-and-summarize, not generative QA).
7. **Test cards:** Instruct model generates comprehension questions from retrieved context; user answers scored by key-phrase overlap.
8. **Evaluation:** Intrinsic metrics computed in `evaluation.py` without reference summaries.

### Experiments

**Suggested thesis protocol:**

1. Collect 5–10 text-based PDFs (e.g. arXiv papers with standard section headings).
2. For each paper, generate a summary and record metrics from the **Thesis metrics** tab.
3. Use **Test cards** to measure comprehension (average score across cards).
4. Optionally export summaries via **Export summary to TXT**.

**Example results table (fill in your runs):**

| Paper | Summary type | Source words | Summary words | Reduction % | Test cards avg % |
|-------|--------------|--------------|---------------|-------------|------------------|
| paper1.pdf | short | | | | |
| paper1.pdf | detailed | | | | |

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

- **OCR disabled by default:** Scanned/image-only PDFs need `MARKER_DISABLE_OCR=0` in `.env` (slower).
- **Heuristic sections:** Non-standard layouts, two-column PDFs, or missing headings reduce section detection accuracy.
- **General-domain summarizer:** DistilBART is trained on news (CNN/DailyMail), not scientific prose; terminology may be simplified or lost.
- **Q&A is not true reasoning:** Answers are abstractive summaries of retrieved chunks, not grounded generative QA.
- **Test card scoring:** Keyword overlap is a simple heuristic; it does not measure semantic correctness fully.
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

- **Scanned PDFs** without a text layer need `MARKER_DISABLE_OCR=0`; OCR is off by default for speed.
- **Q&A** uses retrieval + summarization, not a chat LLM; answers work best for factual questions about paper content.
- First summary on a long paper can take several minutes on CPU.

## License

Use and modify freely for learning and research.
