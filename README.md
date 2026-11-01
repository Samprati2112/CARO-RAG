<<<<<<< HEAD
# CARO RAG: Context-Aware Adaptive Retrieval Orchestration Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LLM: Ollama](https://img.shields.io/badge/LLM-Ollama%20(gemma3:4b)-orange.svg)](https://ollama.com/)
[![Embeddings: BGE-M3](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--m3-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![Tests: Pytest](https://img.shields.io/badge/Tests-54%20Passing-brightgreen.svg)](tests/)

**CARO RAG** is an academic research and production-grade conversational Retrieval-Augmented Generation (RAG) framework designed to overcome the critical failure modes of conventional dense RAG architectures: **blind retrieval overhead**, **topical distractor hallucinations**, **citation fabrication**, and **conversational memory decay**.

The project includes an **Interactive Comparative Evaluation Studio (Web UI)** allowing side-by-side benchmarking of **Baseline Dense RAG vs. CARO RAG** across multi-document uploads (PDF, DOCX, PPTX, TXT, CSV, JSON) and standardized 500-question HotpotQA evaluation sets.

---

## Key Highlights

- **Adaptive Query Routing:** Intelligently routes queries into `DIRECT` (definitional/general knowledge bypass), `MEMORY` (conversational follow-ups), or `RETRIEVE` (corpus/document search), cutting unnecessary retrieval latency by up to 70%.
- **Hybrid Retrieval Strategy:** Dynamically chooses between Dense (BGE-M3 + FAISS), Sparse (BM25), and Hybrid retrieval using Reciprocal Rank Fusion (RRF).
- **Aspect-Level Evidence Sufficiency Gate:** Analyzes retrieved passages at the factual aspect level before generation. Halts hallucination on missing or incomplete information with honest abstention verdicts (`SUFFICIENT`, `PARTIAL`, `INSUFFICIENT`, `CONFLICTING`, `AMBIGUOUS`).
- **Claim-Level Citation Grounding:** Verifies every claim against exact source document IDs and passages, tagging factual provenance (`CORPUS_GROUNDED`, `MEMORY_GROUNDED`, `INFERRED`, `UNSUPPORTED`) and stripping hallucinated citations.
- **Cross-Turn Conversational Memory:** Implements structured `FactRecord` tracking with turn and source provenance, eliminating recency bias during multi-hop anaphoric follow-up resolution.
- **Interactive Comparative Studio:** Side-by-side web application (`http://localhost:8000`) for real-time document upload, evidence inspection, latency breakdown, and live RAG comparison.

---

## System Architecture

```
                       [Incoming User Query]
                                 │
                                 ▼
                 [Module 3: Adaptive Query Router]
                  /              │              \
          DIRECT /        MEMORY │               \ RETRIEVE
                ▼                ▼                ▼
        [Direct Answer]   [Module 6: Memory]  [Module 4: Strategy Selector]
                                 │                 /         │         \
                                 │          Dense /   Sparse │          \ Hybrid
                                 │               ▼           ▼           ▼
                                 │       [Module 5: Hybrid Retrieval Engine]
                                 │          (BGE-M3 + FAISS / BM25 / RRF)
                                 │                           │
                                 │                           ▼
                                 │          [Module 7: Evidence Sufficiency Gate]
                                 │           - Aspect decomposition
                                 │           - Distractor rejection
                                 │           - Multi-doc combination
                                 │           - Sufficiency verdict
                                 │                           │
                                 └───────────────┬───────────┘
                                                 │
                                                 ▼
                              [Module 8: Grounded Answer Synthesis]
                                - Claim-level citation validation
                                - Hallucination prevention
                                - Factual provenance tagging
                                                 │
                                                 ▼
                                        [Verified Response]
```

---

## 8-Stage Modular Pipeline

| Module | Name | Purpose |
| :--- | :--- | :--- |
| **Module 1** | **Data Processing** | Ingestion, chunking, and metadata extraction across heterogeneous documents (PDF, DOCX, PPTX, TXT, CSV, JSON) and benchmark corpora (HotpotQA, QReCC). |
| **Module 2** | **Baseline Dense RAG** | Standard dense retrieval reference implementation utilizing FAISS vector search and dense embeddings for head-to-head empirical benchmarking. |
| **Module 3** | **Adaptive Query Router** | Classifies incoming questions into `DIRECT`, `MEMORY`, or `RETRIEVE` routes based on query characteristics and history. |
| **Module 4** | **Retrieval Strategy Selector** | Selects optimal retrieval modalities (`DENSE`, `SPARSE_BM25`, or `HYBRID`) conditioned on query entity density and specificity. |
| **Module 5** | **Hybrid Retrieval Engine** | Executes dense vector search (BGE-M3 + FAISS) and sparse keyword retrieval (BM25) combined via Reciprocal Rank Fusion (RRF). |
| **Module 6** | **Conversational Memory** | Tracks entities across multi-turn sessions with structured `FactRecord` provenance, eliminating recency bias. |
| **Module 7** | **Evidence Sufficiency Analyzer** | Evaluates whether retrieved evidence is sufficient to answer the prompt before generation, detecting distractors and issuing explicit verdicts. |
| **Module 8** | **Grounded Answer Synthesis** | Generates verified responses with strict claim-level citations and enforces honest abstention when evidence is insufficient. |

---

## Interactive Comparative Studio (Web UI)

The project includes an interactive web application that provides real-time comparative analysis between **Baseline Dense RAG** and **CARO RAG**:


### Features:
- **Side-by-Side Dual Engine:** Runs queries through both Baseline RAG and CARO RAG concurrently, highlighting differences in answer grounding and evidence processing.
- **Multi-Document Ingestion:** Drag-and-drop or select multiple documents (`.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`, `.md`, `.json`). Automatically parses and creates runtime search indices.
- **Evidence Sufficiency Badges:** Real-time visual feedback on evidence sufficiency (`SUFFICIENT`, `PARTIAL`, `INSUFFICIENT`).
- **Telemetry & Metrics:** Displays latency (retrieval vs. generation), token counts, source passage references, and query routing explanations.

### Launching the Studio:
```bash
python server.py
# Alternatively:
python run_ui.py
```
Open **`http://localhost:8000`** in any web browser.

---

## Benchmark Evaluation & Results

The repository includes pre-computed head-to-head benchmark evaluations across **500 HotpotQA queries (Seed 42)** comparing Baseline Dense RAG against CARO RAG:

| Metric | Baseline Dense RAG | CARO RAG | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Answerable Fact Recall / EM** | 41.2% | **73.8%** | **+32.6%** |
| **Citation Precision** | 34.6% | **94.2%** | **+59.6%** |
| **Unanswerable Abstention Accuracy** | 12.0% | **91.4%** | **+79.4%** |
| **Hallucination Prevention Rate** | 22.4% | **96.8%** | **+74.4%** |
| **Average End-to-End Latency** | 2.14s | **1.38s** | **35.5% faster** (via adaptive routing) |

Evaluation datasets, results, and generated figures are located in:
- `eval/results/CARO_ANSWERABLE_seed42.jsonl`: 500-question CARO evaluation outputs
- `eval/results/B1_Dense_ANSWERABLE_seed42.jsonl`: 500-question Baseline evaluation outputs
- `eval/results/testv2_comparison.csv`: Reconciled metrics comparison table
- `eval/results/figures/`: High-resolution comparative visualization charts

---

## Installation & Setup

### Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Python 3.10 through 3.12 (recommended)
- **Ollama:** Running locally with `gemma3:4b`

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/CARO_RAG.git
cd CARO_RAG
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Ollama and Pull the Model
```bash
# In a separate terminal:
ollama serve

# Pull the model:
ollama pull gemma3:4b
```

---

## Quick Start & Usage

### 1. Run the Interactive Web Studio
```bash
python server.py
```
Navigate to `http://localhost:8000`. You can upload custom PDFs, DOCX, or PPTX documents (such as the provided sample documents) and submit test questions.

### 2. Run the Benchmark Evaluations
```bash
# Run full comparative evaluation suite:
python run_all_evaluations.py

# Run targeted evaluation script:
python run_fixed_evaluation.py
```

### 3. Run the Automated Test Suite
```bash
# Run all 54 unit and regression tests:
pytest -v

# Run targeted reliability regression suite:
pytest -v tests/test_integrated_reliability_fixes.py
```

### 4. Explore via Jupyter Notebook
```bash
jupyter notebook CARO_RAG.ipynb
```

---

## Repository Structure

```
CARO_RAG/
├── config/                     # System configuration and hyperparameters
│   └── config.py               # Path definitions, top-k, and threshold settings
├── modules/                    # 8 Core Architectural Modules
│   ├── module1_data_processing/
│   ├── module2_baseline_rag/
│   ├── module3_adaptive_query_router/
│   ├── module4_retrieval_strategy/
│   ├── module5_retrieval/
│   ├── module6_temporary_memory/
│   ├── module7_evidence_sufficiency/
│   └── module8_answer_generation/
├── eval/                       # Benchmark runners, schemas, and test harnesses
│   ├── data/                   # 500-question HotpotQA evaluation sets
│   ├── results/                # Evaluation outputs, JSONL records, CSVs, and figures
│   │   └── figures/            # Comparative analysis performance charts
│   ├── caro_core.py            # Evaluation core logic
│   └── runner.py               # Batch benchmark executor
├── ui/                         # Web Studio Frontend
│   ├── index.html              # Studio interface markup
│   ├── app.js                  # Frontend state management & live charts
│   └── style.css               # Styling and layout
├── tests/                      # Pytest unit and regression test suite
├── server.py                   # High-performance FastAPI server with multi-doc ingestion
├── run_ui.py                   # One-click launcher for the Web Studio
├── run_all_evaluations.py      # Master benchmark evaluation orchestrator
├── run_fixed_evaluation.py     # Targeted evaluation script
├── question.txt                # Reference catalog of HotpotQA questions
├── implementation_report.md    # Technical report on architectural diagnoses and fixes
├── thesis_correction_report.md # Thesis revision notes and theoretical grounding
├── PROJECT_COMPLETION.md       # Project freeze and milestone documentation
├── requirements.txt            # Python dependencies (verified)
├── .gitignore                  # Git exclusions for large index binaries (>100MB)
└── LICENSE                     # MIT License
```

> **Note on Large FAISS Vector Indices:**
> Per GitHub's 100 MB file size limit, raw pre-computed 250,000-document FAISS index binaries (`*.faiss`, `*.jsonl`) are excluded from this repository via `.gitignore`.
> The system operates immediately out-of-the-box using the **Multi-Document Ingestion Engine** (uploading PDF/DOCX/PPTX directly in the Web UI). To build or rebuild full-corpus FAISS indices, execute the ingestion pipeline in `modules/module1_data_processing` and `modules/module2_baseline_rag`.

---

## Academic Context & Credits

- **Project:** Adaptive Retrieval Orchestration for Retrieval-Augmented Generation using the CARO Framework in LLMs
- **Guide:** Dr. Rabindra Kumar Singh
- **Presenter / Author:** Samprati (25MCA1059)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
