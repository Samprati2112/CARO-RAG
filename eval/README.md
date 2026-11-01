# CARO RAG Evaluation Harness

A rigorous, fair, and statistically validated benchmarking suite comparing **CARO RAG** against baseline Retrieval-Augmented Generation (RAG) architectures on the 250,000-document HotpotQA corpus.

---

## 1. Systems Evaluated

All systems implement a common interface: `system.answer(question: str, excluded_doc_ids: Optional[Set[str]] = None) -> Dict[str, Any]`

| System | Retrieval Mechanism | Query Strategy | Abstention Logic | Context Size |
| :--- | :--- | :--- | :--- | :--- |
| **B1: Dense Only** | FAISS `BAAI/bge-m3` | Single primary query | Prompt standard | 8 documents |
| **B2: BM25 Only** | Title-boosted inverted index | Single primary query | Prompt standard | 8 documents |
| **B3: Hybrid RRF** | Dense + BM25 Reciprocal Rank Fusion | Single primary query (no LLM in retrieval) | Prompt standard | 8 documents |
| **B4: RAG-Fusion** | Dense + BM25 RRF | Multi-query LLM expansion + fusion | Prompt standard | 8 documents |
| **B3-abs: Prompt Abstention** | Dense + BM25 RRF | Single primary query | Prompt instruction to abstain if evidence absent | 8 documents |
| **CARO** | Full CARO pipeline | LLM dynamic router + multi-pass search | Dedicated LLM sufficiency judge | 8 documents |

---

## 2. Fair-Fight Constraints

1. **Identical Assets:** All systems query the identical verified 250,000-document FAISS index and metadata.
2. **Identical Generator:** All systems use `gemma3:4b` at `temperature: 0.0`.
3. **Unified Prompt:** The base generation prompt format, document presentation, citation guidelines, and system instructions are identical across all systems (defined in `eval/common.py`).
4. **Context Parity:** Exactly `MAX_CONTEXT_DOCS = 8` documents are passed to the generator for all systems.
5. **Exact Document Exclusion:** For `PARTIAL` and `UNANSWERABLE` test sets, gold supporting documents are excluded after over-fetching (`fetch_k = top_k + len(excluded) + 50`), ensuring full candidate pools without information leakage.
6. **Unbiased Grading:**
   - **Semantic Correctness:** Evaluated by `llama3.1:8b` via Ollama (configured via `JUDGE_MODEL`), avoiding self-grading bias.
   - **Claim Faithfulness:** Evaluated by NLI cross-encoder `cross-encoder/nli-deberta-v3-small`.
7. **Single-Turn Isolation:** CARO's conversation memory is explicitly cleared before every query.
8. **Call & Latency Tracking:** Standardized `CallTracker` context manager measures exact LLM calls, retrieval calls, and wall-clock latency.

---

## 3. Quick Start & Execution

### Run Unit Tests
Verifies answer normalization, token F1/EM, retrieval metrics, abstention math, exclusion filtering, and bootstrap CI computation:
```bash
python -m pytest eval/tests/test_harness.py -v
```

### Run Smoke Test (N=10 Questions)
Executes all 6 systems across ANSWERABLE, PARTIAL, and UNANSWERABLE sets (30 evaluations per system, 180 total queries):
```bash
python -m eval.runner --smoke --seed 42
```

### Run Full Benchmark (N=500 Questions)
```bash
python -m eval.runner --sample_size 500 --seed 42 --workers 4 --device cuda
```
*Note: The runner is fully resumable across systems and sets. If interrupted, restarting the command immediately resumes from where it left off, writing each record immediately.*

### Google Colab Execution
1. Install dependencies:
```bash
pip install -r eval/requirements_colab.txt
```
2. Start Ollama and pull models:
```bash
ollama serve &
ollama pull gemma3:4b
ollama pull llama3.1:8b
```
3. Run evaluation concurrently with CUDA acceleration:
```bash
python -m eval.runner --n 30 --seed 42 --workers 4 --device cuda
# Or run with disjoint seed 123
python -m eval.runner --n 30 --seed 123 --workers 4 --device cuda
```

### Generate Evaluation Report & Artifacts
```bash
python -c "from eval.report import generate_evaluation_report; generate_evaluation_report(seed=42)"
```

### Interactive Notebook
Open and execute `eval/CARO_RAG_Eval.ipynb` in VS Code, JupyterLab, or Google Colab.

---

## 4. Deliverables and Output Files

- `eval/results/<system>_<set>_seed<seed>.jsonl`: Detailed per-query execution and metric records.
- `eval/results/run_config.json`: Run configuration, seed, model parameters, and cryptographic code/prompt hashes.
- `eval/results/human_check.csv`: 50 random judged items exported with a blank column for manual human validation.
- `eval/results/errors_<system>.csv`: Up to 50 failure cases per system exported for error taxonomy analysis.
- `eval/RESULTS.md`: Comprehensive evaluation report with bootstrap 95% CIs, paired hypothesis tests, cost-quality tradeoffs, and honest verdict.
