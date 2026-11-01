# Academic Audit and Technical Correction Report: CARO RAG Thesis

- **Document Audited:** `doc2_updated.docx` (derived from `doc2.docx`)
- **Corrected Output:** `doc2_corrected.docx`
- **Reconciliation Data:** `results_reconciliation.csv`
- **Auditor Role:** Academic Research Editor, RAG Systems Researcher, Experimental-Evaluation Auditor
- **Date:** October 22, 2026

---

## 1. Executive Summary

This audit conducted an end-to-end technical, methodological, and academic review of the MCA thesis on **CARO (Context-Aware Adaptive Retrieval Orchestrator for Retrieval-Augmented Generation)**. The thesis was evaluated against all ground-truth codebase modules, configuration files, vector/sparse indexes, and empirical execution artifacts (`eval/results/*.jsonl`).

### Key Inconsistencies Discovered and Resolved:
1. **Implementation Contradiction Across Chapters:**
   - *Issue:* Early chapters (Chapters 1, 3, 4) and concluding chapters (Chapters 9, 10) stated that Modules M5–M8 were "unimplemented", "planned extensions", or "future work", while Chapter 8 and Appendix 5 reported a full 500-question evaluation of M1–M8.
   - *Resolution:* Audited source code and execution logs. Confirmed that Modules M1–M5, M7, and M8 are fully implemented and were executed across the 500-question HotpotQA benchmark. Module M6 (Temporary Episodic Memory) is implemented and unit-tested in code, but was not exercised in the benchmark because HotpotQA consists of isolated multi-hop queries, and QReCC conversational loading was deferred. All chapters have been harmonized to reflect this precise factual status.
2. **Corpus Size Conflict (5,000 vs. 250,000 Documents):**
   - *Issue:* Multiple sections stated that evaluation was performed on a "5,000-document development index", whereas the execution logs and `eval/caro_core.py` verified that the 500-question benchmark ran against `hotpotqa_full_index.faiss` containing **250,000 documents**.
   - *Resolution:* Disentangled the two indexes. The 5,000-document index was used exclusively for early module unit tests and sanity checks; the 250,000-document index was used for the official 500-question comparative benchmark.
3. **Apparent Evidence-Sufficiency vs. Abstention Discrepancy:**
   - *Issue:* Module 7 reported 398 Sufficient, 75 Insufficient, and 27 Partial evidence determinations (summing to 500), but the thesis reported 98 abstentions and 402 answers.
   - *Resolution:* Demonstrated through record-level cross-tabulation that evidence status (M7) and final answering behavior (M8) are decoupled variables:
     - 75 Insufficient $\rightarrow$ 100% abstained (75).
     - 27 Partial $\rightarrow$ 63.0% answered with qualifications (17), 37.0% abstained (10).
     - 398 Sufficient $\rightarrow$ 96.7% answered (385), 3.3% abstained (13) due to generator-level missing target verification.
     - Total Abstained: $75 + 10 + 13 = 98$. Total Answered: $385 + 17 = 402$.
4. **Unsupported Citation-Grounding and Reliability Claims:**
   - *Issue:* The thesis previously claimed "100% verified citation attribution", "elimination of hallucinations", and implied external LLM judge verification.
   - *Resolution:* In reality, evaluation was run with `--skip-grading` (judge scores and faithfulness fields are null). Furthermore, in **292 queries where evidence was judged SUFFICIENT and an answer was generated, the output was factually incorrect** (Exact Match = False) due to 4B generator reasoning errors or extractive phrasing mismatches. Weakened and qualified all claims: CARO achieves 100% citation *format compliance* and eliminates fabrications on *abstained* queries, but does not guarantee factual truth on generated answers.
5. **Relabeled Retrieval-Ranking Metrics:**
   - *Issue:* Appendix 3 (Table 14) previously relabeled token-level precision/recall (0.4254, 0.4088) as Recall@5 and Precision@5.
   - *Resolution:* Calculated the true document-level retrieval metrics directly from `gold_titles` and `retrieved_titles` in the 500 JSONL records: Baseline achieved Recall@5 = 0.7790, Precision@5 = 0.3752, MRR = 0.9040; CARO achieved Recall@5 = 0.7340, Precision@5 = 0.3456, MRR = 0.8537.
6. **Erroneous "Fine-Tuning" Language in Chapter 7:**
   - *Issue:* Section 7.7 described Steps 3 and 4 as "fine-tuning of the Top-5 RAG baseline" and "fine-tuning of the CARO model", even though no model fine-tuning was performed.
   - *Resolution:* Replaced all instances of "fine-tuning" with "inference execution without fine-tuning" over off-the-shelf zero-shot models (`gemma3:4b` and `BAAI/bge-m3`).
7. **Broken Mathematical Formatting & Missing References:**
   - *Issue:* Equation (13) appeared as `3)` and Equation (22) appeared as `22)`. In-text citations cited `[4]`, `[10]`, `[13]`, `[14]`, and `[15]`, but the reference list omitted entries 13–15 and misattributed entry 4.
   - *Resolution:* Repaired equations (13) and (22). Aligned entry [4] to Karpukhin et al. (DPR), entry [10] to Johnson et al. (FAISS), and added references [13] (Bhat et al.), [14] (Yuan et al.), and [15] (Sharma & Kumaran).

---

## 2. Module Implementation Audit Table

| Module | Intended Functionality | Evidence in Code / Artifacts | Verified Status | Required Thesis Correction |
|---|---|---|---|---|
| **M1: Dataset Processing** | Load, clean, and standardize multi-hop (HotpotQA) and conversational (QReCC) datasets into JSONL. | `modules/module1_data_loader/`, `eval/dataset.py`, `eval/data/questions.jsonl`. 250k documents processed. | **Implemented & Executed** for HotpotQA; **Partially Implemented** for QReCC (upstream loader issue). | Update Chapters 4, 5, and 7 to clarify that HotpotQA is fully processed (250k passages), while QReCC raw source loading had format/network constraints. |
| **M2: Baseline RAG** | Unconstrained Top-K dense retrieval (FAISS + BGE-M3) with direct generation without evidence gating. | `modules/module2_baseline_rag/`, `eval/baselines.py`, `eval/results/B1_Dense_ANSWERABLE_seed42.jsonl`. | **Implemented & Executed** (500 queries). | Confirm full execution; report verified metrics (EM=31.40%, Latency=25.02s, 0% abstention). |
| **M3: Query Router** | Classify query intent and dependencies into Boolean states (`retrieval_needed`, `memory_needed`). | `modules/module3_adaptive_query_router/`, `eval/caro_core.py`, `CARO_ANSWERABLE_seed42.jsonl` (`route`). | **Implemented & Executed** (500 queries, 100% schema valid). | Confirm implementation; remove claims of "learned router" (prompt/rule-based execution). |
| **M4: Strategy Selector** | Deterministically map query features to DENSE, SPARSE, or HYBRID retrieval. | `modules/module4_retrieval_strategy/`, `eval/caro_core.py`, `CARO_ANSWERABLE_seed42.jsonl` (`retrieval_strategy`). | **Implemented & Executed** (361 Hybrid, 112 Dense, 27 Sparse). | Confirm implementation; remove legacy text treating strategy selection as an unverified prototype. |
| **M5: Hybrid Retrieval & Fusion** | Execute dense FAISS, sparse BM25, and Reciprocal Rank Fusion ($k=60$) across query variants. | `modules/module5_retrieval/`, `eval/caro_core.py`, `indexes/hotpotqa/` (FAISS + BM25 cache). | **Implemented & Executed** across 250,000 documents. | Remove all claims that M5 is unimplemented or future work. Report true document retrieval metrics. |
| **M6: Temporary Memory** | Bounded conversational episodic memory store and retriever with session expiry. | `modules/module6_temporary_memory/`, `tests/test_caro_flow.py`. | **Implemented in Code; Unverified on Benchmark.** | Clarify that M6 exists and passes unit tests, but was not evaluated on 500-question HotpotQA run (which consists of single-turn queries). |
| **M7: Evidence Sufficiency** | Assess factual coverage, detect missing bridge entities, categorize (Sufficient/Insufficient/Partial). | `modules/module7_evidence_sufficiency/`, `eval/caro_core.py`, `CARO_ANSWERABLE_seed42.jsonl`. | **Implemented & Executed** (398 Suff, 75 Insuff, 27 Part; 98 abstentions). | Remove claims that M7 is future work. Reconcile sufficiency counts with final abstention behavior. |
| **M8: Grounded Generation** | Schema-constrained generation with `[Document k]` citations, validation, and honest refusal. | `modules/module8_answer_generation/`, `eval/caro_core.py`, `CARO_ANSWERABLE_seed42.jsonl`. | **Implemented & Executed** (402 answers, 98 abstentions, 100% valid JSON). | Remove claims that M8 is future work. Qualify citation formatting vs. factual accuracy; note judge skipping. |

---

## 3. Experimental Results Reconciliation

All empirical runs were audited from raw JSONL files and documented in `results_reconciliation.csv`:

### Summary of Reconciled Benchmark Runs:

| Experiment Identifier | System | Dataset / Conditions | Corpus Size | Query Count | Answered (%) | Abstained (%) | Exact Match | Token F1 | Mean Latency | Doc Recall@5 | Doc MRR | Verification Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXP1: 500Q Main Benchmark** | **B1_Dense** | HotpotQA Answerable (seed 42, CUDA) | 250,000 | 500 | 500 (100.0%) | 0 (0.0%) | 31.40% (157) | 0.4052 | 25.02 s | 0.7790 | 0.9040 | `eval/results/B1_Dense_ANSWERABLE_seed42.jsonl` |
| **EXP1: 500Q Main Benchmark** | **CARO** | HotpotQA Answerable (seed 42, CUDA) | 250,000 | 500 | 402 (80.40%) | 98 (19.60%) | 19.00% (95)* | 0.2713 | 64.48 s | 0.7340 | 0.8537 | `eval/results/CARO_ANSWERABLE_seed42.jsonl` |
| **EXP2: Unanswerable Run** | **B1_Dense** | HotpotQA Unanswerable (seed 42) | 250,000 | 21 | 21 (100.0%) | 0 (0.0%) | 0.00% | 0.0412 | 22.41 s | N/A | N/A | `eval/results/B1_Dense_UNANSWERABLE_seed42.jsonl` |
| **EXP3: Earlier Pilot Run** | **B1_Dense** | HotpotQA Pilot (N=20, CPU) | 250,000 | 20 | 20 (100.0%) | 0 (0.0%) | 0.00% | 0.0700 | 72.36 s | 0.7500 | N/A | `eval/RESULTS.md` |
| **EXP3: Earlier Pilot Run** | **CARO** | HotpotQA Pilot (N=20, CPU) | 250,000 | 20 | 16 (80.0%) | 4 (20.0%) | 0.00% | 0.0800 | 152.06 s | 0.7500 | N/A | `eval/RESULTS.md` |
| **EXP4: Unit Integration** | **CARO** | Curated Sanity Queries (CPU) | 5,000 | 35 | 35 (100.0%) | 0 (0.0%) | N/A | N/A | 16.08 s | N/A | N/A | `implementation_report.md` |

*\*Note: On the answered subset (402 queries), CARO's Exact Match is 95 / 402 = 23.63%. All abstentions are scored as 0 under standard NLP benchmark evaluation.*

---

## 4. Evidence Sufficiency and Abstention Reconciliation

A core finding of this audit resolves the apparent numerical tension in Section 8.2:

```
Total Evaluated Queries: 500
│
├── Module 7 Assessor Output:
│   ├── SUFFICIENT: 398 (79.60%)
│   │   ├── Module 8 Answer Generated: 385 (96.73%)
│   │   └── Module 8 Generator Abstention: 13 (3.27%)  <-- Target entity missing in spans
│   │
│   ├── INSUFFICIENT: 75 (15.00%)
│   │   └── Module 8 Honest Abstention: 75 (100.0%)   <-- Complete factual refusal
│   │
│   └── PARTIAL: 27 (5.40%)
│       ├── Module 8 Answer Generated: 17 (62.96%)     <-- Bounded cautious answer
│       └── Module 8 Honest Abstention: 10 (37.04%)   <-- Missing bridge entity forced refusal
│
└── Reconciled System Behavior:
    ├── Total Answered: 385 + 17 = 402 (80.40%)
    └── Total Honest Abstentions: 75 + 10 + 13 = 98 (19.60%)
```

---

## 5. Claims Weakened or Qualified Due to Evidentiary Constraints

| Previous Thesis Claim | Audited Evidence | Applied Correction in `doc2_corrected.docx` |
|---|---|---|
| *"100% verified citation attribution guarantees factual truth."* | 292 queries had sufficient evidence and valid citation tags, yet generated incorrect answers against gold references. | Replaced with: *"100% citation format compliance ensures source traceability; does not eliminate generator reasoning errors."* |
| *"Elimination of ungrounded hallucinations."* | CARO eliminates fabrications on the 98 abstained queries, but makes reasoning/entity confusion errors on 292 answered queries. | Replaced with: *"Eliminates forced guessing on queries with deficient evidence via deterministic honest refusal."* |
| *"LLM judge verified faithfulness and semantic correctness."* | Execution command used `--skip-grading`. `judge_score`, `judge_verdict`, and `faithfulness` fields are all null. | Replaced with: *"Automated LLM judging was skipped during this run; faithfulness is evaluated via citation syntax adherence and abstention audits."* |
| *"Token precision/recall reported as Recall@5 and Precision@5."* | Token overlap scores were previously inserted into retrieval tables. | Replaced with true document-level retrieval metrics: Baseline Recall@5 = 77.9%, CARO Recall@5 = 73.4%. |
| *"Modules 5–8 remain future work."* | Modules 5, 7, and 8 are fully implemented and executed in `eval/caro_core.py`. | Harmonized Chapters 1, 3, 4, 5, 7, 9, 10 to confirm M1–M5, M7, M8 implementation. |

---

## 6. Section-by-Section Inventory of Applied Changes

1. **Chapter 1 (Introduction):**
   - Updated Contribution 3 (P40) to remove the claim that M5–M8 are "planned extensions", confirming end-to-end execution on 250,000 documents while distinguishing conversational M6.
2. **Chapter 3 (Literature Review):**
   - Updated P133 and P143 to reflect that M5, M7, and M8 have been realized and empirically benchmarked on HotpotQA.
3. **Chapter 4 (Methodology & Validity Threats):**
   - Updated Table 1: Clarified HotpotQA dev index (5,000 docs) vs. full benchmark index (250,000 docs); noted QReCC upstream source loading status.
   - Updated Table 2 (Threats to Validity): Replaced "Unimplemented M5–M8" with "Conversational evaluation deferred (M6)".
4. **Chapter 5 (System Design):**
   - Corrected P225 to eliminate the statement that conclusions must wait for M5–M8 implementation.
5. **Chapter 6 (Mathematical Formulations):**
   - Repaired broken Equation (13) in P232 (previously printed as `3)`).
   - Repaired broken Equation (22) in P251 (previously printed as `22)`).
6. **Chapter 7 (Implementation & Experimental Setup):**
   - Updated Table 7 to accurately specify both the 5,000-doc development corpus and the 250,000-doc evaluation corpus, along with the NVIDIA RTX GPU hardware environment.
   - Corrected P311 to cite the 250,000-document index.
   - Corrected P323: Replaced "fine-tuning" with "inference execution without fine-tuning".
   - Disclosed that the 500-question run was executed with `--skip-grading`.
7. **Chapter 8 (Results & Discussion):**
   - Corrected corpus size in Section 8.1 (P331) to 250,000 documents.
   - Updated Table 8.1 (Table 9) row on Citation Attribution to specify format compliance.
   - Inserted rigorous reconciliation narrative in Section 8.2 (P342) explaining the 398/75/27 to 98/402 relationship.
   - Expanded Section 8.6 (Error Analysis) to detail the 292 queries where evidence was SUFFICIENT yet the answer was incorrect.
8. **Chapter 9 (Limitations):**
   - Rewrote Section 9.1 (P380–P384) to replace the outdated "M5–M8 unimplemented" text with the 5 empirical limitations established by the run: (1) latency overhead (64.48s vs 25.02s), (2) generator reasoning limits on 4B model, (3) conversational M6 deferred, (4) absence proof bounds, and (5) skipped judge scoring.
   - Overhauled Table 16 (Limitation-to-Future-Work Mapping).
9. **Chapter 10 (Conclusion & Future Work):**
   - Rewrote Section 10.1 (P420) to state the verified conclusions and realistic trade-offs.
   - Rewrote Section 10.4 (P430–P435) to replace "Finish Module M5/M7/M8" with genuine future research directions (conversational QReCC evaluation, lightweight learned router, generator scaling to 8B/70B).
10. **References:**
    - Corrected entry [4] to Karpukhin et al. (DPR).
    - Corrected entry [10] to Johnson et al. (FAISS).
    - Appended missing reference entries [13] (Bhat et al., HyDE/Gemma), [14] (Yuan et al., AdaKG-RAG), and [15] (Sharma & Kumaran, Adaptive Retrieval Judgment) to align with all in-text citations.
11. **Appendix 3 (Evaluation Tables):**
    - Updated Table 18 (Table A3.1): Inserted verified document-level Recall@5 (0.7790 vs 0.7340), Precision@5 (0.3752 vs 0.3456), MRR (0.9040 vs 0.8537), and Both Gold Retrieved (0.6080 vs 0.5400).
    - Updated Table 19 (Table A3.2): Inserted Exact Match (31.40% vs 19.00% overall; 23.63% on answered subset), Token F1 (0.4052 vs 0.2713), and qualified faithfulness notes.
    - Updated Table 20 (Table A3.3): Inserted retrieval calls (1.00 vs 8.00), LLM invocations (1.00 vs 4.85), and total execution time (3.47 hrs vs 8.96 hrs).

---

## 7. Unresolved Issues Requiring Student / Advisor Decision

The following two items cannot be decided purely from codebase artifacts and are flagged for your review:

1. **Thesis Scope Framing for Module M6 (Conversational Memory):**
   - *Current Status:* Module M6 is implemented in code (`modules/module6_temporary_memory/`) and passes unit tests, but QReCC evaluation was deferred due to upstream HuggingFace loader/network issues.
   - *Option A (Recommended & Implemented in `doc2_corrected.docx`):* Frame CARO as an adaptive multi-hop RAG orchestrator with a complete, empirically benchmarked multi-hop pipeline (M1–M5, M7, M8 on HotpotQA), presenting M6 as an architectural extension verified at the unit level.
   - *Option B:* Remove Module M6 entirely from the core architecture diagrams and reposition it strictly as a proposed future module.
2. **Offline LLM-Judge Evaluation Campaign:**
   - *Current Status:* The 500-question run used `--skip-grading` to prevent multi-day execution times on local hardware. As a result, semantic LLM-judge scores (`judge_score`) and automated NLI faithfulness metrics are not available.
   - *Action:* The thesis now explicitly discloses this omission. If your committee requires formal LLM-as-a-judge scores, a dedicated offline evaluation script can be run on the existing `CARO_ANSWERABLE_seed42.jsonl` trajectories without re-running retrieval or generation.

---

## 8. Final Quality Checklist

| Checklist Item | Status | Verification Evidence |
|---|---|---|
| 1. Implementation status consistent across all chapters | **COMPLETE** | Harmonized Chapters 1, 3, 4, 5, 7, 9, 10; removed all "unimplemented M5–M8" claims. |
| 2. Baseline and CARO metrics match available artifacts | **COMPLETE** | Exact Match (31.40% vs 19.00%), F1 (0.4052 vs 0.2713), Latency (25.02s vs 64.48s) verified against 500 JSONL records. |
| 3. Answer and abstention counts reconcile | **COMPLETE** | Reconciled: 385 (from Suff) + 17 (from Part) = 402 answered; 75 (Insuff) + 10 (Part) + 13 (Suff) = 98 abstained. |
| 4. Corpus sizes and model configurations correctly attributed | **COMPLETE** | 5,000 docs assigned to dev unit tests; 250,000 docs assigned to full benchmark; Gemma-3-4B and BGE-M3 documented. |
| 5. Retrieval metrics use valid definitions and data | **COMPLETE** | Token scores removed from retrieval tables; true document Recall@5 (77.9% vs 73.4%) and MRR (0.904 vs 0.854) inserted. |
| 6. Citation-grounding and reliability claims supported | **COMPLETE** | Disclosed 292 incorrect answers on sufficient evidence; qualified format compliance vs. factual truth. |
| 7. Mathematical formulations and numbering consistent | **COMPLETE** | Repaired broken equation formatting for Equation (13) in P232 and Equation (22) in P251. |
| 8. References and citations checked | **COMPLETE** | Aligned entries [4] (DPR) and [10] (FAISS); appended missing references [13], [14], [15]. |
| 9. Limitations and conclusion reflect actual findings | **COMPLETE** | Chapters 9 and 10 rewritten with verified trade-offs (latency, model capacity, abstention penalty). |
| 10. Original thesis remains unchanged | **COMPLETE** | Original `doc2.docx` preserved intact; corrected document saved as `doc2_corrected.docx`. |

---
**Verdict:** The thesis document is now technically consistent, empirically verified, methodologically defensible, and ready for your separate humanization and paraphrasing stage.
