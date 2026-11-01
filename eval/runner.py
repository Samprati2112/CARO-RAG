"""
Evaluation Runner for CARO RAG vs Baseline Systems.
Features:
- Resumable: appends results to eval/results/<system>_<set>_seed<seed>.jsonl
- Two-Phase Optimization:
    Phase 1: Generate answers across all systems with Gemma 3 4B (model stays resident in Ollama)
    Phase 2: Grade answers in batch with LLM Judge (Llama 3.1 8B stays resident in Ollama)
    This completely eliminates thrashing/reloading models between generation and grading.
- Per-question try/except with error logging
- CLI support: --smoke (N=10), --sample_size (default 500), --seed (default 42)
- Reuses loaded corpus assets across all systems
- Saves run config with prompt and file hashes to eval/results/run_config.json
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import gc
import time
import json
import hashlib
import argparse
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

from eval.caro_core import (
    CorpusAssets,
    PROJECT_ROOT,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    MAX_CONTEXT_DOCS,
    INITIAL_TOP_K,
)
from eval.baselines import get_all_systems
from eval.dataset import prepare_evaluation_dataset, get_evaluation_items
from eval.common import (
    exact_match_score,
    token_f1_score,
    compute_retrieval_metrics,
    build_shared_generation_prompt,
    print_ram_usage,
)
from eval.judge import LLMJudge, FaithfulnessChecker, JUDGE_MODEL

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_run_config(
    seed: int,
    sample_size: int,
    smoke: bool,
    systems: List[str],
    workers: int = 1,
    device: str = "cpu"
) -> Path:
    """Logs run configuration to eval/results/run_config.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    caro_core_path = PROJECT_ROOT / "eval" / "caro_core.py"
    caro_hash = compute_file_hash(caro_core_path) if caro_core_path.exists() else "unknown"

    sample_prompt = build_shared_generation_prompt("Sample Question?", [])
    prompt_hash = hashlib.sha256(sample_prompt.encode("utf-8")).hexdigest()

    config = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "sample_size": sample_size,
        "smoke_mode": smoke,
        "workers": workers,
        "device": device,
        "llm_model": LLM_MODEL_NAME,
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "max_context_docs": MAX_CONTEXT_DOCS,
        "initial_top_k": INITIAL_TOP_K,
        "temperature": 0.0,
        "caro_core_file_hash": caro_hash,
        "base_prompt_hash": prompt_hash,
        "evaluated_systems": systems
    }

    config_file = RESULTS_DIR / "run_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_file


def run_step1_profiling(seed: int = 42, device: str = "cpu") -> None:
    """
    Step 1 of Reduced Test Mode:
    Runs exactly 3 CARO questions, prints the per-stage timing table,
    and displays the query details.
    """
    print("=" * 80, flush=True)
    print("STEP 1: CARO PROFILING RUN (3 Questions)", flush=True)
    print("=" * 80, flush=True)

    CorpusAssets.load(device=device)
    sampled_questions, _ = prepare_evaluation_dataset(sample_size=3, seed=seed, force_rebuild=True)
    eval_items = get_evaluation_items(sampled_questions, target_sets=("ANSWERABLE",))[:3]

    from eval.baselines import CaroPipelineAdapter
    caro = CaroPipelineAdapter()

    stage_names = [
        "router", "retrieval_primary",
        "retrieval_primary.query_encode", "retrieval_primary.faiss_search",
        "retrieval_primary.bm25_search", "retrieval_primary.exclusion_filter",
        "expansion", "decomposition",
        "retrieval_variants", "retrieval_cross_strategy", "retrieval_fusion",
        "assessment", "generation", "total"
    ]
    timings_by_stage: Dict[str, List[float]] = {st: [] for st in stage_names}
    question_summaries = []

    for idx, item in enumerate(eval_items, 1):
        q_text = item["question"]
        gold_ans = item["gold_answer"]
        gold_titles = set(item["gold_titles"])
        excluded = set(item["excluded_doc_ids"])

        print(f"\n--- Running Question {idx}/3 ---", flush=True)
        print(f"Question: {q_text}", flush=True)
        print(f"Gold Answer: {gold_ans}", flush=True)

        res = caro.answer(q_text, excluded_doc_ids=excluded)
        st_times = res.get("stage_timings", {})

        for st in stage_names:
            timings_by_stage[st].append(float(st_times.get(st, 0.0)))

        final_ans = res["final_answer"]
        raw_ans = res["final_answer_raw"]
        abstained = res["abstained_strict"]
        em = exact_match_score(final_ans, gold_ans) if not abstained else False
        f1, _, _ = token_f1_score(final_ans, gold_ans) if not abstained else (0.0, 0.0, 0.0)

        meta = res.get("metadata", {})
        question_summaries.append({
            "idx": idx,
            "question": q_text,
            "gold": gold_ans,
            "extracted_answer": final_ans,
            "raw_answer": raw_ans,
            "abstained": abstained,
            "em": em,
            "f1": f1,
            "route": meta.get("route"),
            "strategy": meta.get("strategy"),
            "evidence_status": meta.get("evidence_status"),
            "latency": res["latency_seconds"],
            "llm_calls": res["llm_calls"],
            "ret_calls": res["retrieval_calls"],
            "timings": st_times
        })

    # Print Question Summaries
    print("\n" + "=" * 80, flush=True)
    print("STEP 1: CARO EXECUTION SUMMARY", flush=True)
    print("=" * 80, flush=True)
    for qs in question_summaries:
        print(f"Q{qs['idx']}: {qs['question']}", flush=True)
        print(f"  Gold: {qs['gold']} | Extracted: {qs['extracted_answer']}", flush=True)
        print(f"  Raw: {qs['raw_answer']}", flush=True)
        print(f"  Route: {qs['route']} | Strategy: {qs['strategy']} | Status: {qs['evidence_status']}", flush=True)
        print(f"  EM: {qs['em']} | F1: {qs['f1']:.3f} | Abstained: {qs['abstained']}", flush=True)
        print(f"  Total Latency: {qs['latency']:.2f}s | LLM calls: {qs['llm_calls']} | Ret calls: {qs['ret_calls']}", flush=True)

    # Print Per-Stage Timing Table
    mean_total = np.mean(timings_by_stage["total"]) if timings_by_stage["total"] else 1e-9
    print("\n" + "=" * 80, flush=True)
    print("PER-STAGE TIMING BREAKDOWN (SECONDS)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Stage':<28} | {'Q1 (s)':<8} | {'Q2 (s)':<8} | {'Q3 (s)':<8} | {'Mean (s)':<8} | {'% of Total':<10}", flush=True)
    print("-" * 80, flush=True)

    for st in stage_names:
        vals = timings_by_stage[st]
        mean_st = float(np.mean(vals))
        pct = (mean_st / max(mean_total, 1e-9)) * 100.0
        q1_str = f"{vals[0]:.3f}" if len(vals) > 0 else "N/A"
        q2_str = f"{vals[1]:.3f}" if len(vals) > 1 else "N/A"
        q3_str = f"{vals[2]:.3f}" if len(vals) > 2 else "N/A"
        print(f"{st:<28} | {q1_str:<8} | {q2_str:<8} | {q3_str:<8} | {mean_st:<8.3f} | {pct:>8.1f}%", flush=True)

    print("=" * 80, flush=True)
    print("Step 1 complete. Per-stage timing table captured.", flush=True)


def run_evaluation(
    smoke: bool = False,
    sample_size: int = 500,
    seed: int = 42,
    selected_systems: Optional[List[str]] = None,
    target_sets: Tuple[str, ...] = ("ANSWERABLE", "PARTIAL", "UNANSWERABLE"),
    skip_grading: bool = False,
    force: bool = False,
    workers: int = 1,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes the fair evaluation harness across all systems and datasets.
    Resumable per system and set_type.
    """
    effective_sample_size = 10 if smoke else sample_size
    print("=" * 80, flush=True)
    mode_str = 'SMOKE (N=10)' if smoke else f'N={effective_sample_size}'
    print(f"RUNNING RAG EVALUATION HARNESS | Mode: {mode_str} | Seed: {seed} | Workers: {workers} | Device: {device} | Skip-Grading: {skip_grading}", flush=True)
    print(f"Target Systems: {selected_systems or 'ALL'} | Sets: {target_sets}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load shared assets once
    CorpusAssets.load(device=device)

    # 2. Prepare/sample dataset
    sampled_questions, data_stats = prepare_evaluation_dataset(
        sample_size=effective_sample_size,
        seed=seed,
        force_rebuild=force,
        device=device
    )
    if device == "cuda":
        CorpusAssets.load(device="cuda")
    eval_items = get_evaluation_items(sampled_questions, target_sets=target_sets)

    # 3. Instantiate systems
    all_systems_map = get_all_systems()
    system_names = selected_systems or list(all_systems_map.keys())

    # 4. Save configuration
    save_run_config(
        seed=seed,
        sample_size=effective_sample_size,
        smoke=smoke,
        systems=system_names,
        workers=workers,
        device=device
    )

    # -------------------------------------------------------------
    # PHASE 1: GENERATION & RETRIEVAL (Gemma 3 4B stays resident)
    # -------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("PHASE 1: GENERATION & RETRIEVAL (Model: Gemma 3 4B)", flush=True)
    print("=" * 50, flush=True)

    summary_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    gen_record_counter = [0]

    for sys_name in system_names:
        if sys_name not in all_systems_map:
            print(f"[WARNING] Unknown system '{sys_name}', skipping.", flush=True)
            continue
        system = all_systems_map[sys_name]
        print(f"\n>>> System: {sys_name} ({len(eval_items)} items across {len(target_sets)} sets) <<<", flush=True)

        for set_type in target_sets:
            result_file = RESULTS_DIR / f"{sys_name}_{set_type}_seed{seed}.jsonl"
            existing_results = {}
            set_items = [it for it in eval_items if it["set_type"] == set_type]

            # If force or if file exists, load existing records
            if not force and result_file.exists():
                with open(result_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            try:
                                rec = json.loads(line_str)
                                if rec.get("error") is None:
                                    existing_results[rec["question_id"]] = rec
                                else:
                                    print(f"    [Resume] Redoing question {rec.get('question_id')} with prior error: {rec.get('error')}", flush=True)
                            except Exception as e:
                                print(f"    [Warning] Corrupted line in {result_file}: {e}", flush=True)
                for rec in existing_results.values():
                    summary_records[sys_name].append(rec)

            remaining_items = [it for it in set_items if it["question_id"] not in existing_results]
            print(f"  [{sys_name}] Set: {set_type} | Total: {len(set_items)} | Already completed: {len(existing_results)} | Remaining: {len(remaining_items)}", flush=True)

            if not remaining_items:
                continue

            file_mode = "a" if existing_results else "w"
            file_lock = threading.Lock()

            with open(result_file, file_mode, encoding="utf-8") as out_f:
                def process_item(item, idx):
                    qid = item["question_id"]
                    q_text = item["question"]
                    gold_ans = item["gold_answer"]
                    gold_titles = set(item["gold_titles"])
                    excluded = set(item["excluded_doc_ids"])

                    t_start = time.time()
                    try:
                        sys_res = system.answer(q_text, excluded_doc_ids=excluded)
                        final_ans = sys_res["final_answer"]
                        final_raw = sys_res.get("final_answer_raw", final_ans)
                        retrieved = sys_res["retrieved_docs"]
                        abstained_strict = sys_res.get("abstained_strict", sys_res["abstained"])
                        abstained_soft = sys_res.get("abstained_soft", abstained_strict)
                        error_msg = None
                    except Exception as e:
                        print(f"    [ERROR] Question {qid} failed in {sys_name}: {e}", flush=True)
                        sys_res = {
                            "final_answer": "ERROR",
                            "final_answer_raw": "ERROR",
                            "retrieved_docs": [],
                            "abstained": False,
                            "abstained_strict": False,
                            "abstained_soft": False,
                            "latency_seconds": round(time.time() - t_start, 4),
                            "llm_calls": 0,
                            "retrieval_calls": 0,
                            "stage_timings": {},
                            "metadata": {}
                        }
                        final_ans = "ERROR"
                        final_raw = "ERROR"
                        retrieved = []
                        abstained_strict = False
                        abstained_soft = False
                        error_msg = str(e)

                    # Compute Answer Quality on extracted text after ANSWER:
                    em = exact_match_score(final_ans, gold_ans) if not abstained_strict and final_ans != "ERROR" else False
                    f1, prec, rec = token_f1_score(final_ans, gold_ans) if not abstained_strict and final_ans != "ERROR" else (0.0, 0.0, 0.0)

                    # Retrieval Quality (deduped titles)
                    ret_metrics = compute_retrieval_metrics(retrieved, gold_titles)

                    # Faithfulness (NLI cross-encoder, skipped if skip_grading is True)
                    faith_res = None
                    if not skip_grading:
                        faith_res = FaithfulnessChecker.check_faithfulness(
                            final_raw, retrieved, question=q_text, abstained=abstained_strict
                        )

                    sys_meta = sys_res.get("metadata", {})
                    record = {
                        "question_id": qid,
                        "sample_index": item["sample_index"],
                        "set_type": set_type,
                        "system_name": sys_name,
                        "question": q_text,
                        "gold_answer": gold_ans,
                        "gold_titles": list(gold_titles),
                        "excluded_count": len(excluded),
                        "final_answer": final_ans,
                        "final_answer_raw": final_raw,
                        "abstained": abstained_strict,
                        "abstained_strict": abstained_strict,
                        "abstained_soft": abstained_soft,
                        "exact_match": em,
                        "token_f1": f1,
                        "token_precision": prec,
                        "token_recall": rec,
                        "retrieval_metrics": ret_metrics,
                        "route": sys_meta.get("route"),
                        "rewritten_query": sys_meta.get("rewritten_query", q_text),
                        "query_variants": sys_meta.get("query_variants", [q_text]),
                        "evidence_status": sys_meta.get("evidence_status", "UNKNOWN"),
                        "evidence_assessment": sys_meta.get("evidence_assessment"),
                        "parse_ok": (sys_meta.get("evidence_assessment") or {}).get("parse_ok", (sys_meta.get("evidence_assessment") or {}).get("json_parsed", True)) if sys_name == "CARO" else None,
                        "fallback_fired": (sys_meta.get("evidence_assessment") or {}).get("fallback_fired", False) if sys_name == "CARO" else None,
                        "retrieval_strategy": sys_meta.get("strategy") or sys_name,
                        "stage_timings": sys_res.get("stage_timings", {}),
                        "judge_verdict": None,
                        "judge_score": None,
                        "judge_reason": None,
                        "faithfulness": faith_res,
                        "latency_seconds": sys_res["latency_seconds"],
                        "llm_calls": sys_res["llm_calls"],
                        "retrieval_calls": sys_res["retrieval_calls"],
                        "retrieved_titles": [d.get("title", "") for d in retrieved],
                        "error": error_msg
                    }

                    # Write and flush each record immediately
                    with file_lock:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        summary_records[sys_name].append(record)
                        gen_record_counter[0] += 1
                        if gen_record_counter[0] % 100 == 0:
                            print_ram_usage(f"Generation processed {gen_record_counter[0]} records")
                        print(f"    [{sys_name} | {set_type}] Item {idx}/{len(set_items)}: EM={em} F1={f1:.3f} StrictAbs={abstained_strict} SoftAbs={abstained_soft} Latency={sys_res['latency_seconds']:.2f}s", flush=True)
                        st = sys_res.get("stage_timings", {})
                        if "retrieval_primary" in st:
                            print(f"      [retrieval_primary: {st.get('retrieval_primary', 0):.4f}s -> query_encode={st.get('retrieval_primary.query_encode', 0):.4f}s, faiss={st.get('retrieval_primary.faiss_search', 0):.4f}s, bm25={st.get('retrieval_primary.bm25_search', 0):.4f}s, filter={st.get('retrieval_primary.exclusion_filter', 0):.4f}s]", flush=True)
                        ev_assess = sys_meta.get("evidence_assessment", {}) or {}
                        if ev_assess:
                            raw_resp = ev_assess.get("raw_response", "")
                            parsed = ev_assess.get("parse_ok", ev_assess.get("json_parsed", True))
                            fb_fired = ev_assess.get("fallback_fired", False)
                            st_val = ev_assess.get("status", sys_meta.get("evidence_status", "UNKNOWN"))
                            print(f"      [CARO Assessment] status={st_val} | parse_ok={parsed} | fallback_fired={fb_fired} | raw='{raw_resp[:120]}'", flush=True)

                    gc.collect()
                    return record

                if workers > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        future_to_item = {executor.submit(process_item, it, idx): it for idx, it in enumerate(remaining_items, len(existing_results) + 1)}
                        for future in concurrent.futures.as_completed(future_to_item):
                            try:
                                future.result()
                            except Exception as exc:
                                item_failed = future_to_item[future]
                                print(f"    [THREAD ERROR] Question {item_failed['question_id']} failed: {exc}", flush=True)
                else:
                    for idx, it in enumerate(remaining_items, len(existing_results) + 1):
                        process_item(it, idx)

    # -------------------------------------------------------------
    # PHASE 2: BATCH JUDGING (Llama 3.1 8B stays resident)
    # Skipped in --skip-grading mode
    # -------------------------------------------------------------
    if not skip_grading:
        print("\n" + "=" * 50, flush=True)
        print(f"PHASE 2: BATCH JUDGING (Model: {JUDGE_MODEL})", flush=True)
        print("=" * 50, flush=True)

        judge = LLMJudge.get_client(model_name=JUDGE_MODEL)
        judge_workers = min(max(1, workers), 2)
        judge_record_counter = [0]

        for sys_name in system_names:
            for set_type in target_sets:
                result_file = RESULTS_DIR / f"{sys_name}_{set_type}_seed{seed}.jsonl"
                if not result_file.exists():
                    continue

                records = []
                with open(result_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))

                ungraded_indices = [
                    i for i, r in enumerate(records)
                    if r.get("judge_verdict") is None or r.get("error") is not None
                ]
                if not ungraded_indices:
                    continue

                print(f"  Grading {len(ungraded_indices)} records for {sys_name} [{set_type}]...", flush=True)
                judge_lock = threading.Lock()

                # Step 1: Missing faithfulness evaluated with workers=1
                missing_faith = [i for i in ungraded_indices if records[i].get("faithfulness") is None or records[i].get("error") is not None]
                for idx in missing_faith:
                    r = records[idx]
                    try:
                        r["faithfulness"] = FaithfulnessChecker.check_faithfulness(
                            r.get("final_answer_raw") or r.get("final_answer", ""),
                            r.get("retrieved_docs", []),
                            question=r.get("question", ""),
                            abstained=bool(r.get("abstained_strict", False))
                        )
                    except Exception as nli_err:
                        r["error"] = f"NLI error: {nli_err}"

                def grade_record(idx):
                    r = records[idx]
                    try:
                        if r.get("error") and "NLI error" in str(r.get("error")):
                            raise RuntimeError(r["error"])
                        judge_res = judge.judge_answer(r["question"], r["gold_answer"], r["final_answer"])
                        with judge_lock:
                            r["judge_verdict"] = judge_res["verdict"]
                            r["judge_score"] = judge_res["score"]
                            r["judge_reason"] = judge_res["reason"]
                            r["error"] = None
                            judge_record_counter[0] += 1
                            # Write immediately so progress is never lost across runs
                            tmp_file = result_file.with_suffix(".tmp")
                            with open(tmp_file, "w", encoding="utf-8") as f:
                                for rec in records:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            tmp_file.replace(result_file)
                    except Exception as exc:
                        print(f"    [JUDGE ERROR] Grading record failed for {r.get('question_id')}: {exc}", flush=True)
                        with judge_lock:
                            r["error"] = str(exc)
                            judge_record_counter[0] += 1
                            tmp_file = result_file.with_suffix(".tmp")
                            with open(tmp_file, "w", encoding="utf-8") as f:
                                for rec in records:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            tmp_file.replace(result_file)

                    if judge_record_counter[0] % 100 == 0:
                        print_ram_usage(f"Grading processed {judge_record_counter[0]} records")

                if judge_workers > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers) as executor:
                        futures = [executor.submit(grade_record, idx) for idx in ungraded_indices]
                        for f in concurrent.futures.as_completed(futures):
                            try:
                                f.result()
                            except Exception:
                                pass
                else:
                    for idx in ungraded_indices:
                        grade_record(idx)

    # -------------------------------------------------------------
    # SUMMARY REPORT DISPLAY
    # -------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("RUN SUMMARY RESULTS", flush=True)
    print("=" * 80, flush=True)

    # Ensure summary_records has records for all evaluated systems
    for s in system_names:
        if not summary_records[s]:
            for st in target_sets:
                rf = RESULTS_DIR / f"{s}_{st}_seed{seed}.jsonl"
                if rf.exists():
                    with open(rf, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    summary_records[s].append(json.loads(line))
                                except Exception:
                                    pass

    # 1. F1 and EM per system on ANSWERABLE set
    print("\n--- 1. Answer Quality (ANSWERABLE Set) ---", flush=True)
    print(f"{'System':<16} | {'Exact Match':<12} | {'Token F1':<12} | {'Mean Latency (s)':<16}", flush=True)
    print("-" * 62, flush=True)
    for s in system_names:
        ans_recs = [r for r in summary_records[s] if r["set_type"] == "ANSWERABLE"]
        if ans_recs:
            em_rate = np.mean([float(r["exact_match"]) for r in ans_recs])
            f1_mean = np.mean([float(r["token_f1"]) for r in ans_recs])
            lat_mean = np.mean([float(r["latency_seconds"]) for r in ans_recs])
            print(f"{s:<16} | {em_rate:<12.3f} | {f1_mean:<12.3f} | {lat_mean:<16.2f}", flush=True)

    # 2. Abstention counts (Strict and Soft) per set
    print("\n--- 2. Abstention Counts per Set ---", flush=True)
    for s in system_names:
        print(f"\nSystem: {s}", flush=True)
        for st in target_sets:
            st_recs = [r for r in summary_records[s] if r["set_type"] == st]
            strict_count = sum(bool(r.get("abstained_strict", False)) for r in st_recs)
            soft_count = sum(bool(r.get("abstained_soft", False)) for r in st_recs)
            total = len(st_recs)
            print(f"  {st:<14} | Total: {total:<3} | Strict Abstained: {strict_count}/{total} | Soft Abstained: {soft_count}/{total}", flush=True)

    # 2b. Abstention rates stratified by both_gold_retrieved (0 vs 1)
    print("\n--- 2b. Abstention Rates Stratified by both_gold_retrieved (0 vs 1) ---", flush=True)
    for s in system_names:
        print(f"\nSystem: {s}", flush=True)
        for st in target_sets:
            st_recs = [r for r in summary_records[s] if r["set_type"] == st]
            for stratum_val in (1.0, 0.0):
                strat_recs = [r for r in st_recs if float(r.get("retrieval_metrics", {}).get("both_gold_retrieved", 0.0)) == stratum_val]
                strict_cnt = sum(bool(r.get("abstained_strict", False)) for r in strat_recs)
                total_strat = len(strat_recs)
                rate_str = f"{strict_cnt}/{total_strat} ({strict_cnt/total_strat*100:.1f}%)" if total_strat > 0 else "0/0 (N/A)"
                print(f"  {st:<14} | both_gold_retrieved={int(stratum_val)} | Total: {total_strat:<3} | Strict Abstained: {rate_str}", flush=True)

    # 3. Evidence Status and Parse/Fallback distribution for CARO
    caro_recs = summary_records.get("CARO", [])
    if caro_recs:
        print("\n--- 3. CARO Evidence Status & Parser Robustness ---", flush=True)
        tot_fb = sum(bool((r.get("evidence_assessment") or {}).get("fallback_fired", False)) for r in caro_recs)
        tot_parse_ok = sum(bool((r.get("evidence_assessment") or {}).get("parse_ok", True)) for r in caro_recs)
        print(f"  Total CARO Records: {len(caro_recs)} | Parse OK: {tot_parse_ok}/{len(caro_recs)} | Fallback Fired: {tot_fb}/{len(caro_recs)} (Target: 0)", flush=True)
        for st in target_sets:
            st_recs = [r for r in caro_recs if r["set_type"] == st]
            statuses = Counter(r.get("evidence_status", "UNKNOWN") for r in st_recs)
            status_str = ", ".join(f"{k}: {v}" for k, v in statuses.items())
            fb_count = sum(bool((r.get("evidence_assessment") or {}).get("fallback_fired", False)) for r in st_recs)
            p_ok = sum(bool((r.get("evidence_assessment") or {}).get("parse_ok", True)) for r in st_recs)
            total = len(st_recs)
            print(f"  {st:<14} | {status_str} | Parse OK: {p_ok}/{total} | Fallback Fired: {fb_count}/{total}", flush=True)

        unans_caro = [r for r in caro_recs if r["set_type"] == "UNANSWERABLE"]
        suff_wrong = sum(bool(r.get("evidence_status") == "SUFFICIENT" and not r.get("abstained_strict")) for r in unans_caro)
        print(f"\n  CARO SUFFICIENT-but-wrong answers on UNANSWERABLE: {suff_wrong}/{len(unans_caro)}", flush=True)

    # 4. Answers for all records (short)
    print("\n--- 4. All Record Answers (Summary) ---", flush=True)
    print(f"{'System':<8} | {'Set':<12} | {'QID':<24} | {'Gold Answer':<20} | {'Final Answer':<32} | {'EM':<5} | {'Status/Abs':<14}", flush=True)
    print("-" * 125, flush=True)
    for s in system_names:
        for r in summary_records[s]:
            q_id = r["question_id"]
            set_t = r["set_type"]
            g_ans = (r["gold_answer"][:18] + "..") if len(r["gold_answer"]) > 18 else r["gold_answer"]
            f_ans = (r["final_answer"][:30] + "..") if len(r["final_answer"]) > 30 else r["final_answer"]
            em_val = str(r.get("exact_match", False))
            if s == "CARO":
                ev_st = r.get("evidence_status", "")
                status_abs = f"{ev_st} (fb={r.get('fallback_fired', False)})"
            else:
                status_abs = "ABSTAINED" if r.get("abstained_strict") else "ANSWERED"
            print(f"{s:<8} | {set_t:<12} | {q_id:<24} | {g_ans:<20} | {f_ans:<32} | {em_val:<5} | {status_abs:<14}", flush=True)

    # 5. Stage timing table
    all_caro_timings = [r.get("stage_timings", {}) for r in caro_recs if r.get("stage_timings")]
    if all_caro_timings:
        print("\n--- 5. CARO Per-Stage Timing Summary (Mean Seconds) ---", flush=True)
        stage_names = [
            "router", "retrieval_primary",
            "retrieval_primary.query_encode", "retrieval_primary.faiss_search",
            "retrieval_primary.bm25_search", "retrieval_primary.exclusion_filter",
            "expansion", "decomposition",
            "retrieval_variants", "retrieval_cross_strategy", "retrieval_fusion",
            "assessment", "generation", "total"
        ]
        mean_tot = np.mean([t.get("total", 0.0) for t in all_caro_timings]) or 1e-9
        for st in stage_names:
            vals = [t.get(st, 0.0) for t in all_caro_timings]
            m_val = np.mean(vals) if vals else 0.0
            pct = (m_val / max(mean_tot, 1e-9)) * 100.0
            print(f"  {st:<26} | Mean: {m_val:6.3f}s | {pct:5.1f}%", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("EVALUATION RUN COMPLETE", flush=True)
    print("=" * 80, flush=True)
    return {"status": "COMPLETE", "seed": seed, "smoke": smoke}


def run_grade_only(
    selected_systems: Optional[List[str]] = None,
    target_sets: Tuple[str, ...] = ("ANSWERABLE", "PARTIAL", "UNANSWERABLE"),
    seed: Optional[int] = None,
    workers: int = 1,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Grades existing generation records in eval/results without loading corpus assets.
    - No CorpusAssets.load (no FAISS, BM25, BGE-M3).
    - Reads generation records from eval/results.
    - Grades only records with no judge_verdict (or error != null).
    - Writes each record immediately, resumable.
    - Default workers 1 for NLI, up to 2 for judge calls.
    - A record that raised an exception is written with error set, and resume redoes records with error != null.
    - Prints RAM after each model load and every 100 records.
    """
    res_dir = results_dir or RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"RUNNING GRADE-ONLY | Systems: {selected_systems or 'ALL'} | Sets: {target_sets} | Seed: {seed or 'ALL'}", flush=True)
    print("=" * 80, flush=True)

    # 1. Discover target result files
    all_files = sorted(res_dir.glob("*.jsonl"))
    target_files = []
    for f in all_files:
        stem = f.stem
        # Format is typically <system>_<set_type>_seed<seed>.jsonl
        parts = stem.split("_")
        if len(parts) >= 3:
            sys_candidate = "_".join(parts[:-2])
            set_candidate = parts[-2]
            seed_candidate = parts[-1]
            if selected_systems and sys_candidate not in selected_systems:
                continue
            if target_sets and set_candidate not in target_sets:
                continue
            if seed is not None and f"seed{seed}" != seed_candidate and not stem.endswith(f"seed{seed}"):
                continue
            target_files.append(f)
        else:
            if not selected_systems and not target_sets:
                target_files.append(f)

    if not target_files:
        print(f"[Grade-Only] No matching result files found in {res_dir}.", flush=True)
        return {"status": "NO_FILES", "graded_count": 0}

    print(f"[Grade-Only] Found {len(target_files)} target file(s) to grade.", flush=True)

    judge = LLMJudge.get_client(model_name=JUDGE_MODEL)
    judge_workers = min(max(1, workers if workers > 1 else 2), 2)
    total_graded = 0
    graded_counter = [0]

    for result_file in target_files:
        records = []
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        records.append(json.loads(line_str))
                    except Exception as e:
                        print(f"    [Warning] Corrupted line in {result_file}: {e}", flush=True)

        if not records:
            continue

        ungraded_indices = [
            i for i, r in enumerate(records)
            if r.get("judge_verdict") is None or r.get("error") is not None
        ]

        if not ungraded_indices:
            print(f"  [{result_file.name}] All {len(records)} records already graded.", flush=True)
            continue

        print(f"  [{result_file.name}] Grading {len(ungraded_indices)} / {len(records)} records (Judge workers: {judge_workers})...", flush=True)
        file_lock = threading.Lock()

        # Step 1: Missing faithfulness evaluated with workers=1
        missing_faith_indices = [
            i for i in ungraded_indices
            if records[i].get("faithfulness") is None or records[i].get("error") is not None
        ]
        if missing_faith_indices:
            print(f"    Evaluating NLI faithfulness for {len(missing_faith_indices)} records (workers=1)...", flush=True)
            for idx in missing_faith_indices:
                r = records[idx]
                try:
                    q_text = r.get("question", "")
                    ans_raw = r.get("final_answer_raw") or r.get("final_answer", "")
                    ret_docs = r.get("retrieved_docs", [])
                    abstained = bool(r.get("abstained_strict", False))
                    r["faithfulness"] = FaithfulnessChecker.check_faithfulness(
                        ans_raw, ret_docs, question=q_text, abstained=abstained
                    )
                except Exception as nli_err:
                    print(f"    [NLI Error] {r.get('question_id')}: {nli_err}", flush=True)
                    r["error"] = f"NLI error: {nli_err}"

        def grade_single_record(idx: int):
            r = records[idx]
            qid = r.get("question_id", "unknown")
            try:
                if r.get("error") and "NLI error" in str(r.get("error")):
                    raise RuntimeError(r["error"])

                judge_res = judge.judge_answer(
                    r.get("question", ""),
                    r.get("gold_answer", ""),
                    r.get("final_answer", "")
                )
                with file_lock:
                    r["judge_verdict"] = judge_res["verdict"]
                    r["judge_score"] = judge_res["score"]
                    r["judge_reason"] = judge_res["reason"]
                    r["error"] = None
                    graded_counter[0] += 1
                    # Write immediately
                    tmp_file = result_file.with_suffix(".tmp")
                    with open(tmp_file, "w", encoding="utf-8") as f_out:
                        for rec in records:
                            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    tmp_file.replace(result_file)
            except Exception as exc:
                print(f"    [JUDGE ERROR] Question {qid} failed during grading: {exc}", flush=True)
                with file_lock:
                    r["error"] = str(exc)
                    graded_counter[0] += 1
                    # Write immediately with error set
                    tmp_file = result_file.with_suffix(".tmp")
                    with open(tmp_file, "w", encoding="utf-8") as f_out:
                        for rec in records:
                            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    tmp_file.replace(result_file)

            if graded_counter[0] % 100 == 0:
                print_ram_usage(f"Graded {graded_counter[0]} records")

        if judge_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers) as executor:
                futures = [executor.submit(grade_single_record, idx) for idx in ungraded_indices]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception:
                        pass
        else:
            for idx in ungraded_indices:
                grade_single_record(idx)

        total_graded += len(ungraded_indices)

    print("\n" + "=" * 80, flush=True)
    print(f"GRADE-ONLY COMPLETE | Total records processed: {total_graded}", flush=True)
    print("=" * 80, flush=True)
    return {"status": "COMPLETE", "total_graded": total_graded}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fair evaluation harness for CARO RAG vs baselines.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on N=10 questions.")
    parser.add_argument("--n", type=int, default=None, help="Sample size per set (e.g. --n 5).")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of questions to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent worker threads (default 1).")
    parser.add_argument("--device", type=str, default="cpu", help="Device for bge-m3 embedding model ('cpu' locally, 'cuda' on Colab).")
    parser.add_argument("--systems", type=str, default=None, help="Comma or space-separated list of systems to run.")
    parser.add_argument("--sets", type=str, default=None, help="Comma or space-separated list of sets to run.")
    parser.add_argument("--skip-grading", "--skip_grading", action="store_true", help="Skip LLM Judge and NLI Cross-Encoder grading.")
    parser.add_argument("--grade-only", "--grade_only", action="store_true", help="Grade existing generation records without loading corpus assets.")
    parser.add_argument("--step1", action="store_true", help="Run 3 CARO questions first and print per-stage timing table.")
    parser.add_argument("--force", "--overwrite", action="store_true", help="Overwrite existing results instead of resuming.")
    args = parser.parse_args()

    # Parse systems with alias normalization
    SYSTEM_ALIASES = {
        "b1": "B1_Dense", "b1_dense": "B1_Dense",
        "b2": "B2_BM25", "b2_bm25": "B2_BM25",
        "b3": "B3_Hybrid", "b3_hybrid": "B3_Hybrid",
        "b4": "B4_RAGFusion", "b4_ragfusion": "B4_RAGFusion",
        "b3_abs": "B3_abs", "caro": "CARO", "caro_fast": "CARO_fast"
    }
    sel_sys = None
    if args.systems:
        if "," in args.systems:
            raw_sys = [s.strip() for s in args.systems.split(",") if s.strip()]
        else:
            raw_sys = [s.strip() for s in args.systems.split() if s.strip()]
        sel_sys = [SYSTEM_ALIASES.get(s.lower(), s) for s in raw_sys]

    # Parse target sets
    target_sets = ("ANSWERABLE", "PARTIAL", "UNANSWERABLE")
    if args.sets:
        if "," in args.sets:
            target_sets = tuple(s.strip() for s in args.sets.split(",") if s.strip())
        else:
            target_sets = tuple(s.strip() for s in args.sets.split() if s.strip())

    if args.grade_only:
        run_grade_only(
            selected_systems=sel_sys,
            target_sets=target_sets,
            seed=args.seed if "--seed" in sys.argv else None,
            workers=args.workers,
        )
    elif args.step1:
        run_step1_profiling(seed=args.seed, device=args.device)
    else:
        effective_n = args.n if args.n is not None else args.sample_size
        run_evaluation(
            smoke=args.smoke,
            sample_size=effective_n,
            seed=args.seed,
            selected_systems=sel_sys,
            target_sets=target_sets,
            skip_grading=args.skip_grading,
            force=args.force,
            workers=args.workers,
            device=args.device
        )
