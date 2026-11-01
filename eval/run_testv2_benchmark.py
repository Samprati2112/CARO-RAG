"""
CARO RAG vs Baseline RAG: TestV2 Comprehensive 500-Query Benchmark Runner.

Outputs saved to:
  - eval/results/testv2_comparison.csv
  - eval/results/testv2_comparison.jsonl
  - eval/results/testv2_summary_metrics.json

Features:
  - Side-by-side execution of Baseline RAG (B1_Dense) vs Upgraded CARO RAG.
  - Fully resumable: automatically continues from the last completed query.
  - Real-time logging of accuracy, retrieval count, latencies, and comparison metrics.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.caro_core import CorpusAssets, PROJECT_ROOT
from eval.baselines import DenseOnlyRAG, CaroPipelineAdapter
from eval.dataset import prepare_evaluation_dataset
from eval.common import (
    exact_match_score,
    token_f1_score,
    compute_retrieval_metrics,
    normalize_answer,
)

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "testv2_comparison.csv"
JSONL_FILE = RESULTS_DIR / "testv2_comparison.jsonl"
SUMMARY_FILE = RESULTS_DIR / "testv2_summary_metrics.json"

CSV_FIELDNAMES = [
    "query_number",
    "question_id",
    "question",
    "gold_answer",
    "baseline_answer",
    "baseline_is_correct",
    "baseline_f1",
    "baseline_latency_seconds",
    "baseline_recall_at_8",
    "caro_answer",
    "caro_is_correct",
    "caro_f1",
    "caro_latency_seconds",
    "caro_retrieval_calls",
    "caro_strategy",
    "caro_evidence_status",
    "caro_abstained",
    "caro_recall_at_8",
    "winner",
]


def load_completed_ids() -> set:
    completed = set()
    if JSONL_FILE.exists():
        with open(JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed.add(record.get("question_id"))
                    except Exception:
                        pass
    return completed


def init_csv_file():
    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()


def update_summary():
    if not JSONL_FILE.exists():
        return
    records = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        return

    n = len(records)
    b_em = sum(1 for r in records if r["baseline_is_correct"])
    c_em = sum(1 for r in records if r["caro_is_correct"])
    b_f1 = sum(r["baseline_f1"] for r in records) / n
    c_f1 = sum(r["caro_f1"] for r in records) / n
    b_lat = sum(r["baseline_latency_seconds"] for r in records) / n
    c_lat = sum(r["caro_latency_seconds"] for r in records) / n
    c_ret = sum(r["caro_retrieval_calls"] for r in records) / n
    c_abs = sum(1 for r in records if r["caro_abstained"])

    wins_caro = sum(1 for r in records if r["winner"] == "CARO")
    wins_base = sum(1 for r in records if r["winner"] == "BASELINE")
    ties = sum(1 for r in records if "TIE" in r["winner"])

    summary = {
        "total_evaluated_queries": n,
        "baseline_exact_match_count": b_em,
        "baseline_exact_match_pct": round(100.0 * b_em / n, 2),
        "baseline_mean_f1": round(b_f1, 4),
        "baseline_mean_latency_s": round(b_lat, 2),
        "caro_exact_match_count": c_em,
        "caro_exact_match_pct": round(100.0 * c_em / n, 2),
        "caro_mean_f1": round(c_f1, 4),
        "caro_mean_latency_s": round(c_lat, 2),
        "caro_mean_retrievals": round(c_ret, 2),
        "caro_abstention_count": c_abs,
        "caro_abstention_pct": round(100.0 * c_abs / n, 2),
        "head_to_head": {
            "caro_wins": wins_caro,
            "baseline_wins": wins_base,
            "ties": ties,
        },
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TestV2: CARO vs Baseline 500-Query Benchmark")
    parser.add_argument("--n", type=int, default=500, help="Number of queries to run (default 500)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for BGE-M3 (cpu or cuda)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing testv2 output")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(" CARO RAG vs BASELINE RAG: TestV2 Benchmark Suite", flush=True)
    print("=" * 70, flush=True)

    if args.force:
        if CSV_FILE.exists():
            CSV_FILE.unlink()
        if JSONL_FILE.exists():
            JSONL_FILE.unlink()
        if SUMMARY_FILE.exists():
            SUMMARY_FILE.unlink()
        print("Existing testv2 files deleted (--force).", flush=True)

    init_csv_file()
    completed_ids = load_completed_ids()
    print(f"Resuming benchmark: {len(completed_ids)} queries already completed.", flush=True)

    print("\n[Step 1/3] Loading 250,000-document Corpus Assets...", flush=True)
    t0 = time.time()
    CorpusAssets.load(device=args.device)
    print(f"Corpus Assets ready in {time.time() - t0:.2f}s.\n", flush=True)

    print("[Step 2/3] Loading HotpotQA 500-Question Evaluation Dataset...", flush=True)
    questions, _ = prepare_evaluation_dataset(sample_size=args.n, seed=42, device=args.device)
    print(f"Loaded {len(questions)} evaluation questions.\n", flush=True)

    print("[Step 3/3] Initializing Baseline and CARO Adapters...", flush=True)
    baseline = DenseOnlyRAG()
    caro = CaroPipelineAdapter()
    print("Pipelines initialized.\n", flush=True)

    print("=" * 70, flush=True)
    print(" Commencing Head-to-Head Evaluation", flush=True)
    print("=" * 70, flush=True)

    for idx, q_item in enumerate(questions[:args.n], 1):
        q_id = q_item["question_id"]
        question = q_item["question"]
        gold_ans = q_item["gold_answer"]
        gold_titles = q_item.get("gold_titles", [])

        if q_id in completed_ids:
            continue

        print(f"\n--- Query [{idx}/{args.n}] (ID: {q_id}) ---", flush=True)
        print(f"Q: {question}", flush=True)
        print(f"Gold: {gold_ans}", flush=True)

        # 1. Run Baseline RAG
        t_b0 = time.time()
        try:
            b_res = baseline.answer(question)
            b_lat = round(time.time() - t_b0, 2)
            b_ans = b_res.get("final_answer", "")
            b_docs = b_res.get("retrieved_docs", [])
        except Exception as e:
            print(f"  [Baseline Error]: {e}", flush=True)
            b_ans = f"ERROR: {e}"
            b_lat = round(time.time() - t_b0, 2)
            b_docs = []

        # 2. Run CARO RAG
        t_c0 = time.time()
        try:
            c_res = caro.answer(question)
            c_lat = round(time.time() - t_c0, 2)
            c_ans = c_res.get("final_answer", "")
            c_meta = c_res.get("metadata", {})
            c_docs = c_res.get("retrieved_docs", [])
            c_ret_calls = c_res.get("retrieval_calls", 1)
            c_strat = c_meta.get("strategy", "HYBRID")
            c_status = c_meta.get("evidence_status", "SUFFICIENT")
            c_abs = c_res.get("abstained_strict", False)
        except Exception as e:
            print(f"  [CARO Error]: {e}", flush=True)
            c_ans = f"ERROR: {e}"
            c_lat = round(time.time() - t_c0, 2)
            c_docs = []
            c_ret_calls = 1
            c_strat = "ERROR"
            c_status = "ERROR"
            c_abs = True

        # Metrics Calculation
        b_em = exact_match_score(b_ans, gold_ans)
        b_f1, _, _ = token_f1_score(b_ans, gold_ans)
        b_ret_m = compute_retrieval_metrics(b_docs, gold_titles)
        b_rec8 = b_ret_m.get("recall@8", 0.0)

        c_em = exact_match_score(c_ans, gold_ans) and not c_abs
        c_f1, _, _ = token_f1_score(c_ans, gold_ans) if not c_abs else (0.0, 0.0, 0.0)
        c_ret_m = compute_retrieval_metrics(c_docs, gold_titles)
        c_rec8 = c_ret_m.get("recall@8", 0.0)

        # Determine Winner
        if c_em and not b_em:
            winner = "CARO"
        elif b_em and not c_em:
            winner = "BASELINE"
        elif c_em and b_em:
            winner = "TIE_BOTH_CORRECT"
        else:
            if c_f1 > b_f1 + 0.05:
                winner = "CARO"
            elif b_f1 > c_f1 + 0.05:
                winner = "BASELINE"
            else:
                winner = "TIE_BOTH_WRONG"

        print(f"  Baseline: {b_ans} | Correct={b_em} | F1={b_f1:.3f} | Latency={b_lat}s", flush=True)
        print(f"  CARO:     {c_ans} | Correct={c_em} | F1={c_f1:.3f} | Latency={c_lat}s | Passes={c_ret_calls} | Strat={c_strat} | Status={c_status}", flush=True)
        print(f"  --> Winner: {winner}", flush=True)

        record = {
            "query_number": idx,
            "question_id": q_id,
            "question": question,
            "gold_answer": gold_ans,
            "gold_titles": gold_titles,
            "baseline_answer": b_ans,
            "baseline_is_correct": b_em,
            "baseline_f1": round(b_f1, 4),
            "baseline_latency_seconds": b_lat,
            "baseline_recall_at_8": b_rec8,
            "caro_answer": c_ans,
            "caro_is_correct": c_em,
            "caro_f1": round(c_f1, 4),
            "caro_latency_seconds": c_lat,
            "caro_retrieval_calls": c_ret_calls,
            "caro_strategy": c_strat,
            "caro_evidence_status": c_status,
            "caro_abstained": c_abs,
            "caro_recall_at_8": c_rec8,
            "winner": winner,
        }

        # Append to JSONL
        with open(JSONL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Append to CSV
        csv_row = {k: record[k] for k in CSV_FIELDNAMES}
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writerow(csv_row)

        completed_ids.add(q_id)
        update_summary()

    print("\n" + "=" * 70, flush=True)
    print(" Benchmark Run Finished! Summary Metrics:", flush=True)
    print("=" * 70, flush=True)
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            print(f.read(), flush=True)


if __name__ == "__main__":
    main()
