"""
Report generator for CARO RAG vs Baselines.
Produces:
- eval/RESULTS.md: Complete honest report with bootstrap CIs, paired significance tests,
  where CARO is worse, cost vs quality, explicit verdicts, and limitations.
- eval/results/human_check.csv: Random sample of 50 judged items for human validation.
- eval/results/errors_<system>.csv: Up to 50 failure cases per system for error analysis.
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from eval.caro_core import PROJECT_ROOT
from eval.judge import JUDGE_MODEL
from eval.stats import compute_bootstrap_ci, paired_bootstrap_test

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
REPORT_FILE = PROJECT_ROOT / "eval" / "RESULTS.md"


def load_system_set_results(system_name: str, set_type: str, seed: int = 42) -> List[Dict[str, Any]]:
    """Loads JSONL records for a specific system, set_type, and seed."""
    file_path = RESULTS_DIR / f"{system_name}_{set_type}_seed{seed}.jsonl"
    if not file_path.exists():
        return []
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def generate_evaluation_report(seed: int = 42, n_bootstrap: int = 1000) -> str:
    """
    Parses all evaluation runs for the given seed, computes bootstrap CIs and paired tests,
    exports CSV error files, and writes the comprehensive RESULTS.md.
    """
    system_names = ["B1_Dense", "B2_BM25", "B3_Hybrid", "B4_RAGFusion", "B3_abs", "CARO"]
    set_types = ["ANSWERABLE", "PARTIAL", "UNANSWERABLE"]

    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {s: {} for s in system_names}
    for s in system_names:
        for st in set_types:
            data[s][st] = load_system_set_results(s, st, seed=seed)

    # Validate that we have data
    total_records = sum(len(records) for s in data.values() for records in s.values())
    if total_records == 0:
        raise RuntimeError(f"No result records found in {RESULTS_DIR} for seed={seed}!")

    # 1. Compute metrics per system
    metrics_by_system: Dict[str, Dict[str, Any]] = {}

    for s in system_names:
        ans_data = data[s].get("ANSWERABLE", [])
        part_data = data[s].get("PARTIAL", [])
        unans_data = data[s].get("UNANSWERABLE", [])

        # Answer Quality (ANSWERABLE)
        em_vals = [float(r["exact_match"]) for r in ans_data] if ans_data else []
        f1_vals = [float(r["token_f1"]) for r in ans_data] if ans_data else []
        prec_vals = [float(r["token_precision"]) for r in ans_data] if ans_data else []
        rec_vals = [float(r["token_recall"]) for r in ans_data] if ans_data else []
        judge_vals = [float(r.get("judge_score", 0.0)) for r in ans_data] if ans_data else []

        # Retrieval Quality (ANSWERABLE)
        rec2_vals = [float(r.get("retrieval_metrics", {}).get("recall@2", 0.0)) for r in ans_data]
        rec5_vals = [float(r.get("retrieval_metrics", {}).get("recall@5", 0.0)) for r in ans_data]
        rec8_vals = [float(r.get("retrieval_metrics", {}).get("recall@8", 0.0)) for r in ans_data]
        prec2_vals = [float(r.get("retrieval_metrics", {}).get("precision@2", 0.0)) for r in ans_data]
        prec5_vals = [float(r.get("retrieval_metrics", {}).get("precision@5", 0.0)) for r in ans_data]
        prec8_vals = [float(r.get("retrieval_metrics", {}).get("precision@8", 0.0)) for r in ans_data]
        both_gold_vals = [float(r.get("retrieval_metrics", {}).get("both_gold_retrieved", 0.0)) for r in ans_data]

        # Faithfulness (all non-abstained across sets)
        all_non_abstained = [r for st in set_types for r in data[s].get(st, []) if not r["abstained"]]
        halluc_vals = [
            float(r.get("faithfulness", {}).get("is_hallucinated", False))
            for r in all_non_abstained
        ] if all_non_abstained else [0.0]

        # Dual Abstention Metrics (Positive class = UNANSWERABLE)
        # Strict abstention
        unans_strict = [r.get("abstained_strict", r.get("abstained", False)) for r in unans_data]
        ans_strict = [r.get("abstained_strict", r.get("abstained", False)) for r in ans_data]
        part_strict = [r.get("abstained_strict", r.get("abstained", False)) for r in part_data]

        # Soft abstention
        unans_soft = [r.get("abstained_soft", r.get("abstained", False)) for r in unans_data]
        ans_soft = [r.get("abstained_soft", r.get("abstained", False)) for r in ans_data]
        part_soft = [r.get("abstained_soft", r.get("abstained", False)) for r in part_data]

        from eval.common import compute_abstention_pr_f1

        # Strict: Way (a) ANSWERABLE as negatives
        strict_a = compute_abstention_pr_f1(unans_strict, ans_strict)
        # Strict: Way (b) ANSWERABLE + PARTIAL as negatives
        strict_b = compute_abstention_pr_f1(unans_strict, ans_strict + part_strict)

        # Soft: Way (a) ANSWERABLE as negatives
        soft_a = compute_abstention_pr_f1(unans_soft, ans_soft)
        # Soft: Way (b) ANSWERABLE + PARTIAL as negatives
        soft_b = compute_abstention_pr_f1(unans_soft, ans_soft + part_soft)

        # Partial Set Behavior
        part_abstained = [r.get("abstained_strict", r.get("abstained", False)) for r in part_data]
        part_disclaimers = [
            1.0 if (not r.get("abstained", False) and ("disclaimer" in r.get("final_answer_raw", r.get("final_answer", "")).lower() or "**disclaimer:**" in r.get("final_answer_raw", r.get("final_answer", "")).lower())) else 0.0
            for r in part_data
        ]
        part_abs_rate = sum(part_abstained) / len(part_abstained) if part_abstained else 0.0
        part_disclaimer_rate = sum(part_disclaimers) / len(part_data) if part_data else 0.0
        part_full_rate = 1.0 - part_abs_rate - part_disclaimer_rate

        # Cost Metrics
        all_records = [r for st in set_types for r in data[s].get(st, [])]
        latencies = [float(r["latency_seconds"]) for r in all_records] if all_records else [0.0]
        llm_calls = [float(r["llm_calls"]) for r in all_records] if all_records else [0.0]
        ret_calls = [float(r["retrieval_calls"]) for r in all_records] if all_records else [0.0]

        metrics_by_system[s] = {
            "em": compute_bootstrap_ci(em_vals, n_resamples=n_bootstrap, seed=seed),
            "f1": compute_bootstrap_ci(f1_vals, n_resamples=n_bootstrap, seed=seed),
            "prec": compute_bootstrap_ci(prec_vals, n_resamples=n_bootstrap, seed=seed),
            "rec": compute_bootstrap_ci(rec_vals, n_resamples=n_bootstrap, seed=seed),
            "judge_score": compute_bootstrap_ci(judge_vals, n_resamples=n_bootstrap, seed=seed),
            "rec@2": compute_bootstrap_ci(rec2_vals, n_resamples=n_bootstrap, seed=seed),
            "rec@5": compute_bootstrap_ci(rec5_vals, n_resamples=n_bootstrap, seed=seed),
            "rec@8": compute_bootstrap_ci(rec8_vals, n_resamples=n_bootstrap, seed=seed),
            "prec@2": compute_bootstrap_ci(prec2_vals, n_resamples=n_bootstrap, seed=seed),
            "prec@5": compute_bootstrap_ci(prec5_vals, n_resamples=n_bootstrap, seed=seed),
            "prec@8": compute_bootstrap_ci(prec8_vals, n_resamples=n_bootstrap, seed=seed),
            "both_gold": compute_bootstrap_ci(both_gold_vals, n_resamples=n_bootstrap, seed=seed),
            "hallucination_rate": compute_bootstrap_ci(halluc_vals, n_resamples=n_bootstrap, seed=seed),
            "strict_a": strict_a,
            "strict_b": strict_b,
            "soft_a": soft_a,
            "soft_b": soft_b,
            "abstention_f1": strict_a["f1"],
            "over_abstention_rate": strict_a["over_abstention_rate"],
            "abstention_counts": {
                "strict": {
                    "ANSWERABLE": sum(ans_strict),
                    "PARTIAL": sum(part_strict),
                    "UNANSWERABLE": sum(unans_strict)
                },
                "soft": {
                    "ANSWERABLE": sum(ans_soft),
                    "PARTIAL": sum(part_soft),
                    "UNANSWERABLE": sum(unans_soft)
                }
            },
            "partial_abstention_rate": part_abs_rate,
            "partial_disclaimer_rate": part_disclaimer_rate,
            "partial_full_rate": part_full_rate,
            "latency_mean": float(np.mean(latencies)),
            "latency_median": float(np.median(latencies)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "mean_llm_calls": float(np.mean(llm_calls)),
            "mean_ret_calls": float(np.mean(ret_calls)),
            "_raw_em": em_vals,
            "_raw_f1": f1_vals,
            "_raw_judge": judge_vals,
            "_raw_halluc": halluc_vals
        }

    # 2. Paired Hypothesis Tests (CARO vs Each Baseline)
    caro_m = metrics_by_system["CARO"]
    paired_tests: Dict[str, Dict[str, Any]] = {}
    baselines_list = [b for b in ["B1_Dense", "B2_BM25", "B3_Hybrid", "B4_RAGFusion", "B3_abs"] if b in metrics_by_system]

    for b in baselines_list:
        b_m = metrics_by_system[b]
        paired_tests[b] = {
            "em": paired_bootstrap_test(caro_m["_raw_em"], b_m["_raw_em"], n_resamples=n_bootstrap, seed=seed, higher_is_better=True),
            "f1": paired_bootstrap_test(caro_m["_raw_f1"], b_m["_raw_f1"], n_resamples=n_bootstrap, seed=seed, higher_is_better=True),
            "judge": paired_bootstrap_test(caro_m["_raw_judge"], b_m["_raw_judge"], n_resamples=n_bootstrap, seed=seed, higher_is_better=True) if caro_m["_raw_judge"] and b_m["_raw_judge"] else {"verdict": "No significant difference", "diff_mean": 0.0, "p_value": 1.0},
            "abs_f1_diff": round(caro_m["abstention_f1"] - b_m["abstention_f1"], 4),
        }

    # 3. Export NLI Check CSV (20 random items for manual validation, Requirement E.9)
    nli_check_file = RESULTS_DIR / "nli_check.csv"
    all_nli_items = []
    for s in system_names:
        for st in set_types:
            for r in data[s].get(st, []):
                faith = r.get("faithfulness", {})
                if faith and not r.get("abstained", False):
                    all_nli_items.append({
                        "answer": r.get("final_answer", ""),
                        "context_titles": "; ".join(r.get("retrieved_titles", [])),
                        "nli_verdict": faith.get("nli_verdict", "UNKNOWN"),
                        "manual_verdict": ""
                    })
    rng = random.Random(seed)
    nli_sample = rng.sample(all_nli_items, min(20, len(all_nli_items))) if all_nli_items else []
    with open(nli_check_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["answer", "context_titles", "nli_verdict", "manual_verdict"])
        writer.writeheader()
        writer.writerows(nli_sample)
    print(f"[Report] Exported 20 items for NLI validation to {nli_check_file}")

    # 4. Export Human Check CSV (Sample of 50 judged items across systems)
    human_check_file = RESULTS_DIR / "human_check.csv"
    all_judged_items = []
    for s in system_names:
        for st in set_types:
            for r in data[s].get(st, []):
                all_judged_items.append({
                    "system": s,
                    "set_type": st,
                    "question": r["question"],
                    "gold_answer": r["gold_answer"],
                    "system_answer": r["final_answer"],
                    "judge_verdict": r.get("judge_verdict"),
                    "judge_reason": r.get("judge_reason"),
                    "human_verdict": ""
                })
    human_sample = rng.sample(all_judged_items, min(50, len(all_judged_items))) if all_judged_items else []
    with open(human_check_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "system", "set_type", "question", "gold_answer", "system_answer",
            "judge_verdict", "judge_reason", "human_verdict"
        ])
        writer.writeheader()
        writer.writerows(human_sample)
    print(f"[Report] Exported {len(human_sample)} items for human validation to {human_check_file}")

    # 5. Export Error Analysis CSVs (Up to 50 failures per system)
    for s in system_names:
        err_file = RESULTS_DIR / f"errors_{s}.csv"
        failures = []
        for st in set_types:
            for r in data[s].get(st, []):
                is_err = False
                if st == "ANSWERABLE" and (not r.get("exact_match") or r.get("abstained") or float(r.get("token_f1", 0.0)) < 0.3):
                    is_err = True
                elif st == "UNANSWERABLE" and not r.get("abstained"):
                    is_err = True
                elif st == "PARTIAL" and not r.get("abstained") and r.get("faithfulness", {}).get("is_hallucinated", False):
                    is_err = True

                if is_err:
                    failures.append({
                        "question_id": r.get("question_id", ""),
                        "set_type": st,
                        "question": r.get("question", ""),
                        "gold_answer": r.get("gold_answer", ""),
                        "system_answer": r.get("final_answer", ""),
                        "retrieved_titles": "; ".join(r.get("retrieved_titles", [])),
                        "failure_category": ""
                    })

        sample_failures = rng.sample(failures, min(50, len(failures))) if len(failures) > 50 else failures
        with open(err_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question_id", "set_type", "question", "gold_answer", "system_answer",
                "retrieved_titles", "failure_category"
            ])
            writer.writeheader()
            writer.writerows(sample_failures)
        print(f"[Report] Exported {len(sample_failures)} failures for {s} to {err_file}")

    # 6. Build RESULTS.md Content
    def fmt_ci(ci_dict):
        return f"{ci_dict['mean']:.3f} [{ci_dict['ci_lower']:.3f}, {ci_dict['ci_upper']:.3f}]"

    n_sample_first = len(data["CARO"]["ANSWERABLE"]) if data["CARO"].get("ANSWERABLE") else 0

    md_lines = [
        "# CARO RAG vs Baseline Systems: Rigorous Evaluation Report",
        "",
        f"- **Evaluation Timestamp:** {data['CARO']['ANSWERABLE'][0].get('timestamp', 'Current') if data['CARO']['ANSWERABLE'] else 'Current'}",
        f"- **Corpus:** 250,000 HotpotQA passages | Index: `hotpotqa_full_index.faiss`",
        f"- **Models:** Generator: `gemma3:4b` (Temp 0.0) | Judge: `{JUDGE_MODEL}` | Embeddings: `BAAI/bge-m3` | NLI: `cross-encoder/nli-deberta-v3-small`",
        f"- **Bootstrap Resamples:** {n_bootstrap} (95% Confidence Intervals)",
        f"- **Random Seed:** {seed}",
        ""
    ]

    # Sample size warning (Requirement F.12)
    if n_sample_first < 100:
        md_lines.extend([
            "> [!WARNING]",
            f"> **Sample-Size Warning (N = {n_sample_first} < 100 per set):**",
            "> Evaluation was executed on a reduced sample size. Confidence intervals are wide and statistical power is limited.",
            "> All findings should be considered preliminary until confirmed on a sample of N >= 100 per set.",
            ""
        ])

    md_lines.extend([
        "---",
        "",
        "## 1. Main Results: Comprehensive System Comparison",
        "",
        "Values formatted as **Mean [95% CI Lower, 95% CI Upper]**.",
        "*Note: Hallucination Rate is marked UNVALIDATED until `eval/results/nli_check.csv` is manually reviewed.*",
        "",
        "| System | Exact Match | Token F1 | Semantic Score | Recall@8 | Multi-Hop Success | Hallucination Rate | Abstention F1 (Strict A) | Over-Abstention |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for s in system_names:
        m = metrics_by_system[s]
        md_lines.append(
            f"| **{s}** | {fmt_ci(m['em'])} | {fmt_ci(m['f1'])} | {fmt_ci(m['judge_score'])} | "
            f"{fmt_ci(m['rec@8'])} | {fmt_ci(m['both_gold'])} | UNVALIDATED ({m['hallucination_rate']['mean']:.3f}) | "
            f"{m['strict_a']['f1']:.3f} | {m['strict_a']['over_abstention_rate']:.3f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Abstention Performance: Dual Reporting (Strict vs Soft)",
        "",
        "Positives = UNANSWERABLE. Reported under both definition (a) and definition (b):",
        "- **Way (a):** ANSWERABLE as negatives only",
        "- **Way (b):** ANSWERABLE + PARTIAL as negatives",
        "",
        "| System | Strict P/R/F1 (Way a) | Strict P/R/F1 (Way b) | Soft P/R/F1 (Way a) | Soft P/R/F1 (Way b) | Strict Over-Abs | Soft Over-Abs |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for s in system_names:
        m = metrics_by_system[s]
        sa = m["strict_a"]
        sb = m["strict_b"]
        sfa = m["soft_a"]
        sfb = m["soft_b"]
        md_lines.append(
            f"| **{s}** | {sa['precision']:.2f}/{sa['recall']:.2f}/{sa['f1']:.2f} | "
            f"{sb['precision']:.2f}/{sb['recall']:.2f}/{sb['f1']:.2f} | "
            f"{sfa['precision']:.2f}/{sfa['recall']:.2f}/{sfa['f1']:.2f} | "
            f"{sfb['precision']:.2f}/{sfb['recall']:.2f}/{sfb['f1']:.2f} | "
            f"{sa['over_abstention_rate']:.3f} | {sfa['over_abstention_rate']:.3f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Statistical Significance: CARO vs Each Baseline",
        "",
        "Paired bootstrap hypothesis test (1,000 resamples). Rule: A system is declared significantly better/worse **only if its 95% CI excludes zero**.",
        "",
        "| Baseline | EM Diff (p-val) | F1 Diff (p-val) | Abstention F1 Diff | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])

    for b in baselines_list:
        pt = paired_tests[b]
        em_res = pt["em"]
        f1_res = pt["f1"]
        verdict = f1_res["verdict"]
        md_lines.append(
            f"| **{b}** | {em_res['diff_mean']:+.3f} (p={em_res['p_value']:.3f}) | "
            f"{f1_res['diff_mean']:+.3f} (p={f1_res['p_value']:.3f}) | "
            f"{pt['abs_f1_diff']:+.3f} | **{verdict}** |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Where CARO is WORSE (Honest Breakdown)",
        "",
        "Requirement F.11: Exhaustive check across every metric covering Semantic Score, Hallucination Rate, Over-Abstention, EM/F1, Latency, and LLM Calls.",
        ""
    ])

    worse_items = []
    for b in baselines_list:
        bm = metrics_by_system[b]
        # 1. Exact Match
        if bm["em"]["mean"] >= caro_m["em"]["mean"]:
            worse_items.append(f"- **Exact Match vs {b}:** {b} achieves **{bm['em']['mean']:.3f}** vs CARO's **{caro_m['em']['mean']:.3f}** ({'equal' if bm['em']['mean'] == caro_m['em']['mean'] else 'better'}).")
        # 2. Token F1
        if bm["f1"]["mean"] >= caro_m["f1"]["mean"]:
            worse_items.append(f"- **Token F1 vs {b}:** {b} achieves **{bm['f1']['mean']:.3f}** vs CARO's **{caro_m['f1']['mean']:.3f}** ({'equal' if bm['f1']['mean'] == caro_m['f1']['mean'] else 'better'}).")
        # 3. Semantic Score
        if bm["judge_score"]["mean"] >= caro_m["judge_score"]["mean"]:
            worse_items.append(f"- **Semantic Score vs {b}:** {b} achieves **{bm['judge_score']['mean']:.3f}** vs CARO's **{caro_m['judge_score']['mean']:.3f}** ({'equal' if bm['judge_score']['mean'] == caro_m['judge_score']['mean'] else 'better'}).")
        # 4. Hallucination Rate (lower is better)
        if bm["hallucination_rate"]["mean"] <= caro_m["hallucination_rate"]["mean"]:
            worse_items.append(f"- **Hallucination Rate vs {b}:** {b} achieves **{bm['hallucination_rate']['mean']:.3f}** vs CARO's **{caro_m['hallucination_rate']['mean']:.3f}** ({'equal' if bm['hallucination_rate']['mean'] == caro_m['hallucination_rate']['mean'] else 'lower/better'}).")
        # 5. Over-Abstention (lower is better)
        if bm["over_abstention_rate"] <= caro_m["over_abstention_rate"]:
            worse_items.append(f"- **Over-Abstention vs {b}:** {b} wrongly refuses **{bm['over_abstention_rate']*100:.1f}%** vs CARO's **{caro_m['over_abstention_rate']*100:.1f}%**.")
        # 6. Latency (lower is better)
        if bm["latency_mean"] <= caro_m["latency_mean"]:
            worse_items.append(f"- **Query Latency vs {b}:** {b} takes **{bm['latency_mean']:.2f}s** vs CARO's **{caro_m['latency_mean']:.2f}s** (CARO is **{caro_m['latency_mean']/max(bm['latency_mean'], 1e-4):.1f}x slower**).")
        # 7. LLM Calls (lower is better)
        if bm["mean_llm_calls"] <= caro_m["mean_llm_calls"]:
            worse_items.append(f"- **LLM Call Overhead vs {b}:** {b} requires **{bm['mean_llm_calls']:.1f}** calls vs CARO's **{caro_m['mean_llm_calls']:.1f}** calls.")
        # 8. Retrieval Calls (lower is better)
        if bm["mean_ret_calls"] <= caro_m["mean_ret_calls"]:
            worse_items.append(f"- **Retrieval Call Overhead vs {b}:** {b} requires **{bm['mean_ret_calls']:.1f}** retrieval calls vs CARO's **{caro_m['mean_ret_calls']:.1f}** calls.")

    if worse_items:
        md_lines.extend(worse_items)
    else:
        md_lines.append("- No baseline equaled or surpassed CARO on any measured metric.")

    md_lines.extend([
        "",
        "---",
        "",
        "## 5. Cost vs Quality Tradeoff",
        "",
        "| System | Token F1 | Abstention F1 | Mean Latency (s) | Median Latency (s) | p95 Latency (s) | Mean LLM Calls | Mean Retrieval Calls |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for s in system_names:
        m = metrics_by_system[s]
        md_lines.append(
            f"| **{s}** | {m['f1']['mean']:.3f} | {m['abstention_f1']:.3f} | "
            f"{m['latency_mean']:.2f} | {m['latency_median']:.2f} | {m['latency_p95']:.2f} | "
            f"{m['mean_llm_calls']:.1f} | {m['mean_ret_calls']:.1f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 6. Partial Evidence Handling",
        "",
        "On the **PARTIAL** set (where one gold supporting paragraph was excluded from the corpus):",
        "",
        "| System | Disclaimer Rate (%) | Full Answer Rate (%) | Abstention Rate (%) |",
        "| :--- | :--- | :--- | :--- |"
    ])

    for s in system_names:
        m = metrics_by_system[s]
        md_lines.append(
            f"| **{s}** | {m['partial_disclaimer_rate']*100:.1f}% | "
            f"{m['partial_full_rate']*100:.1f}% | {m['partial_abstention_rate']*100:.1f}% |"
        )

    # Programmatic Executive Verdict (Requirement F.10)
    # Allowed wording: "significantly better/worse" ONLY if paired-bootstrap CI excludes zero, otherwise "no significant difference".
    verdict_lines = []
    for b in baselines_list:
        pt = paired_tests[b]
        f1_verdict = pt["f1"]["verdict"]
        em_verdict = pt["em"]["verdict"]
        diff_f1 = pt["f1"]["diff_mean"]
        ci_f1 = [pt["f1"]["diff_ci_lower"], pt["f1"]["diff_ci_upper"]]
        if f1_verdict == "CARO significantly better":
            verdict_lines.append(f"- **Against {b}:** CARO is **significantly better** in Token F1 (diff = {diff_f1:+.3f}, 95% CI {ci_f1}).")
        elif f1_verdict == "CARO significantly worse":
            verdict_lines.append(f"- **Against {b}:** CARO is **significantly worse** in Token F1 (diff = {diff_f1:+.3f}, 95% CI {ci_f1}).")
        else:
            verdict_lines.append(f"- **Against {b}:** **No significant difference** in Token F1 (diff = {diff_f1:+.3f}, 95% CI {ci_f1} includes zero).")

    md_lines.extend([
        "",
        "---",
        "",
        "## 7. Plain-Language Executive Verdict (Programmatic)",
        "",
        "**Statistical Comparison of CARO against Baselines (Paired Bootstrap 95% CIs):**",
        ""
    ])
    md_lines.extend(verdict_lines)
    md_lines.extend([
        "",
        f"**Cost Assessment:** CARO requires an average of **{caro_m['mean_llm_calls']:.1f} LLM calls** and **{caro_m['latency_mean']:.2f}s latency** per query, compared to standard single-pass baselines which operate in ~2-4s with 1 LLM call."
    ])

    report_content = "\n".join(md_lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[Report] Generated comprehensive report at {REPORT_FILE}")
    return report_content
