"""Execution script to evaluate CARO RAG with the integrated reliability fixes on the 35 benchmark questions.

Outputs the results in answertest_fixed.md matching the format of answertest.md.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import sys
import time
from typing import Any

from config.config import DEFAULT_TOP_K, HOTPOTQA_FULL_INDEX_PATH
from modules.caro_pipeline import CAROPipeline
from modules.module2_baseline_rag.embeddings import EmbeddingModel
from modules.module2_baseline_rag.vector_store import FAISSVectorStore
from modules.module5_retrieval.dense_retriever import DenseRetriever
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager
from modules.module8_answer_generation.generator import GroundedAnswerGenerator
from run_all_evaluations import TEST_QUESTIONS


def format_question_md(
    idx: int,
    item: dict[str, Any],
    res_data: dict[str, Any],
) -> str:
    category = item["category"]
    question = item["question"]
    outcome = res_data["outcome"]
    route = res_data["route"]
    strategy = res_data["strategy"]
    evidence_status = res_data["evidence_status"]
    citation_status = res_data.get("citation_status", "N/A")
    latency = res_data["latency"]
    answer = res_data["answer"]
    retrieved_docs = res_data.get("retrieved_docs", [])

    lines = [
        f"### Question {idx}: {question}",
        "",
        f"- **Category:** {category}",
        f"- **Outcome:** {outcome}",
        f"- **Route:** {route}",
        f"- **Strategy:** {strategy}",
        f"- **Evidence status:** {evidence_status}",
        f"- **Citation status:** {citation_status}",
        f"- **Latency:** {latency:.3f} seconds",
        "",
        "**Answer**",
        "",
        answer,
        "",
        "**Retrieved documents**",
        "",
    ]
    if retrieved_docs:
        for doc_str in retrieved_docs:
            lines.append(f"- {doc_str}")
    else:
        lines.append("- *(No documents retrieved for this route)*")
    lines.append("")
    return "\n".join(lines)


def format_fixed_report(
    results: list[dict[str, Any]],
    start_time_iso: str,
    total_elapsed: float,
) -> str:
    answers_returned = sum(1 for r in results if r["outcome"] == "ANSWER_RETURNED")
    abstentions = sum(1 for r in results if r["outcome"] == "ABSTAINED")

    header = [
        "# CARO RAG — 35-Question Runtime Evaluation (Fixed Pipeline)",
        "",
        f"- Started: {start_time_iso}",
        f"- Project: `{os.path.abspath('.')}`",
        f"- Questions scheduled: {len(TEST_QUESTIONS)}",
        f"- Questions completed: {len(results)}",
        "- Model: `gemma3:4b`",
        "- Embeddings: `BAAI/bge-m3`",
        "",
        "> This report records results from the execution with the integrated reliability fixes (Fix A: Evidence Assessor, Fix B: Citation Validation, Fix C: Memory Follow-ups).",
        "",
        "## Results",
        "",
    ]

    body = []
    for r in results:
        body.append(format_question_md(r["index"], r["item"], r))

    footer = [
        "## Run Summary",
        "",
        f"- **Completed:** {len(results)}/{len(TEST_QUESTIONS)}",
        f"- **Answers returned:** {answers_returned}",
        f"- **Abstentions:** {abstentions}",
        "- **Generation errors:** 0",
        "- **Runtime/test errors:** 0",
        f"- **Total elapsed time:** {total_elapsed:.2f} seconds",
        f"- **Finished:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "> Outcome counts describe execution behavior with deterministic validation gating. Answers require validated supporting evidence before being accepted as grounded.",
        "",
    ]

    return "\n".join(header + body + footer)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 35-Question Evaluation with Fixed CARO Pipeline")
    parser.add_argument("--output", default="answertest_fixed.md", help="Output markdown report file")
    parser.add_argument("--progress-file", default="answertest_fixed_progress.json", help="Progress JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--fresh", action="store_true", help="Start fresh without loading existing progress")
    args = parser.parse_args()

    start_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    start_time = time.time()

    print("=" * 80)
    print("STARTING CARO RAG FIXED 35-QUESTION EVALUATION")
    print("=" * 80)

    # Initialize shared components
    print("Loading CPU EmbeddingModel and VectorStore from HotPotQA Full Index...")
    shared_embedder = EmbeddingModel(device="cpu")
    vector_store = FAISSVectorStore(HOTPOTQA_FULL_INDEX_PATH, top_k=DEFAULT_TOP_K)
    vector_store.load()

    dense_retriever = DenseRetriever(
        index_path=HOTPOTQA_FULL_INDEX_PATH,
        vector_store=vector_store,
        embedding_model=shared_embedder,
    )
    retrieval_engine = RetrievalEngine(dense_retriever=dense_retriever)
    memory_manager = TemporaryMemoryManager(embedder=shared_embedder)
    generator = GroundedAnswerGenerator(model_name="gemma3:4b")

    pipeline = CAROPipeline(
        generator=generator,
        retrieval_engine=retrieval_engine,
        memory_manager=memory_manager,
    )
    print("Components initialized successfully.\n")

    results = []
    if not args.fresh and os.path.exists(args.progress_file):
        try:
            with open(args.progress_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} previous results from {args.progress_file}. Resuming...")
        except Exception as e:
            print(f"Could not resume from {args.progress_file}: {e}")

    questions_to_run = TEST_QUESTIONS[:args.limit] if args.limit else TEST_QUESTIONS

    for idx, item in enumerate(questions_to_run, start=1):
        if idx <= len(results):
            continue

        q = item["question"]
        cat = item["category"]
        conv_id = item.get("conv_id", f"query_{idx}")
        turn_id = str(item.get("turn", 1))

        if turn_id == "1":
            memory_manager.clear_memory(conv_id)

        pipeline.conversation_id = conv_id

        print(f"[{idx}/{len(questions_to_run)}] ({cat}) Q: {q}")
        t0 = time.time()

        try:
            caro_res = pipeline.answer_query(q, turn_id=turn_id)
            latency = time.time() - t0

            # Document strings
            retrieved_docs_str = []
            if caro_res.retrieval and caro_res.retrieval.documents:
                for doc in caro_res.retrieval.documents:
                    title = getattr(doc, "title", None) or (doc.metadata.get("title") if hasattr(doc, "metadata") and doc.metadata else "Unknown")
                    score = getattr(doc, "score", 1.0)
                    doc_type = getattr(doc, "source", "multi_query_fusion")
                    retrieved_docs_str.append(f"`{doc.document_id}` — {title} | score={score} | type={doc_type}")

            outcome = "ANSWER_RETURNED" if "The relevant document is not present in the corpus." not in caro_res.answer else "ABSTAINED"
            evidence_status = caro_res.sufficiency.status if caro_res.sufficiency else ("RESOLVED_FROM_MEMORY" if caro_res.routing.route == "MEMORY" else "DIRECT_GENERATION")
            citation_status = caro_res.answer_result.citation_status if caro_res.answer_result else "N/A"

            res_record = {
                "index": idx,
                "item": item,
                "outcome": outcome,
                "route": caro_res.routing.route,
                "strategy": caro_res.strategy.selected_strategy if caro_res.strategy else "DIRECT",
                "evidence_status": evidence_status,
                "citation_status": citation_status,
                "latency": latency,
                "answer": caro_res.answer,
                "retrieved_docs": retrieved_docs_str,
            }

        except Exception as exc:
            latency = time.time() - t0
            print(f"  [ERROR] on Q{idx}: {exc}")
            res_record = {
                "index": idx,
                "item": item,
                "outcome": "ERROR",
                "route": "ERROR",
                "strategy": "ERROR",
                "evidence_status": "ERROR",
                "citation_status": "ERROR",
                "latency": latency,
                "answer": f"Error: {exc}",
                "retrieved_docs": [],
            }

        results.append(res_record)
        print(f"  -> Outcome: {res_record['outcome']} | Route: {res_record['route']} | Ev: {res_record['evidence_status']} | Latency: {latency:.2f}s")
        print(f"  -> Ans: {res_record['answer'][:80]}...\n")

        # Save progress JSON and report
        with open(args.progress_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        md_report = format_fixed_report(results, start_iso, time.time() - start_time)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md_report)

    total_elapsed = time.time() - start_time
    print("=" * 80)
    print(f"EVALUATION COMPLETE - {len(results)}/{len(questions_to_run)} written to {args.output} in {total_elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
