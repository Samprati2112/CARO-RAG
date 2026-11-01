import argparse
import json

from config.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_INDEXING_CHUNK_SIZE,
    DEFAULT_TOP_K,
    HOTPOTQA_DEV_BM25_PATH,
    HOTPOTQA_DEV_INDEX_PATH,
    HOTPOTQA_DOCS_OUTPUT,
    HOTPOTQA_FULL_BM25_PATH,
    HOTPOTQA_FULL_INDEX_META_PATH,
    HOTPOTQA_FULL_INDEX_PATH,
    HOTPOTQA_INDEX_PATH,
    HOTPOTQA_QUESTIONS_OUTPUT,
    MAX_INDEX_DOCUMENTS,
    PROCESSED_DATA_DIR,
    QRECC_CONVERSATIONS_OUTPUT,
    QRECC_DEV_BM25_PATH,
    QRECC_DEV_INDEX_PATH,
    QRECC_DOCS_OUTPUT,
    QRECC_FULL_BM25_PATH,
    QRECC_FULL_INDEX_META_PATH,
    QRECC_FULL_INDEX_PATH,
    QRECC_QUESTIONS_OUTPUT,
)
from modules.module1_data_processing.hotpotqa_processor import HotPotQAProcessor
from modules.module1_data_processing.processor import describe_dataset, read_jsonl
from modules.module1_data_processing.qrecc_processor import QReCCProcessor
from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
from modules.module4_retrieval_strategy.strategy_selector import RetrievalStrategySelector
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module5_retrieval.schemas import RetrievalConfig


def process_hotpotqa() -> dict:
    processor = HotPotQAProcessor()
    dataset = processor.load_dataset()
    print("HotPotQA dataset info:")
    processor.print_dataset_summary(dataset)
    processed = processor.process_dataset(dataset)
    summary = processor.save_outputs(processed)
    return summary


def process_qrecc() -> dict:
    try:
        processor = QReCCProcessor()
        dataset = processor.load_dataset()
        print("QReCC dataset info:")
        processor.print_dataset_summary(dataset)
        processed = processor.process_dataset(dataset)
        summary = processor.save_outputs(processed)
        return summary
    except Exception as exc:
        error_msg = f"QReCC dataset loading failed: {exc}"
        print(f"ERROR: {error_msg}")
        return {
            "dataset": "QReCC",
            "status": "failed",
            "error": error_msg,
            "input_records": 0,
            "processed_questions": 0,
            "processed_documents": 0,
            "validation_errors": 0,
        }


def build_hotpotqa_faiss_index(limit: int | None = MAX_INDEX_DOCUMENTS):
    from modules.module2_baseline_rag.embeddings import EmbeddingModel
    from modules.module2_baseline_rag.vector_store import FAISSVectorStore

    all_documents = read_jsonl(HOTPOTQA_DOCS_OUTPUT)
    if not all_documents:
        raise FileNotFoundError(f"HotPotQA processed documents not found: {HOTPOTQA_DOCS_OUTPUT}")
    documents = all_documents[:limit] if limit is not None else all_documents
    print(f"[1/5] Loading documents... selecting {len(documents)} documents from the processed HotPotQA corpus.")
    model = EmbeddingModel()
    store = FAISSVectorStore(index_path=HOTPOTQA_INDEX_PATH, top_k=DEFAULT_TOP_K)
    store.build_from_documents(documents, model, limit=len(documents))
    print("[3/5] Building FAISS index...")
    store.save()
    print(f"Saved FAISS index to {HOTPOTQA_INDEX_PATH}")
    return {"dataset": "hotpotqa", "index_path": str(HOTPOTQA_INDEX_PATH), "documents": len(documents), "limit": limit}


def build_full_corpus_index(
    dataset: str = "hotpotqa",
    component: str = "both",
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    chunk_size: int = DEFAULT_INDEXING_CHUNK_SIZE,
    limit: int | None = None,
    num_threads: int | None = None,
    show_progress: bool = True,
    resume: bool = True,
    device: str = "auto",
) -> dict:
    """Builds separate full-corpus FAISS dense and BM25 sparse indexes."""
    from modules.module2_baseline_rag.embeddings import EmbeddingModel
    from modules.module2_baseline_rag.vector_store import FAISSVectorStore
    from modules.module5_retrieval.sparse_retriever import SparseRetriever

    datasets_to_build = ["hotpotqa", "qrecc"] if dataset == "all" else [dataset]
    results = {}

    for ds in datasets_to_build:
        ds_results = {}
        if ds == "hotpotqa":
            docs_file = HOTPOTQA_DOCS_OUTPUT
            dense_index_path = HOTPOTQA_FULL_INDEX_PATH
            bm25_cache_path = HOTPOTQA_FULL_BM25_PATH
        elif ds == "qrecc":
            docs_file = QRECC_DOCS_OUTPUT
            dense_index_path = QRECC_FULL_INDEX_PATH
            bm25_cache_path = QRECC_FULL_BM25_PATH
        else:
            raise ValueError(f"Unsupported dataset: {ds}")

        if not docs_file.exists():
            raise FileNotFoundError(f"Processed documents file for {ds} not found: {docs_file}")

        print("==================================================")
        print(f"BUILDING FULL CORPUS INDEX: {ds.upper()}")
        print(f"Source: {docs_file}")
        print(f"Components requested: {component}")
        print(f"Compute device target: {device}")
        if num_threads:
            print(f"CPU threads allocated: {num_threads}")
        print("==================================================")

        # 1. Dense FAISS indexing
        if component in {"both", "dense"}:
            print(f"[{ds}] Initializing dense indexer (FAISS + BGE-M3)...")
            print(f"[{ds}] Target index path: {dense_index_path}")
            model = EmbeddingModel(device=device, num_threads=num_threads)
            store = FAISSVectorStore(index_path=dense_index_path, top_k=DEFAULT_TOP_K)
            dense_res = store.build_from_jsonl(
                jsonl_path=docs_file,
                embedding_model=model,
                limit=limit,
                chunk_size=chunk_size,
                batch_size=batch_size,
                dataset_name=ds,
                show_progress_bar=show_progress,
                resume=resume,
            )
            ds_results["dense"] = dense_res
            print(f"[{ds}] Dense indexing completed: {dense_res['valid_indexed_documents']} documents indexed.")

        # 2. Sparse BM25 indexing
        if component in {"both", "sparse"}:
            print(f"[{ds}] Initializing sparse indexer (BM25)...")
            print(f"[{ds}] Target BM25 cache path: {bm25_cache_path}")
            sparse_res = SparseRetriever.build_bm25_index(
                documents_path=docs_file,
                cache_path=bm25_cache_path,
                max_documents=limit,
                dataset_name=ds,
            )
            ds_results["sparse"] = sparse_res
            print(f"[{ds}] Sparse BM25 indexing completed: {sparse_res['documents_indexed']} documents cached.")

        results[ds] = ds_results

    return results


def get_retrieval_engine(args) -> RetrievalEngine:
    """Builds RetrievalEngine with configuration resolved from CLI flags."""
    dataset = args.dataset if args.dataset in {"hotpotqa", "qrecc"} else "hotpotqa"
    index_type = getattr(args, "index_type", "dev")
    config = RetrievalConfig(
        dataset=dataset,
        index_type=index_type,
        max_documents=None if index_type == "full" else MAX_INDEX_DOCUMENTS,
    )
    return RetrievalEngine(config)


def run_hotpotqa_baseline_query(question: str | None = None):
    from modules.module2_baseline_rag.pipeline import BaselineRAGPipeline

    if question is None:
        question = "What is the capital of France?"
    pipeline = BaselineRAGPipeline(index_path=HOTPOTQA_INDEX_PATH, top_k=DEFAULT_TOP_K)
    if not HOTPOTQA_INDEX_PATH.exists():
        print("Index not found. Building a development FAISS index first...")
        build_hotpotqa_faiss_index(MAX_INDEX_DOCUMENTS)
    pipeline.vector_store.load()
    print(f"Question:\n{question}")
    print("\nRetrieved documents:")
    results = pipeline.query(question, top_k=DEFAULT_TOP_K)
    for index, doc in enumerate(results, start=1):
        text = doc.get("text", "")
        print(f"{index}. {text[:300]}...")
    answer = pipeline.generator.generate(question, results)
    print("\nFinal Answer:\n" + answer)
    return {"question": question, "answer": answer, "retrieved_documents": len(results)}


def main() -> None:
    parser = argparse.ArgumentParser(description="CARO project: Module 1-8 orchestration")
    parser.add_argument("--dataset", choices=["hotpotqa", "qrecc", "all"], default="hotpotqa")
    parser.add_argument("--module", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="1")
    parser.add_argument(
        "--action",
        choices=[
            "build-index",
            "build-full-index",
            "query",
            "route",
            "strategy",
            "retrieve",
            "add",
            "clear",
            "state",
            "answer",
            "sufficiency",
            "generate",
            "validate",
            "evaluate",
        ],
        default=None,
    )
    parser.add_argument(
        "--index-type",
        choices=["dev", "full"],
        default="dev",
        help="Index type to build or query: 'dev' (default 5,000 docs) or 'full' (complete corpus)",
    )
    parser.add_argument(
        "--index-component",
        choices=["both", "dense", "sparse"],
        default="both",
        help="Which index component to build: 'both', 'dense' (FAISS), or 'sparse' (BM25)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=f"Embedding batch size for dense indexing (default {DEFAULT_EMBEDDING_BATCH_SIZE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_INDEXING_CHUNK_SIZE,
        help=f"Document streaming chunk size for dense indexing (default {DEFAULT_INDEXING_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of PyTorch CPU intra-op threads (recommended: 4 or 8 for i5-10300H)",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show sub-chunk progress bar for embedding generation (default: True)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume dense indexing from checkpoint if available (default: True)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device for dense embeddings: 'auto' (detects CUDA, falls back to CPU), 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional document limit for testing or bounded index construction",
    )
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--conversation-id", type=str, default="default")
    parser.add_argument("--turn-id", type=str, default=None)
    parser.add_argument("--response", type=str, default="")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--memory-threshold", type=float, default=0.0)
    args = parser.parse_args()

    # Top-level build-full-index dispatch
    if args.action == "build-full-index":
        result = build_full_corpus_index(
            dataset=args.dataset,
            component=args.index_component,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            limit=args.limit,
            num_threads=args.num_threads,
            show_progress=args.show_progress,
            resume=args.resume,
            device=args.device,
        )
        print("==================================================")
        print("FULL CORPUS INDEX BUILD COMPLETE")
        print("==================================================")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.module == "2":
        from modules.module2_baseline_rag.generator import Generator
        from modules.module2_baseline_rag.pipeline import BaselineRAGPipeline

        if args.action == "build-full-index" or (args.action == "build-index" and args.index_type == "full"):
            result = build_full_corpus_index(
                dataset=args.dataset,
                component=args.index_component,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                limit=args.limit,
                num_threads=args.num_threads,
                show_progress=args.show_progress,
                resume=args.resume,
                device=args.device,
            )
            print("==================================================")
            print("FULL CORPUS INDEX BUILD COMPLETE")
            print("==================================================")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        if args.action == "build-index":
            build_hotpotqa_faiss_index(args.limit or MAX_INDEX_DOCUMENTS)
            return

        if args.action == "query":
            index_path = HOTPOTQA_INDEX_PATH
            if args.index_type == "full":
                index_path = QRECC_FULL_INDEX_PATH if args.dataset == "qrecc" else HOTPOTQA_FULL_INDEX_PATH
                if not index_path.exists():
                    print(f"Full index not found at {index_path}.")
                    print(f"Please build it first with: python main.py --module 2 --action build-full-index --dataset {args.dataset}")
                    return
            else:
                if not HOTPOTQA_INDEX_PATH.exists():
                    print("Index not found. Building a development FAISS index first...")
                    build_hotpotqa_faiss_index(MAX_INDEX_DOCUMENTS)
            pipeline = BaselineRAGPipeline(index_path=index_path, top_k=DEFAULT_TOP_K)
            pipeline.generator = Generator()
            pipeline.vector_store.load()
            result = pipeline.answer_query(args.question or "What is the capital of France?")
            print(f"Question:\n{args.question or 'What is the capital of France?'}")
            print("\nFinal Answer:\n" + result)
            return
        raise ValueError("Module 2 requires --action build-index, build-full-index, or query.")

    if args.module == "3":
        if args.action == "route":
            router = AdaptiveQueryRouter()
            query_text = args.question or "What is the capital of France?"
            result = router.route(query_text)
            print("==================================================")
            print("CARO ADAPTIVE QUERY ROUTER")
            print("==================================================")
            print(f"\nQuestion:\n{result.query}")
            print(f"\nQuestion Type:\n{result.question_type}")
            print(f"\nRoute:\n{result.route}")
            print(f"\nRequires Retrieval:\n{result.requires_retrieval}")
            print(f"\nRequires Memory:\n{result.requires_memory}")
            print(f"\nConfidence:\n{result.confidence}")
            print(f"\nReason:\n{result.reason}")
            return
        raise ValueError("Module 3 requires --action route.")

    if args.module == "4":
        if args.action == "strategy":
            router = AdaptiveQueryRouter()
            selector = RetrievalStrategySelector()
            query_text = args.question or "What is the capital of France?"
            routing_result = router.route(query_text)
            strategy_result = selector.select(routing_result)
            print("==================================================")
            print("CARO RETRIEVAL STRATEGY SELECTOR")
            print("==================================================")
            print(json.dumps(strategy_result.to_dict(), ensure_ascii=False, indent=2))
            return
        raise ValueError("Module 4 requires --action strategy.")

    if args.module == "5":
        if args.action == "retrieve":
            query_text = args.question or "What is the capital of France?"
            routing_result = AdaptiveQueryRouter().route(query_text)
            strategy_result = RetrievalStrategySelector().select(routing_result)
            result = get_retrieval_engine(args).retrieve(strategy_result)
            print("==================================================")
            print("CARO RETRIEVAL AND RELEVANCE FILTER")
            print("==================================================")
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        raise ValueError("Module 5 requires --action retrieve.")

    if args.module == "6":
        from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager

        manager = TemporaryMemoryManager(
            memory_top_k=args.memory_top_k,
            memory_relevance_threshold=args.memory_threshold,
        )
        if args.action == "add":
            record = manager.add_memory(
                conversation_id=args.conversation_id,
                turn_id=args.turn_id or "1",
                user_query=args.question or "",
                assistant_response=args.response,
                context=args.context,
            )
            print("==================================================")
            print("CARO TEMPORARY MEMORY")
            print("==================================================")
            print(json.dumps(record.to_dict(include_embedding=False), ensure_ascii=False, indent=2))
            return
        if args.action == "retrieve":
            result = manager.retrieve_memory(args.question or "", args.conversation_id)
            print("==================================================")
            print("CARO TEMPORARY MEMORY RETRIEVAL")
            print("==================================================")
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        if args.action == "clear":
            removed = manager.clear_memory(args.conversation_id)
            print(json.dumps({"conversation_id": args.conversation_id, "removed": removed}, indent=2))
            return
        if args.action == "state":
            print(json.dumps(manager.get_memory_state(args.conversation_id), indent=2))
            return
        if args.action == "answer":
            from modules.caro_pipeline import CAROPipeline

            result = CAROPipeline(
                conversation_id=args.conversation_id,
                memory_manager=manager,
                retrieval_engine=get_retrieval_engine(args),
            ).answer_query(args.question or "", turn_id=args.turn_id)
            print("==================================================")
            print("CARO END-TO-END QUERY")
            print("==================================================")
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        raise ValueError("Module 6 requires --action add, retrieve, clear, state, or answer.")

    if args.module == "7":
        from modules.module7_evidence_sufficiency.evidence_sufficiency import EvidenceSufficiencyAnalyzer

        if args.action in {"sufficiency", "query", None}:
            query_text = args.question or "What is the capital of France?"
            routing_result = AdaptiveQueryRouter().route(query_text)
            strategy_result = RetrievalStrategySelector().select(routing_result)
            retrieval_result = get_retrieval_engine(args).retrieve(strategy_result) if strategy_result.requires_retrieval else None
            evidence_docs = retrieval_result.documents if retrieval_result else []
            analyzer = EvidenceSufficiencyAnalyzer()
            sufficiency_result = analyzer.analyze(query_text, evidence_docs)
            print("==================================================")
            print("CARO EVIDENCE SUFFICIENCY (MODULE 7)")
            print("==================================================")
            print(json.dumps(sufficiency_result.to_dict(), ensure_ascii=False, indent=2))
            return
        raise ValueError("Module 7 requires --action sufficiency or query.")

    if args.module == "8":
        from modules.module8_answer_generation.generator import GroundedAnswerGenerator

        if args.action in {"generate", "answer", "query", None}:
            query_text = args.question or "What is the capital of France?"
            routing_result = AdaptiveQueryRouter().route(query_text)
            strategy_result = RetrievalStrategySelector().select(routing_result)
            retrieval_result = get_retrieval_engine(args).retrieve(strategy_result) if strategy_result.requires_retrieval else None
            evidence_docs = retrieval_result.documents if retrieval_result else []
            generator = GroundedAnswerGenerator()
            gen_result = generator.generate(query_text, evidence_docs)
            print("==================================================")
            print("CARO ANSWER GENERATION & VALIDATION (MODULE 8)")
            print("==================================================")
            print(json.dumps(gen_result.to_dict(), ensure_ascii=False, indent=2))
            return
        raise ValueError("Module 8 requires --action generate, answer, or query.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"hotpotqa", "all"}:
        print("Schema inspection (HotPotQA):")
        try:
            hotpot_info = describe_dataset("hotpotqa/hotpot_qa", config="distractor")
            print(json.dumps({"dataset": "HotPotQA", "schema": hotpot_info}, ensure_ascii=False, indent=2)[:2000])
        except Exception as exc:
            print(f"WARNING: Could not inspect HotPotQA schema: {exc}")

    if args.dataset in {"qrecc", "all"}:
        print("Schema inspection (QReCC):")
        try:
            qrecc_info = describe_dataset("svakulenk0/qrecc")
            print(json.dumps({"dataset": "QReCC", "schema": qrecc_info}, ensure_ascii=False, indent=2)[:2000])
        except Exception as exc:
            print(f"WARNING: Could not inspect QReCC schema: {exc}")

    if args.dataset in {"hotpotqa", "all"}:
        print("\nProcessing HotPotQA...")
        hotpot_summary = process_hotpotqa()
        print(json.dumps({"dataset": "HotPotQA", "summary": hotpot_summary}, ensure_ascii=False, indent=2))

    if args.dataset in {"qrecc", "all"}:
        print("\nProcessing QReCC...")
        qrecc_summary = process_qrecc()
        print(json.dumps({"dataset": "QReCC", "summary": qrecc_summary}, ensure_ascii=False, indent=2))

        if "status" in qrecc_summary and qrecc_summary["status"] == "failed":
            if args.dataset == "qrecc":
                print("\nERROR: QReCC processing failed and no other dataset was requested.")
                raise RuntimeError(qrecc_summary["error"])
            print("\nWARNING: QReCC processing failed but HotPotQA succeeded. Continuing...")

    print("\nCompleted Module 1 processing.")
    print(f"Processed outputs: {HOTPOTQA_QUESTIONS_OUTPUT}, {HOTPOTQA_DOCS_OUTPUT}, {QRECC_QUESTIONS_OUTPUT}, {QRECC_DOCS_OUTPUT}, {QRECC_CONVERSATIONS_OUTPUT}")


if __name__ == "__main__":
    main()
