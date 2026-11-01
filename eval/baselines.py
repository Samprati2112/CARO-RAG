"""
Implementation of Baseline RAG Systems (B1, B2, B3, B4, B3-abs) and CARO Pipeline Adapter.
All systems expose a common interface:
    system.answer(question: str, excluded_doc_ids: Optional[Set[str]] = None) -> Dict[str, Any]
with uniform call tracking, fair-fight prompt sharing, and document exclusion.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set

from eval.caro_core import (
    search_dense,
    search_sparse,
    search_hybrid,
    expand_query,
    _merge_candidates,
    call_gemma,
    CaroRAGPipeline,
    CallTracker,
    ExclusionContext,
    MAX_CONTEXT_DOCS,
    INITIAL_TOP_K,
    UNANSWERABLE_STRING,
)
from eval.common import (
    build_shared_generation_prompt,
    extract_answer_text,
    is_strict_abstained,
    is_soft_abstained,
)


class BaseRAGSystem(ABC):
    """Abstract base class establishing the common evaluation interface."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _execute(self, question: str) -> Dict[str, Any]:
        """Internal execution logic to be implemented by each system."""
        pass

    def answer(self, question: str, excluded_doc_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Public evaluation entrypoint.
        Wraps execution with CallTracker and ExclusionContext.
        Guarantees returned dictionary contains:
            - final_answer (str: extracted span for evaluation)
            - final_answer_raw (str: full unparsed response for faithfulness)
            - retrieved_docs (list of dicts with doc_id and title)
            - abstained (bool: strict abstention)
            - abstained_strict (bool)
            - abstained_soft (bool)
            - latency_seconds (float)
            - llm_calls (int)
            - retrieval_calls (int)
            - stage_timings (dict)
            - system_name (str)
        """
        start_time = time.time()
        with CallTracker() as tracker, ExclusionContext(excluded_doc_ids):
            result = self._execute(question)

        elapsed = time.time() - start_time
        raw_ans = result.get("final_answer_raw", result.get("final_answer", ""))
        extracted_ans = result.get("final_answer") or extract_answer_text(raw_ans)
        evidence_status = result.get("metadata", {}).get("evidence_status", "")

        abstained_strict = bool(result.get("abstained_strict", is_strict_abstained(raw_ans, result.get("abstained", False), evidence_status)))
        abstained_soft = bool(result.get("abstained_soft", is_soft_abstained(raw_ans, abstained_strict, evidence_status)))

        stage_timings = result.get("stage_timings", {})
        if "total" not in stage_timings:
            stage_timings["total"] = round(elapsed, 4)

        return {
            "system_name": self.name,
            "question": question,
            "final_answer": extracted_ans,
            "final_answer_raw": raw_ans,
            "retrieved_docs": result.get("retrieved_docs", [])[:MAX_CONTEXT_DOCS],
            "abstained": abstained_strict,
            "abstained_strict": abstained_strict,
            "abstained_soft": abstained_soft,
            "latency_seconds": round(elapsed, 4),
            "llm_calls": tracker.llm_calls,
            "retrieval_calls": tracker.retrieval_calls,
            "stage_timings": stage_timings,
            "metadata": result.get("metadata", {})
        }


class DenseOnlyRAG(BaseRAGSystem):
    """B1: Dense top-k only + Gemma generation."""
    def __init__(self):
        super().__init__("B1_Dense")

    def _execute(self, question: str) -> Dict[str, Any]:
        t_ret0 = time.time()
        # Naive single-hop dense baseline (top_k=1) without multi-hop expansion
        docs = search_dense(question, top_k=1)
        t_ret = time.time() - t_ret0

        if not docs:
            return {
                "final_answer": UNANSWERABLE_STRING,
                "final_answer_raw": UNANSWERABLE_STRING,
                "retrieved_docs": [],
                "abstained": True,
                "abstained_strict": True,
                "abstained_soft": True,
                "stage_timings": {"retrieval": round(t_ret, 4), "generation": 0.0, "total": round(t_ret, 4)}
            }

        prompt = build_shared_generation_prompt(question, docs, system_type="standard")
        t_gen0 = time.time()
        res = call_gemma(prompt, json_mode=False, timeout=300, max_retries=2, num_predict=64)
        t_gen = time.time() - t_gen0

        raw_ans = res["raw"].strip() if res["success"] else UNANSWERABLE_STRING
        extracted = extract_answer_text(raw_ans)
        abstained_strict = is_strict_abstained(raw_ans)
        abstained_soft = is_soft_abstained(raw_ans)

        return {
            "final_answer": extracted,
            "final_answer_raw": raw_ans,
            "retrieved_docs": docs,
            "abstained": abstained_strict,
            "abstained_strict": abstained_strict,
            "abstained_soft": abstained_soft,
            "stage_timings": {"retrieval": round(t_ret, 4), "generation": round(t_gen, 4)}
        }


class BM25OnlyRAG(BaseRAGSystem):
    """B2: BM25 top-k only + Gemma generation."""
    def __init__(self):
        super().__init__("B2_BM25")

    def _execute(self, question: str) -> Dict[str, Any]:
        t_ret0 = time.time()
        docs = search_sparse(question, top_k=MAX_CONTEXT_DOCS)
        t_ret = time.time() - t_ret0

        if not docs:
            return {
                "final_answer": UNANSWERABLE_STRING,
                "final_answer_raw": UNANSWERABLE_STRING,
                "retrieved_docs": [],
                "abstained": True,
                "abstained_strict": True,
                "abstained_soft": True,
                "stage_timings": {"retrieval": round(t_ret, 4), "generation": 0.0, "total": round(t_ret, 4)}
            }

        prompt = build_shared_generation_prompt(question, docs, system_type="standard")
        t_gen0 = time.time()
        res = call_gemma(prompt, json_mode=False, timeout=300, max_retries=2, num_predict=64)
        t_gen = time.time() - t_gen0

        raw_ans = res["raw"].strip() if res["success"] else UNANSWERABLE_STRING
        extracted = extract_answer_text(raw_ans)
        abstained_strict = is_strict_abstained(raw_ans)
        abstained_soft = is_soft_abstained(raw_ans)

        return {
            "final_answer": extracted,
            "final_answer_raw": raw_ans,
            "retrieved_docs": docs,
            "abstained": abstained_strict,
            "abstained_strict": abstained_strict,
            "abstained_soft": abstained_soft,
            "stage_timings": {"retrieval": round(t_ret, 4), "generation": round(t_gen, 4)}
        }


class HybridRAG(BaseRAGSystem):
    """B3: Hybrid (dense + BM25, RRF), no LLM calls in retrieval + Gemma generation."""
    def __init__(self):
        super().__init__("B3_Hybrid")

    def _execute(self, question: str) -> Dict[str, Any]:
        t_ret0 = time.time()
        docs = search_hybrid(question, top_k=MAX_CONTEXT_DOCS)
        t_ret = time.time() - t_ret0

        if not docs:
            return {
                "final_answer": UNANSWERABLE_STRING,
                "final_answer_raw": UNANSWERABLE_STRING,
                "retrieved_docs": [],
                "abstained": True,
                "abstained_strict": True,
                "abstained_soft": True,
                "stage_timings": {"retrieval": round(t_ret, 4), "generation": 0.0, "total": round(t_ret, 4)}
            }

        prompt = build_shared_generation_prompt(question, docs, system_type="standard")
        t_gen0 = time.time()
        res = call_gemma(prompt, json_mode=False, timeout=300, max_retries=2, num_predict=64)
        t_gen = time.time() - t_gen0

        raw_ans = res["raw"].strip() if res["success"] else UNANSWERABLE_STRING
        extracted = extract_answer_text(raw_ans)
        abstained_strict = is_strict_abstained(raw_ans)
        abstained_soft = is_soft_abstained(raw_ans)

        return {
            "final_answer": extracted,
            "final_answer_raw": raw_ans,
            "retrieved_docs": docs,
            "abstained": abstained_strict,
            "abstained_strict": abstained_strict,
            "abstained_soft": abstained_soft,
            "stage_timings": {"retrieval": round(t_ret, 4), "generation": round(t_gen, 4)}
        }


class MultiQueryRAGFusion(BaseRAGSystem):
    """
    B4: B3 + multi-query expansion with RRF fusion (RAG-Fusion style).
    No query router, no evidence sufficiency check.
    """
    def __init__(self):
        super().__init__("B4_RAGFusion")

    def _execute(self, question: str) -> Dict[str, Any]:
        t_exp0 = time.time()
        expansions = expand_query(question)
        t_exp = time.time() - t_exp0
        variants = [question] + [e for e in expansions if e.casefold() != question.casefold()][:3]

        t_var0 = time.time()
        pass_results = []
        for i, q_var in enumerate(variants):
            var_docs = search_hybrid(q_var, top_k=INITIAL_TOP_K)
            pass_results.append((f"variant_{i}", var_docs))
        t_var = time.time() - t_var0

        t_fuse0 = time.time()
        fused_docs = _merge_candidates(pass_results)
        final_docs = fused_docs[:MAX_CONTEXT_DOCS]
        t_fuse = time.time() - t_fuse0

        if not final_docs:
            return {
                "final_answer": UNANSWERABLE_STRING,
                "final_answer_raw": UNANSWERABLE_STRING,
                "retrieved_docs": [],
                "abstained": True,
                "abstained_strict": True,
                "abstained_soft": True,
                "stage_timings": {
                    "expansion": round(t_exp, 4),
                    "retrieval_variants": round(t_var, 4),
                    "retrieval_fusion": round(t_fuse, 4),
                    "generation": 0.0
                }
            }

        prompt = build_shared_generation_prompt(question, final_docs, system_type="standard")
        t_gen0 = time.time()
        res = call_gemma(prompt, json_mode=False, timeout=300, max_retries=2, num_predict=64)
        t_gen = time.time() - t_gen0

        raw_ans = res["raw"].strip() if res["success"] else UNANSWERABLE_STRING
        extracted = extract_answer_text(raw_ans)
        abstained_strict = is_strict_abstained(raw_ans)
        abstained_soft = is_soft_abstained(raw_ans)

        return {
            "final_answer": extracted,
            "final_answer_raw": raw_ans,
            "retrieved_docs": final_docs,
            "abstained": abstained_strict,
            "abstained_strict": abstained_strict,
            "abstained_soft": abstained_soft,
            "stage_timings": {
                "expansion": round(t_exp, 4),
                "retrieval_variants": round(t_var, 4),
                "retrieval_fusion": round(t_fuse, 4),
                "generation": round(t_gen, 4)
            },
            "metadata": {"query_variants": variants, "candidate_count": len(fused_docs)}
        }


class HybridPromptAbstentionRAG(BaseRAGSystem):
    """
    B3-abs: B3 hybrid retrieval, but the generation prompt is allowed to say
    'The relevant document is not present in the corpus.' when evidence is insufficient.
    Tests whether CARO's dedicated sufficiency judge beats simple prompt-based abstention.
    """
    def __init__(self):
        super().__init__("B3_abs")

    def _execute(self, question: str) -> Dict[str, Any]:
        t_ret0 = time.time()
        docs = search_hybrid(question, top_k=MAX_CONTEXT_DOCS)
        t_ret = time.time() - t_ret0

        if not docs:
            return {
                "final_answer": UNANSWERABLE_STRING,
                "final_answer_raw": UNANSWERABLE_STRING,
                "retrieved_docs": [],
                "abstained": True,
                "abstained_strict": True,
                "abstained_soft": True,
                "stage_timings": {"retrieval": round(t_ret, 4), "generation": 0.0, "total": round(t_ret, 4)}
            }

        prompt = build_shared_generation_prompt(question, docs, system_type="b3_abs")
        t_gen0 = time.time()
        res = call_gemma(prompt, json_mode=False, timeout=300, max_retries=2, num_predict=64)
        t_gen = time.time() - t_gen0

        raw_ans = res["raw"].strip() if res["success"] else UNANSWERABLE_STRING
        extracted = extract_answer_text(raw_ans)
        abstained_strict = is_strict_abstained(raw_ans)
        abstained_soft = is_soft_abstained(raw_ans)

        final_answer = UNANSWERABLE_STRING if abstained_strict else extracted
        return {
            "final_answer": final_answer,
            "final_answer_raw": raw_ans,
            "retrieved_docs": docs,
            "abstained": abstained_strict,
            "abstained_strict": abstained_strict,
            "abstained_soft": abstained_soft,
            "stage_timings": {"retrieval": round(t_ret, 4), "generation": round(t_gen, 4)}
        }


class CaroPipelineAdapter(BaseRAGSystem):
    """
    CARO RAG full pipeline adapter.
    Preserves exact frozen pipeline behavior.
    Clears memory before every question for single-turn evaluation.
    """
    def __init__(self):
        super().__init__("CARO")
        self.pipeline = CaroRAGPipeline()

    def _execute(self, question: str) -> Dict[str, Any]:
        self.pipeline.clear_memory()
        res = self.pipeline.query(question)

        raw_ans = res.get("final_answer", "")
        evidence_status = res.get("evidence_status", "UNKNOWN")
        extracted_ans = extract_answer_text(raw_ans)
        abstained_strict = is_strict_abstained(raw_ans)
        abstained_soft = is_soft_abstained(raw_ans)

        return {
            "final_answer": extracted_ans,
            "final_answer_raw": raw_ans,
            "retrieved_docs": res.get("retrieved_documents", [])[:MAX_CONTEXT_DOCS],
            "abstained": abstained_strict,
            "abstained_strict": abstained_strict,
            "abstained_soft": abstained_soft,
            "stage_timings": res.get("stage_timings", {}),
            "metadata": {
                "evidence_status": evidence_status,
                "evidence_assessment": res.get("evidence_assessment"),
                "route": res.get("route"),
                "strategy": res.get("strategy"),
                "iterations_run": res.get("iterations_run"),
                "disclaimer": res.get("disclaimer"),
                "rewritten_query": res.get("rewritten_query"),
                "query_variants": res.get("query_variants", [question])
            }
        }


class CaroFastPipelineAdapter(BaseRAGSystem):
    """
    CARO-fast ablation system (Requirement 19):
    - Merged router + expansion call
    - Skip decomposition unless multi-hop
    - Early stop when the first pass is sufficient
    - Separate optional system, never replaces frozen CARO.
    """
    def __init__(self):
        super().__init__("CARO_fast")

    def _execute(self, question: str) -> Dict[str, Any]:
        t_start = time.time()
        stage_timings = {}

        # 1. Merged router + expansion call
        prompt = f"""Analyze this query for RAG retrieval.
Query: {question}
Return JSON:
{{"route": "RETRIEVE"|"DIRECT", "is_multi_hop": true|false, "expansions": ["variant1", "variant2"]}}"""
        t_r0 = time.time()
        r_res = call_gemma(prompt, json_mode=True, timeout=300)
        stage_timings["router_and_expansion"] = round(time.time() - t_r0, 4)

        expansions = []
        is_multi_hop = False
        if r_res["success"] and isinstance(r_res["json"], dict):
            p = r_res["json"]
            expansions = p.get("expansions", [])[:2]
            is_multi_hop = bool(p.get("is_multi_hop", False))

        # 2. First pass: Hybrid search
        t_p0 = time.time()
        first_pass_docs = search_hybrid(question, top_k=MAX_CONTEXT_DOCS)
        stage_timings["retrieval_primary"] = round(time.time() - t_p0, 4)

        # Early check: assess evidence sufficiency on first pass
        from eval.caro_core import assess_evidence_sufficiency, generate_grounded_answer
        t_a0 = time.time()
        assessment = assess_evidence_sufficiency(question, first_pass_docs)
        stage_timings["assessment"] = round(time.time() - t_a0, 4)

        if assessment.status == "SUFFICIENT":
            # Early stop! First pass is sufficient
            t_g0 = time.time()
            ans = generate_grounded_answer(question, first_pass_docs, is_partial=False)
            stage_timings["generation"] = round(time.time() - t_g0, 4)
            stage_timings["total"] = round(time.time() - t_start, 4)
            extracted = extract_answer_text(ans)
            return {
                "final_answer": extracted,
                "final_answer_raw": ans,
                "retrieved_docs": first_pass_docs,
                "abstained": False,
                "abstained_strict": False,
                "abstained_soft": False,
                "stage_timings": stage_timings,
                "metadata": {
                    "early_stopped": True,
                    "evidence_status": "SUFFICIENT",
                    "route": "RETRIEVE",
                    "strategy": "HYBRID",
                    "query_variants": [question]
                }
            }

        # Otherwise expand if not sufficient
        pass_results = [("primary", first_pass_docs)]
        variants = [question] + [e for e in expansions if e != question]
        if is_multi_hop:
            from eval.caro_core import decompose_query
            t_d0 = time.time()
            subq = decompose_query(question)
            stage_timings["decomposition"] = round(time.time() - t_d0, 4)
            variants.extend(subq[:2])

        t_v0 = time.time()
        for i, var in enumerate(variants[1:], 1):
            var_docs = search_hybrid(var, top_k=INITIAL_TOP_K)
            pass_results.append((f"variant_{i}", var_docs))
        stage_timings["retrieval_variants"] = round(time.time() - t_v0, 4)

        fused = _merge_candidates(pass_results)[:MAX_CONTEXT_DOCS]
        t_a1 = time.time()
        assessment2 = assess_evidence_sufficiency(question, fused)
        stage_timings["assessment_final"] = round(time.time() - t_a1, 4)

        t_g0 = time.time()
        if assessment2.status == "SUFFICIENT":
            ans = generate_grounded_answer(question, fused, is_partial=False)
            abstained = False
        elif assessment2.status == "PARTIAL":
            ans = generate_grounded_answer(question, fused, is_partial=True, missing_info=assessment2.missing_information)
            abstained = False
        else:
            ans = UNANSWERABLE_STRING
            abstained = True
        stage_timings["generation"] = round(time.time() - t_g0, 4)
        stage_timings["total"] = round(time.time() - t_start, 4)

        extracted = extract_answer_text(ans)
        return {
            "final_answer": extracted,
            "final_answer_raw": ans,
            "retrieved_docs": fused,
            "abstained": abstained,
            "abstained_strict": is_strict_abstained(ans, evidence_status=assessment2.status),
            "abstained_soft": is_soft_abstained(ans, evidence_status=assessment2.status),
            "stage_timings": stage_timings,
            "metadata": {
                "early_stopped": False,
                "evidence_status": assessment2.status,
                "route": "RETRIEVE",
                "strategy": "HYBRID",
                "query_variants": variants
            }
        }


def get_all_systems() -> Dict[str, BaseRAGSystem]:
    """Returns mapping of all evaluation systems, including optional CARO_fast ablation."""
    return {
        "B1_Dense": DenseOnlyRAG(),
        "B2_BM25": BM25OnlyRAG(),
        "B3_Hybrid": HybridRAG(),
        "B4_RAGFusion": MultiQueryRAGFusion(),
        "B3_abs": HybridPromptAbstentionRAG(),
        "CARO": CaroPipelineAdapter(),
        "CARO_fast": CaroFastPipelineAdapter()
    }
