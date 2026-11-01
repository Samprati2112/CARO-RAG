"""
LLM Judge for Semantic Correctness and NLI Cross-Encoder for Faithfulness.
- Judge model: llama3.1:8b (or configured via JUDGE_MODEL) to avoid self-grading bias.
- Faithfulness: cross-encoder/nli-deberta-v3-small checking claim entailment against context.
"""

import os
import re
import json
import threading
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from eval.caro_core import OLLAMA_BASE_URL, _doc_text, MAX_CONTEXT_DOCS
from eval.common import is_abstained_answer, UNANSWERABLE_STRING, print_ram_usage

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3.1:8b")


class LLMJudge:
    """Evaluates semantic correctness of generated answers against gold answers."""
    _instance: Optional["LLMJudge"] = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = JUDGE_MODEL):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.model_name = model_name
                    instance._call_lock = threading.Lock()
                    print_ram_usage(f"LLMJudge client initialized ({model_name})")
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_client(cls, model_name: str = JUDGE_MODEL) -> "LLMJudge":
        return cls(model_name=model_name)

    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        if not hasattr(self, "_call_lock"):
            self._call_lock = threading.Lock()

    def judge_answer(self, question: str, gold_answer: str, system_answer: str) -> Dict[str, Any]:
        """
        Determines semantic correctness: 'correct', 'partially_correct', or 'incorrect'.
        If system abstained on an answerable question, it is 'incorrect'.
        """
        if is_abstained_answer(system_answer):
            return {
                "verdict": "incorrect",
                "score": 0.0,
                "reason": "System abstained on question.",
                "raw_judge_response": "ABSTAINED"
            }

        prompt = f"""You are an expert, impartial evaluator for question-answering systems.
Evaluate whether the Candidate Answer is correct given the Question and the Reference Gold Answer.

Evaluation Guidelines:
- "correct": The candidate answer conveys the exact same factual meaning and answers the question accurately according to the reference, even if phrased differently or containing minor extra details.
- "partially_correct": The candidate answer correctly provides part of the required answer, but misses another required detail or expresses uncertainty.
- "incorrect": The candidate answer contradicts the reference, provides an entirely different answer, makes false claims, or fails to answer the question.

Question: {question}
Reference Gold Answer: {gold_answer}
Candidate Answer: {system_answer}

Respond ONLY with a JSON object in this exact schema:
{{
  "verdict": "correct" | "partially_correct" | "incorrect",
  "reason": "Brief 1-sentence justification"
}}"""

        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                raw_text = res_json.get("response", "").strip()
                parsed = json.loads(raw_text)
                verdict = str(parsed.get("verdict", "incorrect")).strip().lower()
                if verdict not in ("correct", "partially_correct", "incorrect"):
                    verdict = "incorrect"
                reason = str(parsed.get("reason", "Parsed from judge output."))
                score_map = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
                return {
                    "verdict": verdict,
                    "score": score_map.get(verdict, 0.0),
                    "reason": reason,
                    "raw_judge_response": raw_text
                }
        except Exception as e:
            # Fallback heuristic if judge request fails
            from eval.common import token_f1_score, exact_match_score
            f1, _, _ = token_f1_score(system_answer, gold_answer)
            em = exact_match_score(system_answer, gold_answer)
            if em or f1 >= 0.8:
                verdict = "correct"
            elif f1 >= 0.4:
                verdict = "partially_correct"
            else:
                verdict = "incorrect"
            score_map = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
            return {
                "verdict": verdict,
                "score": score_map[verdict],
                "reason": f"Fallback heuristic due to judge error: {e}",
                "raw_judge_response": f"ERROR: {e}"
            }


class FaithfulnessChecker:
    """
    Checks claim-level entailment against retrieved context using NLI cross-encoder.
    Model: cross-encoder/nli-deberta-v3-small
    """
    _cross_encoder = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls):
        if cls._cross_encoder is None:
            with cls._lock:
                if cls._cross_encoder is None:
                    from sentence_transformers import CrossEncoder
                    cls._cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-small")
                    print_ram_usage("FaithfulnessChecker NLI model loaded (cross-encoder/nli-deberta-v3-small)")
        return cls._cross_encoder

    @staticmethod
    def clean_citations_and_disclaimers(text: str) -> str:
        """Strips citation markers like [Document N] and [n] and disclaimers."""
        cleaned = re.sub(r'\[Document\s+\d+\]', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[\d+\]', '', cleaned)
        cleaned = re.sub(r'\*\*Disclaimer:\*\*.*', '', cleaned, flags=re.DOTALL)
        return ' '.join(cleaned.split())

    @classmethod
    def check_faithfulness(
        cls,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        question: str = "",
        abstained: bool = False
    ) -> Dict[str, Any]:
        """
        Checks declarative statement 'question + ANSWER' against concatenated context using NLI.
        If answer is abstained or empty, hallucination rate is 0 (no hallucination).
        """
        if abstained or is_abstained_answer(answer):
            return {
                "declarative_statement": "",
                "nli_verdict": "ABSTAINED",
                "entailment_prob": 1.0,
                "is_hallucinated": False
            }

        cleaned_ans = cls.clean_citations_and_disclaimers(answer)
        if not cleaned_ans:
            return {
                "declarative_statement": "",
                "nli_verdict": "EMPTY",
                "entailment_prob": 1.0,
                "is_hallucinated": False
            }

        clean_q = str(question or "").strip().rstrip("?")
        declarative_statement = f"{clean_q} {cleaned_ans}." if clean_q else cleaned_ans

        if not retrieved_docs:
            return {
                "declarative_statement": declarative_statement,
                "nli_verdict": "NO_CONTEXT",
                "entailment_prob": 0.0,
                "is_hallucinated": True
            }

        with cls._lock:
            model = cls.get_model()
            # Concatenated context truncated to 600 chars per doc
            context_blocks = []
            for d in retrieved_docs[:MAX_CONTEXT_DOCS]:
                title = d.get("title", "")
                txt = _doc_text(d)[:600]
                context_blocks.append(f"{title}: {txt}" if title else txt)
            concat_context = "\n".join(context_blocks)

            pairs = [(concat_context, declarative_statement)]
            logits = model.predict(pairs)
            if len(logits.shape) == 1:
                logits = np.expand_dims(logits, axis=0)

        # DeBERTa-v3 NLI label mapping: [contradiction, entailment, neutral]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Entailment probability is index 1
        entail_prob = float(probs[0, 1])
        is_supported = bool(entail_prob >= 0.50)

        verdict = "ENTAILED" if is_supported else "UNSUPPORTED"
        is_hallucinated = not is_supported

        return {
            "declarative_statement": declarative_statement,
            "nli_verdict": verdict,
            "entailment_prob": round(entail_prob, 4),
            "is_hallucinated": is_hallucinated
        }
