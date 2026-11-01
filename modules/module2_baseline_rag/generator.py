from __future__ import annotations

import json
import urllib.request

from config.config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
from modules.module2_baseline_rag.prompt import build_rag_prompt


class Generator:
    """Baseline generation component that accepts a query and context."""

    def __init__(self, model_name: str = OLLAMA_MODEL_NAME):
        self.model_name = model_name

    def build_prompt(self, query: str, context_documents):
        return build_rag_prompt(query, context_documents)

    def generate(self, query: str, context_documents):
        prompt = self.build_prompt(query, context_documents)
        if self.model_name == "dummy-model":
            return self._dummy_generate(query, context_documents)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }

        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
                answer = result.get("response", "").strip()
                return answer or "The information is insufficient."
        except Exception as exc:
            return f"Generation failed: {exc}. Please verify Ollama is running with gemma3:4b."

    def _dummy_generate(self, query: str, context_documents):
        if isinstance(context_documents, dict):
            context_documents = context_documents.get("memory_context", []) + context_documents.get("external_evidence", [])
        context_text = "\n".join(
            (doc.get("text") or doc.get("stored_context") or doc.get("stored_response") or "")
            if isinstance(doc, dict) else str(doc)
            for doc in context_documents
        )
        if "Paris" in context_text and "France" in context_text:
            return "Paris is the capital of France."
        if context_documents:
            return f"Answer based on context: {context_text[:200]}"
        return "The information is insufficient."
