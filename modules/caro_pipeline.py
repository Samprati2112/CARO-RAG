from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from modules.module2_baseline_rag.generator import Generator
from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
from modules.module3_adaptive_query_router.schemas import RoutingResult
from modules.module4_retrieval_strategy.schemas import RetrievalStrategyResult
from modules.module4_retrieval_strategy.strategy_selector import RetrievalStrategySelector
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module5_retrieval.schemas import RetrievalResult
from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager
from modules.module6_temporary_memory.schemas import MemoryRecord, MemoryResult
from modules.module7_evidence_sufficiency.evidence_sufficiency import EvidenceSufficiencyAnalyzer, EvidenceSufficiencyResult
from modules.module8_answer_generation.generator import GroundedAnswerGenerator
from modules.module8_answer_generation.schemas import AnswerGenerationResult


@dataclass
class CAROQueryResult:
    """Complete runtime result for one routed conversational query."""

    query: str
    conversation_id: str
    turn_id: str
    routing: RoutingResult
    strategy: RetrievalStrategyResult
    memory: MemoryResult | None
    retrieval: RetrievalResult | None
    generation_context: dict[str, list[dict[str, Any]]]
    answer: str
    stored_memory: MemoryRecord
    sufficiency: EvidenceSufficiencyResult | None = None
    answer_result: AnswerGenerationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "routing": self.routing.__dict__,
            "strategy": self.strategy.to_dict(),
            "memory": self.memory.to_dict() if self.memory else None,
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "sufficiency": self.sufficiency.to_dict() if self.sufficiency else None,
            "generation_context": self.generation_context,
            "answer": self.answer,
            "answer_result": self.answer_result.to_dict() if self.answer_result else None,
            "stored_memory": self.stored_memory.to_dict(include_embedding=False),
        }


class CAROPipeline:
    """Runtime CARO orchestration for routing, retrieval, sufficiency, generation, and memory."""

    def __init__(
        self,
        conversation_id: str = "default",
        *,
        router: AdaptiveQueryRouter | None = None,
        strategy_selector: RetrievalStrategySelector | None = None,
        memory_manager: TemporaryMemoryManager | None = None,
        retrieval_engine: RetrievalEngine | None = None,
        sufficiency_analyzer: EvidenceSufficiencyAnalyzer | None = None,
        generator: GroundedAnswerGenerator | Generator | None = None,
        max_retrieval_iterations: int = 5,
    ):
        self.conversation_id = conversation_id
        self.router = router or AdaptiveQueryRouter()
        self.strategy_selector = strategy_selector or RetrievalStrategySelector()
        self.memory_manager = memory_manager or TemporaryMemoryManager()
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        self.sufficiency_analyzer = sufficiency_analyzer or EvidenceSufficiencyAnalyzer()
        self.generator = generator or GroundedAnswerGenerator()
        self.max_retrieval_iterations = max_retrieval_iterations

    def answer_query(
        self,
        query: str,
        *,
        turn_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        ground_truth: str | None = None,
    ) -> CAROQueryResult:
        routing = self.router.route(query)
        strategy = self.strategy_selector.select(routing)

        # 1. Handle Memory Retrieval if required
        memory = None
        if strategy.requires_memory:
            memory = self.memory_manager.retrieve_memory(
                query,
                self.conversation_id,
                strategy=strategy.selected_strategy,
            )

        # 2. Handle External Retrieval with Iterative Bounded M7 Sufficiency Loop
        retrieval = None
        sufficiency = None
        accumulated_documents = []

        if strategy.requires_retrieval:
            iteration = 0
            current_query = query

            # If conversational memory is present and query contains follow-up pronouns,
            # enrich the retrieval query with the conversational subject from memory
            if memory and memory.memories:
                prior_queries = " ".join(m.original_query for m in memory.memories)
                prior_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", prior_queries)
                if prior_entities and re.search(r"\b(he|she|his|her|him|they|their|those|these)\b", query.lower()):
                    current_query = f"{prior_entities[0]} {query}"

            while iteration < self.max_retrieval_iterations:
                current_strategy = self.strategy_selector.select(
                    self.router.route(current_query)
                ) if current_query != query else strategy

                retrieval = self.retrieval_engine.retrieve(current_strategy)
                for doc in retrieval.documents:
                    if doc.document_id not in [d.document_id for d in accumulated_documents]:
                        accumulated_documents.append(doc)

                # Analyze collective evidence sufficiency with M7
                sufficiency = self.sufficiency_analyzer.analyze(
                    query,
                    accumulated_documents,
                    iteration=iteration,
                    max_iterations=self.max_retrieval_iterations,
                )

                if sufficiency.action == "ANSWER":
                    break
                elif sufficiency.action == "RETRIEVE_AGAIN":
                    iteration += 1
                    if sufficiency.suggested_subquery:
                        current_query = f"{query} {sufficiency.suggested_subquery}"
                    else:
                        break
                else:  # ABSTAIN
                    break

            # Update retrieval documents to the final accumulated collection
            if retrieval is not None:
                retrieval.documents = accumulated_documents
                retrieval.relevant_count = sum(d.selected_for_evidence for d in accumulated_documents)

        # 3. Assemble Generation Context
        generation_context = {
            "route": routing.route,
            "memory_context": [item.to_dict() for item in memory.memories] if memory else [],
            "resolved_facts": [f.to_dict() for f in memory.resolved_facts] if memory else [],
            "primary_entity": memory.primary_entity if memory else "",
            "external_evidence": [document.to_dict() for document in accumulated_documents] if accumulated_documents else [],
        }

        # 4. Generate Answer via M8 (or compatible Generator)
        answer_result = None
        if isinstance(self.generator, GroundedAnswerGenerator):
            answer_result = self.generator.generate(
                query,
                generation_context,
                ground_truth=ground_truth,
                route=routing.route,
                sufficiency_result=sufficiency,
            )
            answer = answer_result.final_answer
        else:
            answer = self.generator.generate(query, generation_context)

        # 5. Persist Turn to Temporary Memory with Validation and Citation Provenance
        effective_turn_id = turn_id or self._next_turn_id()
        stored_context = f"Turn {effective_turn_id}: {query}"
        stored_metadata = dict(metadata or {})
        stored_metadata.update({
            "route": routing.route,
            "strategy": strategy.selected_strategy,
            "sufficient": sufficiency.sufficient if sufficiency else True,
            "validation_status": answer_result.validation.status if answer_result else "UNKNOWN",
            "citation_status": answer_result.citation_status if answer_result else "UNKNOWN",
        })
        stored_memory = self.memory_manager.add_memory(
            conversation_id=self.conversation_id,
            turn_id=effective_turn_id,
            user_query=query,
            assistant_response=answer,
            context=stored_context,
            metadata=stored_metadata,
        )

        return CAROQueryResult(
            query=query,
            conversation_id=self.conversation_id,
            turn_id=effective_turn_id,
            routing=routing,
            strategy=strategy,
            memory=memory,
            retrieval=retrieval,
            generation_context=generation_context,
            answer=answer,
            stored_memory=stored_memory,
            sufficiency=sufficiency,
            answer_result=answer_result,
        )

    def _next_turn_id(self) -> str:
        state = self.memory_manager.get_memory_state(self.conversation_id)
        return str(state["record_count"] + 1)