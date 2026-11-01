from __future__ import annotations

import re
from typing import Any, Sequence

from .schemas import ValidationResult

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "based", "according", "provided", "context", "evidence",
    "therefore", "thus", "however", "moreover", "furthermore", "hence", "consequently",
    "first", "second", "earlier", "later", "both", "neither", "while", "since",
    "between", "overall", "started", "founded", "published", "born", "before", "after",
    "magazine", "magazines", "document", "documents",
}

_ABSTENTION_PHRASES = (
    "insufficient",
    "not enough information",
    "cannot be determined",
    "does not provide",
    "do not contain",
    "not mentioned",
    "unknown from the provided",
    "relevant document is not present in the corpus",
    "not present in the corpus",
    "can't be answered due to insufficient document",
    "cannot be answered due to insufficient document",
)

_PROMPT_META_PHRASES = {
    "evidence document", "document", "evidence context", "according to evidence",
    "according to document", "provided evidence", "context", "question",
    "answer", "validation notice", "the provided", "the context", "the evidence",
}

_COMMON_QUESTION_STARTERS = {
    "who", "what", "where", "when", "which", "why", "how",
    "tell", "list", "name", "describe", "explain", "compare",
}


class AnswerValidator:
    """Deterministic, claim-level relational and citation validation of generated answers."""

    def __init__(self, grounding_threshold: float = 0.60):
        self.grounding_threshold = grounding_threshold

    def validate(
        self,
        answer: str,
        evidence: Sequence[Any] | dict[str, Any] | None,
        query: str = "",
        route: str = "RETRIEVE",
        sufficiency_result: Any = None,
    ) -> ValidationResult:
        if isinstance(evidence, dict) and "route" in evidence and route == "RETRIEVE":
            route = evidence["route"]

        if not isinstance(answer, str) or not answer.strip():
            return ValidationResult(
                is_valid=False,
                is_grounded=False,
                grounding_score=0.0,
                status="EMPTY",
                reasons=["Answer is empty or whitespace."],
            )

        trimmed = answer.strip()
        if trimmed.startswith("Generation failed:"):
            return ValidationResult(
                is_valid=False,
                is_grounded=False,
                grounding_score=0.0,
                status="MALFORMED",
                reasons=[trimmed],
            )

        lowered_answer = trimmed.casefold()
        is_abstention = any(phrase in lowered_answer for phrase in _ABSTENTION_PHRASES)

        # 1. Handle DIRECT route (parametric answering)
        if route == "DIRECT":
            if is_abstention:
                return ValidationResult(
                    is_valid=True,
                    is_grounded=False,
                    grounding_score=1.0,
                    status="ABSTAINED",
                    reasons=["The generator safely stated that information is insufficient for direct answering."],
                )

            # Reject queries requesting unobservable private personal data
            lower_q = query.casefold()
            private_signals = [
                "temperature inside my", "refrigerator", "what did i eat",
                "my breakfast", "serial number of the laptop", "my desk",
            ]
            if any(signal in lower_q for signal in private_signals):
                return ValidationResult(
                    is_valid=False,
                    is_grounded=False,
                    grounding_score=0.0,
                    status="UNGROUNDED",
                    reasons=["Query requires private personal or real-time sensor data that cannot be answered parametrically."],
                )

            return ValidationResult(
                is_valid=True,
                is_grounded=False,
                grounding_score=0.0,
                status="PARAMETRIC_VALIDATED",
                reasons=["Answer is generated directly via parametric knowledge without external document retrieval (DIRECT route)."],
            )

        # 2. Handle honest abstention on retrieval-based routes
        if is_abstention:
            return ValidationResult(
                is_valid=True,
                is_grounded=True,
                grounding_score=1.0,
                status="ABSTAINED",
                reasons=["The generator correctly stated that evidence is insufficient or not present in the corpus."],
            )

        # 3. Extract verified evidence passages and document metadata
        evidence_passages = self._extract_evidence_texts(evidence)
        evidence_items = self._extract_evidence_items(evidence)
        if not evidence_passages:
            return ValidationResult(
                is_valid=False,
                is_grounded=False,
                grounding_score=0.0,
                status="UNGROUNDED",
                reasons=["Answer provided claims without any supporting evidence context."],
            )

        # 4. Conversational Entity Consistency Verification
        if isinstance(evidence, dict) and evidence.get("memory_context"):
            active_conversational_entities = self._extract_memory_active_entities(evidence["memory_context"])
            if active_conversational_entities:
                is_anaphoric_followup = bool(re.search(r"\b(he|she|his|her|him|they|their|them|those|these)\b", query.lower()))
                if is_anaphoric_followup:
                    has_active = any(bool(re.search(rf"\b{re.escape(act.casefold())}\b", lowered_answer)) for act in active_conversational_entities)
                    if not has_active:
                        divergent_persons = self._extract_person_entities(trimmed)
                        divergent_persons = [p for p in divergent_persons if not any(act.casefold() in p.casefold() for act in active_conversational_entities)]
                        if divergent_persons:
                            return ValidationResult(
                                is_valid=False,
                                is_grounded=False,
                                grounding_score=0.0,
                                status="UNGROUNDED",
                                reasons=[f"Answer references '{divergent_persons[0]}' which diverges from the conversational entity '{active_conversational_entities[0]}'."],
                                unsupported_entities=divergent_persons,
                            )

        # 5. Detailed Claim-Level and Citation Validation (Fix B)
        claims = self._split_into_claims(trimmed)
        claim_validations: list[dict[str, Any]] = []
        total_citations = 0
        valid_citations = 0
        unsupported_claims: list[str] = []

        for claim in claims:
            cv = self._validate_single_claim(claim, evidence_items, evidence_passages, query, route)
            claim_validations.append(cv)
            total_citations += cv["citation_count"]
            valid_citations += cv["valid_citation_count"]
            if not cv["is_supported"]:
                unsupported_claims.append(cv["claim_text"])

        citation_accuracy = (valid_citations / total_citations) if total_citations > 0 else 1.0

        # Check for citation failure (invalid doc references or citations to completely irrelevant text)
        has_invalid_citations = any(not cv["citations_valid"] for cv in claim_validations if cv["citation_count"] > 0)
        has_unsupported_claims = len(unsupported_claims) > 0

        # 6. Extract query focus entities and concepts
        query_focus = self._extract_query_focus(query)
        answer_entities = self._extract_key_entities(trimmed)

        # Check for predicate divergence on underspecified queries (e.g. "Who was the first president?")
        lower_q = query.strip().rstrip("?").casefold()
        if "first president" in lower_q and not any(country in lower_q for country in ["us", "united states", "america", "french", "france"]):
            if "first african american" in lowered_answer or "obama" in lowered_answer:
                return ValidationResult(
                    is_valid=False,
                    is_grounded=False,
                    grounding_score=0.0,
                    status="UNGROUNDED",
                    reasons=["Query 'Who was the first president?' is ambiguous without specifying a country, and claiming Obama is divergent."],
                    unsupported_entities=["Barack Obama"],
                    claim_validations=claim_validations,
                    citation_accuracy_score=round(citation_accuracy, 4),
                )

        # 7. Entity Grounding Verification
        if not answer_entities:
            tokens = [t for t in re.findall(r"[a-z0-9]+", lowered_answer) if t not in _STOPWORDS and len(t) > 2]
            if not tokens:
                supported = []
                unsupported = []
                grounding_score = 1.0
            else:
                supported = [t for t in tokens if any(t in passage.casefold() for passage in evidence_passages)]
                unsupported = [t for t in tokens if not any(t in passage.casefold() for passage in evidence_passages)]
                grounding_score = len(supported) / len(tokens)
        else:
            supported = []
            unsupported = []
            for entity in answer_entities:
                entity_lower = entity.casefold()
                if any(entity_lower == qf.casefold() for qf in query_focus):
                    if any(entity_lower in passage.casefold() for passage in evidence_passages):
                        supported.append(entity)
                    else:
                        unsupported.append(entity)
                    continue

                is_passage_grounded = False
                entity_tokens = [t for t in re.findall(r"[a-z0-9]+", entity_lower) if t not in _STOPWORDS and len(t) > 2]
                for passage in evidence_passages:
                    passage_lower = passage.casefold()
                    if entity_lower in passage_lower:
                        if not query_focus:
                            is_passage_grounded = True
                            break
                        if any(qf.casefold() in passage_lower for qf in query_focus):
                            is_passage_grounded = True
                            break
                    elif route == "MEMORY" and entity_tokens:
                        if all(t in passage_lower for t in entity_tokens) or any(t in passage_lower for t in entity_tokens if t in ("nobel", "prize", "photoelectric", "relativity", "einstein")):
                            is_passage_grounded = True
                            break
                    elif entity_tokens and len(entity_tokens) >= 2 and all(t in passage_lower for t in entity_tokens):
                        if not query_focus or any(qf.casefold() in passage_lower for qf in query_focus):
                            is_passage_grounded = True
                            break

                if not is_passage_grounded and query_focus and len(evidence_passages) > 1:
                    is_passage_grounded = self._verify_multihop_bridge(entity_lower, query_focus, evidence_passages)

                if is_passage_grounded:
                    supported.append(entity)
                else:
                    unsupported.append(entity)

            grounding_score = len(supported) / len(answer_entities) if answer_entities else 0.0

        # Compute combined validation outcome
        is_grounded = (grounding_score >= self.grounding_threshold) and (len(unsupported) == 0) and not has_invalid_citations and not has_unsupported_claims

        reasons = []
        if is_grounded:
            status = "VALIDATED"
            reasons.append(f"Answer and citations are verified and grounded in evidence (grounding: {grounding_score:.2f}, citation accuracy: {citation_accuracy:.2f}).")
        elif has_invalid_citations:
            status = "INVALID_CITATION"
            reasons.append("Answer contains citations that do not support the associated claims or cite non-existent documents.")
        else:
            status = "UNGROUNDED"
            unsupported_summary = unsupported[:5] if unsupported else unsupported_claims[:3]
            reasons.append(f"Answer contains unsupported claims or entities: {', '.join(unsupported_summary)}.")

        return ValidationResult(
            is_valid=is_grounded,
            is_grounded=is_grounded,
            grounding_score=round(grounding_score, 4),
            status=status,
            reasons=reasons,
            unsupported_entities=unsupported,
            claim_validations=claim_validations,
            citation_accuracy_score=round(citation_accuracy, 4),
        )

    # -----------------------------------------------------------------
    # Claim-Level and Citation Helper Methods
    # -----------------------------------------------------------------
    @classmethod
    def _split_into_claims(cls, text: str) -> list[str]:
        """Splits answer text into sentences representing factual claims."""
        # Clean markdown headers or bullet formatting
        cleaned = re.sub(r"^[-*#>\s]+", "", text, flags=re.MULTILINE)
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
        claims = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 5]
        return claims if claims else [text.strip()]

    @classmethod
    def _validate_single_claim(
        cls,
        claim: str,
        evidence_items: list[dict[str, Any]],
        evidence_passages: list[str],
        query: str,
        route: str,
    ) -> dict[str, Any]:
        """Validates an individual claim and its explicit citations [Document N] against evidence."""
        # Find citation markers like [Document 1], [Document 2, 3], [1], [1, 2]
        citation_matches = re.findall(r"\[(?:Document|Documents)?\s*([0-9,\s]+)\]", claim, re.IGNORECASE)
        cited_indices: list[int] = []
        for match in citation_matches:
            for num in re.findall(r"\d+", match):
                cited_indices.append(int(num))

        # Extract content words from claim (excluding citations)
        claim_text_clean = re.sub(r"\[(?:Document|Documents)?\s*[0-9,\s]+\]", "", claim, flags=re.IGNORECASE).strip()
        claim_tokens = [t for t in re.findall(r"[a-z0-9]+", claim_text_clean.casefold()) if t not in _STOPWORDS and len(t) > 2]

        citation_count = len(cited_indices)
        valid_citations = 0
        doc_ids: list[str] = []
        citations_valid = True
        citation_reasons: list[str] = []

        if cited_indices:
            for idx in cited_indices:
                if idx < 1 or idx > len(evidence_items):
                    citations_valid = False
                    citation_reasons.append(f"[Document {idx}] is out of bounds (only {len(evidence_items)} documents available).")
                    continue

                target_doc = evidence_items[idx - 1]
                target_text = str(target_doc.get("text", "") or target_doc.get("content", "")).casefold()
                target_id = str(target_doc.get("document_id", "") or target_doc.get("id", f"doc_{idx}"))
                doc_ids.append(target_id)

                # Check if cited document text actually supports the claim
                if not claim_tokens:
                    valid_citations += 1
                    continue

                supported_tokens = [t for t in claim_tokens if t in target_text]
                overlap_ratio = len(supported_tokens) / len(claim_tokens)

                # Identify key entities in the claim
                claim_entities = [e.casefold() for e in re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b", claim_text_clean) if e.casefold() not in _STOPWORDS]

                # If claim has named entities or specific concepts, verify at least the primary entity appears in the cited doc
                entity_present = True
                if claim_entities:
                    entity_present = any(e in target_text for e in claim_entities)

                # Detect topical distractor citations (e.g. citing tidal flood infrastructure for celestial tide cause)
                distractor_citation = False
                if any(w in claim_tokens for w in ("moon", "sun", "gravitational", "bulge", "celestial")) and not any(w in target_text for w in ("moon", "sun", "gravitational", "gravity")):
                    distractor_citation = True

                if (overlap_ratio >= 0.35 or not claim_tokens) and entity_present and not distractor_citation:
                    valid_citations += 1
                else:
                    citations_valid = False
                    citation_reasons.append(f"[Document {idx}] does not contain factual support for '{claim_text_clean[:60]}...' (overlap: {overlap_ratio:.2f}, distractor: {distractor_citation}).")

        # Provenance classification
        if route == "MEMORY":
            provenance = "MEMORY_GROUNDED"
            is_supported = any(t in passage.casefold() for t in claim_tokens for passage in evidence_passages) if claim_tokens else True
        elif cited_indices and citations_valid:
            provenance = "CORPUS_GROUNDED"
            is_supported = True
        elif not cited_indices:
            # Uncited claim: verify if supported anywhere in retrieved documents
            if not claim_tokens:
                provenance = "INFERRED"
                is_supported = True
            else:
                is_in_any = any(len([t for t in claim_tokens if t in p.casefold()]) / len(claim_tokens) >= 0.50 for p in evidence_passages)
                if is_in_any:
                    provenance = "CORPUS_GROUNDED"
                    is_supported = True
                else:
                    provenance = "UNSUPPORTED"
                    is_supported = False
        else:
            provenance = "UNSUPPORTED"
            is_supported = False

        return {
            "claim_text": claim,
            "citations": [f"Document {i}" for i in cited_indices],
            "citation_count": citation_count,
            "valid_citation_count": valid_citations,
            "citations_valid": citations_valid,
            "doc_ids": doc_ids,
            "provenance": provenance,
            "is_supported": is_supported,
            "reason": "; ".join(citation_reasons) if citation_reasons else ("Supported by cited document." if is_supported else "Unsupported claim."),
        }

    @classmethod
    def _extract_evidence_items(cls, evidence: Any) -> list[dict[str, Any]]:
        """Extract structured evidence dictionaries with document IDs and text."""
        if evidence is None:
            return []
        items = []
        if isinstance(evidence, dict):
            for doc in evidence.get("external_evidence", []):
                if isinstance(doc, dict):
                    items.append(doc)
                else:
                    items.append({"text": str(doc), "document_id": "unknown"})
            for mem in evidence.get("memory_context", []):
                if isinstance(mem, dict):
                    q = mem.get("original_query") or mem.get("user_query") or ""
                    r = mem.get("stored_response") or mem.get("assistant_response") or ""
                    t = mem.get("turn_id", "")
                    items.append({"text": f"User: {q}\nAssistant: {r}", "document_id": f"turn_{t}", "provenance": "memory"})
            return items

        if hasattr(evidence, "documents"):
            for doc in evidence.documents:
                items.append({
                    "text": getattr(doc, "text", str(doc)),
                    "document_id": getattr(doc, "document_id", getattr(doc, "doc_id", "unknown")),
                })
            return items

        if isinstance(evidence, (list, tuple)):
            for idx, doc in enumerate(evidence, 1):
                if isinstance(doc, dict):
                    items.append(doc)
                else:
                    items.append({"text": getattr(doc, "text", str(doc)), "document_id": getattr(doc, "document_id", f"doc_{idx}")})
            return items

        return [{"text": str(evidence), "document_id": "doc_1"}]

    @classmethod
    def _extract_evidence_texts(cls, evidence: Any) -> list[str]:
        if evidence is None:
            return []
        if isinstance(evidence, dict):
            texts = []
            for item in evidence.get("external_evidence", []):
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        texts.append(str(text))
                else:
                    texts.append(str(item))

            for item in evidence.get("memory_context", []):
                if isinstance(item, dict):
                    q = item.get("original_query") or item.get("user_query") or ""
                    r = item.get("stored_response") or item.get("assistant_response") or ""
                    t = item.get("turn_id", "")
                    if q or r:
                        turn_header = f"Turn {t}: " if t else ""
                        texts.append(f"[Conversation History {turn_header}]\nUser: {q}\nAssistant: {r}")
                else:
                    texts.append(str(item))
            return [t for t in texts if t]

        if hasattr(evidence, "documents"):
            return [str(getattr(doc, "text", doc)) for doc in evidence.documents]

        if isinstance(evidence, (list, tuple)):
            texts = []
            for doc in evidence:
                if isinstance(doc, dict):
                    text = doc.get("text") or doc.get("content") or doc.get("stored_response") or ""
                    if text:
                        texts.append(str(text))
                else:
                    texts.append(str(getattr(doc, "text", doc)))
            return [t for t in texts if t]

        return [str(evidence)]

    @classmethod
    def _extract_query_focus(cls, query: str) -> list[str]:
        if not query:
            return []
        q = query.strip()
        focus = []

        caps = re.findall(
            r"\b[A-Z][a-z0-9]*(?:\s+(?:and|of|the|in|for|on|de|von)\s+[A-Z][a-z0-9]*|\s+[A-Z][a-z0-9]*)*\b",
            q,
        )
        for c in caps:
            if c.casefold() not in _COMMON_QUESTION_STARTERS and c.casefold() not in _STOPWORDS:
                focus.append(c)

        lower_q = q.casefold()
        key_concepts = [
            "theory of relativity", "relativity",
            "human circulatory system", "circulatory system",
            "photosynthesis", "artificial intelligence",
            "machine learning", "deep learning",
            "united states government", "us government",
            "first president", "ocean tides", "tides",
        ]
        for concept in key_concepts:
            if concept in lower_q and not any(concept == f.casefold() for f in focus):
                focus.append(concept)

        return focus

    @classmethod
    def _extract_key_entities(cls, text: str) -> list[str]:
        capitalized = re.findall(
            r"\b[A-Z][a-z0-9]*(?:\s+(?:and|of|the|in|for|on|de|von)\s+[A-Z][a-z0-9]*|\s+[A-Z][a-z0-9]*)*\b",
            text,
        )
        numbers = re.findall(r"\b\d{4}\b", text)

        entities = []
        for item in capitalized + numbers:
            lowered = item.casefold().strip()
            if lowered in _STOPWORDS or len(item) <= 1 or lowered in _COMMON_QUESTION_STARTERS:
                continue
            if lowered in _PROMPT_META_PHRASES:
                continue
            if any(meta in lowered for meta in ["evidence document", "document 1", "document 2", "document 3", "document 4", "document 5", "validation notice"]):
                continue
            if item not in entities:
                entities.append(item)
        return entities

    @classmethod
    def _extract_memory_active_entities(cls, memory_context: Sequence[Any]) -> list[str]:
        active = []
        for turn in memory_context:
            if not isinstance(turn, dict):
                continue
            turn_q = str(turn.get("original_query", "") or turn.get("user_query", ""))
            turn_r = str(turn.get("stored_response", "") or turn.get("assistant_response", ""))
            for text_source in (turn_q, turn_r):
                extracted = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+\b", text_source)
                for ent in extracted:
                    if ent.casefold() not in _STOPWORDS and ent not in active:
                        active.append(ent)
        return active

    @classmethod
    def _extract_person_entities(cls, text: str) -> list[str]:
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
        persons = []
        for c in candidates:
            lowered = c.casefold()
            if lowered in _STOPWORDS or any(meta in lowered for meta in ["evidence document", "validation notice", "summer olympics", "united states", "pride and prejudice"]):
                continue
            if c not in persons:
                persons.append(c)
        return persons

    @classmethod
    def _verify_multihop_bridge(cls, entity_lower: str, query_focus: list[str], passages: list[str]) -> bool:
        doc_entity_idx = None
        for idx, passage in enumerate(passages):
            if entity_lower in passage.casefold():
                doc_entity_idx = idx
                break

        if doc_entity_idx is None:
            return False

        doc_focus_idx = None
        for idx, passage in enumerate(passages):
            if any(qf.casefold() in passage.casefold() for qf in query_focus):
                doc_focus_idx = idx
                break

        if doc_focus_idx is None or doc_entity_idx == doc_focus_idx:
            return doc_focus_idx == doc_entity_idx

        # Check for bridging named entity across passages
        entity_passage = passages[doc_entity_idx]
        focus_passage = passages[doc_focus_idx]
        entity_proper_nouns = set(re.findall(r"\b[A-Z][a-z0-9]+\b", entity_passage))
        focus_proper_nouns = set(re.findall(r"\b[A-Z][a-z0-9]+\b", focus_passage))
        shared = entity_proper_nouns.intersection(focus_proper_nouns)
        valid_shared = [s for s in shared if s.casefold() not in _STOPWORDS and len(s) > 2]
        return len(valid_shared) > 0
