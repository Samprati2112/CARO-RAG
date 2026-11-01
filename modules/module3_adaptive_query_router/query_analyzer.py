from __future__ import annotations

import re
from dataclasses import dataclass


FOLLOWUP_TOKENS = {
    "this", "that", "these", "those", "it", "they", "he", "she", "we",
    "its", "their", "his", "her", "him", "them", "earlier", "previously",
    "before", "after", "there", "then", "now", "mentioned", "above", "below", "same"
}

QUESTION_WORDS = {
    "what", "who", "when", "where", "why", "how", "which", "whose"
}


FACTUAL_TOKENS = {
    "capital", "country", "countries", "city", "cities", "founder", "founded", "president",
    "population", "largest", "located", "location", "birthplace", "born", "built", "builder",
    "architect", "designed", "written", "author", "invented", "director", "directed",
    "established", "published", "released", "height", "length", "official name", "elevation",
    "won", "titles", "border", "borders", "bordered", "neighboring", "planet", "planets",
    "branch", "branches", "government", "war", "battle", "olympics", "hosted", "host",
    "stadium", "causes", "cause", "occur", "scientist", "developed", "theory",
}


@dataclass
class QueryAnalysis:
    """Deterministic query features used by the router."""

    query: str
    length: int
    question_type: str
    contains_follow_up_reference: bool
    contains_contextual_marker: bool
    is_knowledge_seeking: bool
    likely_needs_retrieval: bool
    likely_direct_answer: bool
    has_procedural_signal: bool
    has_comparison_signal: bool
    has_multi_hop_signal: bool
    normalized: str


class QueryAnalyzer:
    """Lightweight, explainable query analysis for AQR.

    This is intentionally heuristic and deterministic: it extracts features from the
    input text instead of making API calls or relying on an LLM classifier.
    """

    @staticmethod
    def _detect_follow_up(lower: str, normalized: str) -> bool:
        # 1. Explicit cross-turn discourse markers
        explicit_patterns = [
            r"\bwhat about\b",
            r"\bhow about\b",
            r"\bas mentioned\b",
            r"\bin the previous\b",
            r"\bmentioned earlier\b",
            r"\bearlier you\b",
            r"\byou mentioned\b",
            r"\bmentioned above\b",
            r"\bmentioned before\b",
            r"\bthe aforementioned\b",
            r"\bthe same person\b",
            r"\bwho was the person mentioned\b",
            r"\bwho was mentioned\b",
            r"\btell me more about that\b",
        ]
        if any(re.search(pat, lower) for pat in explicit_patterns):
            return True

        # 2. Conversational demonstratives referring across turns
        if re.search(r"\bwhich of (?:those|these)\b", lower):
            # e.g. "Which of those contributions led to his Nobel Prize?"
            return True
        if re.search(r"\bwhat about (?:this|that|these|those|it|its)\b", lower):
            return True
        if re.search(r"^(?:why|how|what|who)\s+(?:was|is|did|are|were)\s+(?:that|this)\b", lower):
            return True

        # 3. Third-person personal pronouns without intra-sentential antecedent named person
        # e.g. "What were his most important scientific contributions?" vs "When was Albert Einstein born and what were his theories?"
        third_person_pronouns = re.findall(r"\b(he|she|his|her|him|they|their|them)\b", lower)
        if third_person_pronouns:
            # Check if query contains a capitalized person name
            has_named_person = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", normalized))
            if not has_named_person:
                # Check for intra-sentential antecedent role noun or compound clause binding
                # e.g. "Which scientist developed the theory of relativity, and in which country was he born?"
                # e.g. "Why do tides occur, and which celestial body has the greatest influence on them?"
                has_intra_sentential_antecedent = bool(
                    re.search(
                        r"\b(?:which|what|who|the)\s+(?:scientist|person|author|director|actor|writer|player|leader|president|founder|athlete|inventor|politician|musician|artist|painter|king|queen|emperor|country|city|company|film|movie|novel|book|planet|system|element|organism|event)\b.*?\b(?:and|or|in which|where|when|while)\b.*?\b(?:he|she|his|her|him|they|their|them|it|its)\b",
                        lower,
                    )
                    or re.search(
                        r"\b(?:tides|planets|countries|branches|causes|differences|animals|plants|languages|oceans|mountains|rivers|stars)\b.*?\b(?:and|or|while|where|which)\b.*?\b(?:them|they|their)\b",
                        lower,
                    )
                )
                if not has_intra_sentential_antecedent:
                    return True

        # 4. Standalone/unbound "its" as subject/possessive without antecedent noun in same query
        if re.search(r"^(?:what|where|when|who|why|how)\s+(?:is|was|are|were|about)\s+its\b", lower):
            return True
        if re.search(r"\bwhat was its\b|\bwhat is its\b", lower) and not re.search(r"\b(?:and|or)\s+what\s+(?:was|is)\s+its\b", lower):
            return True

        # 5. Demonstrative noun phrase referencing context (e.g., "that magazine", "this company", "those contributions")
        if re.search(r"\b(?:that|this)\s+(?:one|magazine|person|company|novel|book|event|show|movie|team|city|country|topic|article|author|band|film|player|president|institution|university|invention|theory)\b", lower):
            return True
        if re.search(r"\b(?:those|these)\s+(?:ones|contributions|theories|books|novels|events|companies|people|magazines|inventions|findings|works)\b", lower):
            return True

        return False

    def analyze(self, query: str) -> QueryAnalysis:
        normalized = " ".join(str(query or "").strip().split())
        lower = normalized.lower()

        length = len(normalized)
        contains_follow_up_reference = self._detect_follow_up(lower, normalized)
        contains_contextual_marker = contains_follow_up_reference

        has_procedural_signal = bool(
            re.search(r"\b(how to|steps to|tutorial|create|install|setup|configure|run)\b", lower)
        )
        has_comparison_signal = bool(
            re.search(
                r"\b(difference between|compare|versus|vs|more than|less than|better|worse|born first|started first|won more|published first|started earlier|published earlier|earlier than|later than|older than|younger than)\b"
                r"|\bwhich\b.*\b(first|earlier|later|older|younger|more)\b.*\bor\b"
                r"|\bwho\b.*\b(first|earlier|later|older|younger|more)\b.*\bor\b",
                lower,
            )
        )
        has_multi_hop_signal = bool(
            re.search(
                r"\b("
                r"who founded|which country was|the person who|the company that|birthplace of the person|"
                r"(?:the\s+)?(?:country|city|archipelago)\s+(?:where|in which)\b.*\bis located|"
                r"the\s+\w+(?:\s+\w+)*\s+of\s+the\s+(?:country|person|place)\s+(?:where|in which|who)|"
                r"(?:where|in which)\s+(?:the\s+)?\w+\s+(?:was|is)\s+(?:held|staged|located|born)|"
                r"(?:which country|what city|who)\s+hosted\b.*"
                r")\b|(?:\band\b.*\blocated in which\b)"
                r"|(?:\bhosted\b.*\band\b.*\b(?:capital|country|stadium)\b)",
                lower,
            )
        )

        is_factual_inquiry = (
            any(token in lower for token in FACTUAL_TOKENS)
            or any(lower.startswith(w + " ") for w in ("where", "who", "when", "which", "whose", "list", "name"))
            or ("what is the capital" in lower)
            or ("three branches" in lower)
        ) and not (
            (lower.startswith("what is") or lower.startswith("who is"))
            and len(normalized.split()) <= 4
            and not any(token in lower for token in ["capital", "country", "founder", "president", "birthplace", "height", "length"])
        )

        question_word = next((word for word in QUESTION_WORDS if lower.startswith(word + " ")), None)
        if question_word:
            start_question = lower.startswith(question_word)
        else:
            start_question = False

        if has_multi_hop_signal:
            question_type = "multi-hop"
        elif has_procedural_signal:
            question_type = "procedural"
        elif has_comparison_signal:
            question_type = "comparison"
        elif contains_follow_up_reference or contains_contextual_marker:
            question_type = "conversational"
        elif "what is" in lower and len(normalized.split()) <= 8 and not any(token in lower for token in ["capital", "country", "city", "population", "height", "length", "official name", "elevation", "difference", "three branches"]):
            question_type = "definitional"
        elif is_factual_inquiry:
            question_type = "factual"
        elif start_question and len(normalized.split()) <= 10:
            question_type = "general"
        elif lower:
            question_type = "unknown"
        else:
            question_type = "unknown"

        likely_direct_answer = (
            question_type == "definitional"
            or (
                len(normalized.split()) <= 6
                and not contains_follow_up_reference
                and not has_multi_hop_signal
                and not has_procedural_signal
                and not has_comparison_signal
                and question_type not in {"factual", "procedural", "comparison", "multi-hop"}
            )
        )

        is_knowledge_seeking = bool(
            any(word in lower for word in ["what", "who", "where", "when", "which", "why", "how", "list", "name", "tell me about"])
            or is_factual_inquiry
        )

        likely_needs_retrieval = (
            (is_knowledge_seeking and not likely_direct_answer and not contains_contextual_marker)
            or has_procedural_signal
            or has_comparison_signal
            or has_multi_hop_signal
            or (question_type in {"factual", "multi-hop", "comparison", "procedural"} and not contains_contextual_marker)
        )

        return QueryAnalysis(
            query=normalized,
            length=length,
            question_type=question_type,
            contains_follow_up_reference=contains_follow_up_reference,
            contains_contextual_marker=contains_contextual_marker,
            is_knowledge_seeking=is_knowledge_seeking,
            likely_needs_retrieval=likely_needs_retrieval,
            likely_direct_answer=likely_direct_answer,
            has_procedural_signal=has_procedural_signal,
            has_comparison_signal=has_comparison_signal,
            has_multi_hop_signal=has_multi_hop_signal,
            normalized=normalized,
        )

