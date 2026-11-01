from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter


def test_simple_general_query_routes_to_direct():
    result = AdaptiveQueryRouter().route("What is Python?")
    assert result.route == "DIRECT"
    assert result.question_type == "definitional"
    assert result.requires_retrieval is False
    assert result.requires_memory is False
    assert result.confidence > 0.0
    assert "simple and self-contained" in result.reason
    assert "DIRECT route" in result.reason
    assert result.signals["retrieval_required"] is False
    assert result.signals["memory_required"] is False


def test_factual_query_routes_to_retrieve():
    result = AdaptiveQueryRouter().route("What is the capital of France?")
    assert result.route == "RETRIEVE"
    assert result.question_type == "factual"
    assert result.requires_retrieval is True
    assert result.requires_memory is False
    assert result.confidence > 0.5


def test_procedural_query_routes_to_retrieve():
    result = AdaptiveQueryRouter().route("How do I create a virtual environment in Python?")
    assert result.route == "RETRIEVE"
    assert result.question_type == "procedural"
    assert result.requires_retrieval is True
    assert "procedural" in result.reason
    assert "RETRIEVE route" in result.reason


def test_comparison_query_routes_to_retrieve():
    result = AdaptiveQueryRouter().route("What is the difference between REST and SOAP?")
    assert result.route == "RETRIEVE"
    assert result.question_type == "comparison"
    assert result.requires_retrieval is True


def test_conversational_follow_up_routes_to_memory():
    result = AdaptiveQueryRouter().route("What about its advantages?")
    assert result.route == "MEMORY"
    assert result.question_type == "conversational"
    assert result.requires_memory is True


def test_context_dependent_query_routes_to_memory():
    result = AdaptiveQueryRouter().route("Who was the person mentioned earlier?")
    assert result.route == "MEMORY"
    assert result.requires_memory is True


def test_multi_hop_query_routes_to_retrieve_without_memory():
    result = AdaptiveQueryRouter().route("Which country was the birthplace of the person who founded the company?")
    assert result.route == "RETRIEVE"
    assert result.question_type == "multi-hop"
    assert result.requires_retrieval is True
    assert result.requires_memory is False
    assert "multi-hop" in result.reason
    assert result.signals["retrieval_required"] is True
    assert result.signals["memory_required"] is False


def test_ambiguous_query_routes_to_direct_or_unknown():
    result = AdaptiveQueryRouter().route("hmm")
    assert result.route in {"DIRECT", "RETRIEVE"}
    assert result.question_type in {"unknown", "general"}


def test_routing_output_is_structured():
    result = AdaptiveQueryRouter().route("What is the capital of France?")
    assert hasattr(result, "query")
    assert hasattr(result, "route")
    assert hasattr(result, "question_type")
    assert hasattr(result, "requires_retrieval")
    assert hasattr(result, "requires_memory")
    assert hasattr(result, "confidence")
    assert hasattr(result, "reason")
