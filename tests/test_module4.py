import json

import main

from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
from modules.module3_adaptive_query_router.schemas import RoutingResult
from modules.module4_retrieval_strategy.strategy_selector import RetrievalStrategySelector


def select(question: str):
    routing_result = AdaptiveQueryRouter().route(question)
    return RetrievalStrategySelector().select(routing_result)


def test_direct_query_selects_direct():
    result = select("What is Python?")
    assert result.route == "DIRECT"
    assert result.selected_strategy == "DIRECT"
    assert "no retrieval requirement" in result.reason
    assert "DIRECT is selected" in result.reason


def test_procedural_query_selects_dense():
    result = select("How do I create a virtual environment in Python?")
    assert result.requires_retrieval is True
    assert result.requires_memory is False
    assert result.selected_strategy == "DENSE"
    assert "procedural" in result.reason
    assert "semantic similarity" in result.reason


def test_exact_entity_query_selects_sparse():
    result = select("What is the population of Reykjavik?")
    assert result.requires_retrieval is True
    assert result.selected_strategy == "SPARSE"


def test_comparison_query_selects_dense_without_memory():
    result = select("What is the difference between REST and SOAP?")
    assert result.requires_retrieval is True
    assert result.requires_memory is False
    assert result.selected_strategy == "DENSE"
    assert "comparison" in result.reason


def test_multi_hop_query_selects_hybrid_without_memory():
    result = select("Which country was the birthplace of the person who founded the company?")
    assert result.question_type == "multi-hop"
    assert result.requires_memory is False
    assert result.selected_strategy == "HYBRID"
    assert "multi_hop signal is true" in result.reason
    assert "HYBRID is selected" in result.reason


def test_memory_only_query_does_not_force_external_retrieval():
    result = select("What about its advantages?")
    assert result.requires_retrieval is False
    assert result.requires_memory is True
    assert result.selected_strategy == "MEMORY"
    assert "previous conversational context" in result.reason
    assert "MEMORY is selected" in result.reason


def test_memory_and_retrieval_strategy_consumes_module3_schema():
    routing_result = RoutingResult(
        query="What did it cost according to the latest report?",
        route="RETRIEVE_AND_MEMORY",
        question_type="conversational",
        requires_retrieval=True,
        requires_memory=True,
        confidence=0.8,
        reason="Test routing result",
    )
    result = RetrievalStrategySelector().select(routing_result)
    assert result.selected_strategy == "MEMORY_AND_RETRIEVAL"
    assert result.route == routing_result.route
    assert result.query == routing_result.query


def test_strategy_selection_is_deterministic():
    question = "Which country was the birthplace of the person who founded the company?"
    assert select(question).to_dict() == select(question).to_dict()


def test_module4_cli_dispatches_strategy(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--module", "4", "--action", "strategy", "--question", "What is Python?"],
    )
    main.main()
    output = capsys.readouterr().out
    payload = json.loads(output.split("==================================================\n", 2)[-1])
    assert payload["selected_strategy"] == "DIRECT"
    assert payload["question_type"] == "definitional"