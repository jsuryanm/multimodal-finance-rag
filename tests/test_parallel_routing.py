from __future__ import annotations


def test_state_has_partial_answers_field():
    """partial_answers must exist and default to empty list."""
    from src.agents.state import FinanceAgentState
    hints = FinanceAgentState.__annotations__
    assert "partial_answers" in hints, "partial_answers field missing from FinanceAgentState"
    assert "active_routes" in hints, "active_routes field missing from FinanceAgentState"


def test_partial_answers_reducer_merges():
    """Annotated[list[dict], add] reducer must combine two lists."""
    from operator import add
    a = [{"route": "summary", "text": "hello"}]
    b = [{"route": "stock_price", "text": "world"}]
    assert add(a, b) == [
        {"route": "summary", "text": "hello"},
        {"route": "stock_price", "text": "world"},
    ]


import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def orchestrator():
    """OrchestratorAgent with all heavy dependencies mocked out."""
    with patch("src.agents.orchestrator_agent.get_llm"), \
         patch("src.agents.orchestrator_agent.SummaryAgent"), \
         patch("src.agents.orchestrator_agent.ChartAgent"), \
         patch("src.agents.orchestrator_agent.ComparsionAgent"), \
         patch("src.agents.orchestrator_agent.get_long_term_memory"):
        from src.agents.orchestrator_agent import OrchestratorAgent
        return OrchestratorAgent()


def test_has_explicit_ticker_known_company(orchestrator):
    assert orchestrator._has_explicit_ticker("What is DBS revenue?") is True


def test_has_explicit_ticker_si_suffix(orchestrator):
    assert orchestrator._has_explicit_ticker("Tell me about D05.SI") is True


def test_has_explicit_ticker_allcaps(orchestrator):
    assert orchestrator._has_explicit_ticker("How is GRAB performing?") is True


def test_has_explicit_ticker_no_match(orchestrator):
    assert orchestrator._has_explicit_ticker("What is the total revenue?") is False


def test_decide_route_single_summary_no_ticker(orchestrator):
    state = {"route": "summary", "question": "What is the revenue?",
             "session_id": "abc", "messages": [], "session_id_b": None}
    assert orchestrator._decide_route(state) == ["summary"]


def test_decide_route_fanout_dbs(orchestrator):
    state = {"route": "summary", "question": "What is DBS revenue?",
             "session_id": "abc", "messages": [], "session_id_b": None}
    result = orchestrator._decide_route(state)
    assert sorted(result) == ["stock_price", "summary"]


def test_decide_route_fanout_grab(orchestrator):
    state = {"route": "summary", "question": "Show me GRAB performance metrics",
             "session_id": "abc", "messages": [], "session_id_b": None}
    result = orchestrator._decide_route(state)
    assert sorted(result) == ["stock_price", "summary"]


def test_decide_route_chart_no_fanout(orchestrator):
    state = {"route": "chart", "question": "Show chart on page 5",
             "session_id": "abc", "messages": [], "session_id_b": None}
    assert orchestrator._decide_route(state) == ["chart"]


def test_decide_route_forces_comparision_with_session_b(orchestrator):
    state = {"route": "summary", "question": "What is revenue?",
             "session_id": "abc", "messages": [], "session_id_b": "def"}
    assert orchestrator._decide_route(state) == ["comparision"]


def test_decide_route_comparision_without_session_b_falls_back(orchestrator):
    state = {"route": "comparision", "question": "Compare both",
             "session_id": "abc", "messages": [], "session_id_b": None}
    assert orchestrator._decide_route(state) == ["summary"]


def test_decide_route_unknown_falls_back(orchestrator):
    state = {"route": "unknown_xyz", "question": "something",
             "session_id": "abc", "messages": [], "session_id_b": None}
    assert orchestrator._decide_route(state) == ["summary"]
