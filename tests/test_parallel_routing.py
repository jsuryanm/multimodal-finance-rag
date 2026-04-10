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
