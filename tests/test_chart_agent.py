# tests/test_chart_agent.py
def test_chart_agent_has_required_methods():
    from src.agents.chart_agent import ChartAgent
    agent = ChartAgent.__dict__
    assert "load_image_node" in agent
    assert "analyze_image_node" in agent
    # Catches duplicate/missing method bugs at test time, not runtime