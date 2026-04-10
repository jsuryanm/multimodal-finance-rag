# Parallel Fan-Out Routing Design

**Date:** 2026-04-10  
**Status:** Approved

---

## Goal

Enable the orchestrator to route a single user query to multiple agents in parallel when the query has mixed intent — specifically, financial document questions that also reference a live stock ticker.

Only one fan-out combination is supported: `summary + stock_price`. All other route combinations remain single-path.

---

## Architecture

LangGraph's native list-return fan-out replaces the current single-string `_decide_route`. When `_decide_route` returns `["summary", "stock_price"]`, LangGraph dispatches both nodes in parallel. A new `merge_node` always runs after all agent nodes complete, collects `partial_answers`, and writes the final `answer` to state.

```
START → load_memory → route → _decide_route (list[str])
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼              ▼  ▼
                summary       chart        comparision  stock_price
                    │             │              │       │
                    └─────────────┴──────────────┴───────┘
                                  ▼
                              merge_node
                                  ▼
                             save_memory → END
```

---

## Multi-Intent Detection

The heuristic lives entirely in `_decide_route` — no LLM call needed:

- LLM routes to `"summary"` (the primary intent)
- `_extract_ticker()` finds a real ticker in the question (returns anything other than the `"D05.SI"` fallback)
- If both conditions met → return `["summary", "stock_price"]`
- All other routes always return a single-element list

**Examples:**
- "What is DBS revenue?" → `["summary", "stock_price"]` (DBS maps to D05.SI but via KNOWN_TICKERS match, not fallback)
- "What is the revenue?" → `["summary"]` (no ticker found, fallback D05.SI triggered)
- "What is GRAB stock price?" → `["stock_price"]` (LLM routes to stock_price directly, heuristic not applied)
- "Compare both companies" → `["comparision"]` (session_id_b present)

**Note:** `_extract_ticker` returns `"D05.SI"` as the fallback when no ticker is found. The heuristic checks `ticker != "D05.SI"` to guard against false positives — but "DBS" in KNOWN_TICKERS maps to `"D05.SI"`, so DBS questions DO trigger fan-out. This is correct behaviour: "What is DBS revenue?" genuinely has dual intent.

---

## State Changes (`src/agents/state.py`)

Two new fields added. All existing fields unchanged.

```python
# Collectors for parallel agent writes
partial_answers: Annotated[list[dict], add] = []
# Each item: {"route": "summary" | "stock_price" | "chart" | "comparision", "text": str}

active_routes: Annotated[list[str], add] = []
# Records which routes actually ran; used by orchestrator run() response
```

`answer: Optional[str] = None` remains unchanged — written **only** by `merge_node` (single writer, no reducer needed).

---

## Agent Node Wrapper Changes (`src/agents/orchestrator_agent.py`)

All four orchestrator wrapper nodes (`_summary_node`, `_chart_node`, `_comparision_node`, `_stock_price_node`) stop returning `"answer"` and `"route"` directly. Instead they return `partial_answers` and `active_routes`.

Pattern:
```python
return {
    "partial_answers": [{"route": "<route_name>", "text": result.get("answer", "")}],
    "active_routes": ["<route_name>"],
    # node-specific fields (documents, docs_a/b, image_b64, structured_responses, messages)
}
```

The underlying agent classes (`SummaryAgent`, `ChartAgent`, `ComparsionAgent`) are **not modified** — only the orchestrator wrapper nodes change.

---

## `_decide_route` Signature Change

```python
# Before
def _decide_route(self, state: FinanceAgentState) -> str:

# After
def _decide_route(self, state: FinanceAgentState) -> list[str]:
```

All existing override rules preserved (comparision forcing, unknown route fallback). Multi-intent fan-out appended as the last rule before the default return.

`add_conditional_edges` mapping is unchanged — LangGraph resolves each element of the returned list through the same dict.

---

## Merge Node

```python
_SECTION_ORDER = {"stock_price": 0, "summary": 1, "chart": 2, "comparision": 3}
_SECTION_LABEL = {
    "stock_price": "📈 Stock Price",
    "summary":     "📄 Summary",
    "chart":       "🖼 Chart Analysis",
    "comparision": "⚖️ Comparison",
}

async def _merge_node(self, state: FinanceAgentState) -> dict:
    parts = sorted(
        state.get("partial_answers", []),
        key=lambda x: _SECTION_ORDER.get(x["route"], 99),
    )
    if len(parts) == 1:
        return {"answer": parts[0]["text"]}
    sections = [
        f"**{_SECTION_LABEL[p['route']]}**\n\n{p['text']}"
        for p in parts
    ]
    return {"answer": "\n\n---\n\n".join(sections)}
```

Single-route: copies `partial_answers[0]["text"]` to `answer` with no formatting.  
Multi-route: sorts by `_SECTION_ORDER` (stock price first, summary second), formats with bold labels and `---` dividers.

---

## Graph Wiring Changes

```python
# Add merge node
graph.add_node("merge", self._merge_node)

# Conditional edges unchanged (same mapping dict, list returns now supported)
graph.add_conditional_edges("route", self._decide_route, {
    "summary": "summary",
    "chart": "chart",
    "comparision": "comparision",
    "stock_price": "stock_price",
})

# Replace direct agent → save_memory edges with agent → merge
for agent_node in ["summary", "chart", "comparision", "stock_price"]:
    graph.add_edge(agent_node, "merge")   # was: graph.add_edge(agent_node, "save_memory")

graph.add_edge("merge", "save_memory")
```

---

## Streaming Changes (`stream()` method)

Two flags track parallel state during streaming:

```python
stock_price_complete = False   # True after stock_price node on_chain_end fires
stock_price_text = ""          # buffered stock price answer
summary_started = False        # True after first summary LLM token
```

| Event | Action |
|-------|--------|
| `on_chain_end` + node=`"stock_price"` | Buffer `stock_price_text`, set `stock_price_complete=True` |
| `on_chat_model_stream` + node=`"summary"` (first token, `stock_price_complete=True`) | Yield stock price section header + text + summary header, set `summary_started=True` |
| `on_chat_model_stream` + node=`"summary"` (subsequent tokens) | Yield token as normal |
| `on_chain_end` + node=`"merge"` + `stock_price_complete=True` + `summary_started=False` | Yield `stock_price_text` (single stock_price route — no LLM tokens, answer emitted here) |
| All other `on_chat_model_stream` (chart, comparision) | Yield token as normal (unchanged) |

`"merge"` added to the node skip list for `on_chain_error` handling.

---

## `run()` Method Change

```python
return {
    "answer": result.get("answer", ""),                      # set by merge_node
    "route": ", ".join(result.get("active_routes", [""])),   # e.g. "summary, stock_price"
    "session_id": session_id,
}
```

---

## What Does NOT Change

- `SummaryAgent`, `ChartAgent`, `ComparsionAgent` — no modifications
- `RouterDecision` schema — `route: str` stays single string
- `_route_node` — no changes
- `save_memory` node — reads `state["answer"]` which merge_node now always populates
- All frontend code — no changes
- All backend API code — no changes
- All langchain/langgraph package versions — no changes

---

## Files Modified

| File | Change |
|------|--------|
| `src/agents/state.py` | Add `partial_answers` and `active_routes` fields |
| `src/agents/orchestrator_agent.py` | `_decide_route` → list return; all 4 wrapper nodes → `partial_answers`/`active_routes`; add `_merge_node`; rewire graph; update `stream()` and `run()` |
