# Skills (MCP Tools)

External capabilities are exposed to the orchestrator via an MCP (Model Context Protocol) server running as a stdio subprocess. The server is defined in `src/mcp_server/server.py` and built with [FastMCP](https://github.com/jlowin/fastmcp).

The orchestrator connects at startup via `MultiServerMCPClient` and stores references to each tool by name in `self._mcp_tools`.

---

## get_stock_price

**Trigger route:** `stock_price`

Fetches the current stock price and key metrics for a given ticker symbol using `yfinance`.

### Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker` | `str` | Stock ticker symbol, e.g. `"D05.SI"`, `"AAPL"` |

Use the `.SI` suffix for SGX-listed stocks:

| Company | Ticker |
|---------|--------|
| DBS Bank | `D05.SI` |
| OCBC | `O39.SI` |
| UOB | `U11.SI` |
| Singtel | `Z74.SI` |
| CapitaLand | `9CI.SI` |
| Keppel | `BN4.SI` |
| Grab | `GRAB` |
| Sea Limited | `SE` |

The orchestrator auto-extracts the ticker from the user's question via `_extract_ticker()` before calling this tool.

### Output

```python
{
    "ticker": "D05.SI",
    "company": "DBS Group Holdings Ltd",
    "price": 57.20,
    "currency": "SGD",
    "change_percent": -0.17,
    "market_cap": 162300000000,
    "pe_ratio": 10.3,
    "dividend_yield": 0.062,
    "week_52_high": 62.40,
    "week_52_low": 38.50
}
```

Returns `{"error": "..."}` if the ticker is invalid or yfinance fails.

### Formatted Answer Example

```
DBS Group Holdings Ltd
Price: SGD 57.20 (-0.17%)
Market Cap: SGD 162.3B
P/E Ratio: 10.3x
Dividend Yield: 6.20%
52-Week Range: 38.50 – 62.40
```

### Important: MCP Return Format

`langchain_mcp_adapters` returns tool results as `list[TextContent]`, not a raw dict. The orchestrator's `_stock_price_node` extracts and parses the JSON from `result[0].text` before passing to `_format_stock_response`.

---

## search_financial_news

**Trigger route:** Not automatically triggered — available for future use or direct invocation.

Searches for recent financial news using the [Tavily](https://tavily.com) web search API. Useful when the annual report doesn't cover recent events (post-publication earnings, analyst upgrades, regulatory news).

### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | — | Search query, e.g. `"DBS Bank Q1 2025 results"` |
| `max_results` | `int` | `3` | Number of results to return (1–5) |

Requires `TAVILY_API_KEY` set in `.env`. Returns a graceful error message if the key is missing.

### Output

```python
[
    {
        "title": "DBS posts record Q1 profit...",
        "url": "https://...",
        "content": "DBS Group reported a record first-quarter net profit..."  # up to 500 chars
    },
    ...
]
```

Returns `[{"error": "..."}]` on failure.

---

## Running the MCP Server

The MCP server runs as a subprocess started by the orchestrator on `build_graph()`. It can also be run standalone for development:

```bash
# Interactive MCP inspector (test tools in browser UI)
uv run mcp dev src/mcp_server/server.py

# Run as stdio server directly
uv run python -m src.mcp_server.server
```

### Adding New Tools

Add a new async function decorated with `@mcp.tool()` in `src/mcp_server/server.py`:

```python
@mcp.tool()
async def my_new_tool(param: str) -> dict:
    """Docstring shown to the LLM as the tool description."""
    ...
    return {"result": ...}
```

The tool is automatically picked up by the orchestrator on the next `build_graph()` call. To route user questions to it, add a new graph node and update `_decide_route` and the conditional edge map in `build_graph()`.
