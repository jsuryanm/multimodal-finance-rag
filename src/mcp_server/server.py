from __future__ import annotations
import re
import yfinance as yf 

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient 

from src.settings.config import settings 
from src.logger.custom_logger import logger 

mcp = FastMCP(name="finance-tools",
              instructions=("Financial tools for annual report analysis"
                            "Use get_stock_prices for live prices."))



def _clean_query(query: str) -> str:
    """Remove noise words from user query."""
    query = query.lower()

    noise = ["stock", "price", "share", "current", "latest", "what", "is"]
    for word in noise:
        query = query.replace(word, "")

    return query.strip()


def _extract_ticker_from_text(text: str) -> str | None:
    """
    Extract ticker like D05.SI or AAPL from text.
    """
    match = re.search(r"\b[A-Z]{1,5}(?:\.SI)?\b", text)
    return match.group(0) if match else None


def _resolve_ticker_with_tavily(query: str) -> str | None:
    """
    Use Tavily search to resolve company → ticker.
    """
    if not settings.TAVILY_API_KEY:
        return None

    try:
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        search_query = f"{query} stock ticker symbol"
        response = client.search(
            query=search_query,
            search_depth="basic",
            max_results=3
        )

        for result in response.get("results", []):
            text = f"{result.get('title', '')} {result.get('content', '')}"
            ticker = _extract_ticker_from_text(text.upper())

            if ticker:
                return ticker

    except Exception as e:
        logger.warning(f"Tavily ticker resolution failed: {e}")

    return None


@mcp.tool()
async def get_stock_price(query: str) -> dict:
    """
    Get stock price using natural language query.

    Flow:
    1. Try direct yfinance lookup
    2. If fails → resolve ticker using Tavily
    3. Retry with resolved ticker
    """

    try:
        query_clean = _clean_query(query)

        stock = yf.Ticker(query_clean)
        info = stock.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")

        if price:
            logger.info(f"Resolved directly via yfinance: {query_clean}")

            return {
                "ticker": info.get("symbol", query_clean.upper()),
                "company": info.get("longName", query_clean),
                "price": price,
                "currency": info.get("currency", "USD"),
                "change_percent": info.get("regularMarketChangePercent"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "week_52_high": info.get("fiftyTwoWeekHigh"),
                "week_52_low": info.get("fiftyTwoWeekLow"),
            }

        ticker = _resolve_ticker_with_tavily(query_clean)

        if not ticker:
            return {
                "error": f"Could not resolve ticker for '{query}'",
                "hint": "Try specifying ticker like D05.SI or AAPL"
            }

        logger.info(f"Tavily resolved ticker: {ticker}")

        stock = yf.Ticker(ticker)
        info = stock.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")

        if not price:
            return {"error": f"No price data found for ticker {ticker}"}

        return {
            "ticker": ticker,
            "company": info.get("longName", ticker),
            "price": price,
            "currency": info.get("currency", "USD"),
            "change_percent": info.get("regularMarketChangePercent"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
        }

    except Exception as e:
        logger.error(f"MCP get_stock_price error [{query}]: {e}")
        return {"error": str(e)}
    
@mcp.tool()
async def search_financial_news(query: str,max_results: int = 3) -> list[dict]:
    """
    Search for recent financial news using Tavily web search.

    Use this when:
    - The annual report doesn't have the answer (e.g., events after report date)
    - User asks about recent analyst ratings or market sentiment
    - User asks about news not covered in the uploaded PDF

    Args:
        query:       Search query (e.g., "DBS Bank Q1 2025 results")
        max_results: Number of results to return (1-5, default 3)

    Returns:
        List of dicts with title, url, and content snippet.
        Returns [{"error": "..."}] if Tavily is not configured.
    """
    try:
        if not settings.TAVILY_API_KEY:
            return [{
                "error": "TAVILY_API_KEY not set in .env file.",
                "hint": "Get a free key at https://tavily.com"
            }]
        
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        response = client.search(query=query,
                                search_depth="basic",
                                max_results=max_results)
        
        results = [{"title":item.get('title'),
                    "url":item.get('url'),
                    "content":item.get('content','')[:500]}
                    for item in response.get("results",[])]
        return results 
    
    except Exception as e:
        logger.error(f"MCP search_financial_news error: {e}")
        return [{"error":str(e)}]

if __name__ == "__main__":
    mcp.run(transport="stdio")