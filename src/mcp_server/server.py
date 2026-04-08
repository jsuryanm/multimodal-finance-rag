from __future__ import annotations

import yfinance as yf 

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient 

from src.settings.config import settings 
from src.logger.custom_logger import logger 

mcp = FastMCP(name="finance-tools",
              instructions=("Financial tools for annual report analysis"
                            "Use get_stock_prices for live prices."))

@mcp.tool()
async def get_stock_price(ticker: str) -> dict:
    """
    Get the current stock price and key metrics for a ticker symbol.

    Use '.SI' suffix for SGX-listed stocks:
    - DBS Bank → D05.SI
    - OCBC     → O39.SI
    - UOB      → U11.SI
    - Singtel  → Z74.SI

    Args:
        ticker: Stock ticker symbol (e.g., "D05.SI", "AAPL")

    Returns:
        dict with price, currency, change%, market cap, P/E ratio.
        Returns {"error": "..."} if the ticker is invalid.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")

        if not price:
            return {"error":f"No price found for {ticker}. SGX tickers end in .SI (e.g, DO5.SI for DBS)"}

        result = {"ticker":ticker,
                  "company":info.get("longName",ticker),
                  "price":price,
                  "currency":info.get("currency","SGD"),
                  "change_percent":info.get("regularMarketChangePercent"),
                  "market_cap":info.get("marketCap"),
                  "pe_ratio":info.get("trailingPE"),
                  "dividend_yield":info.get("dividendYield"),
                  "week_52_high": info.get("fiftyTwoWeekHigh"),
                  "week_52_low": info.get("fiftyTwoWeekLow"),}
        
        logger.info(f"MCP get_stock_price {ticker} -> {result['currency']} {price}")
        return result
    
    except Exception as e:
        logger.error(f"MCP get_stock_price error [{ticker}]: {e}")
        return {"error":str(e)}
    
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