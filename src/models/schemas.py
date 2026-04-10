from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional,Literal

# Agent Outputs

class RouterDecision(BaseModel):
    # The orchestrators router decision
    route: str = Field(description="One of: summary, chart, comparision, stock_price")
    reasoning: str = Field(description="Brief reason for this routing decision")
    

class FinancialSummary(BaseModel):
    # Structured financial metrics extracted from annual report 
    revenue: Optional[str] = Field(default=None,
                                   description="Total revenue with currency, units and year. E.g. 'SGD 22.3 billion (2024)'")
    
    net_profit: Optional[str] = Field(default=None,
                                      description="Net profit with currency,units and year")
    
    operating_profit: Optional[str] = Field(default=None,
                                            description="Operating profit or profit before allowances")
    
    total_assets: Optional[str] = Field(default=None,description="Total assets with currency and year")
    
    total_liabilities: Optional[str] = Field(default=None,description="Total debt or borrowings")
    
    eps: Optional[str] = Field(default=None,description="Earnings per share with currency and year")
    
    roe: Optional[str] = Field(default=None,description="Return on equity percentage")
    
    dividend: Optional[str] = Field(default=None,
                                    description="Dividend per share with currency")
    
    yoy_growth: Optional[str] = Field(default=None,
                                      description="Year-over-year growth percentage for key percentage")
    
    key_risks: Optional[str] = Field(default=None, description="Main risk factors mentioned")

    @field_validator("key_risks", mode="before")
    @classmethod
    def coerce_key_risks(cls, v):
        """LLMs sometimes return key_risks as a list. Join it into a single string."""
        if isinstance(v, list):
            return "\n".join(f"- {item}" for item in v)
        return v

    summary: Optional[str] = Field(description="2-3 sentence analyst style summary of the financial performance")


CHART_TYPES = Literal[
    "bar", "line", "histogram", "pie", "scatter",
    "waterfall", "area", "heatmap", "table", "combo", "unknown"
]

class ChartIntent(BaseModel):
    """Extracted intent from the user's chart question.
    
    Used to (1) score page selection and (2) inject type-specific
    guidance into the vision LLM prompt so it knows what to look for.
    """
    chart_type: CHART_TYPES = "unknown"
    explicit_page: Optional[int] = None      # set if user said "page 12"
    topic_keywords: list[str] = []           # e.g. ["revenue", "segment"]



class ChartAnalysis(BaseModel):
    visual_type: Optional[str] = Field(default=None, description="Type of visual")
    title: Optional[str] = Field(default=None, description="Chart or table title if visible")
    time_period: Optional[str] = Field(default=None, description="Time period covered")
    
    # These were the broken fields — make them Optional with fallback defaults
    key_values: Optional[str] = Field(default=None, description="Most important numerical values shown")
    trend: Optional[str] = Field(default=None, description="Overall trend direction and magnitude")
    key_insight: Optional[str] = Field(default=None, description="Single most important insight from the visual")
    explanation: Optional[str] = Field(default=None, description="Full explanation in financial terms")

class ComparisionRow(BaseModel):
    # A single row in the comparision table
    metric: str 
    company_a: str
    company_b: str
    insight: str

class ComparisionSummary(BaseModel):
    """Side by side comparision of 2 companies"""
    company_a_name: str = Field(description="Name of Company A")
    company_b_name: str = Field(description="Name of Company B")
    
    rows: list[ComparisionRow] = Field(description="Comparision rows for each financial metric")
    
    revenue_comparision: str = Field(description="Who has higher revenue and by how much")
    profit_comparison: str = Field(description="Who is more profitable and why")
    debt_comparison: str = Field(description="Debt levels and financial leverage comparison")
    growth_comparison: str = Field(description="Growth trajectory comparison")
    
    final_verdict: str = Field(description="Which company appears stronger overall and why, based only on the data")
