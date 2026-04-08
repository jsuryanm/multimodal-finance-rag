from __future__ import annotations 
from pydantic import BaseModel,Field 
from typing import Optional 

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
    
    key_risks: Optional[str] = Field(default=None,description="Main risk factors mentioned") 
    
    summary: Optional[str] = Field(description="2-3 sentence analyst style summary of the financial performance")


class ChartAnalysis(BaseModel):
    # Analysis of a chart or table from pdf page 
    visual_type: Optional[str] = Field(description="Type of visual: bar chart, line chart, pie chart, table, etc")
    
    title: Optional[str] = Field(default=None,
                                 description="Chart or table title if visible")
    
    time_period: Optional[str] = Field(default=None,
                                       description="Time period covered, eg. '2020-2024'")
    
    key_values: str = Field(description="Most important numerical values shown")

    trend: str = Field(description="Most important numerical values shown")
    
    key_insight: str = Field(description="The most single most important insight from the visual")
    explanation: str = Field(description="Full explanation of what the visual shows in financial terms")


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