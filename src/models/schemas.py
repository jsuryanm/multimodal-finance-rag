from pydantic import BaseModel,Field 
from typing import Optional 

class FinancialSummary(BaseModel):
    revenue: Optional[str] = Field(default=None,description="Company revenue")
    net_profit: Optional[str]
    total_assets: Optional[str]
    total_debt: Optional[str]
    eps: Optional[str]
    summary: str

class ComparisionSummary(BaseModel):
    company_a: str
    company_b: str
    revenue_comparision: str 
    profit_comparision: str
    debt_comparision: str 
    final_verdict: str 

class ChartSummary(BaseModel):
    chart_type: str 
    key_trend: str 
    important_values: str 
    explanation: str
