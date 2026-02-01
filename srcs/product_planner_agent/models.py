"""Pydantic models for data contracts between coordinators."""
from pydantic import BaseModel
from typing import List, Dict, Any


class FigmaComponent(BaseModel):
    name: str
    description: str


class FigmaAnalysisResult(BaseModel):
    product_overview: str
    target_users: List[str]
    core_features: List[FigmaComponent]
    design_principles: List[str]
    technical_context: str


class PRDResult(BaseModel):
    prd_content: str
    file_info: Dict[str, Any]


class BusinessPlan(BaseModel):
    business_model: Dict[str, Any]
    gtm_strategy: Dict[str, Any]
    financial_plan: Dict[str, Any]


class KPISet(BaseModel):
    north_star: str
    user_kpis: List[str]
    business_kpis: List[str]


class MarketingStrategy(BaseModel):
    target_audience: str
    channels: List[str]
    positioning: str


class ProjectPlan(BaseModel):
    roadmap: str
    sprint_plan: str
    resource_plan: str


class OperationsPlan(BaseModel):
    infrastructure_plan: str
    monitoring_plan: str
    support_plan: str
