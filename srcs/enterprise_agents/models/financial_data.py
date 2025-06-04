"""
Financial Data Models
금융 데이터 모델

Pydantic models for financial data structures used across enterprise agents
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class AssetCategory(str, Enum):
    """Asset category enumeration"""
    STOCKS = "stocks"
    ETF = "etf"
    CRYPTO = "crypto"
    REAL_ESTATE = "real_estate"
    SAVINGS = "savings"
    BONDS = "bonds"

class RiskLevel(int, Enum):
    """Risk level enumeration (1-5)"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

class Currency(str, Enum):
    """Supported currencies"""
    KRW = "KRW"
    USD = "USD"
    EUR = "EUR"
    JPY = "JPY"

class DynamicFinancialData(BaseModel):
    """Dynamic financial data structure with validation"""
    symbol: str = Field(..., description="Financial instrument symbol")
    name: str = Field(..., description="Display name")
    price: float = Field(..., gt=0, description="Current price")
    change_percent: float = Field(..., description="Price change percentage")
    volume: int = Field(0, ge=0, description="Trading volume")
    market_cap: Optional[float] = Field(None, ge=0, description="Market capitalization")
    pe_ratio: Optional[float] = Field(None, gt=0, description="Price-to-earnings ratio")
    dividend_yield: Optional[float] = Field(None, ge=0, le=100, description="Dividend yield percentage")
    sector: Optional[str] = Field(None, description="Industry sector")
    country: Optional[str] = Field("KR", description="Country code")
    currency: Currency = Field(Currency.KRW, description="Price currency")
    data_source: str = Field("unknown", description="Data source identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Symbol cannot be empty')
        return v.strip().upper()
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DynamicProduct(BaseModel):
    """Dynamic financial product structure with validation"""
    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    category: AssetCategory = Field(..., description="Asset category")
    subcategory: str = Field(..., description="Asset subcategory")
    provider: str = Field(..., description="Product provider/issuer")
    expected_return: float = Field(..., description="Expected annual return percentage")
    risk_level: RiskLevel = Field(..., description="Risk level (1-5)")
    min_investment: int = Field(..., gt=0, description="Minimum investment amount")
    features: List[str] = Field(default_factory=list, description="Product features")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Product rating (0-5)")
    data_source: str = Field("unknown", description="Data source identifier")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @validator('features')
    def validate_features(cls, v):
        return [feature.strip() for feature in v if feature.strip()]
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MarketInsight(BaseModel):
    """Market insight and analysis with validation"""
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    impact_score: float = Field(..., ge=0, le=1, description="Impact score (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    related_symbols: List[str] = Field(default_factory=list, description="Related financial symbols")
    source: str = Field(..., description="Insight source")
    timestamp: datetime = Field(default_factory=datetime.now, description="Insight timestamp")
    
    @validator('related_symbols')
    def validate_symbols(cls, v):
        return [symbol.strip().upper() for symbol in v if symbol.strip()]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserProfile(BaseModel):
    """User financial profile with validation"""
    age: int = Field(..., ge=18, le=100, description="User age")
    risk_tolerance: RiskLevel = Field(RiskLevel.MEDIUM, description="Risk tolerance level")
    monthly_income: int = Field(..., gt=0, description="Monthly income (KRW)")
    monthly_expense: int = Field(..., ge=0, description="Monthly expenses (KRW)")
    current_assets: int = Field(0, ge=0, description="Current total assets (KRW)")
    investment_experience: int = Field(1, ge=1, le=5, description="Investment experience level (1-5)")
    preferred_categories: List[AssetCategory] = Field(default_factory=list, description="Preferred asset categories")
    goals: List[str] = Field(default_factory=list, description="Financial goals")
    
    @validator('monthly_expense')
    def validate_expense_vs_income(cls, v, values):
        if 'monthly_income' in values and v > values['monthly_income']:
            raise ValueError('Monthly expenses cannot exceed monthly income')
        return v
    
    @property
    def monthly_surplus(self) -> int:
        """Calculate monthly surplus"""
        return self.monthly_income - self.monthly_expense
    
    @property
    def savings_rate(self) -> float:
        """Calculate savings rate percentage"""
        if self.monthly_income == 0:
            return 0.0
        return (self.monthly_surplus / self.monthly_income) * 100
    
    class Config:
        use_enum_values = True

class FinancialHealthResult(BaseModel):
    """Financial health analysis result"""
    health_score: float = Field(..., ge=0, le=100, description="Overall health score (0-100)")
    grade: str = Field(..., description="Health grade (A-F)")
    analysis: Dict[str, Any] = Field(default_factory=dict, description="Detailed analysis")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    suitable_products: Dict[str, List[DynamicProduct]] = Field(default_factory=dict, description="Suitable products by category")
    market_insights: List[MarketInsight] = Field(default_factory=list, description="Relevant market insights")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    @validator('grade')
    def validate_grade(cls, v):
        valid_grades = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']
        if v not in valid_grades:
            raise ValueError(f'Grade must be one of {valid_grades}')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 