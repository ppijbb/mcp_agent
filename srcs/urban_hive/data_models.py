"""Urban Hive data model definitions.

This module centralises typed structures shared between the Urban Hive agent and
Streamlit UI.  It purposefully contains *no* heavyweight dependencies so that
importing it never introduces side-effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List


class UrbanDataCategory(str, Enum):
    """Supported urban analytics categories."""

    TRAFFIC_FLOW = "traffic_flow"
    PUBLIC_SAFETY = "public_safety"
    ILLEGAL_DUMPING = "illegal_dumping"
    COMMUNITY_EVENTS = "community_events"
    URBAN_PLANNING = "urban_planning"
    ENVIRONMENTAL = "environmental"
    REAL_ESTATE_TRENDS = "real_estate_trends"


class UrbanThreatLevel(str, Enum):
    """Qualitative risk/severity levels used across analyses."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UrbanAnalysisResult:
    """Structured representation of an urban data analysis output."""

    data_category: UrbanDataCategory
    threat_level: UrbanThreatLevel
    overall_score: float

    key_metrics: Dict[str, Any] = field(default_factory=dict)
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    affected_areas: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    geographic_data: Dict[str, Any] = field(default_factory=dict)
    predicted_trends: List[str] = field(default_factory=list)


@dataclass
class UrbanActionPlan:
    """Actionable plan derived from an `UrbanAnalysisResult`."""

    plan_id: str
    target_areas: List[str] = field(default_factory=list)
    immediate_actions: List[str] = field(default_factory=list)
    short_term_strategies: List[str] = field(default_factory=list)
    long_term_planning: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: str = ""
    implementation_timeline: Dict[str, Any] = field(default_factory=dict)
    stakeholders: List[str] = field(default_factory=list)


__all__ = [
    "UrbanDataCategory",
    "UrbanThreatLevel",
    "UrbanAnalysisResult",
    "UrbanActionPlan",
] 