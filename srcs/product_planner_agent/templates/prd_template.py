"""PRD (Product Requirements Document) template for Product Planner Agent."""

from typing import Dict, Any, List


DEFAULT_PRD_TEMPLATE = """# Product Requirements Document

## Overview
**Product Name:** {product_name}
**Version:** {version}
**Date:** {date}

## Goals
{goals}

## User Stories
{user_stories}

## Requirements

### Functional Requirements
{functional_requirements}

### Non-Functional Requirements
{non_functional_requirements}

## Success Metrics
{success_metrics}

## Timeline
{timeline}

## Risks & Mitigations
{risks}
"""


def render_prd(
    product_name: str,
    goals: str,
    user_stories: str = "",
    functional_requirements: str = "",
    non_functional_requirements: str = "",
    success_metrics: str = "",
    timeline: str = "",
    risks: str = "",
    version: str = "1.0"
) -> str:
    """Render a PRD document from the template.

    Args:
        product_name: Name of the product.
        goals: High-level goals of the product.
        user_stories: User stories (optional).
        functional_requirements: Functional requirements (optional).
        non_functional_requirements: Non-functional requirements (optional).
        success_metrics: Success metrics (optional).
        timeline: Project timeline (optional).
        risks: Risks and mitigations (optional).
        version: Product version (default: "1.0").

    Returns:
        Rendered PRD document string.
    """
    from datetime import datetime
    return DEFAULT_PRD_TEMPLATE.format(
        product_name=product_name,
        version=version,
        date=datetime.now().strftime("%Y-%m-%d"),
        goals=goals,
        user_stories=user_stories or "- TBD",
        functional_requirements=functional_requirements or "- TBD",
        non_functional_requirements=non_functional_requirements or "- TBD",
        success_metrics=success_metrics or "- TBD",
        timeline=timeline or "- TBD",
        risks=risks or "- TBD"
    )
