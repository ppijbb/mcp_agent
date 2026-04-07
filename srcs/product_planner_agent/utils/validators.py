"""Input validation utilities for Product Planner Agent."""

from typing import Any, Dict, List, Optional


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_prd_input(prd_data: Dict[str, Any]) -> bool:
    """Validate PRD (Product Requirements Document) input data.

    Args:
        prd_data: Dictionary containing PRD data.

    Returns:
        True if validation passes.

    Raises:
        ValidationError: If validation fails.
    """
    required_fields = ["title", "description", "goals"]
    for field in required_fields:
        if field not in prd_data or not prd_data[field]:
            raise ValidationError(f"Missing required field: {field}")
    return True


def validate_feature(feature: Dict[str, Any]) -> bool:
    """Validate a feature specification.

    Args:
        feature: Dictionary containing feature data.

    Returns:
        True if validation passes.

    Raises:
        ValidationError: If validation fails.
    """
    required_fields = ["name", "description"]
    for field in required_fields:
        if field not in feature or not feature[field]:
            raise ValidationError(f"Feature missing required field: {field}")
    return True


def validate_milestone(milestone: Dict[str, Any]) -> bool:
    """Validate a milestone specification.

    Args:
        milestone: Dictionary containing milestone data.

    Returns:
        True if validation passes.

    Raises:
        ValidationError: If validation fails.
    """
    required_fields = ["name", "target_date"]
    for field in required_fields:
        if field not in milestone or not milestone[field]:
            raise ValidationError(f"Milestone missing required field: {field}")
    return True


def validate_roadmap(roadmap: Dict[str, Any]) -> bool:
    """Validate a complete roadmap structure.

    Args:
        roadmap: Dictionary containing roadmap data.

    Returns:
        True if validation passes.

    Raises:
        ValidationError: If validation fails.
    """
    if "milestones" not in roadmap:
        raise ValidationError("Roadmap must contain 'milestones' field")
    if not isinstance(roadmap["milestones"], list):
        raise ValidationError("Roadmap 'milestones' must be a list")
    for milestone in roadmap["milestones"]:
        validate_milestone(milestone)
    return True
