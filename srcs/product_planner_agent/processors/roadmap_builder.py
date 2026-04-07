"""Roadmap builder for product planning and tracking."""

from typing import Any, Dict, List, Optional


class RoadmapBuilder:
    """Builds product roadmaps from PRD specifications."""

    def __init__(self, template: Optional[str] = None):
        """Initialize the RoadmapBuilder.

        Args:
            template: Optional roadmap template string.
        """
        self.template = template

    def build(
        self,
        features: List[Dict[str, Any]],
        milestones: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build a roadmap from features and milestones.

        Args:
            features: List of feature dictionaries.
            milestones: List of milestone dictionaries.

        Returns:
            A roadmap dictionary containing features and milestones.
        """
        return {
            "features": features,
            "milestones": milestones,
            "status": "draft"
        }

    def to_markdown(self, roadmap: Dict[str, Any]) -> str:
        """Convert roadmap to markdown format.

        Args:
            roadmap: Roadmap dictionary.

        Returns:
            Markdown string representation of the roadmap.
        """
        lines = ["# Product Roadmap\n"]
        for milestone in roadmap.get("milestones", []):
            lines.append(f"## {milestone.get('name', 'Unnamed')}\n")
        return "\n".join(lines)
