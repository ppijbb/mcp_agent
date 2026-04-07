"""Roadmap template for Product Planner Agent."""

from typing import Dict, Any, List


DEFAULT_ROADMAP_TEMPLATE = """# Product Roadmap

## {title}

**Last Updated:** {last_updated}

## Vision
{vision}

## Milestones

{milestones}

## Quarterly Roadmap

{quarterly_roadmap}
"""


MILESTONE_TEMPLATE = """### {name}
- **Target Date:** {target_date}
- **Description:** {description}
- **Key Deliverables:** {deliverables}
- **Status:** {status}
"""


def render_roadmap(
    title: str,
    vision: str = "",
    milestones: List[Dict[str, Any]] = None,
    quarterly_roadmap: str = ""
) -> str:
    """Render a roadmap document from the template.

    Args:
        title: Roadmap title.
        vision: Product vision statement.
        milestones: List of milestone dictionaries.
        quarterly_roadmap: Quarterly breakdown text.

    Returns:
        Rendered roadmap document string.
    """
    from datetime import datetime
    
    milestones = milestones or []
    milestone_blocks = []
    for m in milestones:
        milestone_blocks.append(MILESTONE_TEMPLATE.format(
            name=m.get("name", "Unnamed"),
            target_date=m.get("target_date", "TBD"),
            description=m.get("description", "TBD"),
            deliverables="- " + "\n- ".join(m.get("deliverables", ["TBD"])),
            status=m.get("status", "planned")
        ))

    return DEFAULT_ROADMAP_TEMPLATE.format(
        title=title,
        last_updated=datetime.now().strftime("%Y-%m-%d"),
        vision=vision or "TBD",
        milestones="\n\n".join(milestone_blocks) if milestone_blocks else "- TBD",
        quarterly_roadmap=quarterly_roadmap or "- TBD"
    )
