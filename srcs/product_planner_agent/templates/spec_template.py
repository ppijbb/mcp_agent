"""Technical specification template for Product Planner Agent."""

from typing import Dict, Any


DEFAULT_SPEC_TEMPLATE = """# Technical Specification

## {title}

**Author:** {author}
**Created:** {created}
**Status:** {status}

## Overview
{overview}

## Technical Approach
{technical_approach}

## API Specification

### Endpoints
{endpoints}

### Data Models
{data_models}

## Architecture

### Components
{components}

### Data Flow
{data_flow}

## Dependencies
{dependencies}

## Testing Strategy
{testing_strategy}

## Deployment
{deployment}
"""


def render_spec(
    title: str,
    author: str = "Product Planner Agent",
    overview: str = "",
    technical_approach: str = "",
    endpoints: str = "",
    data_models: str = "",
    components: str = "",
    data_flow: str = "",
    dependencies: str = "",
    testing_strategy: str = "",
    deployment: str = "",
    status: str = "Draft"
) -> str:
    """Render a technical specification document from the template.

    Args:
        title: Specification title.
        author: Author name.
        overview: Overview section.
        technical_approach: Technical approach description.
        endpoints: API endpoint specifications.
        data_models: Data model definitions.
        components: System components.
        data_flow: Data flow description.
        dependencies: Project dependencies.
        testing_strategy: Testing approach.
        deployment: Deployment information.
        status: Document status.

    Returns:
        Rendered specification document string.
    """
    from datetime import datetime
    return DEFAULT_SPEC_TEMPLATE.format(
        title=title,
        author=author,
        created=datetime.now().strftime("%Y-%m-%d"),
        status=status,
        overview=overview or "TBD",
        technical_approach=technical_approach or "- TBD",
        endpoints=endpoints or "- TBD",
        data_models=data_models or "- TBD",
        components=components or "- TBD",
        data_flow=data_flow or "- TBD",
        dependencies=dependencies or "- TBD",
        testing_strategy=testing_strategy or "- TBD",
        deployment=deployment or "- TBD"
    )
