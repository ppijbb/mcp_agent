from typing import Callable
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from srcs.common.llm import create_fallback_llm_factory

def get_llm_factory() -> Callable[[], GoogleAugmentedLLM]:
    """
    Returns a factory function for creating instances of the GoogleAugmentedLLM.
    This helps in centralizing the LLM creation logic for the product planner agents,
    using the exact model specified.
    Fallback support included for API errors (503, overloaded, etc.)
    """
    return create_fallback_llm_factory(
        primary_model="gemini-2.5-flash-lite-preview-06-07"
    ) 