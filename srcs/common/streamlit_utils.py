"""
Streamlit Utilities Module

Provides Streamlit-specific utilities for agent integration including
state management and agent initialization helpers.

Classes:
    AgentState: Container for agent and its associated LLM

Functions:
    get_agent_state: Get or create agent state with session persistence

Example:
    >>> async def init_agent():
    ...     state = await get_agent_state(
    ...         key="my_agent",
    ...         agent_class=MyAgent,
    ...         llm_class=OpenAIAugmentedLLM
    ...     )
    ...     return state.agent
"""

import streamlit as st
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Any
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

T = TypeVar("T", bound=OpenAIAugmentedLLM)


@dataclass
class AgentState:
    """
    Container for agent and its associated LLM.
    
    Attributes:
        agent: The agent instance
        llm: Optional LLM instance attached to the agent
    
    Example:
        >>> state = AgentState(agent=my_agent, llm=my_llm)
        >>> print(state.agent.name)
    """

    agent: Agent
    llm: Optional[OpenAIAugmentedLLM] = None


async def get_agent_state(
    key: str,
    agent_class: Type[Agent],
    llm_class: Optional[Type[T]] = None,
    **agent_kwargs: Any,
) -> AgentState:
    """
    Get or create agent state, reinitializing connections if retrieved from session.

    Args:
        key: Session state key
        agent_class: Agent class to instantiate
        llm_class: Optional LLM class to attach
        **agent_kwargs: Arguments for agent instantiation
        
    Returns:
        AgentState containing the agent and optional LLM instance
    """
    if key not in st.session_state:
        agent = agent_class(
            connection_persistence=False,
            **agent_kwargs,
        )
        await agent.initialize()

        llm: Optional[OpenAIAugmentedLLM] = None
        if llm_class:
            llm = await agent.attach_llm(llm_class)

        state: AgentState = AgentState(agent=agent, llm=llm)
        st.session_state[key] = state
    else:
        state = st.session_state[key]

    return state
