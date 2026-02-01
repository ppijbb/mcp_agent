"""
Multi Persona Agent Configuration

Centralized configuration management for multi-persona dialogue system.
All hardcoded values are moved to this configuration file.
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class PersonaConfig:
    """Persona-specific settings"""
    enabled_personas: List[str] = None

    def __post_init__(self):
        if self.enabled_personas is None:
            self.enabled_personas = ["Advocate", "Critic", "Skeptic", "Synthesizer", "MetaObserver"]


@dataclass
class DialogueConfig:
    """Dialogue management settings"""
    max_rounds: int = int(os.getenv("DIALOGUE_MAX_ROUNDS", "3"))
    turns_per_round: int = int(os.getenv("DIALOGUE_TURNS_PER_ROUND", "3"))


@dataclass
class LLMConfig:
    """LLM configuration for different tasks"""
    dialogue_model: str = os.getenv("DIALOGUE_MODEL", "gemini-2.5-flash-lite")
    dialogue_temperature: float = float(os.getenv("DIALOGUE_TEMP", "0.7"))
    synthesis_model: str = os.getenv("SYNTHESIS_MODEL", "gemini-2.5-flash-lite")
    synthesis_temperature: float = float(os.getenv("SYNTHESIS_TEMP", "0.5"))
    meta_model: str = os.getenv("META_MODEL", "gemini-2.5-flash-lite")
    meta_temperature: float = float(os.getenv("META_TEMP", "0.3"))


@dataclass
class MultiPersonaSystemConfig:
    """Overall system configuration"""
    persona: PersonaConfig = None
    dialogue: DialogueConfig = None
    llm: LLMConfig = None

    def __post_init__(self):
        if self.persona is None:
            self.persona = PersonaConfig()
        if self.dialogue is None:
            self.dialogue = DialogueConfig()
        if self.llm is None:
            self.llm = LLMConfig()


# Global configuration instance
config = MultiPersonaSystemConfig()


def get_persona_config() -> PersonaConfig:
    """Get persona configuration."""
    return config.persona


def get_dialogue_config() -> DialogueConfig:
    """Get dialogue configuration."""
    return config.dialogue


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return config.llm


def update_config_from_env():
    """Update configuration from environment variables."""
    # Persona settings
    if os.getenv("ENABLED_PERSONAS"):
        personas_str = os.getenv("ENABLED_PERSONAS")
        config.persona.enabled_personas = [p.strip() for p in personas_str.split(",")]

    # Dialogue settings
    if os.getenv("DIALOGUE_MAX_ROUNDS"):
        config.dialogue.max_rounds = int(os.getenv("DIALOGUE_MAX_ROUNDS"))
    if os.getenv("DIALOGUE_TURNS_PER_ROUND"):
        config.dialogue.turns_per_round = int(os.getenv("DIALOGUE_TURNS_PER_ROUND"))

    # LLM settings
    if os.getenv("DIALOGUE_MODEL"):
        config.llm.dialogue_model = os.getenv("DIALOGUE_MODEL")
    if os.getenv("DIALOGUE_TEMP"):
        config.llm.dialogue_temperature = float(os.getenv("DIALOGUE_TEMP"))
    if os.getenv("SYNTHESIS_MODEL"):
        config.llm.synthesis_model = os.getenv("SYNTHESIS_MODEL")
    if os.getenv("SYNTHESIS_TEMP"):
        config.llm.synthesis_temperature = float(os.getenv("SYNTHESIS_TEMP"))
    if os.getenv("META_MODEL"):
        config.llm.meta_model = os.getenv("META_MODEL")
    if os.getenv("META_TEMP"):
        config.llm.meta_temperature = float(os.getenv("META_TEMP"))


# Initialize configuration from environment
update_config_from_env()
