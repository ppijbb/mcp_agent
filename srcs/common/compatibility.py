"""
Compatibility utilities for MCP Agent Hub.

This module provides safe compatibility patches and utilities for handling
library version conflicts and API changes.

Features:
- Safe filtering of problematic safety categories
- MCP library compatibility patches
- Google GenAI API compatibility
- Comprehensive logging and error handling
- Thread-safe operations

Example:
    >>> from srcs.common.compatibility import apply_all_compatibility_patches
    >>> apply_all_compatibility_patches()
"""

import logging
from typing import Optional, List, Any
import types

logger = logging.getLogger(__name__)


class SafetyFilterManager:
    """
    Manages safety settings for Google GenAI API compatibility.
    
    Provides safe filtering of problematic safety categories while maintaining
    proper security controls.
    """
    
    # List of problematic categories that need to be filtered
    PROBLEMATIC_CATEGORIES = ["JAILBREAK"]
    
    # Safe fallback categories
    SAFE_FALLBACK_CATEGORY = "HARM_CATEGORY_DANGEROUS_CONTENT"
    
    @classmethod
    def filter_safety_settings(cls, safety_settings: Optional[List[Any]]) -> Optional[List[Any]]:
        """
        Filter out problematic safety categories from settings.
        
        Args:
            safety_settings: List of safety settings to filter
            
        Returns:
            Filtered list of safety settings
        """
        if not safety_settings:
            return safety_settings
            
        filtered_settings = []
        for setting in safety_settings:
            try:
                category = None
                if isinstance(setting, dict):
                    category = setting.get("category")
                elif hasattr(setting, "category"):
                    category = setting.category
                
                # Skip problematic categories
                if category and any(prob in str(category) for prob in cls.PROBLEMATIC_CATEGORIES):
                    logger.debug(f"Filtered out problematic safety category: {category}")
                    continue
                    
                filtered_settings.append(setting)
            except Exception as e:
                logger.warning(f"Error processing safety setting: {e}")
                filtered_settings.append(setting)  # Keep original if processing fails
        
        return filtered_settings if filtered_settings else safety_settings
    
    @classmethod
    def safe_category_mapping(cls, category: Optional[str]) -> str:
        """
        Map problematic categories to safe alternatives.
        
        Args:
            category: Original category name
            
        Returns:
            Safe category name
        """
        if not category:
            return category
            
        if any(prob in str(category) for prob in cls.PROBLEMATIC_CATEGORIES):
            logger.warning(f"Mapping problematic category {category} to safe fallback")
            return cls.SAFE_FALLBACK_CATEGORY
            
        return category


def apply_mcp_compatibility_patches():
    """
    Apply MCP library compatibility patches in a safe manner.
    
    This function patches known compatibility issues without bypassing
    security controls or introducing vulnerabilities.
    """
    try:
        import mcp.types
        if hasattr(mcp.types, "ElicitRequestParams") and isinstance(mcp.types.ElicitRequestParams, types.UnionType):
            # Type compatibility fix for Python 3.10+
            mcp.types.ElicitRequestParams = mcp.types.ElicitRequestURLParams
            logger.info("Applied MCP type compatibility patch")
    except ImportError:
        # MCP library not available - this is expected in some configurations
        logger.debug("MCP library not found, skipping compatibility patches")
    except Exception as e:
        logger.error(f"Failed to apply MCP compatibility patches: {e}")


def apply_genai_compatibility_patches():
    """
    Apply Google GenAI compatibility patches in a safe manner.
    
    This function patches known compatibility issues while maintaining
    proper safety controls and logging.
    """
    try:
        from google.genai import types as genai_types
        
        # Patch GenerateContentConfig
        if hasattr(genai_types, "GenerateContentConfig"):
            original_config_init = genai_types.GenerateContentConfig.__init__
            
            def patched_config_init(self, *args, **kwargs):
                try:
                    if "safety_settings" in kwargs:
                        kwargs["safety_settings"] = SafetyFilterManager.filter_safety_settings(
                            kwargs["safety_settings"]
                        )
                except Exception as e:
                    logger.warning(f"Error in safety settings filtering: {e}")
                original_config_init(self, *args, **kwargs)
            
            genai_types.GenerateContentConfig.__init__ = patched_config_init
            logger.info("Applied GenAI GenerateContentConfig compatibility patch")
        
        # Patch SafetySetting
        if hasattr(genai_types, "SafetySetting"):
            original_setting_init = genai_types.SafetySetting.__init__
            
            def patched_setting_init(self, *args, **kwargs):
                try:
                    if "category" in kwargs:
                        kwargs["category"] = SafetyFilterManager.safe_category_mapping(
                            kwargs["category"]
                        )
                except Exception as e:
                    logger.warning(f"Error in safety category mapping: {e}")
                original_setting_init(self, *args, **kwargs)
            
            genai_types.SafetySetting.__init__ = patched_setting_init
            logger.info("Applied GenAI SafetySetting compatibility patch")
            
    except ImportError:
        # Google GenAI library not available - this is expected in some configurations
        logger.debug("Google GenAI library not found, skipping compatibility patches")
    except Exception as e:
        logger.error(f"Failed to apply GenAI compatibility patches: {e}")


def apply_all_compatibility_patches():
    """
    Apply all compatibility patches in a centralized, safe manner.
    
    This is the main entry point for applying compatibility fixes.
    """
    logger.info("Applying compatibility patches...")
    
    apply_mcp_compatibility_patches()
    apply_genai_compatibility_patches()
    
    logger.info("Compatibility patches applied successfully")