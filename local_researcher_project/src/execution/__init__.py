"""
Script Execution Package

This package provides functionality for executing various types of scripts
and code snippets as part of the research process.
"""

from .script_executor import ScriptExecutor, ExecutionResult, ExecutionStatus

__all__ = [
    'ScriptExecutor',
    'ExecutionResult',
    'ExecutionStatus'
]
