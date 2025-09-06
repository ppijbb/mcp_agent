"""
Script Executor

This module provides the base functionality for executing various types of scripts.
"""

import asyncio
import logging
import subprocess
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutionStatus:
    """Execution status constants."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of script execution."""
    script_id: str
    status: str
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    output_files: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.metadata is None:
            self.metadata = {}


class ScriptExecutor(ABC):
    """Base class for script executors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the script executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.timeout = config.get('timeout', 300)
        self.max_memory = config.get('max_memory', '1GB')
        self.working_directory = config.get('working_directory', tempfile.gettempdir())
        self.enabled = config.get('enabled', True)
        
        # Create working directory if it doesn't exist
        Path(self.working_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized script executor: {self.name}")
    
    @abstractmethod
    async def execute(self, script: str, **kwargs) -> ExecutionResult:
        """Execute a script.
        
        Args:
            script: Script content to execute
            **kwargs: Additional execution parameters
            
        Returns:
            Execution result
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.
        
        Returns:
            List of supported extensions
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if the executor is enabled."""
        return self.enabled
    
    def get_config(self) -> Dict[str, Any]:
        """Get executor configuration."""
        return self.config
    
    def _generate_script_id(self) -> str:
        """Generate a unique script ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"script_{timestamp}"
    
    def _create_temp_file(self, content: str, extension: str) -> str:
        """Create a temporary file with the given content.
        
        Args:
            content: File content
            extension: File extension
            
        Returns:
            Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'.{extension}',
            dir=self.working_directory,
            delete=False
        ) as f:
            f.write(content)
            return f.name
    
    def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary file.
        
        Args:
            file_path: Path to the file to clean up
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {file_path}: {e}")
    
    def _parse_memory_limit(self, memory_str: str) -> int:
        """Parse memory limit string to bytes.
        
        Args:
            memory_str: Memory limit string (e.g., '1GB', '512MB')
            
        Returns:
            Memory limit in bytes
        """
        memory_str = memory_str.upper()
        if memory_str.endswith('GB'):
            return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
        elif memory_str.endswith('MB'):
            return int(float(memory_str[:-2]) * 1024 * 1024)
        elif memory_str.endswith('KB'):
            return int(float(memory_str[:-2]) * 1024)
        else:
            return int(float(memory_str))
    
    def _format_execution_time(self, start_time: datetime, end_time: datetime) -> float:
        """Format execution time in seconds.
        
        Args:
            start_time: Start time
            end_time: End time
            
        Returns:
            Execution time in seconds
        """
        return (end_time - start_time).total_seconds()
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"
    
    def __repr__(self) -> str:
        return self.__str__()
