"""
Logging system for Local Researcher

This module provides a centralized logging system with configurable
log levels, formats, and output destinations with color support.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    # Agent-specific colors
    AGENT_COLORS = {
        'autonomous_orchestrator': '\033[94m',    # Blue
        'task_analyzer': '\033[95m',              # Magenta
        'task_decomposer': '\033[96m',            # Cyan
        'research_agent': '\033[92m',             # Light Green
        'evaluation_agent': '\033[93m',           # Light Yellow
        'validation_agent': '\033[91m',           # Light Red
        'synthesis_agent': '\033[97m',            # White
        'mcp_integration': '\033[90m',            # Gray
        'llm_methods': '\033[94m',                # Blue
        'RESET': '\033[0m'                        # Reset
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        if not self.use_colors or not hasattr(record, 'levelname'):
            return super().format(record)
        
        # Get base format
        formatted = super().format(record)
        
        # Add colors
        level_color = self.COLORS.get(record.levelname, '')
        agent_color = self.AGENT_COLORS.get(record.name, '')
        reset_color = self.COLORS['RESET']
        
        # Apply colors to different parts
        if level_color:
            formatted = formatted.replace(record.levelname, f"{level_color}{record.levelname}{reset_color}")
        
        if agent_color and record.name in self.AGENT_COLORS:
            formatted = formatted.replace(record.name, f"{agent_color}{record.name}{reset_color}")
        
        return formatted


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    detailed_format: bool = True,
    use_colors: bool = True
) -> logging.Logger:
    """Setup and configure a logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        detailed_format: Whether to use detailed formatting
        use_colors: Whether to use colors in console output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatters
    if detailed_format:
        console_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
        file_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
    else:
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console formatter with colors
    console_formatter = ColoredFormatter(
        console_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        use_colors=use_colors
    )
    
    # File formatter without colors
    file_formatter = logging.Formatter(
        file_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        try:
            # Create logs directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Parse max file size
            max_bytes = _parse_size(max_file_size)
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # Fallback to basic file handler if rotation fails
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(level)
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e2:
                logger.warning(f"Failed to setup file logging: {e2}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def _parse_size(size_str: str) -> int:
    """Parse size string to bytes.
    
    Args:
        size_str: Size string (e.g., "10MB", "1GB", "100KB")
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith('B'):
        return int(float(size_str[:-1]))
    else:
        # Assume bytes if no unit specified
        return int(float(size_str))


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(logger_name: str, level: str):
    """Set log level for a specific logger.
    
    Args:
        logger_name: Name of the logger
        level: Log level to set
    """
    logger = logging.getLogger(logger_name)
    level_enum = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level_enum)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level_enum)


def add_file_handler(
    logger_name: str,
    log_file: str,
    level: str = "INFO",
    max_file_size: str = "10MB",
    backup_count: int = 5,
    detailed_format: bool = True
):
    """Add a file handler to an existing logger.
    
    Args:
        logger_name: Name of the logger
        log_file: Path to log file
        level: Log level for the file handler
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files
        detailed_format: Whether to use detailed formatting
    """
    logger = logging.getLogger(logger_name)
    
    # Create formatter
    if detailed_format:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    try:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max file size
        max_bytes = _parse_size(max_file_size)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.warning(f"Failed to add file handler: {e}")


def remove_file_handler(logger_name: str, log_file: str):
    """Remove a specific file handler from a logger.
    
    Args:
        logger_name: Name of the logger
        log_file: Path to the log file to remove
    """
    logger = logging.getLogger(logger_name)
    
    # Find and remove handlers for the specific file
    handlers_to_remove = []
    for handler in logger.handlers:
        if (isinstance(handler, logging.FileHandler) and 
            handler.baseFilename == str(Path(log_file).absolute())):
            handlers_to_remove.append(handler)
    
    for handler in handlers_to_remove:
        logger.removeHandler(handler)
        handler.close()


def setup_default_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    use_colors: bool = True
):
    """Setup default logging configuration for the application.
    
    Args:
        log_level: Default log level
        log_file: Default log file path
        console_output: Whether to output to console
        use_colors: Whether to use colors in console output
    """
    if log_file is None:
        # Create default log file in logs directory
        project_root = Path(__file__).parent.parent.parent
        log_file = str(project_root / "logs" / "local_researcher.log")
    
    # Setup root logger
    root_logger = setup_logger(
        "local_researcher",
        log_level=log_level,
        log_file=log_file,
        console_output=console_output,
        use_colors=use_colors
    )
    
    # Setup common loggers
    setup_logger("research_orchestrator", log_level=log_level, console_output=console_output, use_colors=use_colors)
    setup_logger("gemini_cli_integration", log_level=log_level, console_output=console_output, use_colors=use_colors)
    setup_logger("open_deep_research_adapter", log_level=log_level, console_output=console_output, use_colors=use_colors)
    
    return root_logger


# Convenience function for quick logging setup
def quick_logger(name: str, use_colors: bool = True) -> logging.Logger:
    """Quick setup for a logger with default settings.
    
    Args:
        name: Logger name
        use_colors: Whether to use colors in console output
        
    Returns:
        Logger instance
    """
    return setup_logger(name, console_output=True, use_colors=use_colors)
