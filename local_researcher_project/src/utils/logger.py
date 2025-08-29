"""
Logging system for Local Researcher

This module provides a centralized logging system with configurable
log levels, formats, and output destinations.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_file_size: str = "10MB",
    backup_count: int = 5
) -> logging.Logger:
    """Setup and configure a logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
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
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
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
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # Fallback to basic file handler if rotation fails
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
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
    backup_count: int = 5
):
    """Add a file handler to an existing logger.
    
    Args:
        logger_name: Name of the logger
        log_file: Path to log file
        level: Log level for the file handler
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files
    """
    logger = logging.getLogger(logger_name)
    
    # Create formatter
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
    console_output: bool = True
):
    """Setup default logging configuration for the application.
    
    Args:
        log_level: Default log level
        log_file: Default log file path
        console_output: Whether to output to console
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
        console_output=console_output
    )
    
    # Setup common loggers
    setup_logger("research_orchestrator", log_level=log_level, console_output=console_output)
    setup_logger("gemini_cli_integration", log_level=log_level, console_output=console_output)
    setup_logger("open_deep_research_adapter", log_level=log_level, console_output=console_output)
    
    return root_logger


# Convenience function for quick logging setup
def quick_logger(name: str) -> logging.Logger:
    """Quick setup for a logger with default settings.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name, console_output=True)
