"""
Configuration Manager with Real-time File Monitoring

Monitors YAML/JSON configuration files for changes and automatically reloads.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Thread-safe configuration manager with file watching.
    
    Features:
    - YAML/JSON file support
    - Real-time file change detection
    - Automatic reload on changes
    - Change event callbacks
    - Configuration validation
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        watch: bool = True,
        interval: float = 5.0,
        debounce: float = 1.0
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            watch: Enable file watching (default: True)
            interval: Polling interval in seconds (default: 5.0)
            debounce: Debounce time after change detected (default: 1.0)
        """
        self.config_path = Path(config_path)
        self.watch = watch
        self.interval = interval
        self.debounce = debounce
        
        self._config: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._last_mtime: Optional[float] = None
        
        # Callbacks
        self._pre_reload_callbacks: list[Callable[[], None]] = []
        self._post_reload_callbacks: list[Callable[[Dict[str, Any]], None]] = []
        
        # File watcher
        self._observer: Optional[Observer] = None
        self._watching = False
        
        # Load initial configuration
        if self.config_path.exists():
            self._load_from_file()
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            self._config = {}
    
    def _load_from_file(self) -> bool:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f) or {}
                elif self.config_path.suffix == '.json':
                    data = json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {self.config_path.suffix}")
                    return False
            
            with self._lock:
                old_config = self._config.copy()
                self._config = data
                self._last_mtime = self.config_path.stat().st_mtime
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
            # Fire change callbacks
            if old_config:
                self._fire_callbacks(old_config, data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return False
    
    def _fire_callbacks(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Fire pre and post reload callbacks."""
        # Pre-reload callbacks
        for callback in self._pre_reload_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Pre-reload callback error: {e}")
        
        # Post-reload callbacks
        for callback in self._post_reload_callbacks:
            try:
                callback(new_config)
            except Exception as e:
                logger.error(f"Post-reload callback error: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        with self._lock:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        return default
                else:
                    return default
            
            return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """Set configuration value (supports dot notation)."""
        with self._lock:
            keys = key.split('.')
            config = self._config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        with self._lock:
            return self._config.copy()
    
    def reload(self) -> bool:
        """Manually reload configuration from file."""
        return self._load_from_file()
    
    def on_reload(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback to run after successful reload."""
        self._post_reload_callbacks.append(callback)
    
    def on_before_reload(self, callback: Callable[[], None]):
        """Register callback to run before reload."""
        self._pre_reload_callbacks.append(callback)
    
    def start_watching(self):
        """Start watching for file changes."""
        if not self.watch or self._watching:
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, manager: ConfigManager):
                self.manager = manager
                self.last_trigger = 0
            
            def on_modified(self, event):
                if event.is_directory:
                    return
                
                if event.src_path == str(self.manager.config_path):
                    # Debounce
                    now = time.time()
                    if now - self.last_trigger < self.manager.debounce:
                        return
                    
                    self.last_trigger = now
                    logger.info(f"Config file changed: {event.src_path}")
                    
                    # Wait for file to be fully written
                    time.sleep(self.manager.debounce)
                    
                    # Reload configuration
                    self.manager.reload()
        
        try:
            self._observer = Observer()
            handler = ConfigFileHandler(self)
            self._observer.schedule(
                handler,
                str(self.config_path.parent),
                recursive=False
            )
            self._observer.start()
            self._watching = True
            logger.info(f"Started watching config file: {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
    
    def stop_watching(self):
        """Stop watching for file changes."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._watching = False
            logger.info("Stopped watching config file")
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate configuration structure.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check required sections
            required_sections = ['system', 'accounts', 'payout', 'agents']
            
            for section in required_sections:
                if section not in self._config:
                    return False, f"Missing required section: {section}"
            
            # Validate payout settings
            payout = self._config.get('payout', {})
            if payout.get('enabled', False):
                if 'threshold' not in payout:
                    return False, "Payout threshold not specified"
                if 'schedule' not in payout:
                    return False, "Payout schedule not specified"
            
            # Validate agents
            agents = self._config.get('agents', {})
            if not isinstance(agents, dict):
                return False, "Agents section must be a dictionary"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class AgentsConfigManager:
    """Manager for agent-specific configuration files."""
    
    def __init__(self, config_dir: Union[str, Path]):
        """
        Initialize agents configuration manager.
        
        Args:
            config_dir: Directory containing agent configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.agents_config: Dict[str, ConfigManager] = {}
        self._load_agent_configs()
    
    def _load_agent_configs(self):
        """Load all agent configuration files."""
        config_file = self.config_dir / "agents_config.yaml"
        
        if config_file.exists():
            manager = ConfigManager(config_file, watch=True)
            self.agents_config['_main'] = manager
        else:
            logger.warning(f"Agents config file not found: {config_file}")
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        if '_main' in self.agents_config:
            main_config = self.agents_config['_main'].get_all()
            agents = main_config.get('agents', {})
            return agents.get(agent_name, {})
        return {}
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if agent is enabled."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get('enabled', False)
    
    def get_agent_priority(self, agent_name: str) -> int:
        """Get agent priority."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get('priority', 999)

