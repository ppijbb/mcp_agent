"""
간단한 플러그인 시스템
Table Game Mate의 확장성을 위한 기본 플러그인 아키텍처
"""

import asyncio
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging

from ..utils.logger import get_logger


class PluginType(Enum):
    """플러그인 타입"""
    GAME_RULES = "game_rules"
    UI_COMPONENT = "ui_component"
    AI_STRATEGY = "ai_strategy"


class PluginStatus(Enum):
    """플러그인 상태"""
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"


@dataclass
class PluginMetadata:
    """플러그인 메타데이터"""
    name: str
    version: str
    description: str
    plugin_type: PluginType
    author: str = "Unknown"


@dataclass
class PluginInfo:
    """플러그인 정보"""
    metadata: PluginMetadata
    status: PluginStatus
    load_time: Optional[datetime] = None
    instance: Optional[Any] = None


class BasePlugin:
    """플러그인 기본 클래스"""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.logger = get_logger(f"plugin.{metadata.name}")
    
    async def initialize(self) -> bool:
        """플러그인 초기화"""
        try:
            self.logger.info(f"Plugin {self.metadata.name} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin {self.metadata.name}: {e}")
            return False
    
    async def cleanup(self):
        """플러그인 정리"""
        try:
            self.logger.info(f"Plugin {self.metadata.name} cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin {self.metadata.name}: {e}")


class SimplePluginManager:
    """간단한 플러그인 관리자"""
    
    def __init__(self):
        self.logger = get_logger("plugin_manager")
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugins_by_type: Dict[PluginType, List[str]] = {pt: [] for pt in PluginType}
    
    def register_plugin(self, plugin_class: Type[BasePlugin], metadata: PluginMetadata):
        """플러그인 등록"""
        plugin_name = metadata.name
        
        if plugin_name in self.plugins:
            self.logger.warning(f"Plugin {plugin_name} already registered")
            return False
        
        # 플러그인 인스턴스 생성
        plugin_instance = plugin_class(metadata)
        
        # 플러그인 정보 저장
        plugin_info = PluginInfo(
            metadata=metadata,
            status=PluginStatus.LOADED,
            instance=plugin_instance
        )
        
        self.plugins[plugin_name] = plugin_info
        self.plugins_by_type[metadata.plugin_type].append(plugin_name)
        
        self.logger.info(f"Plugin {plugin_name} registered successfully")
        return True
    
    async def initialize_plugin(self, plugin_name: str) -> bool:
        """플러그인 초기화"""
        if plugin_name not in self.plugins:
            return False
        
        plugin_info = self.plugins[plugin_name]
        
        try:
            if await plugin_info.instance.initialize():
                plugin_info.status = PluginStatus.ACTIVE
                plugin_info.load_time = datetime.now()
                return True
            else:
                plugin_info.status = PluginStatus.ERROR
                return False
        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            self.logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """플러그인 인스턴스 조회"""
        plugin_info = self.plugins.get(plugin_name)
        if plugin_info and plugin_info.status == PluginStatus.ACTIVE:
            return plugin_info.instance
        return None
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """타입별 플러그인 목록"""
        return self.plugins_by_type.get(plugin_type, [])
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """플러그인 통계"""
        total_plugins = len(self.plugins)
        status_counts = {status.value: 0 for status in PluginStatus}
        type_counts = {pt.value: 0 for pt in PluginType}
        
        for plugin_info in self.plugins.values():
            status_counts[plugin_info.status.value] += 1
            type_counts[plugin_info.metadata.plugin_type.value] += 1
        
        return {
            "total_plugins": total_plugins,
            "status_distribution": status_counts,
            "type_distribution": type_counts
        }


# 전역 플러그인 매니저
_plugin_manager = None

def get_plugin_manager() -> SimplePluginManager:
    """전역 플러그인 매니저 인스턴스"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = SimplePluginManager()
    return _plugin_manager


# 플러그인 등록 데코레이터
def register_plugin(plugin_type: PluginType):
    """플러그인 등록 데코레이터"""
    def decorator(cls):
        if not hasattr(cls, 'metadata'):
            cls.metadata = PluginMetadata(
                name=cls.__name__,
                version="1.0.0",
                description=cls.__doc__ or f"Plugin {cls.__name__}",
                plugin_type=plugin_type
            )
        
        # 플러그인 매니저에 등록
        plugin_manager = get_plugin_manager()
        plugin_manager.register_plugin(cls, cls.metadata)
        
        return cls
    return decorator
