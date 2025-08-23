"""
플러그인 관리 시스템
Table Game Mate의 확장성을 위한 플러그인 아키텍처
"""

import asyncio
import importlib
import inspect
import json
import os
from typing import Dict, List, Any, Optional, Type, Callable, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import logging

from ..utils.logger import get_logger
from .error_handler import ErrorHandler, ErrorSeverity, ErrorCategory


class PluginType(Enum):
    """플러그인 타입"""
    GAME_RULES = "game_rules"           # 게임 규칙 플러그인
    UI_COMPONENT = "ui_component"       # UI 컴포넌트 플러그인
    AI_STRATEGY = "ai_strategy"         # AI 전략 플러그인
    MCP_SERVER = "mcp_server"           # MCP 서버 플러그인
    GAME_MECHANIC = "game_mechanic"     # 게임 메커니즘 플러그인
    INTEGRATION = "integration"         # 외부 시스템 연동 플러그인


class PluginStatus(Enum):
    """플러그인 상태"""
    LOADED = "loaded"           # 로드됨
    ACTIVE = "active"           # 활성화됨
    ERROR = "error"             # 에러 발생
    DISABLED = "disabled"       # 비활성화됨
    UNLOADED = "unloaded"       # 언로드됨


@dataclass
class PluginMetadata:
    """플러그인 메타데이터"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    license: Optional[str] = None
    min_system_version: Optional[str] = None
    max_system_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "homepage": self.homepage,
            "license": self.license,
            "min_system_version": self.min_system_version,
            "max_system_version": self.max_system_version
        }


@dataclass
class PluginInfo:
    """플러그인 정보"""
    metadata: PluginMetadata
    file_path: str
    module_name: str
    status: PluginStatus
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None
    instance: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "metadata": self.metadata.to_dict(),
            "file_path": self.file_path,
            "module_name": self.module_name,
            "status": self.status.value,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "error_message": self.error_message,
            "has_instance": self.instance is not None
        }


class BasePlugin:
    """플러그인 기본 클래스"""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.logger = get_logger(f"plugin.{metadata.name}")
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """플러그인 초기화"""
        try:
            self.is_initialized = True
            self.logger.info(f"Plugin {self.metadata.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin {self.metadata.name}: {e}")
            return False
    
    async def cleanup(self):
        """플러그인 정리"""
        try:
            self.is_initialized = False
            self.logger.info(f"Plugin {self.metadata.name} cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin {self.metadata.name}: {e}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """플러그인 기능 반환"""
        return {
            "name": self.metadata.name,
            "type": self.metadata.plugin_type.value,
            "version": self.metadata.version,
            "initialized": self.is_initialized
        }


class GameRulesPlugin(BasePlugin):
    """게임 규칙 플러그인"""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.supported_games: List[str] = []
        self.rule_parsers: Dict[str, Callable] = {}
    
    async def parse_game_rules(self, game_name: str, rules_text: str) -> Dict[str, Any]:
        """게임 규칙 파싱"""
        if game_name not in self.supported_games:
            raise ValueError(f"Game {game_name} not supported by this plugin")
        
        parser = self.rule_parsers.get(game_name)
        if not parser:
            raise ValueError(f"No parser available for game {game_name}")
        
        return await parser(rules_text)
    
    def register_game_parser(self, game_name: str, parser_func: Callable):
        """게임별 규칙 파서 등록"""
        self.rule_parsers[game_name] = parser_func
        if game_name not in self.supported_games:
            self.supported_games.append(game_name)


class UIComponentPlugin(BasePlugin):
    """UI 컴포넌트 플러그인"""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.components: Dict[str, Type] = {}
        self.templates: Dict[str, Dict[str, Any]] = {}
    
    def register_component(self, name: str, component_class: Type):
        """UI 컴포넌트 등록"""
        self.components[name] = component_class
    
    def register_template(self, name: str, template: Dict[str, Any]):
        """UI 템플릿 등록"""
        self.templates[name] = template
    
    def get_component(self, name: str) -> Optional[Type]:
        """UI 컴포넌트 조회"""
        return self.components.get(name)
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """UI 템플릿 조회"""
        return self.templates.get(name)


class AIStrategyPlugin(BasePlugin):
    """AI 전략 플러그인"""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.strategies: Dict[str, Callable] = {}
        self.persona_templates: Dict[str, Dict[str, Any]] = {}
    
    def register_strategy(self, name: str, strategy_func: Callable):
        """AI 전략 등록"""
        self.strategies[name] = strategy_func
    
    def register_persona_template(self, name: str, template: Dict[str, Any]):
        """페르소나 템플릿 등록"""
        self.persona_templates[name] = template
    
    async def execute_strategy(self, strategy_name: str, game_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """AI 전략 실행"""
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        if asyncio.iscoroutinefunction(strategy):
            return await strategy(game_state, **kwargs)
        else:
            return strategy(game_state, **kwargs)


class PluginManager:
    """플러그인 관리자"""
    
    def __init__(self, plugins_dir: str = "./plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.logger = get_logger("plugin_manager")
        self.error_handler = ErrorHandler()
        
        # 플러그인 저장소
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugins_by_type: Dict[PluginType, List[str]] = {pt: [] for pt in PluginType}
        
        # 플러그인 디렉토리 생성
        self.plugins_dir.mkdir(exist_ok=True)
        
        # 자동 검색 활성화
        self.auto_discover = True
        self.auto_load = False
    
    async def discover_plugins(self) -> List[str]:
        """플러그인 자동 검색"""
        discovered = []
        
        try:
            # 플러그인 디렉토리 스캔
            for plugin_file in self.plugins_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                
                plugin_name = plugin_file.stem
                if plugin_name not in self.plugins:
                    discovered.append(plugin_name)
            
            self.logger.info(f"Discovered {len(discovered)} new plugins: {discovered}")
            
        except Exception as e:
            await self.error_handler.handle_error(
                error=e,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM_ERROR,
                context={"operation": "plugin_discovery"}
            )
        
        return discovered
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """플러그인 로드"""
        try:
            plugin_file = self.plugins_dir / f"{plugin_name}.py"
            
            if not plugin_file.exists():
                self.logger.error(f"Plugin file not found: {plugin_file}")
                return False
            
            # 모듈 동적 로드
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            if not spec or not spec.loader:
                self.logger.error(f"Failed to create module spec for {plugin_name}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 플러그인 클래스 찾기
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                self.logger.error(f"No valid plugin class found in {plugin_name}")
                return False
            
            # 메타데이터 추출
            metadata = self._extract_metadata(plugin_class, plugin_name)
            if not metadata:
                self.logger.error(f"Failed to extract metadata from {plugin_name}")
                return False
            
            # 플러그인 인스턴스 생성
            plugin_instance = plugin_class(metadata)
            
            # 플러그인 정보 저장
            plugin_info = PluginInfo(
                metadata=metadata,
                file_path=str(plugin_file),
                module_name=plugin_name,
                status=PluginStatus.LOADED,
                load_time=datetime.now(),
                instance=plugin_instance
            )
            
            self.plugins[plugin_name] = plugin_info
            self.plugins_by_type[metadata.plugin_type].append(plugin_name)
            
            self.logger.info(f"Plugin {plugin_name} loaded successfully")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                error=e,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SYSTEM_ERROR,
                context={"plugin_name": plugin_name, "operation": "plugin_loading"}
            )
            
            # 에러 상태로 플러그인 정보 저장
            if plugin_name in self.plugins:
                self.plugins[plugin_name].status = PluginStatus.ERROR
                self.plugins[plugin_name].error_message = str(e)
            
            return False
    
    def _extract_metadata(self, plugin_class: Type, plugin_name: str) -> Optional[PluginMetadata]:
        """플러그인 메타데이터 추출"""
        try:
            # 클래스 속성에서 메타데이터 찾기
            if hasattr(plugin_class, 'metadata'):
                return plugin_class.metadata
            
            # 클래스 docstring에서 메타데이터 파싱
            doc = plugin_class.__doc__ or ""
            
            # 기본 메타데이터 생성
            metadata = PluginMetadata(
                name=plugin_name,
                version="1.0.0",
                description=doc.strip().split('\n')[0] if doc else f"Plugin {plugin_name}",
                author="Unknown",
                plugin_type=PluginType.GAME_RULES  # 기본값
            )
            
            # 클래스 속성에서 추가 정보 추출
            if hasattr(plugin_class, 'PLUGIN_TYPE'):
                metadata.plugin_type = PluginType(plugin_class.PLUGIN_TYPE)
            
            if hasattr(plugin_class, 'VERSION'):
                metadata.version = plugin_class.VERSION
            
            if hasattr(plugin_class, 'AUTHOR'):
                metadata.author = plugin_class.AUTHOR
            
            if hasattr(plugin_class, 'DESCRIPTION'):
                metadata.description = plugin_class.DESCRIPTION
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {plugin_name}: {e}")
            return None
    
    async def initialize_plugin(self, plugin_name: str) -> bool:
        """플러그인 초기화"""
        if plugin_name not in self.plugins:
            self.logger.error(f"Plugin {plugin_name} not found")
            return False
        
        plugin_info = self.plugins[plugin_name]
        
        if plugin_info.status != PluginStatus.LOADED:
            self.logger.error(f"Plugin {plugin_name} is not in LOADED state")
            return False
        
        try:
            if plugin_info.instance:
                success = await plugin_info.instance.initialize()
                if success:
                    plugin_info.status = PluginStatus.ACTIVE
                    self.logger.info(f"Plugin {plugin_name} initialized successfully")
                    return True
                else:
                    plugin_info.status = PluginStatus.ERROR
                    plugin_info.error_message = "Initialization failed"
                    return False
            
        except Exception as e:
            await self.error_handler.handle_error(
                error=e,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM_ERROR,
                context={"plugin_name": plugin_name, "operation": "plugin_initialization"}
            )
            
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = str(e)
            return False
        
        return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """플러그인 언로드"""
        if plugin_name not in self.plugins:
            return False
        
        plugin_info = self.plugins[plugin_name]
        
        try:
            # 플러그인 정리
            if plugin_info.instance:
                await plugin_info.instance.cleanup()
            
            # 상태 업데이트
            plugin_info.status = PluginStatus.UNLOADED
            plugin_info.instance = None
            
            # 타입별 목록에서 제거
            if plugin_name in self.plugins_by_type[plugin_info.metadata.plugin_type]:
                self.plugins_by_type[plugin_info.metadata.plugin_type].remove(plugin_name)
            
            self.logger.info(f"Plugin {plugin_name} unloaded successfully")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                error=e,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM_ERROR,
                context={"plugin_name": plugin_name, "operation": "plugin_unloading"}
            )
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """플러그인 인스턴스 조회"""
        plugin_info = self.plugins.get(plugin_name)
        if plugin_info and plugin_info.status == PluginStatus.ACTIVE:
            return plugin_info.instance
        return None
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """타입별 플러그인 목록 조회"""
        return self.plugins_by_type.get(plugin_type, [])
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """플러그인 정보 조회"""
        return self.plugins.get(plugin_name)
    
    def get_all_plugins(self) -> List[PluginInfo]:
        """모든 플러그인 정보 조회"""
        return list(self.plugins.values())
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """플러그인 재로드"""
        # 언로드
        await self.unload_plugin(plugin_name)
        
        # 다시 로드
        if await self.load_plugin(plugin_name):
            return await self.initialize_plugin(plugin_name)
        
        return False
    
    async def auto_discover_and_load(self):
        """자동 검색 및 로드"""
        if not self.auto_discover:
            return
        
        discovered = await self.discover_plugins()
        
        if self.auto_load:
            for plugin_name in discovered:
                await self.load_plugin(plugin_name)
    
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
            "type_distribution": type_counts,
            "auto_discover": self.auto_discover,
            "auto_load": self.auto_load
        }


# 전역 플러그인 매니저 인스턴스
_plugin_manager = None

def get_plugin_manager() -> PluginManager:
    """전역 플러그인 매니저 인스턴스 반환"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


# 플러그인 등록 데코레이터
def register_plugin(plugin_type: PluginType):
    """플러그인 등록 데코레이터"""
    def decorator(cls):
        # 메타데이터 설정
        if not hasattr(cls, 'metadata'):
            cls.metadata = PluginMetadata(
                name=cls.__name__,
                version="1.0.0",
                description=cls.__doc__ or f"Plugin {cls.__name__}",
                author="Unknown",
                plugin_type=plugin_type
            )
        
        # 플러그인 타입 설정
        cls.PLUGIN_TYPE = plugin_type.value
        
        return cls
    
    return decorator


# 플러그인 기능 등록 데코레이터
def register_component(name: str):
    """UI 컴포넌트 등록 데코레이터"""
    def decorator(cls):
        # 플러그인 매니저에 컴포넌트 등록
        plugin_manager = get_plugin_manager()
        # TODO: 현재 활성 플러그인에 컴포넌트 등록
        return cls
    
    return decorator


def register_strategy(name: str):
    """AI 전략 등록 데코레이터"""
    def decorator(func):
        # 플러그인 매니저에 전략 등록
        plugin_manager = get_plugin_manager()
        # TODO: 현재 활성 플러그인에 전략 등록
        return func
    
    return decorator
