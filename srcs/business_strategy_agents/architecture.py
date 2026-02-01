"""
Most Hooking Business Strategy Agent - Core Architecture

This module defines the core architecture for a comprehensive business intelligence agent
that monitors global digital trends and generates actionable business insights.

Author: AI Assistant
Date: 2024
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import logging
from datetime import datetime
import subprocess
import os

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """지역 분류"""
    EAST_ASIA = "east_asia"
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    GLOBAL = "global"


class ContentType(Enum):
    """컨텐츠 타입"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    COMMUNITY = "community"
    TREND = "trend"
    BUSINESS_DATA = "business_data"


class BusinessOpportunityLevel(Enum):
    """비즈니스 기회 레벨"""
    CRITICAL = "critical"  # 즉시 실행 필요
    HIGH = "high"         # 높은 잠재력
    MEDIUM = "medium"     # 모니터링 필요
    LOW = "low"          # 참고용


@dataclass
class DataSource:
    """데이터 소스 정의"""
    name: str
    type: ContentType
    region: RegionType
    api_endpoint: str
    rate_limit: int  # requests per minute
    mcp_server_config: Dict[str, Any]
    is_active: bool = True


@dataclass
class RawContent:
    """원시 컨텐츠 데이터"""
    source: str
    content_type: ContentType
    region: RegionType
    title: str
    content: str
    url: str
    timestamp: datetime
    author: Optional[str] = None
    engagement_metrics: Dict[str, int] = None
    metadata: Dict[str, Any] = None


@dataclass
class ProcessedInsight:
    """처리된 인사이트"""
    content_id: str
    hooking_score: float  # 0.0 - 1.0
    business_opportunity: BusinessOpportunityLevel
    region: RegionType
    category: str  # tech, marketing, finance, etc.
    key_topics: List[str]
    sentiment_score: float  # -1.0 to 1.0
    trend_direction: str  # rising, falling, stable
    market_size_estimate: Optional[str]
    competitive_landscape: List[str]
    actionable_insights: List[str]
    timestamp: datetime


@dataclass
class BusinessStrategy:
    """생성된 비즈니스 전략"""
    strategy_id: str
    title: str
    opportunity_level: BusinessOpportunityLevel
    region: RegionType
    category: str
    description: str
    key_insights: List[str]
    action_items: List[Dict[str, Any]]
    timeline: str
    resource_requirements: Dict[str, Any]
    roi_prediction: Dict[str, float]
    risk_factors: List[str]
    success_metrics: List[str]
    related_trends: List[str]
    created_at: datetime


class DataCollectorInterface(ABC):
    """데이터 수집기 인터페이스"""

    @abstractmethod
    async def collect(self, keywords: List[str], region: RegionType) -> List[RawContent]:
        """데이터 수집"""

    @abstractmethod
    def get_rate_limit(self) -> int:
        """Rate limit 반환"""


class ContentProcessorInterface(ABC):
    """컨텐츠 처리기 인터페이스"""

    @abstractmethod
    async def process(self, raw_content: RawContent) -> ProcessedInsight:
        """컨텐츠 처리 및 인사이트 생성"""


class StrategyGeneratorInterface(ABC):
    """전략 생성기 인터페이스"""

    @abstractmethod
    async def generate_strategy(self, insights: List[ProcessedInsight]) -> BusinessStrategy:
        """인사이트를 기반으로 비즈니스 전략 생성"""


class OutputHandlerInterface(ABC):
    """출력 처리기 인터페이스"""

    @abstractmethod
    async def deliver(self, strategy: BusinessStrategy) -> bool:
        """전략을 최종 목적지로 전달"""


class MCPServerManager:
    """MCP 서버 연결 관리자"""

    def __init__(self):
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, Any] = {}
        self.health_status: Dict[str, bool] = {}

    async def register_server(self, name: str, config: Dict[str, Any]) -> bool:
        """MCP 서버 등록"""
        try:
            self.servers[name] = config
            # TODO: 실제 MCP 서버 연결 로직
            self.health_status[name] = True
            logger.info(f"MCP server '{name}' registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register MCP server '{name}': {e}")
            return False

    async def health_check(self) -> Dict[str, bool]:
        """모든 서버 헬스체크"""
        for server_name in self.servers:
            try:
                # TODO: 실제 헬스체크 로직
                self.health_status[server_name] = True
            except Exception as e:
                logger.warning(f"Health check failed for '{server_name}': {e}")
                self.health_status[server_name] = False

        return self.health_status.copy()

    def get_active_servers(self) -> List[str]:
        """활성 서버 목록 반환"""
        return [name for name, status in self.health_status.items() if status]


class DataSourceRegistry:
    """데이터 소스 레지스트리"""

    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self._initialize_default_sources()

    def _initialize_default_sources(self):
        """기본 데이터 소스들 초기화"""
        default_sources = [
            # News Sources
            DataSource(
                name="reuters_api",
                type=ContentType.NEWS,
                region=RegionType.GLOBAL,
                api_endpoint="https://api.reuters.com/v1/",
                rate_limit=100,
                mcp_server_config={"type": "rest_api", "auth": "api_key"}
            ),
            DataSource(
                name="bloomberg_api",
                type=ContentType.NEWS,
                region=RegionType.GLOBAL,
                api_endpoint="https://api.bloomberg.com/v1/",
                rate_limit=50,
                mcp_server_config={"type": "rest_api", "auth": "oauth"}
            ),

            # Social Media Sources
            DataSource(
                name="twitter_api",
                type=ContentType.SOCIAL_MEDIA,
                region=RegionType.GLOBAL,
                api_endpoint="https://api.twitter.com/2/",
                rate_limit=300,
                mcp_server_config={"type": "rest_api", "auth": "bearer_token"}
            ),
            DataSource(
                name="linkedin_api",
                type=ContentType.SOCIAL_MEDIA,
                region=RegionType.GLOBAL,
                api_endpoint="https://api.linkedin.com/v2/",
                rate_limit=100,
                mcp_server_config={"type": "rest_api", "auth": "oauth"}
            ),

            # Community Sources
            DataSource(
                name="reddit_api",
                type=ContentType.COMMUNITY,
                region=RegionType.GLOBAL,
                api_endpoint="https://www.reddit.com/api/v1/",
                rate_limit=60,
                mcp_server_config={"type": "rest_api", "auth": "oauth"}
            ),
            DataSource(
                name="hackernews_api",
                type=ContentType.COMMUNITY,
                region=RegionType.GLOBAL,
                api_endpoint="https://hacker-news.firebaseio.com/v0/",
                rate_limit=1000,
                mcp_server_config={"type": "rest_api", "auth": "none"}
            ),

            # Trend Sources
            DataSource(
                name="google_trends",
                type=ContentType.TREND,
                region=RegionType.GLOBAL,
                api_endpoint="https://trends.googleapis.com/trends/api/",
                rate_limit=100,
                mcp_server_config={"type": "rest_api", "auth": "api_key"}
            ),

            # Business Data Sources
            DataSource(
                name="crunchbase_api",
                type=ContentType.BUSINESS_DATA,
                region=RegionType.GLOBAL,
                api_endpoint="https://api.crunchbase.com/api/v4/",
                rate_limit=200,
                mcp_server_config={"type": "rest_api", "auth": "api_key"}
            ),

            # Regional Sources - East Asia
            DataSource(
                name="naver_news",
                type=ContentType.NEWS,
                region=RegionType.EAST_ASIA,
                api_endpoint="https://openapi.naver.com/v1/search/",
                rate_limit=25000,
                mcp_server_config={"type": "rest_api", "auth": "client_id"}
            ),
            DataSource(
                name="weibo_api",
                type=ContentType.SOCIAL_MEDIA,
                region=RegionType.EAST_ASIA,
                api_endpoint="https://api.weibo.com/2/",
                rate_limit=150,
                mcp_server_config={"type": "rest_api", "auth": "oauth"}
            )
        ]

        for source in default_sources:
            self.sources[source.name] = source

    def get_sources_by_region(self, region: RegionType) -> List[DataSource]:
        """지역별 데이터 소스 반환"""
        return [source for source in self.sources.values()
                if source.region == region or source.region == RegionType.GLOBAL]

    def get_sources_by_type(self, content_type: ContentType) -> List[DataSource]:
        """타입별 데이터 소스 반환"""
        return [source for source in self.sources.values()
                if source.type == content_type and source.is_active]


class CoreArchitecture:
    """핵심 아키텍처 클래스"""

    def __init__(self):
        self.mcp_manager = MCPServerManager()
        self.data_registry = DataSourceRegistry()
        self.collectors: Dict[str, DataCollectorInterface] = {}
        self.processors: List[ContentProcessorInterface] = []
        self.strategy_generators: List[StrategyGeneratorInterface] = []
        self.output_handlers: List[OutputHandlerInterface] = []

        # Configuration
        self.config = {
            'collection_interval': 300,  # 5 minutes
            'max_concurrent_collectors': 10,
            'hooking_score_threshold': 0.7,
            'supported_regions': [RegionType.EAST_ASIA, RegionType.NORTH_AMERICA],
            'monitoring_keywords': [
                'AI', 'machine learning', 'startup', 'investment', 'technology',
                'market trend', 'consumer behavior', 'digital transformation',
                'e-commerce', 'fintech', 'healthcare tech', 'sustainability'
            ]
        }

    async def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            logger.info("Initializing Most Hooking Business Strategy Agent Architecture...")

            # MCP 서버들 등록
            await self._register_mcp_servers()

            # 컴포넌트 초기화
            await self._initialize_components()

            # 헬스체크
            health_status = await self.mcp_manager.health_check()
            active_servers = len([s for s in health_status.values() if s])

            logger.info(f"Architecture initialized successfully. Active MCP servers: {active_servers}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize architecture: {e}")
            return False

    async def _register_mcp_servers(self):
        """MCP 서버 등록"""
        for source in self.data_registry.sources.values():
            await self.mcp_manager.register_server(
                source.name,
                source.mcp_server_config
            )

    async def _initialize_components(self):
        """컴포넌트 초기화"""
        # TODO: 실제 컴포넌트 구현체들로 교체
        logger.info("Components initialized (placeholder)")

    def register_collector(self, name: str, collector: DataCollectorInterface):
        """데이터 수집기 등록"""
        self.collectors[name] = collector
        logger.info(f"Data collector '{name}' registered")

    def register_processor(self, processor: ContentProcessorInterface):
        """컨텐츠 처리기 등록"""
        self.processors.append(processor)
        logger.info("Content processor registered")

    def register_strategy_generator(self, generator: StrategyGeneratorInterface):
        """전략 생성기 등록"""
        self.strategy_generators.append(generator)
        logger.info("Strategy generator registered")

    def register_output_handler(self, handler: OutputHandlerInterface):
        """출력 처리기 등록"""
        self.output_handlers.append(handler)
        logger.info("Output handler registered")

    async def run_collection_cycle(self) -> List[BusinessStrategy]:
        """데이터 수집 및 전략 생성 사이클 실행"""
        try:
            logger.info("Starting collection cycle...")

            # 1. 데이터 수집
            raw_contents = await self._collect_data()
            logger.info(f"Collected {len(raw_contents)} raw contents")

            # 2. 컨텐츠 처리
            insights = await self._process_contents(raw_contents)
            logger.info(f"Generated {len(insights)} insights")

            # 3. 후킹 포인트 필터링
            hooking_insights = self._filter_hooking_insights(insights)
            logger.info(f"Found {len(hooking_insights)} hooking insights")

            # 4. 비즈니스 전략 생성
            strategies = await self._generate_strategies(hooking_insights)
            logger.info(f"Generated {len(strategies)} business strategies")

            # 5. 출력 전달
            await self._deliver_strategies(strategies)

            return strategies

        except Exception as e:
            logger.error(f"Collection cycle failed: {e}")
            return []

    async def _collect_data(self) -> List[RawContent]:
        """데이터 수집"""
        raw_contents = []

        # 지역별로 수집
        for region in self.config['supported_regions']:
            region_sources = self.data_registry.get_sources_by_region(region)

            for source in region_sources:
                if source.name in self.collectors:
                    try:
                        collector = self.collectors[source.name]
                        contents = await collector.collect(
                            self.config['monitoring_keywords'],
                            region
                        )
                        raw_contents.extend(contents)
                    except Exception as e:
                        logger.warning(f"Failed to collect from {source.name}: {e}")

        return raw_contents

    async def _process_contents(self, raw_contents: List[RawContent]) -> List[ProcessedInsight]:
        """컨텐츠 처리"""
        insights = []

        for content in raw_contents:
            for processor in self.processors:
                try:
                    insight = await processor.process(content)
                    insights.append(insight)
                except Exception as e:
                    logger.warning(f"Failed to process content {content.title}: {e}")

        return insights

    def _filter_hooking_insights(self, insights: List[ProcessedInsight]) -> List[ProcessedInsight]:
        """후킹 인사이트 필터링"""
        threshold = self.config['hooking_score_threshold']
        return [insight for insight in insights if insight.hooking_score >= threshold]

    async def _generate_strategies(self, insights: List[ProcessedInsight]) -> List[BusinessStrategy]:
        """전략 생성"""
        strategies = []

        for generator in self.strategy_generators:
            try:
                strategy = await generator.generate_strategy(insights)
                strategies.append(strategy)
            except Exception as e:
                logger.warning(f"Failed to generate strategy: {e}")

        return strategies

    async def _deliver_strategies(self, strategies: List[BusinessStrategy]):
        """전략 전달"""
        for strategy in strategies:
            for handler in self.output_handlers:
                try:
                    await handler.deliver(strategy)
                except Exception as e:
                    logger.warning(f"Failed to deliver strategy {strategy.strategy_id}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            'mcp_servers': len(self.mcp_manager.servers),
            'active_servers': len(self.mcp_manager.get_active_servers()),
            'data_sources': len(self.data_registry.sources),
            'collectors': len(self.collectors),
            'processors': len(self.processors),
            'strategy_generators': len(self.strategy_generators),
            'output_handlers': len(self.output_handlers),
            'supported_regions': [r.value for r in self.config['supported_regions']],
            'monitoring_keywords': len(self.config['monitoring_keywords'])
        }


# 싱글톤 인스턴스
architecture = CoreArchitecture()


def get_architecture() -> CoreArchitecture:
    """아키텍처 인스턴스 반환"""
    return architecture


async def connect_to_mcp_server(server_name: str, config: Dict[str, Any]) -> bool:
    """실제 MCP 서버 연결 로직"""

    try:
        pass

        # MCP 서버 설정 검증
        if not config.get('command'):
            raise ValueError(f"MCP 서버 '{server_name}' 설정에 command가 없습니다.")

        command = config['command']
        args = config.get('args', [])
        port = config.get('port')

        # 서버별 연결 로직
        if server_name == "urban-hive":
            # Urban Hive MCP 서버 연결
            return await connect_urban_hive_server(command, args, port)

        elif server_name in ["g-search", "fetch", "filesystem", "puppeteer", "brave"]:
            # 표준 MCP 서버들 연결
            return await connect_standard_mcp_server(server_name, command, args)

        elif server_name == "interpreter":
            # Docker 기반 인터프리터 서버 연결
            return await connect_docker_mcp_server(command, args)

        else:
            # 일반적인 MCP 서버 연결
            return await connect_generic_mcp_server(server_name, command, args)

    except Exception as e:
        print(f"MCP 서버 '{server_name}' 연결 실패: {e}")
        return False


async def connect_urban_hive_server(command: str, args: List[str], port: int = 8002) -> bool:
    """Urban Hive MCP 서버 연결"""

    try:
        # 포트 사용 중인지 확인
        if is_port_in_use(port):
            print(f"Urban Hive 서버가 이미 포트 {port}에서 실행 중입니다.")
            return True

        # 서버 시작
        full_command = [command] + args
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # 프로젝트 루트
        )

        # 서버 시작 대기 (최대 10초)
        for _ in range(20):  # 0.5초씩 20번 = 10초
            if is_port_in_use(port):
                print(f"Urban Hive MCP 서버가 포트 {port}에서 성공적으로 시작되었습니다.")
                return True
            await asyncio.sleep(0.5)

        # 시작 실패 시 프로세스 종료
        process.terminate()
        return False

    except Exception as e:
        print(f"Urban Hive 서버 시작 실패: {e}")
        return False


async def connect_standard_mcp_server(server_name: str, command: str, args: List[str]) -> bool:
    """표준 MCP 서버 연결 (npx, uvx 기반)"""

    try:
        # 명령어 존재 확인
        if command == "npx":
            # Node.js 패키지 매니저 확인
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"npm이 설치되지 않았습니다. {server_name} 서버를 시작할 수 없습니다.")
                return False

        elif command == "uvx":
            # uv 패키지 매니저 확인
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"uv가 설치되지 않았습니다. {server_name} 서버를 시작할 수 없습니다.")
                return False

        # 서버 연결 테스트 (실제로는 패키지 설치 확인)
        full_command = [command] + args + ["--help"]
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(f"{server_name} MCP 서버 패키지가 사용 가능합니다.")
            return True
        else:
            print(f"{server_name} MCP 서버 패키지를 찾을 수 없습니다.")
            return False

    except subprocess.TimeoutExpired:
        print(f"{server_name} 서버 연결 테스트 시간 초과")
        return False
    except Exception as e:
        print(f"{server_name} 서버 연결 실패: {e}")
        return False


async def connect_docker_mcp_server(command: str, args: List[str]) -> bool:
    """Docker 기반 MCP 서버 연결"""

    try:
        # Docker 설치 확인
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Docker가 설치되지 않았습니다.")
            return False

        # Docker 이미지 존재 확인
        if len(args) >= 4 and args[0] == "run":
            image_name = args[3]  # docker run -i --rm <image_name>

            result = subprocess.run(
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                print(f"Docker 이미지 '{image_name}'가 사용 가능합니다.")
                return True
            else:
                print(f"Docker 이미지 '{image_name}'를 찾을 수 없습니다.")
                # 이미지 풀 시도
                pull_result = subprocess.run(
                    ["docker", "pull", image_name],
                    capture_output=True,
                    text=True
                )

                if pull_result.returncode == 0:
                    print(f"Docker 이미지 '{image_name}'를 성공적으로 다운로드했습니다.")
                    return True
                else:
                    print(f"Docker 이미지 '{image_name}' 다운로드 실패")
                    return False

        return False

    except Exception as e:
        print(f"Docker MCP 서버 연결 실패: {e}")
        return False


async def connect_generic_mcp_server(server_name: str, command: str, args: List[str]) -> bool:
    """일반적인 MCP 서버 연결"""

    try:
        # 명령어 존재 확인
        result = subprocess.run(
            ["which", command] if command != "python" else ["python", "--version"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"{server_name} MCP 서버 명령어 '{command}'가 사용 가능합니다.")
            return True
        else:
            print(f"{server_name} MCP 서버 명령어 '{command}'를 찾을 수 없습니다.")
            return False

    except Exception as e:
        print(f"{server_name} 서버 연결 실패: {e}")
        return False


def is_port_in_use(port: int) -> bool:
    """포트 사용 중인지 확인"""

    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except Exception:
        return False


async def validate_mcp_configuration(config_path: str = None) -> Dict[str, bool]:
    """MCP 설정 검증 및 연결 테스트"""

    try:
        import yaml
        from pathlib import Path

        # 설정 파일 경로
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / "configs" / "mcp_agent.config.yaml"

        # 설정 파일 로드
        if not Path(config_path).exists():
            print(f"MCP 설정 파일을 찾을 수 없습니다: {config_path}")
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        mcp_servers = config.get('mcp', {}).get('servers', {})

        # 각 서버 연결 테스트
        connection_results = {}

        for server_name, server_config in mcp_servers.items():
            print(f"MCP 서버 '{server_name}' 연결 테스트 중...")
            connection_results[server_name] = await connect_to_mcp_server(server_name, server_config)

        # 결과 요약
        successful_connections = sum(connection_results.values())
        total_servers = len(connection_results)

        print(f"\nMCP 서버 연결 결과: {successful_connections}/{total_servers} 성공")

        for server_name, success in connection_results.items():
            status = "✅ 성공" if success else "❌ 실패"
            print(f"  {server_name}: {status}")

        return connection_results

    except Exception as e:
        print(f"MCP 설정 검증 실패: {e}")
        return {}
