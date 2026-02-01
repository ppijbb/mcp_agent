import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseAgentSimple(ABC):
    """간단한 BaseAgent - mcp_agent 의존성 없이 동작"""

    def __init__(self, name: str = "base_agent"):
        self.name = name
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"agent.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    async def run_workflow(self, *args, **kwargs) -> Any:
        """에이전트의 핵심 워크플로우를 실행하는 추상 메서드"""

    async def run(self, *args, **kwargs):
        """에이전트 워크플로우를 실행하고 결과를 반환"""
        self.logger.info(f"'{self.name}' 에이전트 워크플로우를 시작합니다.")
        try:
            result = await self.run_workflow(*args, **kwargs)
            self.logger.info(f"'{self.name}' 에이전트 워크플로우를 성공적으로 완료했습니다.")
            return result
        except Exception as e:
            self.logger.error(f"'{self.name}' 워크플로우 중 오류 발생: {e}", exc_info=True)
            raise
