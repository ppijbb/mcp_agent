"""
향상된 메인 실행 파일
MCP와 A2A 프로토콜을 지원하는 향상된 에이전트 협업 시스템을 실행합니다.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List
from datetime import datetime
import uuid

from .enhanced_graph import enhanced_workflow, start_enhanced_workflow, stop_enhanced_workflow
from .mcp_integration import mcp_registry, mcp_executor
from .a2a_protocol import a2a_message_broker
from .security import security_manager, privacy_manager, audit_logger

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMultiAgentSystem:
    """향상된 멀티 에이전트 시스템"""
    
    def __init__(self):
        self.workflow = enhanced_workflow
        self.is_running = False
        self.execution_history = []
    
    async def start_system(self):
        """시스템 시작"""
        try:
            logger.info("Starting Enhanced Multi-Agent System...")
            
            # 환경 변수 로드
            load_dotenv()
            
            # API 키 확인
            self._validate_environment()
            
            # 향상된 워크플로우 시작
            await start_enhanced_workflow()
            
            # 시스템 상태 설정
            self.is_running = True
            
            logger.info("Enhanced Multi-Agent System started successfully!")
            
        except Exception as e:
            logger.error(f"Failed to start system: {str(e)}")
            raise
    
    async def stop_system(self):
        """시스템 중지"""
        try:
            logger.info("Stopping Enhanced Multi-Agent System...")
            
            # 향상된 워크플로우 중지
            await stop_enhanced_workflow()
            
            # 시스템 상태 설정
            self.is_running = False
            
            logger.info("Enhanced Multi-Agent System stopped successfully!")
            
        except Exception as e:
            logger.error(f"Failed to stop system: {str(e)}")
            raise
    
    def _validate_environment(self):
        """환경 변수 검증"""
        required_keys = ['OPENAI_API_KEY', 'TAVILY_API_KEY']
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_keys)}\n"
                "Please set them in your .env file."
            )
    
    async def execute_workflow(self, query: str) -> Dict[str, Any]:
        """워크플로우 실행"""
        if not self.is_running:
            raise RuntimeError("System is not running. Please start the system first.")
        
        try:
            logger.info(f"Executing workflow for query: {query}")
            
            # 감사 로그 기록
            audit_logger.log_access(
                agent_id="user",
                resource="enhanced_workflow",
                action="execute",
                success=True,
                details={"query": query}
            )
            
            # 워크플로우 실행
            inputs = {"query": query}
            
            # 스트리밍 실행으로 진행 상황 모니터링
            execution_events = []
            async for event in self.workflow.app.astream(inputs):
                for key, value in event.items():
                    execution_events.append({
                        "step": key,
                        "value": value,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                    
                    logger.info(f"Workflow step completed: {key}")
                    print(f"--- 워크플로우 단계: {key} ---")
                    print(f"결과: {value}")
                    print("=" * 50)
            
            # 최종 결과 조회
            final_state = await self.workflow.app.ainvoke(inputs)
            
            # 실행 히스토리 저장
            execution_record = {
                "query": query,
                "events": execution_events,
                "final_state": final_state,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.execution_history.append(execution_record)
            
            # 워크플로우 완료 알림
            await self._notify_workflow_completion(query, final_state)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            
            # 보안 이벤트 로그 기록
            audit_logger.log_security_event(
                event_type="workflow_execution_failed",
                severity="high",
                details={"error": str(e), "query": query}
            )
            
            raise
    
    async def _notify_workflow_completion(self, query: str, final_state: Dict[str, Any]):
        """워크플로우 완료 알림"""
        try:
            # A2A 메시지로 완료 알림
            from .a2a_protocol import A2AMessage, MessageType, MessagePriority
            
            notification = A2AMessage(
                message_id=str(uuid.uuid4()),
                sender_id="enhanced_main",
                receiver_id="*",
                message_type=MessageType.NOTIFICATION,
                priority=MessagePriority.NORMAL,
                content={
                    "event_type": "workflow_execution_completed",
                    "query": query,
                    "has_final_report": "final_report" in final_state,
                    "execution_success": True
                },
                timestamp=datetime.now()
            )
            
            await a2a_message_broker.publish_message(notification)
            
        except Exception as e:
            logger.warning(f"Failed to send completion notification: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            "is_running": self.is_running,
            "registered_agents": len(a2a_message_broker.list_agents()),
            "available_mcp_tools": len(mcp_registry.list_tools()),
            "active_security_sessions": len(security_manager.active_sessions),
            "execution_history_count": len(self.execution_history),
            "privacy_encryption_enabled": privacy_manager.encryption_enabled
        }
    
    def get_agent_cards(self) -> List[Dict[str, Any]]:
        """등록된 에이전트 카드 조회"""
        agents = a2a_message_broker.list_agents()
        return [agent.to_dict() for agent in agents]
    
    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 MCP 도구 목록 조회"""
        tools = mcp_registry.list_tools()
        return [tool.to_dict() for tool in tools]
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """실행 히스토리 조회"""
        return self.execution_history[-limit:] if limit else self.execution_history

async def main():
    """메인 실행 함수"""
    system = EnhancedMultiAgentSystem()
    
    try:
        # 시스템 시작
        await system.start_system()
        
        # 시스템 상태 출력
        status = system.get_system_status()
        print("\n=== 향상된 멀티 에이전트 시스템 상태 ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        
        # 등록된 에이전트 정보 출력
        agent_cards = system.get_agent_cards()
        print(f"\n=== 등록된 에이전트 ({len(agent_cards)}개) ===")
        for agent in agent_cards:
            print(f"- {agent['name']} ({agent['agent_id']}): {agent['description']}")
        
        # 사용 가능한 MCP 도구 정보 출력
        mcp_tools = system.get_mcp_tools()
        print(f"\n=== 사용 가능한 MCP 도구 ({len(mcp_tools)}개) ===")
        for tool in mcp_tools:
            print(f"- {tool['name']}: {tool['description']}")
        
        # 사용자 입력 받기
        print("\n" + "="*50)
        query = input("안녕하세요! 어떤 주제에 대한 향상된 보고서를 작성해드릴까요?\n> ")
        
        if query.strip():
            # 워크플로우 실행
            final_state = await system.execute_workflow(query)
            
            # 최종 결과 출력
            print("\n\n=== 최종 향상된 보고서 ===")
            if "final_report" in final_state:
                print(final_state["final_report"])
            else:
                print("보고서 생성에 실패했습니다.")
            
            # 협업 데이터 출력
            if "collaboration_data" in final_state:
                print(f"\n=== 협업 정보 ===")
                collab_data = final_state["collaboration_data"]
                print(f"전략: {collab_data.get('strategy', 'N/A')}")
                print(f"협업 파트너: {', '.join(collab_data.get('partners', []))}")
        
        else:
            print("쿼리가 입력되지 않았습니다.")
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        print(f"\n시스템 오류가 발생했습니다: {str(e)}")
    finally:
        # 시스템 정리
        try:
            await system.stop_system()
        except Exception as e:
            logger.error(f"Failed to stop system: {str(e)}")

if __name__ == "__main__":
    # asyncio 이벤트 루프 실행
    asyncio.run(main())
