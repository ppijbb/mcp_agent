"""
향상된 그래프 워크플로우
MCP와 A2A 프로토콜을 지원하는 향상된 에이전트 협업 시스템
"""

import asyncio
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolExecutor
from langchain_core.runnables import Runnable

from .enhanced_agents import (
    enhanced_supervisor, enhanced_search, enhanced_analyst,
    create_enhanced_agent
)
from .mcp_integration import mcp_registry, mcp_executor
from .a2a_protocol import a2a_message_broker, MessageType, MessagePriority
from .security import security_manager, privacy_manager, audit_logger

# 향상된 상태 정의
class EnhancedAgentState(TypedDict):
    query: str
    supervisor_feedback: Optional[str] = None
    search_queries: Optional[List[str]] = None
    search_results: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    collaboration_data: Optional[Dict[str, Any]] = None
    final_report: Optional[str] = None
    security_context: Optional[str] = None
    audit_trail: Optional[List[Dict[str, Any]]] = None

class EnhancedWorkflowGraph:
    """향상된 워크플로우 그래프"""
    
    def __init__(self):
        self.state_graph = StateGraph(EnhancedAgentState)
        self.memory = MemorySaver()
        self.tool_executor = ToolExecutor([])
        self.setup_graph()
    
    def setup_graph(self):
        """그래프 설정"""
        # 노드 추가
        self.state_graph.add_node("enhanced_supervisor", self.enhanced_supervisor_node)
        self.state_graph.add_node("enhanced_search", self.enhanced_search_node)
        self.state_graph.add_node("enhanced_analyst", self.enhanced_analyst_node)
        self.state_graph.add_node("collaboration_orchestrator", self.collaboration_orchestrator_node)
        self.state_graph.add_node("report_generator", self.report_generator_node)
        
        # 엣지 설정
        self.state_graph.set_entry_point("enhanced_supervisor")
        self.state_graph.add_edge("enhanced_supervisor", "enhanced_search")
        self.state_graph.add_edge("enhanced_search", "enhanced_analyst")
        self.state_graph.add_conditional_edge(
            "enhanced_analyst", 
            self.route_after_analysis,
            {
                "collaboration": "collaboration_orchestrator",
                "generate_report": "report_generator"
            }
        )
        self.state_graph.add_edge("collaboration_orchestrator", "enhanced_supervisor")
        self.state_graph.add_edge("report_generator", END)
        
        # 그래프 컴파일
        self.app = self.state_graph.compile(
            checkpointer=self.memory,
            interrupt_before=["enhanced_analyst", "collaboration_orchestrator"]
        )
    
    async def enhanced_supervisor_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """향상된 감독관 노드"""
        try:
            # 보안 컨텍스트 생성
            security_context = security_manager.create_session(
                agent_id="workflow_supervisor",
                permissions=["query_generation", "project_management"]
            )
            state["security_context"] = security_context.session_id
            
            # 감사 로그 기록
            audit_logger.log_access(
                agent_id="workflow_supervisor",
                resource="enhanced_supervisor",
                action="start",
                success=True
            )
            
            # 향상된 감독관 에이전트와 협업
            task_result = await enhanced_supervisor.execute_task(
                state["query"],
                context="workflow_initiation"
            )
            
            state["search_queries"] = task_result["queries"]
            state["collaboration_data"] = {
                "strategy": task_result["strategy"],
                "partners": task_result["collaboration_partners"]
            }
            
            # A2A 메시지로 워크플로우 시작 알림
            await self._notify_workflow_start(state)
            
            return state
            
        except Exception as e:
            audit_logger.log_security_event(
                event_type="supervisor_node_failed",
                severity="high",
                agent_id="workflow_supervisor",
                details={"error": str(e)}
            )
            raise
    
    async def enhanced_search_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """향상된 검색 노드"""
        try:
            # 감사 로그 기록
            audit_logger.log_access(
                agent_id="workflow_search",
                resource="enhanced_search",
                action="execute",
                success=True
            )
            
            # 향상된 검색 에이전트와 협업
            search_result = await enhanced_search.execute_task(
                "Execute search queries",
                queries=state["search_queries"]
            )
            
            state["search_results"] = search_result
            
            # 프라이버시 검사 및 데이터 정리
            if privacy_manager.encryption_enabled:
                state["search_results"] = self._sanitize_search_results(
                    state["search_results"]
                )
            
            return state
            
        except Exception as e:
            audit_logger.log_security_event(
                event_type="search_node_failed",
                severity="medium",
                agent_id="workflow_search",
                details={"error": str(e)}
            )
            raise
    
    async def enhanced_analyst_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """향상된 분석 노드"""
        try:
            # 감사 로그 기록
            audit_logger.log_access(
                agent_id="workflow_analyst",
                resource="enhanced_analyst",
                action="execute",
                success=True
            )
            
            # 향상된 분석 에이전트와 협업
            analysis_result = await enhanced_analyst.execute_task(
                "Analyze search results",
                search_results=state["search_results"]
            )
            
            state["analysis"] = analysis_result
            
            # 품질 점수에 따른 라우팅 결정
            if analysis_result["quality_score"] < 0.7:
                state["supervisor_feedback"] = analysis_result["recommendations"]
            
            return state
            
        except Exception as e:
            audit_logger.log_security_event(
                event_type="analyst_node_failed",
                severity="medium",
                agent_id="workflow_analyst",
                details={"error": str(e)}
            )
            raise
    
    async def collaboration_orchestrator_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """협업 오케스트레이터 노드"""
        try:
            # 감사 로그 기록
            audit_logger.log_access(
                agent_id="workflow_orchestrator",
                resource="collaboration_orchestrator",
                action="execute",
                success=True
            )
            
            # A2A 프로토콜을 통한 에이전트 간 협업
            collaboration_result = await self._orchestrate_collaboration(state)
            
            state["collaboration_data"].update(collaboration_result)
            
            # 협업 결과를 바탕으로 검색 쿼리 개선
            if state["supervisor_feedback"]:
                improved_queries = await self._improve_search_queries(
                    state["search_queries"],
                    state["supervisor_feedback"]
                )
                state["search_queries"] = improved_queries
            
            return state
            
        except Exception as e:
            audit_logger.log_sestrity_event(
                event_type="collaboration_orchestrator_failed",
                severity="medium",
                agent_id="workflow_orchestrator",
                details={"error": str(e)}
            )
            raise
    
    async def report_generator_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """보고서 생성 노드"""
        try:
            # 감사 로그 기록
            audit_logger.log_access(
                agent_id="workflow_report",
                resource="report_generator",
                action="execute",
                success=True
            )
            
            # 최종 보고서 생성
            final_report = await self._generate_final_report(state)
            state["final_report"] = final_report
            
            # 워크플로우 완료 알림
            await self._notify_workflow_completion(state)
            
            return state
            
        except Exception as e:
            audit_logger.log_security_event(
                event_type="report_generator_failed",
                severity="medium",
                agent_id="workflow_report",
                details={"error": str(e)}
            )
            raise
    
    def route_after_analysis(self, state: EnhancedAgentState) -> str:
        """분석 후 라우팅 결정"""
        if state.get("supervisor_feedback"):
            return "collaboration"
        else:
            return "generate_report"
    
    async def _notify_workflow_start(self, state: EnhancedAgentState):
        """워크플로우 시작 알림"""
        from .a2a_protocol import A2AMessage
        
        notification = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id="workflow_engine",
            receiver_id="*",
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.NORMAL,
            content={
                "event_type": "workflow_started",
                "query": state["query"],
                "search_queries": state["search_queries"]
            },
            timestamp=datetime.now()
        )
        
        await a2a_message_broker.publish_message(notification)
    
    async def _notify_workflow_completion(self, state: EnhancedAgentState):
        """워크플로우 완료 알림"""
        from .a2a_protocol import A2AMessage
        
        notification = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id="workflow_engine",
            receiver_id="*",
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.NORMAL,
            content={
                "event_type": "workflow_completed",
                "query": state["query"],
                "final_report_length": len(state["final_report"]) if state["final_report"] else 0
            },
            timestamp=datetime.now()
        )
        
        await a2a_message_broker.publish_message(notification)
    
    async def _orchestrate_collaboration(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """에이전트 간 협업 오케스트레이션"""
        collaboration_data = {
            "participants": [],
            "decisions": [],
            "improvements": []
        }
        
        # 사용 가능한 에이전트들과 협업
        available_agents = a2a_message_broker.list_agents()
        
        for agent in available_agents:
            if agent.agent_id != "workflow_engine":
                # 각 에이전트에게 개선 제안 요청
                try:
                    response = await enhanced_supervisor.send_request(
                        receiver_id=agent.agent_id,
                        content={
                            "request_type": "improvement_suggestions",
                            "current_state": state,
                            "analysis": state["analysis"]
                        },
                        priority=MessagePriority.HIGH
                    )
                    
                    if response:
                        collaboration_data["participants"].append(agent.name)
                        collaboration_data["decisions"].append({
                            "agent": agent.name,
                            "suggestion": "Improvement suggestion received"
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to collaborate with agent {agent.name}: {str(e)}")
        
        return collaboration_data
    
    async def _improve_search_queries(self, current_queries: List[str], 
                                    feedback: List[str]) -> List[str]:
        """검색 쿼리 개선"""
        # LLM을 사용한 쿼리 개선
        from .utils import model
        
        prompt = f"""
        Current search queries: {current_queries}
        Feedback for improvement: {feedback}
        
        Please improve these search queries based on the feedback.
        Return only the improved queries, one per line.
        """
        
        response = model.invoke(prompt)
        improved_queries = [line.strip() for line in response.content.split('\n') if line.strip()]
        
        return improved_queries[:5]  # 최대 5개
    
    async def _generate_final_report(self, state: EnhancedAgentState) -> str:
        """최종 보고서 생성"""
        from .utils import model
        
        prompt = f"""
        Based on the following information, generate a comprehensive final report:
        
        Original Query: {state['query']}
        Search Results: {state['search_results']}
        Analysis: {state['analysis']}
        Collaboration Data: {state['collaboration_data']}
        
        The report should be well-structured, comprehensive, and address the original query effectively.
        """
        
        response = model.invoke(prompt)
        return response.content
    
    def _sanitize_search_results(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """검색 결과 정리 (프라이버시 보호)"""
        if not search_results:
            return search_results
        
        sanitized = {}
        for key, value in search_results.items():
            if isinstance(value, str):
                sanitized[key] = privacy_manager.sanitize_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    privacy_manager.sanitize_data(str(item)) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized

# 향상된 워크플로우 인스턴스 생성
enhanced_workflow = EnhancedWorkflowGraph()

# 워크플로우 시작 함수
async def start_enhanced_workflow():
    """향상된 워크플로우 시작"""
    try:
        # A2A 메시지 브로커 시작
        await a2a_message_broker.start()
        
        # MCP 도구들 등록
        from .mcp_integration import register_default_tools
        register_default_tools()
        
        # 보안 컨텍스트 초기화
        security_manager.create_session(
            agent_id="workflow_engine",
            permissions=["workflow_management", "agent_collaboration"]
        )
        
        logger.info("Enhanced workflow started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start enhanced workflow: {str(e)}")
        raise

# 워크플로우 중지 함수
async def stop_enhanced_workflow():
    """향상된 워크플로우 중지"""
    try:
        # A2A 메시지 브로커 중지
        await a2a_message_broker.stop()
        
        # 보안 세션 정리
        for session_id in list(security_manager.active_sessions.keys()):
            security_manager.revoke_session(session_id)
        
        logger.info("Enhanced workflow stopped successfully")
        
    except Exception as e:
        logger.error(f"Failed to stop enhanced workflow: {str(e)}")
        raise
