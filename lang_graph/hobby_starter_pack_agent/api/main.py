from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

from ..autogen.agents import HSPAutoGenAgents
from ..langgraph_workflow.workflow import HSPLangGraphWorkflow
from ..langgraph_workflow.vector_store import HSPVectorStore
from ..bridge.a2a_bridge import A2AProtocolBridge, A2AMessage
from ..mcp.manager import MCPServerManager

# 새로운 API 라우터들
from .auth import router as auth_router
from .users import router as users_router
from .hobbies import router as hobbies_router
from .communities import router as communities_router
from .notifications import router as notifications_router

app = FastAPI(title="Hobby Starter Pack Agent API", version="2.0.0")

# Global instances
autogen_agents = HSPAutoGenAgents()
vector_store = HSPVectorStore()
langgraph_workflow = HSPLangGraphWorkflow(vector_store=vector_store)
a2a_bridge = A2AProtocolBridge()
mcp_manager = MCPServerManager()

# 라우터 등록
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(hobbies_router)
app.include_router(communities_router)
app.include_router(notifications_router)

class AgentConsensusRequest(BaseModel):
    agents: List[str]
    topic: str
    context: Dict[str, Any]
    user_profile: Optional[Dict[str, Any]] = None

class MCPCallRequest(BaseModel):
    server_name: str
    capability: str
    params: Dict[str, Any]

class A2AMessageRequest(BaseModel):
    sender_agent: str
    receiver_agent: str
    message_type: str
    payload: Dict[str, Any]
    session_id: Optional[str] = None

class WorkflowRequest(BaseModel):
    user_input: str
    user_profile: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None

class StartConversationRequest(BaseModel):
    initial_input: str
    user_profile: Optional[Dict[str, Any]] = None

class AnswerQuestionRequest(BaseModel):
    session_id: str
    answer: str

class CompleteConversationRequest(BaseModel):
    session_id: str

@app.post("/api/agents/consensus")
async def create_agent_consensus(request: AgentConsensusRequest):
    """에이전트 간 합의 프로세스 시작"""
    try:
        session_id = await a2a_bridge.create_consensus_session(
            participants=request.agents,
            topic=request.topic
        )
        
        # AutoGen 에이전트들 등록
        for agent_name in request.agents:
            await a2a_bridge.register_agent(
                agent_id=agent_name,
                agent_type=agent_name.lower(),
                framework="autogen"
            )
        
        # 합의 프로세스 시작
        consensus_result = await autogen_agents.run_consensus(
            agents=request.agents,
            topic=request.topic,
            context=request.context,
            user_profile=request.user_profile,
            bridge=a2a_bridge,
            session_id=session_id
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "consensus_result": consensus_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consensus failed: {str(e)}")

@app.post("/api/mcp/call")
async def call_mcp_server(request: MCPCallRequest):
    """MCP 서버 기능 호출"""
    try:
        result = await mcp_manager.call_mcp_server(
            server_name=request.server_name,
            capability=request.capability,
            params=request.params
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return {
            "status": "success",
            "server": request.server_name,
            "capability": request.capability,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP call failed: {str(e)}")

@app.post("/api/a2a/send-message")
async def send_a2a_message(request: A2AMessageRequest):
    """A2A 프로토콜을 통한 에이전트 간 메시지 전송"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        message = A2AMessage(
            sender_agent=request.sender_agent,
            receiver_agent=request.receiver_agent,
            message_type=request.message_type,
            payload=request.payload,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
        result = await a2a_bridge.send_message(message)
        
        return {
            "status": "success",
            "message_id": session_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Message sending failed: {str(e)}")

@app.post("/api/workflow/run")
async def run_workflow(request: WorkflowRequest):
    """LangGraph 워크플로우 실행"""
    try:
        # A2A 브리지와 MCP 매니저를 워크플로우에 연결
        workflow_result = await langgraph_workflow.run_workflow(
            user_input=request.user_input,
            user_profile=request.user_profile,
            preferences=request.preferences,
            a2a_bridge=a2a_bridge,
            mcp_manager=mcp_manager
        )
        
        return {
            "status": "success",
            "workflow_result": workflow_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.post("/api/workflow/start-conversation")
async def start_conversation(request: StartConversationRequest):
    """대화 세션 시작"""
    try:
        session_id = str(uuid.uuid4())
        
        # 초기 상태로 워크플로우 실행 (대화 단계까지)
        initial_state = {
            "user_input": request.initial_input,
            "user_profile": request.user_profile or {},
            "agent_session": {"session_id": session_id},
            "workflow_context": {"started_at": datetime.now().isoformat()},
            "current_decision_point": "collect_preferences",
            "conversation_history": [],
            "collected_preferences": {},
            "question_completeness_score": 0.0,
            "current_question": None,
            "waiting_for_user_response": False
        }
        
        # 대화 수집 노드 실행
        from ..langgraph_workflow.state import StateManager
        state_manager = StateManager()
        
        # 첫 질문 생성
        question_result = await autogen_agents.generate_adaptive_question(
            conversation_history=[],
            collected_preferences={},
            user_input=request.initial_input
        )
        
        initial_state["current_question"] = question_result["next_question"]
        initial_state["waiting_for_user_response"] = True
        initial_state["conversation_history"] = [{
            "question": question_result["next_question"],
            "category": question_result["category"],
            "timestamp": datetime.now().isoformat()
        }]
        initial_state["collected_preferences"] = question_result.get("collected_preferences", {})
        initial_state["question_completeness_score"] = question_result.get("completeness_score", 0.0)
        initial_state["session_id"] = session_id
        initial_state["current_step"] = "collect_user_preferences"
        initial_state["current_step_result"] = {}
        initial_state["hobby_recommendations"] = []
        initial_state["cached_data_keys"] = []
        
        # 상태 저장
        state_manager.save_state(initial_state)
        
        return {
            "status": "success",
            "session_id": session_id,
            "question": question_result["next_question"],
            "category": question_result["category"],
            "completeness_score": question_result["completeness_score"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation start failed: {str(e)}")

@app.post("/api/workflow/answer-question")
async def answer_question(request: AnswerQuestionRequest):
    """사용자 답변 제출 및 다음 질문 받기"""
    try:
        # 세션 상태 로드 (실제로는 Redis나 DB에서 로드해야 함)
        # 여기서는 간단히 워크플로우를 통해 처리
        from ..langgraph_workflow.state import StateManager
        state_manager = StateManager()
        
        # 세션 상태 로드 시도
        session_state = state_manager.load_state(request.session_id)
        
        if not session_state:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
        
        conversation_history = session_state.get("conversation_history", [])
        collected_preferences = session_state.get("collected_preferences", {})
        
        # 사용자 답변을 히스토리에 추가
        if conversation_history:
            conversation_history[-1]["answer"] = request.answer
        
        # 다음 질문 생성
        question_result = await autogen_agents.generate_adaptive_question(
            conversation_history=conversation_history,
            collected_preferences=collected_preferences,
            user_input=request.answer
        )
        
        # 새로운 질문 추가
        conversation_history.append({
            "question": question_result["next_question"],
            "category": question_result["category"],
            "timestamp": datetime.now().isoformat()
        })
        
        # 상태 업데이트
        session_state["conversation_history"] = conversation_history
        session_state["collected_preferences"] = question_result["collected_preferences"]
        session_state["current_question"] = question_result["next_question"]
        session_state["question_completeness_score"] = question_result["completeness_score"]
        session_state["waiting_for_user_response"] = True
        
        # 상태 저장
        state_manager.save_state(session_state)
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "question": question_result["next_question"],
            "category": question_result["category"],
            "completeness_score": question_result["completeness_score"],
            "should_continue": question_result["should_continue"],
            "collected_preferences": question_result["collected_preferences"],
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer processing failed: {str(e)}")

@app.post("/api/workflow/complete-conversation")
async def complete_conversation(request: CompleteConversationRequest):
    """대화 완료 후 워크플로우 진행"""
    try:
        # 세션 상태 로드
        from ..langgraph_workflow.state import StateManager
        state_manager = StateManager()
        
        session_state = state_manager.load_state(request.session_id)
        
        if not session_state:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
        
        # 수집된 정보로 프로필 구성
        collected_preferences = session_state.get("collected_preferences", {})
        user_profile = session_state.get("user_profile", {})
        user_profile.update(collected_preferences)
        
        # 워크플로우 실행 (대화 단계를 건너뛰고 프로필 분석부터 시작)
        workflow_result = await langgraph_workflow.run_workflow(
            user_input=session_state.get("user_input", ""),
            user_profile=user_profile,
            preferences={},
            a2a_bridge=a2a_bridge,
            mcp_manager=mcp_manager
        )
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "workflow_result": workflow_result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation completion failed: {str(e)}")

@app.get("/api/mcp/capabilities")
async def get_mcp_capabilities():
    """사용 가능한 MCP 서버 기능 목록"""
    capabilities = mcp_manager.get_available_capabilities()
    return {
        "status": "success",
        "capabilities": capabilities,
        "server_count": len(capabilities),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/agents/status")
async def get_agents_status():
    """등록된 에이전트들의 상태 확인"""
    return {
        "status": "success",
        "agents": a2a_bridge.agent_registry,
        "active_sessions": len(a2a_bridge.active_sessions),
        "timestamp": datetime.now().isoformat()
    }

class VectorSearchRequest(BaseModel):
    query: str
    user_profile: Dict[str, Any]
    search_type: str = "hobbies"  # "hobbies" or "communities"
    top_k: int = 5

@app.post("/api/vector/search")
async def vector_search(request: VectorSearchRequest):
    """벡터 데이터베이스 검색"""
    try:
        if request.search_type == "hobbies":
            results = vector_store.search_similar_hobbies(
                request.query, 
                request.user_profile, 
                request.top_k
            )
        elif request.search_type == "communities":
            results = vector_store.search_communities(
                request.query, 
                request.user_profile, 
                request.top_k
            )
        else:
            raise HTTPException(status_code=400, detail="잘못된 검색 타입입니다. 'hobbies' 또는 'communities'를 사용하세요.")
        
        return {
            "status": "success",
            "search_type": request.search_type,
            "query": request.query,
            "results": results,
            "result_count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 검색 실패: {str(e)}")

@app.get("/api/vector/stats")
async def get_vector_store_stats():
    """벡터 스토어 통계 조회"""
    try:
        stats = vector_store.get_vector_store_stats()
        return {
            "status": "success",
            "vector_store_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 스토어 통계 조회 실패: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    API 헬스 체크 및 시스템 상태 확인
    
    Returns:
        시스템 상태 및 메트릭 정보
    """
    try:
        # 각 컴포넌트 상태 확인
        vector_stats = vector_store.get_vector_store_stats()
        error_stats = langgraph_workflow.error_handler.get_error_statistics() if hasattr(langgraph_workflow, 'error_handler') else {}
        
        # 컴포넌트 상태
        components_status = {
            "autogen_agents": "active",
            "langgraph_workflow": "active", 
            "a2a_bridge": "active",
            "mcp_manager": "active",
            "vector_store": "active" if not vector_stats.get("error") else "inactive"
        }
        
        # 전체 상태 결정
        all_healthy = all(
            status == "active" 
            for status in components_status.values()
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components_status,
            "metrics": {
                "error_statistics": error_stats,
                "vector_store_stats": vector_stats,
                "active_sessions": len(a2a_bridge.active_sessions) if hasattr(a2a_bridge, 'active_sessions') else 0
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/metrics")
async def get_metrics():
    """
    시스템 메트릭 수집
    
    Returns:
        상세 메트릭 정보
    """
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # 에러 통계
        if hasattr(langgraph_workflow, 'error_handler'):
            metrics["error_statistics"] = langgraph_workflow.error_handler.get_error_statistics()
        
        # 벡터 스토어 통계
        metrics["vector_store"] = vector_store.get_vector_store_stats()
        
        # A2A 브리지 상태
        if hasattr(a2a_bridge, 'agent_registry'):
            metrics["a2a_bridge"] = {
                "registered_agents": len(a2a_bridge.agent_registry),
                "active_sessions": len(a2a_bridge.active_sessions) if hasattr(a2a_bridge, 'active_sessions') else 0
            }
        
        # MCP 서버 상태
        if hasattr(mcp_manager, 'get_available_capabilities'):
            capabilities = mcp_manager.get_available_capabilities()
            metrics["mcp_servers"] = {
                "available_capabilities": len(capabilities),
                "capabilities": list(capabilities.keys()) if isinstance(capabilities, dict) else []
            }
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메트릭 수집 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 