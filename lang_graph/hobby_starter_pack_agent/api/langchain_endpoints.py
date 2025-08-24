"""
LangChain 기반 워크플로우 API 엔드포인트
새로운 LangChain 컴포넌트들을 활용한 API
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from ..langchain_workflow.workflow import HSPLangChainWorkflow

router = APIRouter(prefix="/api/langchain", tags=["LangChain Workflow"])

# LangChain 워크플로우 인스턴스
langchain_workflow = HSPLangChainWorkflow()

class LangChainWorkflowRequest(BaseModel):
    user_input: str
    user_id: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None

class MemoryQueryRequest(BaseModel):
    user_id: str
    query: str
    limit: Optional[int] = 10

class VectorSearchRequest(BaseModel):
    query: str
    user_profile: Dict[str, Any]
    search_type: str = "hobbies"  # "hobbies" or "communities"
    top_k: int = 5

class AgentExecutionRequest(BaseModel):
    user_input: str
    context: Optional[Dict[str, Any]] = None
    agent_type: str = "general"  # "general", "hobby_recommendation", "community_matching", "schedule_integration"

@router.post("/workflow/run")
async def run_langchain_workflow(request: LangChainWorkflowRequest):
    """LangChain 기반 워크플로우 실행"""
    try:
        result = await langchain_workflow.run_workflow(
            user_input=request.user_input,
            user_id=request.user_id,
            user_profile=request.user_profile,
            preferences=request.preferences
        )
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain 워크플로우 실행 실패: {str(e)}")

@router.get("/workflow/status/{session_id}")
async def get_workflow_status(session_id: str):
    """워크플로우 상태 조회"""
    try:
        status = await langchain_workflow.get_workflow_status(session_id)
        
        return {
            "status": "success",
            "workflow_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"워크플로우 상태 조회 실패: {str(e)}")

@router.get("/workflow/stats")
async def get_workflow_stats():
    """워크플로우 통계 조회"""
    try:
        stats = langchain_workflow.get_workflow_stats()
        
        return {
            "status": "success",
            "workflow_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"워크플로우 통계 조회 실패: {str(e)}")

@router.post("/memory/query")
async def query_memory(request: MemoryQueryRequest):
    """메모리에서 대화 기록 조회"""
    try:
        conversation_history = langchain_workflow.memory_manager.get_user_conversation_history(
            request.user_id, 
            request.limit
        )
        
        return {
            "status": "success",
            "user_id": request.user_id,
            "conversation_history": conversation_history,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메모리 조회 실패: {str(e)}")

@router.get("/memory/stats")
async def get_memory_stats():
    """메모리 사용 통계 조회"""
    try:
        stats = langchain_workflow.memory_manager.get_memory_stats()
        
        return {
            "status": "success",
            "memory_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메모리 통계 조회 실패: {str(e)}")

@router.post("/vector/search")
async def vector_search(request: VectorSearchRequest):
    """벡터 데이터베이스 검색"""
    try:
        if request.search_type == "hobbies":
            results = langchain_workflow.vector_store.search_similar_hobbies(
                request.query, 
                request.user_profile, 
                request.top_k
            )
        elif request.search_type == "communities":
            results = langchain_workflow.vector_store.search_communities(
                request.query, 
                request.user_profile, 
                request.top_k
            )
        else:
            raise HTTPException(status_code=400, detail="잘못된 검색 타입입니다.")
        
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

@router.get("/vector/stats")
async def get_vector_store_stats():
    """벡터 스토어 통계 조회"""
    try:
        stats = langchain_workflow.vector_store.get_vector_store_stats()
        
        return {
            "status": "success",
            "vector_store_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 스토어 통계 조회 실패: {str(e)}")

@router.post("/agent/execute")
async def execute_agent(request: AgentExecutionRequest):
    """LangChain 에이전트 실행"""
    try:
        result = await langchain_workflow.langchain_agent.run_agent(
            request.user_input, 
            request.context
        )
        
        return {
            "status": "success",
            "agent_type": request.agent_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 실행 실패: {str(e)}")

@router.get("/agent/status")
async def get_agent_status():
    """에이전트 상태 조회"""
    try:
        status = langchain_workflow.langchain_agent.get_agent_status()
        
        return {
            "status": "success",
            "agent_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 상태 조회 실패: {str(e)}")

@router.post("/agent/hobby-recommendation")
async def get_hobby_recommendations(request: LangChainWorkflowRequest):
    """취미 추천 에이전트 실행"""
    try:
        if not request.user_profile:
            raise HTTPException(status_code=400, detail="사용자 프로필이 필요합니다.")
        
        result = await langchain_workflow.langchain_agent.get_hobby_recommendations(
            user_profile=request.user_profile,
            preferences=request.preferences.get("hobby_preferences", []) if request.preferences else [],
            constraints={
                "time": request.user_profile.get("time_availability", "flexible"),
                "budget": request.user_profile.get("budget_range", "medium"),
                "location": request.user_profile.get("location_preference", "Seoul")
            }
        )
        
        return {
            "status": "success",
            "hobby_recommendations": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"취미 추천 실패: {str(e)}")

@router.post("/agent/community-matching")
async def match_communities(request: LangChainWorkflowRequest):
    """커뮤니티 매칭 에이전트 실행"""
    try:
        if not request.user_profile:
            raise HTTPException(status_code=400, detail="사용자 프로필이 필요합니다.")
        
        # 취미 정보 추출 (실제로는 이전 단계에서 가져와야 함)
        hobby_interests = request.user_profile.get("interests", [])
        
        result = await langchain_workflow.langchain_agent.match_communities(
            user_profile=request.user_profile,
            hobby_interests=hobby_interests,
            location=request.user_profile.get("location_preference", "Seoul")
        )
        
        return {
            "status": "success",
            "community_matches": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"커뮤니티 매칭 실패: {str(e)}")

@router.post("/agent/schedule-integration")
async def integrate_schedule(request: LangChainWorkflowRequest):
    """스케줄 통합 에이전트 실행"""
    try:
        if not request.user_profile:
            raise HTTPException(status_code=400, detail="사용자 프로필이 필요합니다.")
        
        # 현재 스케줄 (실제로는 MCP 서버에서 가져와야 함)
        current_schedule = {
            "weekdays": {"evening": ["퇴근", "저녁식사"], "night": ["휴식"]},
            "weekends": {"morning": ["아침식사"], "afternoon": ["자유시간"], "evening": ["자유시간"]}
        }
        
        # 취미 활동 (실제로는 이전 단계에서 가져와야 함)
        hobby_activities = [
            {"name": "독서", "duration": "1 hour", "frequency": "daily"},
            {"name": "등산", "duration": "3 hours", "frequency": "weekly"}
        ]
        
        result = await langchain_workflow.langchain_agent.integrate_schedule(
            current_schedule=current_schedule,
            hobby_activities=hobby_activities,
            time_constraints={
                "available_time": request.user_profile.get("time_availability", "flexible"),
                "preferred_duration": "1-2 hours"
            }
        )
        
        return {
            "status": "success",
            "schedule_integration": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스케줄 통합 실패: {str(e)}")

@router.post("/data/sample")
async def add_sample_data():
    """샘플 데이터를 벡터 스토어에 추가"""
    try:
        success = await langchain_workflow.add_sample_data()
        
        if success:
            return {
                "status": "success",
                "message": "샘플 데이터가 성공적으로 추가되었습니다.",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="샘플 데이터 추가에 실패했습니다.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"샘플 데이터 추가 실패: {str(e)}")

@router.delete("/data/clear")
async def clear_workflow_data():
    """워크플로우 데이터 초기화"""
    try:
        langchain_workflow.clear_workflow_data()
        
        return {
            "status": "success",
            "message": "워크플로우 데이터가 초기화되었습니다.",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 초기화 실패: {str(e)}")

@router.get("/health")
async def langchain_health_check():
    """LangChain 컴포넌트 헬스 체크"""
    try:
        # 각 컴포넌트 상태 확인
        memory_stats = langchain_workflow.memory_manager.get_memory_stats()
        vector_stats = langchain_workflow.vector_store.get_vector_store_stats()
        agent_status = langchain_workflow.langchain_agent.get_agent_status()
        workflow_stats = langchain_workflow.get_workflow_stats()
        
        return {
            "status": "healthy",
            "components": {
                "memory_manager": "active" if memory_stats else "inactive",
                "vector_store": "active" if not vector_stats.get("error") else "inactive",
                "langchain_agent": "active" if agent_status.get("initialized") else "inactive",
                "workflow": "active" if workflow_stats else "inactive"
            },
            "component_details": {
                "memory": memory_stats,
                "vector_store": vector_stats,
                "agent": agent_status,
                "workflow": workflow_stats
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
