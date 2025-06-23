from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import uuid
from datetime import datetime

from ..autogen.agents import HSPAutoGenAgents
from ..langgraph_workflow.workflow import HSPLangGraphWorkflow
from ..bridge.a2a_bridge import A2AProtocolBridge, A2AMessage
from ..mcp.manager import MCPServerManager

app = FastAPI(title="Hobby Starter Pack Agent API", version="1.0.0")

# Global instances
autogen_agents = HSPAutoGenAgents()
langgraph_workflow = HSPLangGraphWorkflow()
a2a_bridge = A2AProtocolBridge()
mcp_manager = MCPServerManager()

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

@app.get("/api/health")
async def health_check():
    """API 헬스 체크"""
    return {
        "status": "healthy",
        "components": {
            "autogen_agents": "active",
            "langgraph_workflow": "active", 
            "a2a_bridge": "active",
            "mcp_manager": "active"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 