"""
Multi-Agent Coordinator

Graph Generator Agent와 RAG Agent를 조율하는 코디네이터
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Union
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import logging

from agents.graph_generator_agent import GraphGeneratorAgent, GraphGeneratorConfig
from agents.rag_agent import RAGAgent, RAGAgentConfig


class CoordinatorState(TypedDict):
    """Multi-Agent Coordinator State"""
    messages: Annotated[List, "Messages in the conversation"]
    current_step: Annotated[str, "Current step in workflow"]
    user_query: Annotated[str, "User query"]
    data_file_path: Annotated[str, "Path to data file"]
    text_units: Annotated[pd.DataFrame, "Text units dataframe"]
    knowledge_graph: Annotated[Any, "Generated knowledge graph"]
    final_response: Annotated[str, "Final response to user"]
    agent_communications: Annotated[List[Dict], "Inter-agent communications"]
    workflow_status: Annotated[str, "Overall workflow status"]
    error_message: Annotated[Optional[str], "Error message if any"]


class MultiAgentConfig(BaseModel):
    """Configuration for Multi-Agent System"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    data_file_path: str = Field(..., description="Path to the data file")
    graph_model_name: str = Field(default="gpt-4o-mini", description="Model for graph generation")
    rag_model_name: str = Field(default="gpt-4o-mini", description="Model for RAG responses")
    max_search_results: int = Field(default=5, description="Max search results for RAG")
    context_window_size: int = Field(default=4000, description="Context window size")


class MultiAgentCoordinator:
    """멀티 에이전트 시스템 코디네이터"""
    
    def __init__(self, config: MultiAgentConfig):
        """
        Initialize Multi-Agent Coordinator
        
        Args:
            config: MultiAgentConfig object with all necessary parameters
        """
        self.config = config
        
        # Initialize agents
        self._initialize_agents()
        
        # Build the coordinator graph
        self.graph = self._build_coordinator_graph()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_agents(self):
        """Initialize the specialized agents"""
        # Graph Generator Agent
        graph_config = GraphGeneratorConfig(
            openai_api_key=self.config.openai_api_key,
            model_name=self.config.graph_model_name,
            temperature=0.0,
            cache_file="graph_cache.db",
            max_concurrency=1
        )
        self.graph_generator = GraphGeneratorAgent(graph_config)
        
        # RAG Agent
        rag_config = RAGAgentConfig(
            openai_api_key=self.config.openai_api_key,
            model_name=self.config.rag_model_name,
            temperature=0.1,
            max_search_results=self.config.max_search_results,
            context_window_size=self.config.context_window_size
        )
        self.rag_agent = RAGAgent(rag_config)
    
    def _build_coordinator_graph(self) -> StateGraph:
        """Build the Multi-Agent Coordinator workflow"""
        workflow = StateGraph(CoordinatorState)
        
        # Add nodes
        workflow.add_node("initialize_workflow", self._initialize_workflow)
        workflow.add_node("load_data", self._load_data)
        workflow.add_node("delegate_to_graph_generator", self._delegate_to_graph_generator) 
        workflow.add_node("delegate_to_rag_agent", self._delegate_to_rag_agent)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # Define workflow
        workflow.set_entry_point("initialize_workflow")
        workflow.add_edge("initialize_workflow", "load_data")
        workflow.add_edge("load_data", "delegate_to_graph_generator")
        workflow.add_edge("delegate_to_graph_generator", "delegate_to_rag_agent")
        workflow.add_edge("delegate_to_rag_agent", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    async def _initialize_workflow(self, state: CoordinatorState) -> CoordinatorState:
        """Initialize the multi-agent workflow"""
        user_query = state.get("user_query", "")
        
        state["workflow_status"] = "initializing"
        state["current_step"] = "initialize_workflow"
        state["agent_communications"] = []
        
        # Log the start of workflow
        communication = {
            "timestamp": asyncio.get_event_loop().time(),
            "from": "coordinator",
            "to": "system",
            "message": f"🚀 멀티 에이전트 워크플로우 시작: '{user_query[:50]}...'"
        }
        state["agent_communications"].append(communication)
        
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="🔄 멀티 에이전트 시스템을 초기화하고 있습니다...")
        ]
        
        return state

    async def _load_data(self, state: CoordinatorState) -> CoordinatorState:
        """Load data from the specified file"""
        if state["workflow_status"] != "initializing":
            return state
        
        data_file_path = state.get("data_file_path", self.config.data_file_path)
        
        try:
            # Load data from CSV file
            df = pd.read_csv(data_file_path)
            
            # Validate required columns
            required_columns = ["id", "document_id", "text_unit"]
            if not all(col in df.columns for col in required_columns):
                # Try to create a basic structure if columns are missing
                if "text" in df.columns:
                    df = df.rename(columns={"text": "text_unit"})
                if "id" not in df.columns:
                    df["id"] = range(len(df))
                if "document_id" not in df.columns:
                    df["document_id"] = "doc_1"
                if "text_unit" not in df.columns:
                    raise ValueError("No text column found in the data file")
            
            state["text_units"] = df
            state["workflow_status"] = "data_loaded"
            state["current_step"] = "load_data"
            
            communication = {
                "timestamp": asyncio.get_event_loop().time(),
                "from": "coordinator",
                "to": "system",
                "message": f"📂 데이터 로딩 완료: {len(df)} 행의 데이터"
            }
            state["agent_communications"].append(communication)
            
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"📊 데이터 로딩 완료: {len(df)}개의 텍스트 단위를 처리할 준비가 되었습니다.")
            ]
            
        except FileNotFoundError:
            state["workflow_status"] = "error"
            state["error_message"] = f"Data file not found: {data_file_path}"
            state["text_units"] = pd.DataFrame(columns=["id", "document_id", "text_unit"])
            
        except Exception as e:
            state["workflow_status"] = "error"
            state["error_message"] = f"Data loading failed: {str(e)}"
            state["text_units"] = pd.DataFrame(columns=["id", "document_id", "text_unit"])
        
        return state

    async def _delegate_to_graph_generator(self, state: CoordinatorState) -> CoordinatorState:
        """Delegate graph generation to the Graph Generator Agent"""
        if state["workflow_status"] not in ["data_loaded", "error"]:
            return state
        
        text_units = state.get("text_units", pd.DataFrame())
        
        if text_units.empty:
            state["knowledge_graph"] = None
            state["workflow_status"] = "graph_generation_skipped"
            return state
        
        try:
            # Communicate with Graph Generator Agent
            communication = {
                "timestamp": asyncio.get_event_loop().time(),
                "from": "coordinator",
                "to": "graph_generator",
                "message": f"Knowledge Graph 생성 요청: {len(text_units)} 텍스트 단위"
            }
            state["agent_communications"].append(communication)
            
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="🧠 Graph Generator Agent에게 지식 그래프 생성을 요청하고 있습니다...")
            ]
            
            # Call Graph Generator Agent
            result = await self.graph_generator.process_text_units(
                text_units=text_units,
                thread_id="coordinator_session"
            )
            
            if result["status"] == "completed":
                state["knowledge_graph"] = result["knowledge_graph"]
                state["workflow_status"] = "graph_generated"
                
                # Log successful generation
                communication = {
                    "timestamp": asyncio.get_event_loop().time(),
                    "from": "graph_generator",
                    "to": "coordinator", 
                    "message": f"✅ 지식 그래프 생성 완료: {result.get('stats', {})}"
                }
                state["agent_communications"].append(communication)
                
                # Add agent messages to state
                if result.get("messages"):
                    state["messages"].extend(result["messages"])
                
            else:
                state["workflow_status"] = "graph_generation_failed"
                state["error_message"] = result.get("error", "Unknown error in graph generation")
                
        except Exception as e:
            state["workflow_status"] = "graph_generation_failed"
            state["error_message"] = f"Graph generation delegation failed: {str(e)}"
            
        state["current_step"] = "delegate_to_graph_generator"
        return state

    async def _delegate_to_rag_agent(self, state: CoordinatorState) -> CoordinatorState:
        """Delegate query processing to the RAG Agent"""
        if state["workflow_status"] not in ["graph_generated", "graph_generation_failed", "graph_generation_skipped"]:
            return state
        
        user_query = state.get("user_query", "")
        knowledge_graph = state.get("knowledge_graph")
        
        if not user_query:
            state["final_response"] = "사용자 질문이 제공되지 않았습니다."
            state["workflow_status"] = "completed"
            return state
        
        if not knowledge_graph:
            state["final_response"] = """
            죄송합니다. 지식 그래프 생성에 실패했거나 데이터가 없어서 질문에 답변할 수 없습니다.
            데이터 파일을 확인하고 다시 시도해 주세요.
            """
            state["workflow_status"] = "completed"
            return state
        
        try:
            # Communicate with RAG Agent
            communication = {
                "timestamp": asyncio.get_event_loop().time(),
                "from": "coordinator",
                "to": "rag_agent",
                "message": f"RAG 질의 요청: '{user_query[:30]}...'"
            }
            state["agent_communications"].append(communication)
            
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="🎯 RAG Agent에게 질문 처리를 요청하고 있습니다...")
            ]
            
            # Call RAG Agent
            result = await self.rag_agent.query_knowledge_graph(
                user_query=user_query,
                knowledge_graph=knowledge_graph,
                thread_id="coordinator_session"
            )
            
            if result["status"] == "completed":
                state["final_response"] = result["response"]
                state["workflow_status"] = "completed"
                
                # Log successful query processing
                communication = {
                    "timestamp": asyncio.get_event_loop().time(),
                    "from": "rag_agent",
                    "to": "coordinator",
                    "message": "✅ RAG 질의 처리 완료"
                }
                state["agent_communications"].append(communication)
                
                # Add agent messages to state
                if result.get("messages"):
                    state["messages"].extend(result["messages"])
                    
            else:
                state["final_response"] = f"RAG 처리 중 오류가 발생했습니다: {result.get('error', 'Unknown error')}"
                state["workflow_status"] = "rag_failed"
                
        except Exception as e:
            state["final_response"] = f"RAG Agent 처리 중 오류가 발생했습니다: {str(e)}"
            state["workflow_status"] = "rag_failed"
            
        state["current_step"] = "delegate_to_rag_agent"
        return state

    async def _finalize_response(self, state: CoordinatorState) -> CoordinatorState:
        """Finalize the response and cleanup"""
        final_response = state.get("final_response", "")
        communications = state.get("agent_communications", [])
        
        # Add workflow summary
        summary = f"""
        
        ═══════════════════════════════════════
        🤖 멀티 에이전트 처리 요약
        ═══════════════════════════════════════
        
        📊 에이전트 간 통신: {len(communications)}회
        🔄 처리 상태: {state.get('workflow_status', 'unknown')}
        📝 최종 응답 길이: {len(final_response)}자
        
        """
        
        state["final_response"] = final_response + summary
        state["current_step"] = "finalize_response"
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="🎉 멀티 에이전트 워크플로우가 완료되었습니다!")
        ]
        
        return state

    async def process_query(
        self,
        user_query: str,
        data_file_path: Optional[str] = None,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process user query using multi-agent system
        
        Args:
            user_query: User's question
            data_file_path: Optional path to data file (overrides config)
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dict containing the response and metadata
        """
        initial_state = CoordinatorState(
            messages=[HumanMessage(content=user_query)],
            current_step="start",
            user_query=user_query,
            data_file_path=data_file_path or self.config.data_file_path,
            text_units=pd.DataFrame(),
            knowledge_graph=None,
            final_response="",
            agent_communications=[],
            workflow_status="initialized",
            error_message=None
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Run the multi-agent workflow
            result = await self.graph.ainvoke(initial_state, config)
            
            return {
                "response": result.get("final_response", ""),
                "status": result.get("workflow_status", "unknown"),
                "error": result.get("error_message"),
                "agent_communications": result.get("agent_communications", []),
                "messages": result.get("messages", []),
                "knowledge_graph_stats": self._get_graph_stats(result.get("knowledge_graph"))
            }
            
        except Exception as e:
            self.logger.error(f"Multi-agent workflow failed: {str(e)}")
            return {
                "response": f"멀티 에이전트 시스템 처리 중 오류가 발생했습니다: {str(e)}",
                "status": "error",
                "error": str(e),
                "agent_communications": [],
                "messages": [AIMessage(content=f"❌ 시스템 오류: {str(e)}")],
                "knowledge_graph_stats": None
            }
    
    def _get_graph_stats(self, knowledge_graph) -> Optional[Dict[str, int]]:
        """Get statistics about the knowledge graph"""
        if not knowledge_graph:
            return None
        
        try:
            return {
                "nodes": len(knowledge_graph.nodes),
                "edges": len(knowledge_graph.edges),
                "entity_types": len(set(getattr(node, 'type', 'unknown') for node in knowledge_graph.nodes)),
                "relationship_types": len(set(getattr(edge, 'type', 'unknown') for edge in knowledge_graph.edges))
            }
        except Exception:
            return None

    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        return {
            "graph_generator": "ready",
            "rag_agent": "ready",
            "coordinator": "ready"
        } 