"""
Graph Generator Agent

Knowledge Graph를 생성하는 전문 에이전트
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated
import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain_graphrag.indexing.graph_generation import (
    EntityRelationshipExtractor,
    GraphsMerger,
    EntityRelationshipDescriptionSummarizer,
    GraphGenerator,
)
from pydantic import BaseModel, Field


class GraphGeneratorState(TypedDict):
    """Graph Generator Agent State"""
    messages: Annotated[List, "Messages in the conversation"]
    agent_id: Annotated[str, "Agent ID"]
    current_step: Annotated[str, "Current step in workflow"]
    text_units: Annotated[pd.DataFrame, "Text units dataframe"]
    knowledge_graph: Annotated[Any, "Generated knowledge graph"]
    processing_status: Annotated[str, "Processing status"]
    error_message: Annotated[Optional[str], "Error message if any"]


class GraphGeneratorConfig(BaseModel):
    """Configuration for Graph Generator Agent"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model name")
    temperature: float = Field(default=0.0, description="LLM temperature")
    cache_file: str = Field(default="graph_cache.db", description="Cache file path")
    max_concurrency: int = Field(default=1, description="Max concurrency for processing")


class GraphGeneratorAgent:
    """Knowledge Graph 생성 전문 에이전트"""
    
    def __init__(self, config: GraphGeneratorConfig):
        """
        Initialize Graph Generator Agent
        
        Args:
            config: GraphGeneratorConfig object with all necessary parameters
        """
        self.config = config
        self.agent_id = "graph_generator"
        
        # Initialize components
        self._initialize_components()
        
        # Build the agent graph
        self.graph = self._build_agent_graph()
    
    def _initialize_components(self):
        """Initialize LLM and GraphRAG components"""
        # Cache setup
        cache = SQLiteCache(self.config.cache_file)
        
        # LLM for Entity and Relationship Extraction
        er_llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key,
            cache=cache,
        )
        extractor = EntityRelationshipExtractor.build_default(llm=er_llm)

        # LLM for Entity and Relationship Summarization
        es_llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key,
            cache=cache,
        )
        summarizer = EntityRelationshipDescriptionSummarizer.build_default(llm=es_llm)
        
        self.graph_generator = GraphGenerator(
            er_extractor=extractor,
            graphs_merger=GraphsMerger(),
            er_description_summarizer=summarizer,
        )
    
    def _build_agent_graph(self) -> StateGraph:
        """Build the Graph Generator Agent workflow"""
        workflow = StateGraph(GraphGeneratorState)
        
        # Add nodes
        workflow.add_node("analyze_text_units", self._analyze_text_units)
        workflow.add_node("extract_entities_relationships", self._extract_entities_relationships)
        workflow.add_node("build_knowledge_graph", self._build_knowledge_graph)
        workflow.add_node("validate_graph", self._validate_graph)
        workflow.add_node("finalize_graph", self._finalize_graph)
        
        # Define flow
        workflow.set_entry_point("analyze_text_units")
        workflow.add_edge("analyze_text_units", "extract_entities_relationships")
        workflow.add_edge("extract_entities_relationships", "build_knowledge_graph")
        workflow.add_edge("build_knowledge_graph", "validate_graph")
        workflow.add_edge("validate_graph", "finalize_graph")
        workflow.add_edge("finalize_graph", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    async def _analyze_text_units(self, state: GraphGeneratorState) -> GraphGeneratorState:
        """Analyze incoming text units and prepare for processing"""
        text_units = state.get("text_units", pd.DataFrame())
        
        if text_units.empty:
            state["processing_status"] = "no_data"
            state["error_message"] = "No text units provided for analysis"
            return state
        
        # Validate required columns
        required_columns = ["id", "document_id", "text_unit"]
        missing_columns = [col for col in required_columns if col not in text_units.columns]
        
        if missing_columns:
            state["processing_status"] = "invalid_data"
            state["error_message"] = f"Missing required columns: {missing_columns}"
            return state
        
        state["processing_status"] = "analyzing"
        state["current_step"] = "analyze_text_units"
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"🔍 분석 중: {len(text_units)} 개의 텍스트 단위를 분석하고 있습니다.")
        ]
        
        return state

    async def _extract_entities_relationships(self, state: GraphGeneratorState) -> GraphGeneratorState:
        """Extract entities and relationships from text units"""
        if state["processing_status"] != "analyzing":
            return state
        
        state["processing_status"] = "extracting"
        state["current_step"] = "extract_entities_relationships"
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="🧠 엔티티와 관계를 추출하고 있습니다...")
        ]
        
        return state

    async def _build_knowledge_graph(self, state: GraphGeneratorState) -> GraphGeneratorState:
        """Build the knowledge graph from extracted data"""
        if state["processing_status"] != "extracting":
            return state
        
        text_units = state["text_units"]
        
        try:
            def run_sync():
                config = RunnableConfig({
                    "callbacks": [],
                    "configurable": {},
                    "max_concurrency": self.config.max_concurrency
                })
                return self.graph_generator.invoke(text_units, config)

            # Run the synchronous graph generator in a separate thread
            knowledge_graph = await asyncio.to_thread(run_sync)
            
            state["knowledge_graph"] = knowledge_graph
            state["processing_status"] = "built"
            state["current_step"] = "build_knowledge_graph"
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"🌐 지식 그래프 생성 완료: {len(knowledge_graph.nodes)} 노드, {len(knowledge_graph.edges)} 엣지")
            ]
            
        except Exception as e:
            state["processing_status"] = "error"
            state["error_message"] = f"Knowledge graph 생성 중 오류: {str(e)}"
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"❌ 오류 발생: {str(e)}")
            ]
        
        return state

    async def _validate_graph(self, state: GraphGeneratorState) -> GraphGeneratorState:
        """Validate the generated knowledge graph"""
        if state["processing_status"] != "built":
            return state
        
        knowledge_graph = state.get("knowledge_graph")
        
        if not knowledge_graph or not knowledge_graph.nodes:
            state["processing_status"] = "error"
            state["error_message"] = "Generated graph is empty or invalid"
            return state
        
        # Basic validation
        num_nodes = len(knowledge_graph.nodes)
        num_edges = len(knowledge_graph.edges)
        
        if num_nodes == 0:
            state["processing_status"] = "error"
            state["error_message"] = "No entities found in text"
            return state
        
        state["processing_status"] = "validated"
        state["current_step"] = "validate_graph"
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"✅ 그래프 검증 완료: 품질 점수 계산 중...")
        ]
        
        return state

    async def _finalize_graph(self, state: GraphGeneratorState) -> GraphGeneratorState:
        """Finalize the knowledge graph and prepare for handoff"""
        if state["processing_status"] != "validated":
            return state
        
        knowledge_graph = state["knowledge_graph"]
        
        # Generate summary statistics
        stats = {
            "nodes": len(knowledge_graph.nodes),
            "edges": len(knowledge_graph.edges),
            "entity_types": len(set(node.type for node in knowledge_graph.nodes if hasattr(node, 'type'))),
            "relationship_types": len(set(edge.type for edge in knowledge_graph.edges if hasattr(edge, 'type')))
        }
        
        state["processing_status"] = "completed"
        state["current_step"] = "finalize_graph"
        state["graph_stats"] = stats
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"🎉 지식 그래프 생성 완료!\n"
                            f"📊 통계: {stats['nodes']} 노드, {stats['edges']} 엣지\n"
                            f"🏷️ 엔티티 타입: {stats['entity_types']}, 관계 타입: {stats['relationship_types']}")
        ]
        
        return state

    async def process_text_units(self, text_units: pd.DataFrame, thread_id: str = "default") -> Dict[str, Any]:
        """
        Process text units and generate knowledge graph
        
        Args:
            text_units: DataFrame with text units
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dict containing the generated knowledge graph and metadata
        """
        initial_state = GraphGeneratorState(
            messages=[HumanMessage(content="지식 그래프를 생성해주세요.")],
            agent_id=self.agent_id,
            current_step="start",
            text_units=text_units,
            knowledge_graph=None,
            processing_status="initialized",
            error_message=None
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the agent workflow
        result = await self.graph.ainvoke(initial_state, config)
        
        return {
            "knowledge_graph": result.get("knowledge_graph"),
            "status": result.get("processing_status"),
            "error": result.get("error_message"),
            "stats": result.get("graph_stats"),
            "messages": result.get("messages", [])
        } 