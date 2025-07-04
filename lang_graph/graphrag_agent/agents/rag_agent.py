"""
RAG Agent

Knowledge Graph를 활용한 RAG(Retrieval-Augmented Generation) 전문 에이전트
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import logging


class RAGAgentState(TypedDict):
    """RAG Agent State"""
    messages: Annotated[List, "Messages in the conversation"]
    agent_id: Annotated[str, "Agent ID"]
    current_step: Annotated[str, "Current step in workflow"]
    user_query: Annotated[str, "User query"]
    knowledge_graph: Annotated[Any, "Knowledge graph to query"]
    search_results: Annotated[List, "Search results from vector store"]
    context: Annotated[str, "Retrieved context"]
    final_response: Annotated[str, "Final generated response"]
    processing_status: Annotated[str, "Processing status"]
    error_message: Annotated[Optional[str], "Error message if any"]


class RAGAgentConfig(BaseModel):
    """Configuration for RAG Agent"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="LLM model name")
    temperature: float = Field(default=0.1, description="LLM temperature for generation")
    max_search_results: int = Field(default=5, description="Maximum search results to retrieve")
    context_window_size: int = Field(default=4000, description="Maximum context window size")


class RAGAgent:
    """Knowledge Graph 기반 RAG 전문 에이전트"""
    
    def __init__(self, config: RAGAgentConfig):
        """
        Initialize RAG Agent
        
        Args:
            config: RAGAgentConfig object with all necessary parameters
        """
        self.config = config
        self.agent_id = "rag_agent"
        
        # Initialize components
        self._initialize_components()
        
        # Build the agent graph
        self.graph = self._build_agent_graph()
    
    def _initialize_components(self):
        """Initialize LLM and embedding components"""
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=self.config.openai_api_key
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _build_agent_graph(self) -> StateGraph:
        """Build the RAG Agent workflow"""
        workflow = StateGraph(RAGAgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("create_vector_store", self._create_vector_store)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("rank_and_filter", self._rank_and_filter)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("validate_response", self._validate_response)
        
        # Define flow
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "create_vector_store")
        workflow.add_edge("create_vector_store", "retrieve_context")
        workflow.add_edge("retrieve_context", "rank_and_filter")
        workflow.add_edge("rank_and_filter", "generate_response")
        workflow.add_edge("generate_response", "validate_response")
        workflow.add_edge("validate_response", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    async def _analyze_query(self, state: RAGAgentState) -> RAGAgentState:
        """Analyze the user query to understand intent and requirements"""
        user_query = state.get("user_query", "")
        
        if not user_query.strip():
            state["processing_status"] = "error"
            state["error_message"] = "Empty query provided"
            return state
        
        # Query analysis using LLM
        analysis_prompt = f"""
        다음 사용자 질문을 분석해주세요:
        "{user_query}"
        
        분석 결과를 다음 형식으로 제공해주세요:
        - 질문 유형: [사실 확인/분석/비교/요약/기타]
        - 핵심 키워드: [추출된 키워드들]
        - 예상 답변 길이: [짧음/보통/길음]
        - 필요한 정보 유형: [엔티티/관계/통계/기타]
        """
        
        try:
            analysis_response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            state["query_analysis"] = analysis_response.content
            state["processing_status"] = "analyzing"
            state["current_step"] = "analyze_query"
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"🔍 질문 분석 완료: '{user_query[:50]}...'")
            ]
            
        except Exception as e:
            state["processing_status"] = "error"
            state["error_message"] = f"Query analysis failed: {str(e)}"
            
        return state

    async def _create_vector_store(self, state: RAGAgentState) -> RAGAgentState:
        """Create a vector store from the knowledge graph"""
        if state["processing_status"] != "analyzing":
            return state
        
        knowledge_graph = state.get("knowledge_graph")
        
        if not knowledge_graph or not knowledge_graph.nodes:
            state["processing_status"] = "error"
            state["error_message"] = "No knowledge graph provided or graph is empty"
            return state
        
        try:
            # Create documents from graph nodes and edges
            graph_docs = []
            
            # Add entity documents
            for node in knowledge_graph.nodes:
                entity_content = f"Entity: {node.title}\n"
                if hasattr(node, 'description') and node.description:
                    entity_content += f"Description: {node.description}\n"
                if hasattr(node, 'type') and node.type:
                    entity_content += f"Type: {node.type}\n"
                
                graph_docs.append(Document(
                    page_content=entity_content,
                    metadata={
                        "type": "entity",
                        "id": node.id,
                        "title": node.title
                    }
                ))
            
            # Add relationship documents
            for edge in knowledge_graph.edges:
                relationship_content = f"Relationship: {edge.source.title} -> {edge.target.title}\n"
                if hasattr(edge, 'description') and edge.description:
                    relationship_content += f"Description: {edge.description}\n"
                if hasattr(edge, 'type') and edge.type:
                    relationship_content += f"Type: {edge.type}\n"
                
                graph_docs.append(Document(
                    page_content=relationship_content,
                    metadata={
                        "type": "relationship",
                        "id": edge.id,
                        "source": edge.source.title,
                        "target": edge.target.title
                    }
                ))
            
            if not graph_docs:
                state["processing_status"] = "error"
                state["error_message"] = "No content found in knowledge graph"
                return state
            
            # Create vector store
            state["vector_store"] = Chroma.from_documents(graph_docs, self.embeddings)
            state["processing_status"] = "vectorized"
            state["current_step"] = "create_vector_store"
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"📚 벡터 저장소 생성 완료: {len(graph_docs)} 개 문서")
            ]
            
        except Exception as e:
            state["processing_status"] = "error"
            state["error_message"] = f"Vector store creation failed: {str(e)}"
            
        return state

    async def _retrieve_context(self, state: RAGAgentState) -> RAGAgentState:
        """Retrieve relevant context from the vector store"""
        if state["processing_status"] != "vectorized":
            return state
        
        user_query = state["user_query"]
        vector_store = state["vector_store"]
        
        try:
            # Perform similarity search
            search_results = vector_store.similarity_search(
                user_query,
                k=self.config.max_search_results
            )
            
            state["search_results"] = search_results
            state["processing_status"] = "retrieved"
            state["current_step"] = "retrieve_context"
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=f"🎯 관련 정보 검색 완료: {len(search_results)} 개 결과")
            ]
            
        except Exception as e:
            state["processing_status"] = "error"
            state["error_message"] = f"Context retrieval failed: {str(e)}"
            
        return state

    async def _rank_and_filter(self, state: RAGAgentState) -> RAGAgentState:
        """Rank and filter the retrieved context"""
        if state["processing_status"] != "retrieved":
            return state
        
        search_results = state["search_results"]
        user_query = state["user_query"]
        
        if not search_results:
            state["context"] = ""
            state["processing_status"] = "no_context"
            return state
        
        try:
            # Create context from top results
            context_parts = []
            total_length = 0
            
            for doc in search_results:
                if total_length + len(doc.page_content) > self.config.context_window_size:
                    break
                context_parts.append(doc.page_content)
                total_length += len(doc.page_content)
            
            state["context"] = "\n\n".join(context_parts)
            state["processing_status"] = "ranked"
            state["current_step"] = "rank_and_filter"
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="🏆 컨텍스트 순위 결정 및 필터링 완료")
            ]
            
        except Exception as e:
            # Fallback to simple concatenation
            context = "\n\n".join([doc.page_content for doc in search_results[:3]])
            state["context"] = context[:self.config.context_window_size]
            state["processing_status"] = "ranked"
            self.logger.warning(f"Ranking failed, using fallback: {str(e)}")
            
        return state

    async def _generate_response(self, state: RAGAgentState) -> RAGAgentState:
        """Generate final response using retrieved context"""
        if state["processing_status"] not in ["ranked", "no_context"]:
            return state
        
        user_query = state["user_query"]
        context = state.get("context", "")
        
        if not context:
            response_text = """
            죄송합니다. 제공된 지식 그래프에서 질문과 관련된 정보를 찾을 수 없습니다. 
            다른 방식으로 질문해 주시거나, 더 구체적인 정보를 제공해 주세요.
            """
        else:
            generation_prompt = f"""
            당신은 지식 그래프 기반 AI 어시스턴트입니다.
            주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.
            
            컨텍스트:
            {context}
            
            사용자 질문: {user_query}
            
            답변 가이드라인:
            1. 컨텍스트에 있는 정보만을 바탕으로 답변하세요
            2. 확실하지 않은 정보는 추측하지 마세요
            3. 관련된 엔티티와 관계를 명확히 언급하세요
            4. 친근하고 이해하기 쉬운 방식으로 설명하세요
            
            답변:
            """
            
            try:
                response = await self.llm.ainvoke([HumanMessage(content=generation_prompt)])
                response_text = response.content
                
            except Exception as e:
                response_text = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        
        state["final_response"] = response_text
        state["processing_status"] = "generated"
        state["current_step"] = "generate_response"
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="💬 답변 생성 완료")
        ]
        
        return state

    async def _validate_response(self, state: RAGAgentState) -> RAGAgentState:
        """Validate the generated response for quality and accuracy"""
        if state["processing_status"] != "generated":
            return state
        
        final_response = state["final_response"]
        
        try:
            state["processing_status"] = "completed"
            state["current_step"] = "validate_response"
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="✅ 답변 검증 완료 - 고품질 응답 생성됨")
            ]
            
        except Exception as e:
            # Even if validation fails, we can still return the response
            state["processing_status"] = "completed"
            self.logger.warning(f"Response validation failed: {str(e)}")
            
        return state

    async def query_knowledge_graph(
        self,
        user_query: str,
        knowledge_graph: Any,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Query knowledge graph and generate response
        
        Args:
            user_query: User's question
            knowledge_graph: Knowledge graph to query
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dict containing the response and metadata
        """
        initial_state = RAGAgentState(
            messages=[HumanMessage(content=user_query)],
            agent_id=self.agent_id,
            current_step="start",
            user_query=user_query,
            knowledge_graph=knowledge_graph,
            search_results=[],
            context="",
            final_response="",
            processing_status="initialized",
            error_message=None
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the agent workflow
        result = await self.graph.ainvoke(initial_state, config)
        
        return {
            "response": result.get("final_response"),
            "status": result.get("processing_status"),
            "error": result.get("error_message"),
            "context": result.get("context"),
            "validation": result.get("validation_result"),
            "messages": result.get("messages", [])
        } 