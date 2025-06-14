import asyncio
from qdrant_client import QdrantClient
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, List, Dict, Any
import streamlit as st
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import (
    AugmentedLLM,
)
import json
import os
from datetime import datetime
from pathlib import Path

T = TypeVar("T", bound=AugmentedLLM)

# Global app instance
app = MCPApp(
    name="mcp_rag_agent",
    settings=get_settings("configs/mcp_agent.config.yaml")
)

@dataclass
class AgentState:
    """Container for agent and its associated LLM"""

    agent: Agent
    llm: Optional[AugmentedLLM] = None


async def get_agent_state(
    key: str,
    agent_class: Type[Agent],
    llm_class: Optional[Type[T]] = None,
    **agent_kwargs,
) -> AgentState:
    """
    Get or create agent state, reinitializing connections if retrieved from session.

    Args:
        key: Session state key
        agent_class: Agent class to instantiate
        llm_class: Optional LLM class to attach
        **agent_kwargs: Arguments for agent instantiation
    """
    if key not in st.session_state:
        # Create new agent
        agent = agent_class(
            connection_persistence=False,
            **agent_kwargs,
        )
        await agent.initialize()

        # Attach LLM if specified
        llm = None
        if llm_class:
            llm = await agent.attach_llm(llm_class)

        state: AgentState = AgentState(agent=agent, llm=llm)
        st.session_state[key] = state
    else:
        state = st.session_state[key]

    return state


SAMPLE_TEXTS = [
    "Today, we're open-sourcing the Model Context Protocol (MCP), a new standard for connecting AI assistants to the systems where data lives, including content repositories, business tools, and development environments",
    "Its aim is to help frontier models produce better, more relevant responses",
    "As AI assistants gain mainstream adoption, the industry has invested heavily in model capabilities, achieving rapid advances in reasoning and quality",
    "Yet even the most sophisticated models are constrained by their isolation from data—trapped behind information silos and legacy systems",
    "Every new data source requires its own custom implementation, making truly connected systems difficult to scale",
    "MCP addresses this challenge",
    "It provides a universal, open standard for connecting AI systems with data sources, replacing fragmented integrations with a single protocol",
    "The result is a simpler, more reliable way to give AI systems access to the data they need",
    "Model Context Protocol\nThe Model Context Protocol is an open standard that enables developers to build secure, two-way connections between their data sources and AI-powered tools",
    "The architecture is straightforward: developers can either expose their data through MCP servers or build AI applications (MCP clients) that connect to these servers",
    "Today, we're introducing three major components of the Model Context Protocol for developers:\n\nThe Model Context Protocol specification and SDKs\nLocal MCP server support in the Claude Desktop apps\nAn open-source repository of MCP servers\nClaude 3",
    "5 Sonnet is adept at quickly building MCP server implementations, making it easy for organizations and individuals to rapidly connect their most important datasets with a range of AI-powered tools",
    "To help developers start exploring, we're sharing pre-built MCP servers for popular enterprise systems like Google Drive, Slack, GitHub, Git, Postgres, and Puppeteer",
    "Early adopters like Block and Apollo have integrated MCP into their systems, while development tools companies including Zed, Replit, Codeium, and Sourcegraph are working with MCP to enhance their platforms—enabling AI agents to better retrieve relevant information to further understand the context around a coding task and produce more nuanced and functional code with fewer attempts",
    "'At Block, open source is more than a development model—it's the foundation of our work and a commitment to creating technology that drives meaningful change and serves as a public good for all,' said Dhanji R",
    "Prasanna, Chief Technology Officer at Block",
    "Open technologies like the Model Context Protocol are the bridges that connect AI to real-world applications, ensuring innovation is accessible, transparent, and rooted in collaboration",
    "We are excited to partner on a protocol and use it to build agentic systems, which remove the burden of the mechanical so people can focus on the creative",
    "\n\nInstead of maintaining separate connectors for each data source, developers can now build against a standard protocol",
    "As the ecosystem matures, AI systems will maintain context as they move between different tools and datasets, replacing today's fragmented integrations with a more sustainable architecture",
    "Getting started\nDevelopers can start building and testing MCP connectors today",
    "All Claude",
    "ai plans support connecting MCP servers to the Claude Desktop app",
    "Claude for Work customers can begin testing MCP servers locally, connecting Claude to internal systems and datasets",
    "We'll soon provide developer toolkits for deploying remote production MCP servers that can serve your entire Claude for Work organization",
    "To start building:\n\nInstall pre-built MCP servers through the Claude Desktop app\nFollow our quickstart guide to build your first MCP server\nContribute to our open-source repositories of connectors and implementations\nAn open community\nWe're committed to building MCP as a collaborative, open-source project and ecosystem, and we're eager to hear your feedback",
    "Whether you're an AI tool developer, an enterprise looking to leverage existing data, or an early adopter exploring the frontier, we invite you to build the future of context-aware AI together",
]


def initialize_collection():
    """Create and add data to collection."""
    client = QdrantClient("http://localhost:6333")
    client.set_model("BAAI/bge-small-en-v1.5")

    if client.collection_exists("my_collection"):
        return

    client.add(
        collection_name="my_collection",
        documents=SAMPLE_TEXTS,
    )


async def main():
    """Main RAG agent function for Streamlit integration"""
    
    # Initialize the app
    await app.initialize()

    # Get agent state
    state = await get_agent_state(
        key="agent",
        agent_class=Agent,
        llm_class=OpenAIAugmentedLLM,
        name="agent",
        instruction="""You are an intelligent assistant equipped with a 
        "find memories" tool that allows you to access information 
        about Model Context Protocol (MCP). Your primary role is to assist 
        users with queries about MCP by actively using the "find memories" 
        tool to retrieve and provide accurate responses. Always utilize the 
        "find memories" tool whenever necessary to ensure accurate information.
        """,
        server_names=["qdrant"],
    )

    # Get available tools
    tools = await state.agent.list_tools()

    # Streamlit UI
    st.title("💬 RAG Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by mcp-agent")

    with st.expander("View Tools"):
        if tools and hasattr(tools, 'tools'):
            tool_descriptions = [f"- **{tool.name}**: {tool.description}\n\n" for tool in tools.tools]
            st.markdown("".join(tool_descriptions))
        else:
            st.markdown("- **find memories**: Search for relevant information in the knowledge base")

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you with Model Context Protocol?"}
        ]

    # Display chat messages
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = ""
            with st.spinner("Thinking..."):
                try:
                    if state.llm:
                        response = await state.llm.generate_str(
                            message=prompt, 
                            request_params=RequestParams(use_history=True)
                        )
                    else:
                        response = "LLM not available. Please check the agent configuration."
                except Exception as e:
                    response = f"Error generating response: {str(e)}"
            
            st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})


def run_streamlit_rag():
    """Run RAG agent in Streamlit environment"""
    try:
        # Initialize collection first
        initialize_collection()
        st.success("✅ Collection initialized successfully!")
        
        # Run the main async function
        asyncio.run(main())
        
    except Exception as e:
        st.error(f"Error running RAG agent: {e}")
        st.info("Please make sure Qdrant server is running and OpenAI API key is set.")


if __name__ == "__main__":
    # For standalone execution
    initialize_collection()
    asyncio.run(main())

def load_collection_types() -> List[str]:
    """사용 가능한 컬렉션 타입 로드"""
    return [
        "document",
        "knowledge_base", 
        "chat_history",
        "embeddings",
        "vector_store",
        "memory"
    ]

def load_document_formats() -> List[Dict[str, str]]:
    """지원하는 문서 형식 로드"""
    return [
        {"format": "txt", "description": "Plain Text Files", "extension": ".txt"},
        {"format": "pdf", "description": "PDF Documents", "extension": ".pdf"},
        {"format": "docx", "description": "Word Documents", "extension": ".docx"},
        {"format": "md", "description": "Markdown Files", "extension": ".md"},
        {"format": "json", "description": "JSON Data", "extension": ".json"},
        {"format": "csv", "description": "CSV Data", "extension": ".csv"},
        {"format": "html", "description": "HTML Documents", "extension": ".html"},
        {"format": "xml", "description": "XML Documents", "extension": ".xml"}
    ]

def get_qdrant_status() -> Dict[str, Any]:
    """Qdrant 서버 상태 확인"""
    try:
        client = QdrantClient("http://localhost:6333")
        
        # 서버 연결 테스트
        collections = client.get_collections()
        
        return {
            "status": "connected",
            "server_url": "http://localhost:6333",
            "collections_count": len(collections.collections) if collections else 0,
            "timestamp": datetime.now().isoformat(),
            "message": "Qdrant server is running and accessible"
        }
        
    except Exception as e:
        return {
            "status": "disconnected",
            "server_url": "http://localhost:6333",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "message": "Cannot connect to Qdrant server"
        }

def get_available_collections() -> List[Dict[str, Any]]:
    """사용 가능한 컬렉션 목록 조회"""
    try:
        client = QdrantClient("http://localhost:6333")
        collections = client.get_collections()
        
        collection_list = []
        
        if collections and hasattr(collections, 'collections'):
            for collection in collections.collections:
                try:
                    # 컬렉션 상세 정보 조회
                    info = client.get_collection(collection.name)
                    collection_list.append({
                        "name": collection.name,
                        "vectors_count": info.vectors_count if info else 0,
                        "status": info.status if info else "unknown",
                        "created": datetime.now().isoformat()
                    })
                except Exception as e:
                    collection_list.append({
                        "name": collection.name,
                        "vectors_count": 0,
                        "status": "error",
                        "error": str(e)
                    })
        
        return collection_list
        
    except Exception as e:
        return [{"error": f"Failed to retrieve collections: {str(e)}"}]

def save_rag_conversation(messages: List[Dict[str, str]], filename: str) -> str:
    """RAG 대화 내용을 파일로 저장"""
    try:
        # 설정에서 보고서 경로 가져오기
        try:
            from configs.settings import get_reports_path
            reports_dir = get_reports_path('rag_agent')
        except ImportError:
            reports_dir = "rag_reports"
        
        # 디렉토리 생성
        os.makedirs(reports_dir, exist_ok=True)
        
        # 파일명에 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename.endswith('.json'):
            filename = f"{filename}_{timestamp}.json"
        
        file_path = os.path.join(reports_dir, filename)
        
        # 대화 데이터 구조화
        conversation_data = {
            "conversation_id": f"rag_chat_{timestamp}",
            "created_at": datetime.now().isoformat(),
            "agent_type": "RAG Agent",
            "messages_count": len(messages),
            "messages": messages,
            "metadata": {
                "qdrant_status": get_qdrant_status(),
                "available_collections": get_available_collections()
            }
        }
        
        # JSON 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        return file_path
        
    except Exception as e:
        raise Exception(f"대화 저장 실패: {str(e)}")

def generate_rag_response(question: str) -> str:
    """RAG 기반 응답 생성"""
    try:
        # Qdrant 연결 상태 확인
        status = get_qdrant_status()
        if status["status"] != "connected":
            return f"❌ Qdrant 서버에 연결할 수 없습니다: {status.get('error', 'Unknown error')}"
        
        # 컬렉션 확인
        collections = get_available_collections()
        if not collections or (len(collections) == 1 and "error" in collections[0]):
            return "❌ 사용 가능한 컬렉션이 없습니다. 먼저 컬렉션을 초기화해주세요."
        
        # Qdrant 클라이언트로 검색
        client = QdrantClient("http://localhost:6333")
        client.set_model("BAAI/bge-small-en-v1.5")
        
        # 컬렉션에서 검색
        search_results = client.query(
            collection_name="my_collection",
            query_text=question,
            limit=3
        )
        
        if not search_results:
            return "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요."
        
        # 검색 결과를 기반으로 응답 생성
        context_texts = []
        for result in search_results:
            if hasattr(result, 'document') and result.document:
                context_texts.append(result.document)
            elif hasattr(result, 'payload') and 'document' in result.payload:
                context_texts.append(result.payload['document'])
        
        if not context_texts:
            return "검색 결과에서 관련 정보를 추출할 수 없습니다."
        
        # 간단한 응답 생성 (실제로는 LLM을 사용해야 함)
        context = "\n".join(context_texts[:2])  # 상위 2개 결과만 사용
        
        response = f"""**질문**: {question}

**관련 정보를 찾았습니다:**

{context}

**응답**: 위의 정보를 바탕으로, MCP(Model Context Protocol)에 관한 질문에 답변드리겠습니다. 더 구체적인 질문이 있으시면 언제든 물어보세요!

*검색된 문서 수: {len(search_results)}개*"""
        
        return response
        
    except Exception as e:
        return f"❌ RAG 응답 생성 중 오류가 발생했습니다: {str(e)}"
