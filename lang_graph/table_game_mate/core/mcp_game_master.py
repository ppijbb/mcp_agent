#!/usr/bin/env python3
"""
MCP 통합 게임 마스터

MultiServerMCPClient를 사용한 실제 MCP 서버 통합
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

# MCP 관련 임포트 (실제 패키지가 설치되면 사용)
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_mcp_adapters.prompts import load_mcp_prompt
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️ langchain_mcp_adapters 패키지가 설치되지 않음. 시뮬레이션 모드로 실행")
    MCP_AVAILABLE = False

class MCPGameMaster:
    """MCP 서버들과 통합된 게임 마스터"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.mcp_client = None
        self.current_session = None
        self.game_state = {}
        
    async def initialize_mcp_servers(self):
        """MCP 서버들 초기화"""
        
        if not MCP_AVAILABLE:
            print("⚠️ MCP 패키지 없음 - 기본 모드로 실행")
            return False
        
        try:
            # MCP 서버 설정
            self.mcp_client = MultiServerMCPClient({
                "game_rules": {
                    "command": "python",
                    "args": ["mcp_servers/game_rules_server.py"],
                    "transport": "stdio"
                }
                # 추가 서버들은 나중에 구현
                # "player_management": {...},
                # "game_state": {...}
            })
            
            print("✅ MCP 클라이언트 초기화 완료")
            return True
            
        except Exception as e:
            print(f"❌ MCP 초기화 실패: {e}")
            return False
    
    async def create_mcp_graph(self):
        """MCP 도구들과 통합된 LangGraph 생성"""
        
        if MCP_AVAILABLE and self.mcp_client:
            return await self._create_real_mcp_graph()
        else:
            return await self._create_simulated_graph()
    
    async def _create_real_mcp_graph(self):
        """실제 MCP 서버와 통합된 그래프"""
        
        try:
            # MCP 세션 시작
            async with self.mcp_client.session("game_rules") as game_rules_session:
                
                # MCP 도구들 로드
                tools = await load_mcp_tools(game_rules_session)
                
                # LLM에 도구 바인딩
                llm_with_tools = self.llm_client.bind_tools(tools)
                
                # 시스템 프롬프트 (기본값)
                system_prompt = """
당신은 테이블 게임 메이트 AI입니다.
BGG에서 게임 정보를 검색하고 규칙을 이해하여 게임을 진행할 수 있습니다.
사용 가능한 도구들을 활용하여 사용자가 원하는 게임을 플레이해주세요.
"""
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("messages")
                ])
                
                chat_llm = prompt_template | llm_with_tools
                
                # 상태 정의
                class State(TypedDict):
                    messages: Annotated[List[AnyMessage], add_messages]
                    game_context: Dict[str, Any]
                
                # 노드 정의
                def chat_node(state: State) -> State:
                    """채팅 노드"""
                    response = chat_llm.invoke({
                        "messages": state["messages"]
                    })
                    state["messages"] = [response]
                    return state
                
                # 그래프 구축
                graph_builder = StateGraph(State)
                graph_builder.add_node("chat_node", chat_node)
                graph_builder.add_node("tool_node", ToolNode(tools=tools))
                
                graph_builder.add_edge(START, "chat_node")
                graph_builder.add_conditional_edges(
                    "chat_node", 
                    tools_condition, 
                    {"tools": "tool_node", "__end__": END}
                )
                graph_builder.add_edge("tool_node", "chat_node")
                
                graph = graph_builder.compile(checkpointer=MemorySaver())
                
                print("✅ 실제 MCP 그래프 생성 완료")
                return graph
                
        except Exception as e:
            print(f"❌ MCP 그래프 생성 실패: {e}")
            return await self._create_simulated_graph()
    
    async def _create_simulated_graph(self):
        """시뮬레이션 그래프 (MCP 없이)"""
        
        # 시뮬레이션 도구들
        simulated_tools = [
            {
                "name": "search_bgg_game",
                "description": "BGG에서 게임 검색",
                "function": self._sim_search_bgg_game
            },
            {
                "name": "get_bgg_game_details", 
                "description": "BGG 게임 상세 정보",
                "function": self._sim_get_game_details
            }
        ]
        
        # 간단한 시뮬레이션 LLM
        async def sim_llm_call(messages):
            last_message = messages[-1] if messages else ""
            
            if "게임" in str(last_message) and "검색" in str(last_message):
                return "BGG에서 게임을 검색하겠습니다."
            elif "시작" in str(last_message):
                return "게임을 시작하겠습니다!"
            else:
                return "어떤 게임을 플레이하고 싶으신가요?"
        
        # 상태 정의
        class State(TypedDict):
            messages: Annotated[List[Any], add_messages]
            game_context: Dict[str, Any]
        
        # 노드
        def sim_chat_node(state: State) -> State:
            response = asyncio.create_task(sim_llm_call(state["messages"]))
            state["messages"] = [{"content": response, "role": "assistant"}]
            return state
        
        # 그래프 구축
        graph_builder = StateGraph(State)
        graph_builder.add_node("chat_node", sim_chat_node)
        graph_builder.add_edge(START, "chat_node")
        graph_builder.add_edge("chat_node", END)
        
        graph = graph_builder.compile()
        
        print("✅ 시뮬레이션 그래프 생성 완료")
        return graph
    
    # 시뮬레이션 함수들
    async def _sim_search_bgg_game(self, game_name: str) -> str:
        """BGG 게임 검색 시뮬레이션"""
        return json.dumps({
            "success": True,
            "game_id": "123456",
            "name": game_name,
            "message": f"'{game_name}' 게임을 찾았습니다 (시뮬레이션)"
        })
    
    async def _sim_get_game_details(self, game_id: str) -> str:
        """게임 상세 정보 시뮬레이션"""
        return json.dumps({
            "success": True,
            "name": "Test Game",
            "min_players": 2,
            "max_players": 4,
            "playing_time": 60,
            "mechanics": ["Card Drafting", "Set Collection"],
            "description": "테스트 게임입니다."
        })
    
    async def start_game_session(self, game_name: str, player_count: int) -> Dict[str, Any]:
        """게임 세션 시작"""
        
        print(f"🎮 '{game_name}' 게임 세션 시작 ({player_count}명)")
        
        # MCP 그래프 생성
        graph = await self.create_mcp_graph()
        
        if graph is None:
            return {"success": False, "error": "그래프 생성 실패"}
        
        # 초기 게임 상태
        initial_state = {
            "messages": [f"{game_name} 게임을 {player_count}명이서 플레이하고 싶습니다."],
            "game_context": {
                "game_name": game_name,
                "player_count": player_count,
                "status": "initializing"
            }
        }
        
        try:
            # 그래프 실행
            config = {"configurable": {"thread_id": f"game_{game_name}_{player_count}"}}
            result = await graph.ainvoke(initial_state, config=config)
            
            print("✅ 게임 세션 초기화 완료")
            
            return {
                "success": True,
                "game_name": game_name,
                "player_count": player_count,
                "graph": graph,
                "config": config,
                "initial_response": result.get("messages", ["게임 준비 완료"])[-1]
            }
            
        except Exception as e:
            print(f"❌ 게임 세션 시작 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_user_input(self, session: Dict, user_input: str) -> str:
        """사용자 입력 처리"""
        
        if not session.get("success"):
            return "게임 세션이 준비되지 않았습니다."
        
        try:
            graph = session["graph"]
            config = session["config"]
            
            # 새 메시지로 그래프 실행
            new_state = {
                "messages": [user_input],
                "game_context": session.get("game_context", {})
            }
            
            result = await graph.ainvoke(new_state, config=config)
            
            # 응답 추출
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    return last_message.get("content", "응답을 처리할 수 없습니다.")
                else:
                    return str(last_message)
            
            return "응답을 받지 못했습니다."
            
        except Exception as e:
            print(f"❌ 사용자 입력 처리 실패: {e}")
            return f"입력 처리 중 오류가 발생했습니다: {str(e)}"


# 사용 예시
async def demo_mcp_game_master():
    """MCP 게임 마스터 데모"""
    
    print("🚀 MCP 게임 마스터 데모 시작")
    
    # 간단한 LLM 클라이언트 (실제로는 OpenAI 등 사용)
    class MockLLMClient:
        def bind_tools(self, tools):
            return self
        
        async def invoke(self, inputs):
            return {"content": "MCP 도구를 사용하여 게임을 처리하겠습니다.", "role": "assistant"}
    
    # 게임 마스터 초기화
    game_master = MCPGameMaster(MockLLMClient())
    
    # MCP 서버 초기화
    await game_master.initialize_mcp_servers()
    
    # 게임 세션 시작
    session = await game_master.start_game_session("Azul", 3)
    
    if session["success"]:
        print(f"✅ 게임 시작: {session['initial_response']}")
        
        # 사용자 입력 처리
        response = await game_master.process_user_input(session, "게임 규칙을 설명해주세요")
        print(f"🤖 AI 응답: {response}")
    else:
        print(f"❌ 게임 시작 실패: {session['error']}")


if __name__ == "__main__":
    asyncio.run(demo_mcp_game_master()) 