#!/usr/bin/env python3
"""
Interactive CLI for GraphRAG Agent

This provides a conversational interface for graph operations.
Users can interact with the graph using natural language commands.
"""

import asyncio
import sys
import os
from typing import Dict, Any, Optional
from config import ConfigManager
from agents.natural_language_agent import NaturalLanguageAgent
from agents.graphrag_agent import GraphRAGAgent


class InteractiveCLI:
    """Interactive command-line interface for GraphRAG Agent"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = None
        self.graph_agent = None
        self.nl_agent = None
        self.graph_state = {
            "node_count": 0,
            "edge_count": 0,
            "last_updated": None,
            "knowledge_graph": None
        }
        self.running = True
        
    async def initialize(self):
        """Initialize the CLI and load configuration"""
        try:
            print("🚀 GraphRAG Agent 초기화 중...")
            
            # Load configuration
            self.config = self.config_manager.load_config()
            if not self.config_manager.validate_config(self.config):
                print("❌ 설정 검증 실패")
                return False
            
            # Initialize agents
            self.graph_agent = GraphRAGAgent(self.config)
            self.nl_agent = NaturalLanguageAgent(self.config)
            
            print("✅ GraphRAG Agent 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            return False
    
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*60)
        print("🧠 GraphRAG Agent - 자연어 그래프 관리 시스템")
        print("="*60)
        print("자연어로 그래프를 생성, 수정, 조회할 수 있습니다.")
        print("'도움말' 또는 'help'를 입력하면 사용법을 볼 수 있습니다.")
        print("'종료' 또는 'quit'를 입력하면 프로그램을 종료합니다.")
        print("="*60 + "\n")
    
    def print_prompt(self):
        """Print command prompt"""
        print("💬 ", end="", flush=True)
    
    def print_response(self, response: Dict[str, Any]):
        """Print formatted response"""
        if response.get("status") == "completed":
            print(f"✅ {response.get('message', '완료되었습니다.')}")
            
            # Print additional information if available
            if "graph_info" in response:
                info = response["graph_info"]
                print(f"   📊 노드 수: {info.get('nodes', 0)}")
                print(f"   🔗 엣지 수: {info.get('edges', 0)}")
                print(f"   🕒 마지막 업데이트: {info.get('last_updated', 'Unknown')}")
            
            if "help_text" in response:
                print(response["help_text"])
                
        elif response.get("status") == "error":
            print(f"❌ {response.get('message', '오류가 발생했습니다.')}")
            
            if "suggestions" in response:
                print("💡 제안 명령어:")
                for suggestion in response["suggestions"]:
                    print(f"   - {suggestion}")
        else:
            print(f"ℹ️  {response.get('message', '처리 중...')}")
    
    async def process_command(self, user_input: str) -> bool:
        """Process user command and return whether to continue"""
        if user_input.lower() in ['종료', 'quit', 'exit', 'q']:
            print("👋 GraphRAG Agent를 종료합니다. 안녕히 가세요!")
            return False
        
        if not user_input.strip():
            return True
        
        try:
            # Parse natural language command
            parsed_command = self.nl_agent.parse_command(user_input)
            
            # Execute command
            response = self.nl_agent.execute_command(parsed_command, self.graph_state)
            
            # Print response
            self.print_response(response)
            
            # Update graph state if needed
            await self._update_graph_state(response)
            
        except Exception as e:
            print(f"❌ 명령 처리 중 오류: {e}")
        
        return True
    
    async def _update_graph_state(self, response: Dict[str, Any]):
        """Update graph state based on response"""
        action = response.get("action")
        
        if action == "create_graph":
            # Initialize new graph
            self.graph_state["node_count"] = 0
            self.graph_state["edge_count"] = 0
            self.graph_state["last_updated"] = "방금 전"
            
        elif action == "add_nodes":
            # Add nodes to graph
            entities = response.get("entities", [])
            self.graph_state["node_count"] += len(entities)
            self.graph_state["last_updated"] = "방금 전"
            
        elif action == "add_relations":
            # Add relations to graph
            relations = response.get("relations", [])
            self.graph_state["edge_count"] += len(relations)
            self.graph_state["last_updated"] = "방금 전"
            
        elif action == "query_graph":
            # Query doesn't change graph state
            pass
            
        elif action == "visualize_graph":
            # Visualization doesn't change graph state
            pass
            
        elif action == "optimize_graph":
            # Optimization might change graph state
            self.graph_state["last_updated"] = "방금 전"
    
    async def run(self):
        """Run the interactive CLI"""
        # Initialize
        if not await self.initialize():
            return
        
        # Print welcome message
        self.print_welcome()
        
        # Main loop
        while self.running:
            try:
                self.print_prompt()
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                self.running = await self.process_command(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 GraphRAG Agent를 종료합니다. 안녕히 가세요!")
                break
            except EOFError:
                print("\n\n👋 GraphRAG Agent를 종료합니다. 안녕히 가세요!")
                break
            except Exception as e:
                print(f"\n❌ 예상치 못한 오류: {e}")
                print("계속하려면 Enter를 누르세요...")
                try:
                    input()
                except:
                    break


async def main():
    """Main entry point"""
    cli = InteractiveCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
