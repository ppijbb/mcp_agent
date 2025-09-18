"""
Standalone GraphRAG Agent

This module provides a standalone agent that can run independently
without requiring an API server. It supports A2A protocol for
inter-agent communication when needed.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any, Optional
from config import ConfigManager
from agents.graphrag_agent import GraphRAGAgent
from agents.natural_language_agent import NaturalLanguageAgent
from utils.a2a_client import A2AClient, A2AServer, A2AMessage


class StandaloneGraphRAGAgent:
    """Standalone GraphRAG Agent with A2A protocol support"""
    
    def __init__(self, config_path: str = None, agent_id: str = "graphrag_agent"):
        self.agent_id = agent_id
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        self.config = config_manager.load_config()
        
        if not self.config:
            raise ValueError("Failed to load configuration")
        
        # Initialize components
        self.graphrag_agent = GraphRAGAgent(self.config)
        self.nl_agent = NaturalLanguageAgent(self.config.agent)
        
        # Initialize A2A client and server
        self.a2a_client = A2AClient(agent_id)
        self.a2a_server = A2AServer(agent_id)
        
        # Register A2A message handlers
        self._register_a2a_handlers()
        
        self.logger.info(f"Standalone GraphRAG Agent {agent_id} initialized")
    
    def _register_a2a_handlers(self):
        """Register A2A protocol message handlers"""
        self.a2a_server.register_handler('graph_request', self._handle_graph_request)
        self.a2a_server.register_handler('query_request', self._handle_query_request)
        self.a2a_server.register_handler('status_request', self._handle_status_request)
        self.a2a_server.register_handler('visualization_request', self._handle_visualization_request)
        self.a2a_server.register_handler('heartbeat', self._handle_heartbeat)
    
    async def _handle_graph_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle graph operation requests"""
        try:
            payload = message.payload
            operation = payload.get('operation', 'create')
            graph_data = payload.get('graph_data', {})
            
            # Update config with provided data
            if 'data_file' in graph_data:
                self.config.data_file = graph_data['data_file']
            if 'user_intent' in graph_data:
                self.config.user_intent = graph_data['user_intent']
            
            # Execute graph operation
            success = await self.graphrag_agent.run()
            
            return {
                'status': 'success' if success else 'error',
                'operation': operation,
                'message_id': message.message_id,
                'result': {
                    'success': success,
                    'graph_path': getattr(self.config, 'graph_path', None),
                    'output_path': getattr(self.config, 'output_path', None)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Graph request handling failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message_id': message.message_id
            }
    
    async def _handle_query_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle query requests"""
        try:
            payload = message.payload
            query = payload.get('query', '')
            context = payload.get('context', {})
            
            # Parse natural language query
            parsed_command = self.nl_agent.parse_command(query)
            result = self.nl_agent.execute_command(parsed_command, context)
            
            return {
                'status': 'success',
                'query': query,
                'message_id': message.message_id,
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"Query request handling failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message_id': message.message_id
            }
    
    async def _handle_status_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle status requests"""
        try:
            status_info = {
                'agent_id': self.agent_id,
                'status': 'running',
                'config_loaded': self.config is not None,
                'components_initialized': {
                    'graphrag_agent': self.graphrag_agent is not None,
                    'nl_agent': self.nl_agent is not None,
                    'a2a_client': self.a2a_client is not None,
                    'a2a_server': self.a2a_server is not None
                }
            }
            
            return {
                'status': 'success',
                'message_id': message.message_id,
                'result': status_info
            }
            
        except Exception as e:
            self.logger.error(f"Status request handling failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message_id': message.message_id
            }
    
    async def _handle_visualization_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle visualization requests"""
        try:
            payload = message.payload
            graph_data = payload.get('graph_data', {})
            format = payload.get('format', 'png')
            
            # Update config for visualization
            if 'graph_path' in graph_data:
                self.config.graph_path = graph_data['graph_path']
            if 'output_path' in graph_data:
                self.config.output_path = graph_data['output_path']
            
            # Execute visualization
            success = await self.graphrag_agent.run()
            
            return {
                'status': 'success' if success else 'error',
                'format': format,
                'message_id': message.message_id,
                'result': {
                    'success': success,
                    'output_path': getattr(self.config, 'output_path', None)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Visualization request handling failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message_id': message.message_id
            }
    
    async def _handle_heartbeat(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle heartbeat messages"""
        return {
            'status': 'success',
            'message_id': message.message_id,
            'result': {
                'agent_id': self.agent_id,
                'timestamp': message.timestamp,
                'alive': True
            }
        }
    
    async def run_standalone(self, command: str = None, interactive: bool = False):
        """Run the agent in standalone mode"""
        try:
            if interactive:
                await self._run_interactive_mode()
            elif command:
                await self._run_single_command(command)
            else:
                await self._run_default_mode()
                
        except Exception as e:
            self.logger.error(f"Standalone execution failed: {e}")
            return False
        
        return True
    
    async def _run_interactive_mode(self):
        """Run in interactive mode"""
        print(f"🤖 GraphRAG Agent {self.agent_id} - Interactive Mode")
        print("Type 'help' for available commands, 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    await self._show_help()
                    continue
                
                if not user_input:
                    continue
                
                # Process command
                await self._process_command(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def _run_single_command(self, command: str):
        """Run a single command"""
        print(f"🤖 GraphRAG Agent {self.agent_id} - Single Command Mode")
        await self._process_command(command)
    
    async def _run_default_mode(self):
        """Run in default mode"""
        print(f"🤖 GraphRAG Agent {self.agent_id} - Default Mode")
        success = await self.graphrag_agent.run()
        if success:
            print("✅ GraphRAG operation completed successfully")
        else:
            print("❌ GraphRAG operation failed")
    
    async def _process_command(self, command: str):
        """Process a natural language command"""
        try:
            # Parse command
            parsed_command = self.nl_agent.parse_command(command)
            
            if parsed_command.command_type.value == "unknown":
                print(f"❌ 명령어를 이해할 수 없습니다: {command}")
                return
            
            # Execute command
            result = self.nl_agent.execute_command(parsed_command, {})
            
            if result.get("status") == "completed":
                print(f"✅ {result.get('message', '완료되었습니다.')}")
                
                # If it's a graph creation command, actually create the graph
                if result.get("action") == "create_graph":
                    print(f"🎯 사용자 의도: {parsed_command.user_intent}")
                    success = await self.graphrag_agent.run()
                    if success:
                        print("✅ 그래프 생성 완료")
                    else:
                        print("❌ 그래프 생성 실패")
            else:
                print(f"❌ {result.get('message', '오류가 발생했습니다.')}")
                
        except Exception as e:
            print(f"❌ 명령 처리 실패: {e}")
    
    async def _show_help(self):
        """Show help information"""
        help_text = """
🤖 GraphRAG Agent - 사용 가능한 명령어들:

📊 그래프 생성:
  - "그래프 생성해줘" / "새로운 그래프 만들어줘"
  - "지식 그래프 생성"
  - "tech_companies.csv로 그래프 생성해줘"

🔍 그래프 검색:
  - "Apple에 대해 알려줘"
  - "AI 관련 정보 찾아줘"

📈 시각화:
  - "그래프 시각화해줘"
  - "그래프를 PNG로 그려줘"

⚡ 최적화:
  - "그래프 최적화해줘"

📊 상태 확인:
  - "현재 상태 보기"
  - "그래프 정보 알려줘"

🎯 사용자 의도 기반:
  - "회사들의 관계를 중심으로 그래프 생성해줘"
  - "시간 순서대로 이벤트들을 정리해줘"
  - "인물들의 협력 관계를 보여줘"

❓ 기타:
  - "help" - 이 도움말 보기
  - "quit" - 종료
        """
        print(help_text)
    
    def send_a2a_message(self, target_agent: str, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send A2A message to another agent"""
        return self.a2a_client.send_message(target_agent, message_type, payload)
    
    def start_a2a_server(self):
        """Start A2A server for inter-agent communication"""
        return self.a2a_server.start_server()


async def main():
    """Main entry point for standalone agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone GraphRAG Agent')
    parser.add_argument('--config', '-c', type=str, help='Configuration file path')
    parser.add_argument('--agent-id', '-a', type=str, default='graphrag_agent', help='Agent ID')
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive mode')
    parser.add_argument('--command', type=str, help='Execute single command')
    parser.add_argument('--a2a-server', action='store_true', help='Start A2A server')
    
    args = parser.parse_args()
    
    try:
        # Initialize standalone agent
        agent = StandaloneGraphRAGAgent(args.config, args.agent_id)
        
        # Start A2A server if requested
        if args.a2a_server:
            agent.start_a2a_server()
        
        # Run agent
        success = await agent.run_standalone(
            command=args.command,
            interactive=args.interactive
        )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
