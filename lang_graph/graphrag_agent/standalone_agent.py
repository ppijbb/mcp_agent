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
from datetime import datetime
from config import ConfigManager
from agents.graphrag_agent import GraphRAGAgent
from agents.natural_language_agent import NaturalLanguageAgent
from utils.a2a_client import A2AClient, A2AServer, A2AMessage

# Import intelligent agent components with error handling
try:
    from agents.intelligent_agent import IntelligentGraphRAGAgent
    INTELLIGENT_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: IntelligentGraphRAGAgent not available: {e}")
    INTELLIGENT_AGENT_AVAILABLE = False

try:
    from agents.autonomous_behavior import AutonomousBehaviorEngine
    AUTONOMOUS_BEHAVIOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AutonomousBehaviorEngine not available: {e}")
    AUTONOMOUS_BEHAVIOR_AVAILABLE = False


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
        
        # Initialize intelligent agent components if available
        if INTELLIGENT_AGENT_AVAILABLE:
            self.intelligent_agent = IntelligentGraphRAGAgent(self.config.agent)
        else:
            self.intelligent_agent = None
            
        if AUTONOMOUS_BEHAVIOR_AVAILABLE:
            self.autonomous_behavior = AutonomousBehaviorEngine(self.config.agent)
        else:
            self.autonomous_behavior = None
        
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
        """Process a natural language command with intelligent agent capabilities"""
        try:
            if self.intelligent_agent:
                print(f"🧠 지능형 에이전트가 명령을 분석하고 있습니다...")
                
                # Use intelligent agent for deep understanding and autonomous execution
                context = {
                    "user_input": command,
                    "agent_state": "processing",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Process with intelligent agent
                intelligent_result = await self.intelligent_agent.process_user_input(command, context)
                
                if intelligent_result.get("status") == "success":
                    result = intelligent_result.get("result", {})
                    suggestions = intelligent_result.get("suggestions", [])
                    insights = intelligent_result.get("agent_insights", {})
                    
                    print(f"✅ {result.get('message', '지능형 에이전트가 작업을 완료했습니다.')}")
                    
                    # Show agent insights
                    if insights:
                        print(f"🤖 에이전트 인사이트:")
                        for key, value in insights.items():
                            print(f"   - {key}: {value}")
                    
                    # Show proactive suggestions
                    if suggestions:
                        print(f"💡 제안사항:")
                        for suggestion in suggestions:
                            print(f"   {suggestion}")
                    
                    # Execute autonomous actions if any
                    await self._execute_autonomous_actions(context)
                    
                else:
                    # Fallback to traditional processing
                    await self._process_command_traditional(command)
            else:
                # Use traditional processing if intelligent agent not available
                print(f"🤖 기본 에이전트가 명령을 처리하고 있습니다...")
                await self._process_command_traditional(command)
                
        except Exception as e:
            print(f"❌ 명령 처리 실패: {e}")
            # Fallback to traditional processing
            await self._process_command_traditional(command)
    
    async def _process_command_traditional(self, command: str):
        """Traditional command processing as fallback"""
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
    
    async def _execute_autonomous_actions(self, context: Dict[str, Any]):
        """Execute autonomous actions based on current context"""
        try:
            if not self.autonomous_behavior:
                return
                
            # Analyze context for autonomous opportunities
            autonomous_actions = await self.autonomous_behavior.analyze_context_and_act(context)
            
            if autonomous_actions:
                print(f"🤖 {len(autonomous_actions)}개의 자율적 행동을 발견했습니다:")
                
                for action in autonomous_actions[:3]:  # Show top 3 actions
                    print(f"   - {action.description} (우선순위: {action.priority}, 신뢰도: {action.confidence:.2f})")
                
                # Execute the highest priority action
                if autonomous_actions:
                    top_action = autonomous_actions[0]
                    print(f"🚀 자율적 행동 실행: {top_action.description}")
                    
                    execution_result = await self.autonomous_behavior.execute_autonomous_action(top_action)
                    
                    if execution_result.get("success"):
                        print(f"✅ 자율적 행동 완료: {execution_result.get('execution_time', 0):.2f}초")
                    else:
                        print(f"❌ 자율적 행동 실패: {execution_result.get('error', 'Unknown error')}")
                        
        except Exception as e:
            print(f"❌ 자율적 행동 실행 실패: {e}")
    
    async def _show_help(self):
        """Show help information"""
        help_text = """
🧠 지능형 GraphRAG Agent - 사용 가능한 명령어들:

🤖 지능형 에이전트 기능:
  - 자율적 의도 이해 및 해석
  - 능동적 데이터 처리 및 분석
  - 유연한 그래프 생성 및 시각화
  - 지속적인 학습 및 개선
  - 예측적 제안 및 최적화

📊 그래프 생성:
  - "그래프 생성해줘" / "새로운 그래프 만들어줘"
  - "지식 그래프 생성"
  - "tech_companies.csv로 그래프 생성해줘"
  - "scientific_research.csv 파일로 그래프 만들어줘"

🔍 그래프 검색:
  - "Apple에 대해 알려줘"
  - "AI 관련 정보 찾아줘"
  - "그래프에서 Microsoft 검색"

📈 시각화:
  - "그래프 시각화해줘"
  - "그래프를 PNG로 그려줘"

⚡ 최적화:
  - "그래프 최적화해줘"
  - "고품질로 그래프 개선"

📊 상태 확인:
  - "현재 상태 보기"
  - "그래프 정보 알려줘"
  - "에이전트 상태 보기"

🎯 사용자 의도 기반:
  - "회사들의 관계를 중심으로 그래프 생성해줘"
  - "시간 순서대로 이벤트들을 정리해줘"
  - "인물들의 협력 관계를 보여줘"

🧠 지능형 기능:
  - "데이터 품질을 개선해줘"
  - "그래프를 더 정확하게 만들어줘"
  - "사용자 경험을 개선해줘"
  - "시스템을 최적화해줘"

❓ 기타:
  - "help" - 이 도움말 보기
  - "quit" - 종료
  - "status" - 에이전트 상태 확인
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
