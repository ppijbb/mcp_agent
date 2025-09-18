"""
LLM-powered Natural Language Agent for GraphRAG

This agent handles natural language commands using LLM for intent understanding
and converts them to structured operations.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from config import AgentConfig
from .llm_processor import LLMProcessor


class CommandType(Enum):
    """Types of natural language commands"""
    CREATE_GRAPH = "create_graph"
    ADD_NODES = "add_nodes"
    ADD_RELATIONS = "add_relations"
    QUERY_GRAPH = "query_graph"
    VISUALIZE_GRAPH = "visualize_graph"
    OPTIMIZE_GRAPH = "optimize_graph"
    SHOW_STATUS = "show_status"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Parsed natural language command"""
    command_type: CommandType
    entities: List[str]
    relations: List[Tuple[str, str]]
    query: str
    parameters: Dict[str, Any]
    confidence: float
    user_intent: str = ""


class NaturalLanguageAgent:
    """LLM-powered natural language interface for graph operations"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_processor = LLMProcessor(config)
        
    def parse_command(self, user_input: str) -> ParsedCommand:
        """Parse natural language input using LLM"""
        try:
            prompt = self._build_command_parsing_prompt(user_input)
            response = self.llm_processor._call_llm(prompt)
            return self._parse_command_response(user_input, response)
        except Exception as e:
            self.logger.error(f"Command parsing failed: {e}")
            return ParsedCommand(
                command_type=CommandType.UNKNOWN,
                entities=[],
                relations=[],
                query=user_input,
                parameters={},
                confidence=0.0,
                user_intent=""
            )
    
    def _build_command_parsing_prompt(self, user_input: str) -> str:
        """Build prompt for command parsing"""
        prompt = f"""
You are an expert at understanding natural language commands for a knowledge graph system. 
Analyze the following user input and determine the intended action.

User Input: "{user_input}"

Please parse this command and return it in the following JSON format:
{{
    "command_type": "create_graph|add_nodes|add_relations|query_graph|visualize_graph|optimize_graph|show_status|help|unknown",
    "entities": ["entity1", "entity2"],
    "relations": [["source1", "target1"], ["source2", "target2"]],
    "parameters": {{
        "format": "png|svg|html",
        "quality": "high|normal|fast",
        "data_file": "filename.csv"
    }},
    "confidence": 0.0-1.0,
    "user_intent": "brief description of what the user wants to achieve"
}}

Available command types:
- create_graph: Create a new knowledge graph from data
- add_nodes: Add specific entities to the graph
- add_relations: Add relationships between entities
- query_graph: Search for information in the graph
- visualize_graph: Generate visual representation of the graph
- optimize_graph: Improve graph structure and quality
- show_status: Display current graph status
- help: Show available commands
- unknown: Command not recognized

Guidelines:
- Extract all relevant entities mentioned in the input
- Identify relationships between entities if mentioned
- Determine appropriate parameters (format, quality, data file)
- Provide confidence score based on clarity of the command
- Infer user intent even if not explicitly stated
- Support both Korean and English inputs
"""
        return prompt
    
    def _parse_command_response(self, user_input: str, response: str) -> ParsedCommand:
        """Parse LLM response for command"""
        try:
            data = json.loads(response)
            
            # Convert command type string to enum
            command_type_str = data.get("command_type", "unknown")
            command_type = CommandType.UNKNOWN
            for cmd_type in CommandType:
                if cmd_type.value == command_type_str:
                    command_type = cmd_type
                    break
            
            # Extract entities
            entities = data.get("entities", [])
            if not isinstance(entities, list):
                entities = []
            
            # Extract relations
            relations = data.get("relations", [])
            if not isinstance(relations, list):
                relations = []
            
            # Convert relations to tuples
            relation_tuples = []
            for rel in relations:
                if isinstance(rel, list) and len(rel) == 2:
                    relation_tuples.append((rel[0], rel[1]))
            
            # Extract parameters
            parameters = data.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}
            
            # Extract confidence and user intent
            confidence = data.get("confidence", 0.5)
            user_intent = data.get("user_intent", "")
            
            return ParsedCommand(
                command_type=command_type,
                entities=entities,
                relations=relation_tuples,
                query=user_input,
                parameters=parameters,
                confidence=confidence,
                user_intent=user_intent
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse command response: {e}")
            return ParsedCommand(
                command_type=CommandType.UNKNOWN,
                entities=[],
                relations=[],
                query=user_input,
                parameters={},
                confidence=0.0,
                user_intent=""
            )
    
    def execute_command(self, parsed_command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parsed command and return result"""
        try:
            if parsed_command.command_type == CommandType.CREATE_GRAPH:
                return self._handle_create_graph(parsed_command, graph_state)
            elif parsed_command.command_type == CommandType.ADD_NODES:
                return self._handle_add_nodes(parsed_command, graph_state)
            elif parsed_command.command_type == CommandType.ADD_RELATIONS:
                return self._handle_add_relations(parsed_command, graph_state)
            elif parsed_command.command_type == CommandType.QUERY_GRAPH:
                return self._handle_query_graph(parsed_command, graph_state)
            elif parsed_command.command_type == CommandType.VISUALIZE_GRAPH:
                return self._handle_visualize_graph(parsed_command, graph_state)
            elif parsed_command.command_type == CommandType.OPTIMIZE_GRAPH:
                return self._handle_optimize_graph(parsed_command, graph_state)
            elif parsed_command.command_type == CommandType.SHOW_STATUS:
                return self._handle_show_status(parsed_command, graph_state)
            elif parsed_command.command_type == CommandType.HELP:
                return self._handle_help(parsed_command, graph_state)
            else:
                return {
                    "status": "error",
                    "message": f"알 수 없는 명령어입니다: {parsed_command.query}",
                    "suggestions": self._get_suggestions(parsed_command.query)
                }
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {
                "status": "error",
                "message": f"명령 실행 중 오류가 발생했습니다: {str(e)}"
            }
    
    def _handle_create_graph(self, command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph creation command"""
        # Check if data file is specified in parameters or entities
        data_file = command.parameters.get("data_file")
        if not data_file and command.entities:
            # Look for file names in entities
            for entity in command.entities:
                if entity.lower().endswith('.csv'):
                    data_file = entity
                    break
        
        return {
            "status": "completed",
            "message": f"새로운 그래프를 생성합니다. {'데이터 파일: ' + data_file if data_file else '기본 샘플 데이터 사용'}",
            "action": "create_graph",
            "entities": command.entities,
            "data_file": data_file,
            "user_intent": command.user_intent
        }
    
    def _handle_add_nodes(self, command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add nodes command"""
        return {
            "status": "completed",
            "message": f"다음 노드들을 그래프에 추가합니다: {', '.join(command.entities)}",
            "action": "add_nodes",
            "entities": command.entities,
            "user_intent": command.user_intent
        }
    
    def _handle_add_relations(self, command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add relations command"""
        return {
            "status": "completed",
            "message": f"다음 관계들을 그래프에 추가합니다: {[f'{r[0]} - {r[1]}' for r in command.relations]}",
            "action": "add_relations",
            "relations": command.relations,
            "user_intent": command.user_intent
        }
    
    def _handle_query_graph(self, command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph query command"""
        return {
            "status": "completed",
            "message": f"그래프에서 '{', '.join(command.entities)}'에 대한 정보를 검색합니다.",
            "action": "query_graph",
            "query": command.query,
            "entities": command.entities,
            "user_intent": command.user_intent
        }
    
    def _handle_visualize_graph(self, command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph visualization command"""
        format_pref = command.parameters.get("format", "png")
        return {
            "status": "completed",
            "message": f"그래프를 {format_pref} 형식으로 시각화합니다.",
            "action": "visualize_graph",
            "format": format_pref,
            "user_intent": command.user_intent
        }
    
    def _handle_optimize_graph(self, command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph optimization command"""
        quality = command.parameters.get("quality", "normal")
        return {
            "status": "completed",
            "message": f"그래프를 {quality} 품질로 최적화합니다.",
            "action": "optimize_graph",
            "quality": quality,
            "user_intent": command.user_intent
        }
    
    def _handle_show_status(self, command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle show status command"""
        return {
            "status": "completed",
            "message": "현재 그래프 상태를 표시합니다.",
            "action": "show_status",
            "graph_info": {
                "nodes": graph_state.get("node_count", 0),
                "edges": graph_state.get("edge_count", 0),
                "last_updated": graph_state.get("last_updated", "Unknown")
            },
            "user_intent": command.user_intent
        }
    
    def _handle_help(self, command: ParsedCommand, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help command"""
        help_text = """
사용 가능한 명령어들:

📊 그래프 생성:
  - "그래프 생성해줘" / "새로운 그래프 만들어줘"
  - "지식 그래프 생성"
  - "tech_companies.csv로 그래프 생성해줘"
  - "scientific_research.csv 파일로 그래프 만들어줘"

➕ 노드 추가:
  - "Apple을 그래프에 추가해줘"
  - "Microsoft 노드 추가"

🔗 관계 추가:
  - "Apple과 Microsoft의 관계 추가"
  - "Google은 Alphabet의 자회사이다"

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

📁 데이터 파일:
  - CSV 파일만 바꿔서 새로운 그래프 생성 가능
  - 자동으로 컬럼 구조 감지
  - 다양한 형식 지원 (tech_companies.csv, news_articles.csv 등)

❓ 도움말:
  - "도움말" / "명령어 보기"

💡 사용자 의도 기반 생성:
  - "회사들의 관계를 중심으로 그래프 만들어줘"
  - "시간 순서대로 이벤트들을 정리해줘"
  - "인물들의 협력 관계를 보여줘"
        """
        
        return {
            "status": "completed",
            "message": "도움말을 표시합니다.",
            "action": "help",
            "help_text": help_text
        }
    
    def _get_suggestions(self, user_input: str) -> List[str]:
        """Get command suggestions for unknown input"""
        suggestions = [
            "그래프 생성해줘",
            "Apple을 그래프에 추가해줘",
            "Apple에 대해 알려줘",
            "그래프 시각화해줘",
            "현재 상태 보기"
        ]
        return suggestions