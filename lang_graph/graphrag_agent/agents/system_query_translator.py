"""
System Ontology Query Translator

This module provides specialized query translation for system ontology:
- Goal achievement path queries
- Executable task queries
- State transition queries
- Dependency resolution queries
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from config import AgentConfig
from .llm_processor import LLMProcessor
from models.system_ontology import SystemOntology, GoalStatus, TaskStatus


@dataclass
class SystemQueryTranslation:
    """Translation result for system ontology queries"""
    natural_language: str
    cypher_query: str
    query_category: str  # goal_achievement, executable_tasks, state_query, dependency_query
    goal_id: Optional[str] = None
    task_ids: List[str] = None
    confidence: float = 0.5
    reasoning: str = ""
    
    def __post_init__(self):
        if self.task_ids is None:
            self.task_ids = []


class SystemQueryTranslator:
    """
    System ontology query translator
    
    Translates system-related queries to Cypher:
    - Goal achievement queries
    - Executable task queries
    - State and transition queries
    - Dependency queries
    """
    
    def __init__(self, config: AgentConfig, llm_processor: LLMProcessor):
        """
        Initialize system query translator
        
        Args:
            config: Agent configuration
            llm_processor: LLM processor for query generation
        """
        self.config = config
        self.llm_processor = llm_processor
        self.logger = logging.getLogger(__name__)
    
    def translate_system_query(
        self,
        natural_language_query: str,
        system_ontology: SystemOntology,
        context: Optional[Dict[str, Any]] = None
    ) -> SystemQueryTranslation:
        """
        Translate natural language query to system ontology Cypher query
        
        Args:
            natural_language_query: User's natural language query
            system_ontology: System ontology for context
            context: Optional context information
            
        Returns:
            SystemQueryTranslation with Cypher query
        """
        # Classify query type
        query_category = self._classify_query(natural_language_query)
        
        if query_category == "goal_achievement":
            return self._translate_goal_achievement_query(
                natural_language_query, system_ontology, context
            )
        elif query_category == "executable_tasks":
            return self._translate_executable_tasks_query(
                natural_language_query, system_ontology, context
            )
        elif query_category == "state_query":
            return self._translate_state_query(
                natural_language_query, system_ontology, context
            )
        elif query_category == "dependency_query":
            return self._translate_dependency_query(
                natural_language_query, system_ontology, context
            )
        else:
            return self._translate_generic_system_query(
                natural_language_query, system_ontology, context
            )
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        goal_keywords = ["goal", "achieve", "accomplish", "complete", "목표", "달성"]
        task_keywords = ["task", "execute", "run", "perform", "작업", "실행"]
        state_keywords = ["state", "status", "condition", "상태"]
        dependency_keywords = ["depend", "require", "need", "의존", "필요"]
        
        if any(kw in query_lower for kw in goal_keywords):
            return "goal_achievement"
        elif any(kw in query_lower for kw in task_keywords):
            if "executable" in query_lower or "ready" in query_lower or "실행 가능" in query_lower:
                return "executable_tasks"
            return "dependency_query"
        elif any(kw in query_lower for kw in state_keywords):
            return "state_query"
        elif any(kw in query_lower for kw in dependency_keywords):
            return "dependency_query"
        else:
            return "generic"
    
    def _translate_goal_achievement_query(
        self,
        query: str,
        system_ontology: SystemOntology,
        context: Optional[Dict[str, Any]]
    ) -> SystemQueryTranslation:
        """Translate goal achievement query"""
        
        # Extract goal name from query
        goal_name = self._extract_goal_name(query, system_ontology)
        
        if goal_name:
            goal_id = None
            for gid, goal in system_ontology.goals.items():
                if goal.name.lower() == goal_name.lower():
                    goal_id = gid
                    break
            
            if goal_id:
                cypher = f"""
                MATCH (goal:Goal {{id: '{goal_id}'}})
                OPTIONAL MATCH path = (goal)-[:HAS_SUBGOAL*]->(subgoal:Goal)-[:ACHIEVED_BY]->(task:Task)
                WITH goal, collect(DISTINCT task) as tasks
                MATCH (goal)-[:ACHIEVED_BY]->(directTask:Task)
                WITH goal, tasks + collect(DISTINCT directTask) as allTasks
                UNWIND allTasks as task
                OPTIONAL MATCH (task)-[:REQUIRES]->(pre:Precondition)
                OPTIONAL MATCH (task)-[:DEPENDS_ON]->(dep:Task)
                RETURN goal, 
                       collect(DISTINCT task) as tasks,
                       collect(DISTINCT pre) as preconditions,
                       collect(DISTINCT dep) as dependencies
                """
            else:
                cypher = f"""
                MATCH (goal:Goal)
                WHERE goal.name CONTAINS '{goal_name}'
                OPTIONAL MATCH (goal)-[:ACHIEVED_BY]->(task:Task)
                RETURN goal, collect(DISTINCT task) as tasks
                """
        else:
            # General goal achievement query
            cypher = """
            MATCH (goal:Goal {status: 'pending'})
            OPTIONAL MATCH (goal)-[:ACHIEVED_BY]->(task:Task)
            RETURN goal, collect(DISTINCT task) as tasks
            ORDER BY goal.priority DESC
            """
        
        return SystemQueryTranslation(
            natural_language=query,
            cypher_query=cypher.strip(),
            query_category="goal_achievement",
            goal_id=goal_id if 'goal_id' in locals() else None,
            confidence=0.8,
            reasoning=f"Translated goal achievement query for: {goal_name or 'all goals'}"
        )
    
    def _translate_executable_tasks_query(
        self,
        query: str,
        system_ontology: SystemOntology,
        context: Optional[Dict[str, Any]]
    ) -> SystemQueryTranslation:
        """Translate executable tasks query"""
        
        cypher = """
        MATCH (task:Task)
        WHERE task.status IN ['ready', 'pending']
        AND NOT EXISTS {
            (task)-[:REQUIRES]->(pre:Precondition)
            WHERE pre.satisfied = false
        }
        AND NOT EXISTS {
            (task)-[:DEPENDS_ON]->(dep:Task)
            WHERE dep.status <> 'completed'
        }
        AND NOT EXISTS {
            (task)-[:CONSTRAINED_BY]->(constraint:Constraint)
            WHERE constraint.violated = true AND constraint.severity = 'hard'
        }
        RETURN task
        ORDER BY task.priority DESC, task.execution_time ASC
        """
        
        return SystemQueryTranslation(
            natural_language=query,
            cypher_query=cypher.strip(),
            query_category="executable_tasks",
            confidence=0.9,
            reasoning="Translated executable tasks query with precondition and dependency validation"
        )
    
    def _translate_state_query(
        self,
        query: str,
        system_ontology: SystemOntology,
        context: Optional[Dict[str, Any]]
    ) -> SystemQueryTranslation:
        """Translate state query"""
        
        cypher = """
        MATCH (state:State)
        OPTIONAL MATCH (task:Task)-[:TRANSITIONS_TO]->(state)
        RETURN state, collect(DISTINCT task) as transitionTasks
        ORDER BY state.timestamp DESC
        """
        
        return SystemQueryTranslation(
            natural_language=query,
            cypher_query=cypher.strip(),
            query_category="state_query",
            confidence=0.8,
            reasoning="Translated state query with transition information"
        )
    
    def _translate_dependency_query(
        self,
        query: str,
        system_ontology: SystemOntology,
        context: Optional[Dict[str, Any]]
    ) -> SystemQueryTranslation:
        """Translate dependency query"""
        
        # Extract task name if mentioned
        task_name = self._extract_task_name(query, system_ontology)
        
        if task_name:
            cypher = f"""
            MATCH (task:Task)
            WHERE task.name CONTAINS '{task_name}'
            OPTIONAL MATCH (task)-[:DEPENDS_ON*]->(dep:Task)
            OPTIONAL MATCH (dependent:Task)-[:DEPENDS_ON*]->(task)
            RETURN task,
                   collect(DISTINCT dep) as dependencies,
                   collect(DISTINCT dependent) as dependents
            """
        else:
            cypher = """
            MATCH (task:Task)-[r:DEPENDS_ON]->(dep:Task)
            RETURN task, dep, r
            ORDER BY task.priority DESC
            """
        
        return SystemQueryTranslation(
            natural_language=query,
            cypher_query=cypher.strip(),
            query_category="dependency_query",
            confidence=0.8,
            reasoning=f"Translated dependency query for: {task_name or 'all tasks'}"
        )
    
    def _translate_generic_system_query(
        self,
        query: str,
        system_ontology: SystemOntology,
        context: Optional[Dict[str, Any]]
    ) -> SystemQueryTranslation:
        """Translate generic system query using LLM"""
        
        # Get ontology summary for context
        ontology_summary = {
            "goals": [{"id": g.id, "name": g.name} for g in list(system_ontology.goals.values())[:10]],
            "tasks": [{"id": t.id, "name": t.name, "status": t.status.value} for t in list(system_ontology.tasks.values())[:10]]
        }
        
        prompt = f"""
Translate this natural language query to a Neo4j Cypher query for system ontology.

Query: "{query}"

System Ontology Context:
{json.dumps(ontology_summary, indent=2)}

Available Node Types:
- Goal (properties: id, name, description, priority, status)
- Task (properties: id, name, description, status, priority, execution_time, success_rate)
- Precondition (properties: id, description, condition, satisfied)
- Postcondition (properties: id, description, condition, achieved)
- State (properties: id, name, state_type, value, timestamp)
- Resource (properties: id, name, resource_type, availability, capacity)
- Constraint (properties: id, constraint_type, condition, severity, violated)

Available Relationship Types:
- (:Goal)-[:HAS_SUBGOAL]->(:Goal)
- (:Goal)-[:ACHIEVED_BY]->(:Task)
- (:Task)-[:REQUIRES]->(:Precondition)
- (:Task)-[:PRODUCES]->(:Postcondition)
- (:Task)-[:DEPENDS_ON]->(:Task)
- (:Task)-[:CONSUMES]->(:Resource)
- (:Task)-[:TRANSITIONS_TO]->(:State)
- (:State)-[:PRECEDES]->(:State)
- (:Task)-[:CONSTRAINED_BY]->(:Constraint)

Return JSON:
{{
    "cypher_query": "MATCH ... RETURN ...",
    "query_category": "goal_achievement|executable_tasks|state_query|dependency_query|generic",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}
"""
        
        try:
            response = self.llm_processor._call_llm(prompt)
            translation_data = json.loads(response)
            
            return SystemQueryTranslation(
                natural_language=query,
                cypher_query=translation_data.get("cypher_query", ""),
                query_category=translation_data.get("query_category", "generic"),
                confidence=float(translation_data.get("confidence", 0.5)),
                reasoning=translation_data.get("reasoning", "")
            )
        except Exception as e:
            self.logger.error(f"Failed to translate query: {e}")
            return SystemQueryTranslation(
                natural_language=query,
                cypher_query="",
                query_category="generic",
                confidence=0.0,
                reasoning=f"Translation failed: {e}"
            )
    
    def _extract_goal_name(self, query: str, system_ontology: SystemOntology) -> Optional[str]:
        """Extract goal name from query"""
        # Simple extraction - can be improved with NER
        for goal in system_ontology.goals.values():
            if goal.name.lower() in query.lower():
                return goal.name
        
        # Try LLM extraction
        prompt = f"""
Extract the goal name from this query.

Query: "{query}"

Available Goals:
{chr(10).join([f"- {g.name}" for g in list(system_ontology.goals.values())[:20]])}

Return JSON:
{{
    "goal_name": "extracted goal name or null"
}}
"""
        try:
            response = self.llm_processor._call_llm(prompt)
            data = json.loads(response)
            return data.get("goal_name")
        except Exception:
            return None
    
    def _extract_task_name(self, query: str, system_ontology: SystemOntology) -> Optional[str]:
        """Extract task name from query"""
        # Simple extraction
        for task in system_ontology.tasks.values():
            if task.name.lower() in query.lower():
                return task.name
        
        return None
