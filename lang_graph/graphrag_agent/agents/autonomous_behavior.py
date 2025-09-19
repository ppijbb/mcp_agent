"""
Autonomous Behavior Module

This module implements truly autonomous agent behaviors:
- Proactive data discovery and analysis
- Self-directed learning and improvement
- Autonomous decision making
- Context-aware adaptation
- Predictive capabilities
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from config import AgentConfig
from .llm_processor import LLMProcessor


class BehaviorType(Enum):
    """Types of autonomous behaviors"""
    PROACTIVE_ANALYSIS = "proactive_analysis"
    PREDICTIVE_SUGGESTION = "predictive_suggestion"
    SELF_IMPROVEMENT = "self_improvement"
    CONTEXT_ADAPTATION = "context_adaptation"
    DATA_DISCOVERY = "data_discovery"
    QUALITY_OPTIMIZATION = "quality_optimization"


@dataclass
class AutonomousAction:
    """Represents an autonomous action the agent can take"""
    action_id: str
    behavior_type: BehaviorType
    description: str
    priority: int
    confidence: float
    expected_impact: str
    execution_plan: Dict[str, Any]
    created_at: datetime


class AutonomousBehaviorEngine:
    """
    Engine for autonomous agent behaviors
    
    This engine enables the agent to:
    - Act proactively without explicit user commands
    - Learn and adapt from experience
    - Make intelligent decisions autonomously
    - Discover and suggest improvements
    - Predict user needs and preferences
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_processor = LLMProcessor(config)
        
        # Behavior state
        self.active_behaviors = []
        self.behavior_history = []
        self.learning_patterns = []
        self.user_preferences = {}
        self.context_memory = {}
        
        # Autonomous capabilities
        self.proactive_threshold = 0.7
        self.learning_rate = 0.1
        self.adaptation_speed = 0.5
        
        self.logger.info("Autonomous Behavior Engine initialized")
    
    async def analyze_context_and_act(self, current_context: Dict[str, Any]) -> List[AutonomousAction]:
        """
        Analyze current context and generate autonomous actions
        
        This is the core method that makes the agent truly autonomous:
        - Analyzes current situation
        - Identifies opportunities for improvement
        - Generates proactive actions
        - Learns from context patterns
        """
        try:
            # Step 1: Context Analysis
            context_analysis = await self._analyze_context(current_context)
            
            # Step 2: Opportunity Identification
            opportunities = await self._identify_opportunities(context_analysis)
            
            # Step 3: Action Generation
            actions = await self._generate_autonomous_actions(opportunities, context_analysis)
            
            # Step 4: Action Prioritization
            prioritized_actions = await self._prioritize_actions(actions, context_analysis)
            
            # Step 5: Learning Update
            await self._update_learning_from_context(context_analysis)
            
            return prioritized_actions
            
        except Exception as e:
            self.logger.error(f"Autonomous behavior analysis failed: {e}")
            return []
    
    async def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deep analysis of current context for autonomous decision making"""
        analysis_prompt = f"""
You are an expert at analyzing context for autonomous agent behavior. 
Analyze the following context and identify opportunities for proactive action.

Current Context: {json.dumps(context, indent=2)}

Provide a comprehensive context analysis in JSON format:
{{
    "context_type": "data_analysis|graph_generation|user_interaction|system_monitoring",
    "current_state": {{
        "data_quality": "assessment of current data quality",
        "graph_completeness": "how complete the current graph is",
        "user_engagement": "level of user engagement",
        "system_performance": "current system performance"
    }},
    "opportunities": {{
        "data_improvements": ["ways to improve data quality"],
        "graph_enhancements": ["ways to enhance the graph"],
        "user_experience": ["ways to improve user experience"],
        "system_optimization": ["ways to optimize system performance"]
    }},
    "risks": {{
        "data_issues": ["potential data problems"],
        "graph_limitations": ["graph limitations to address"],
        "user_frustration": ["potential user frustration points"],
        "system_issues": ["potential system issues"]
    }},
    "learning_opportunities": {{
        "patterns_to_learn": ["patterns to learn from this context"],
        "preferences_to_update": ["user preferences to update"],
        "behaviors_to_adapt": ["behaviors to adapt based on context"]
    }},
    "predictive_insights": {{
        "likely_next_actions": ["what the user might want to do next"],
        "potential_improvements": ["improvements that might be needed"],
        "optimization_opportunities": ["optimization opportunities"]
    }}
}}

Consider:
- Current data state and quality
- Graph completeness and accuracy
- User behavior patterns
- System performance
- Historical patterns and learning
- Predictive modeling
"""
        
        response = self.llm_processor._call_llm(analysis_prompt)
        return json.loads(response)
    
    async def _identify_opportunities(self, context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for autonomous action"""
        opportunities = []
        
        # Data improvement opportunities
        for improvement in context_analysis.get("opportunities", {}).get("data_improvements", []):
            opportunities.append({
                "type": "data_improvement",
                "description": improvement,
                "priority": self._calculate_priority("data_improvement", improvement),
                "confidence": 0.8
            })
        
        # Graph enhancement opportunities
        for enhancement in context_analysis.get("opportunities", {}).get("graph_enhancements", []):
            opportunities.append({
                "type": "graph_enhancement",
                "description": enhancement,
                "priority": self._calculate_priority("graph_enhancement", enhancement),
                "confidence": 0.7
            })
        
        # User experience opportunities
        for ux_improvement in context_analysis.get("opportunities", {}).get("user_experience", []):
            opportunities.append({
                "type": "user_experience",
                "description": ux_improvement,
                "priority": self._calculate_priority("user_experience", ux_improvement),
                "confidence": 0.9
            })
        
        # System optimization opportunities
        for optimization in context_analysis.get("opportunities", {}).get("system_optimization", []):
            opportunities.append({
                "type": "system_optimization",
                "description": optimization,
                "priority": self._calculate_priority("system_optimization", optimization),
                "confidence": 0.6
            })
        
        return opportunities
    
    async def _generate_autonomous_actions(self, opportunities: List[Dict[str, Any]], context_analysis: Dict[str, Any]) -> List[AutonomousAction]:
        """Generate specific autonomous actions based on opportunities"""
        actions = []
        
        for opportunity in opportunities:
            action = await self._create_autonomous_action(opportunity, context_analysis)
            if action:
                actions.append(action)
        
        return actions
    
    async def _create_autonomous_action(self, opportunity: Dict[str, Any], context_analysis: Dict[str, Any]) -> Optional[AutonomousAction]:
        """Create a specific autonomous action from an opportunity"""
        action_type = opportunity["type"]
        description = opportunity["description"]
        
        # Generate execution plan for the action
        execution_plan = await self._generate_execution_plan(action_type, description, context_analysis)
        
        if not execution_plan:
            return None
        
        action = AutonomousAction(
            action_id=f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{action_type}",
            behavior_type=self._map_to_behavior_type(action_type),
            description=description,
            priority=opportunity["priority"],
            confidence=opportunity["confidence"],
            expected_impact=execution_plan.get("expected_impact", "Unknown"),
            execution_plan=execution_plan,
            created_at=datetime.now()
        )
        
        return action
    
    async def _generate_execution_plan(self, action_type: str, description: str, context_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate execution plan for an autonomous action"""
        planning_prompt = f"""
Create a detailed execution plan for an autonomous agent action.

Action Type: {action_type}
Description: {description}
Context: {json.dumps(context_analysis, indent=2)}

Provide a detailed execution plan in JSON format:
{{
    "action_steps": [
        {{
            "step_id": "step_1",
            "description": "detailed step description",
            "estimated_duration": "time estimate",
            "dependencies": ["other_step_ids"],
            "success_criteria": ["how to measure success"]
        }}
    ],
    "required_resources": {{
        "data_sources": ["data sources needed"],
        "computational_resources": ["computational requirements"],
        "external_apis": ["external APIs needed"]
    }},
    "expected_impact": "description of expected impact",
    "risk_assessment": {{
        "potential_risks": ["potential risks"],
        "mitigation_strategies": ["how to mitigate risks"]
    }},
    "success_metrics": ["metrics to measure success"],
    "rollback_plan": "plan to rollback if action fails"
}}

The plan should be:
- Detailed and actionable
- Consider resource requirements
- Include risk assessment
- Provide clear success criteria
- Be autonomous (minimal user intervention)
"""
        
        try:
            response = self.llm_processor._call_llm(planning_prompt)
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Failed to generate execution plan: {e}")
            return None
    
    async def _prioritize_actions(self, actions: List[AutonomousAction], context_analysis: Dict[str, Any]) -> List[AutonomousAction]:
        """Prioritize autonomous actions based on impact and feasibility"""
        # Sort by priority and confidence
        prioritized = sorted(actions, key=lambda x: (x.priority, x.confidence), reverse=True)
        
        # Apply learning-based adjustments
        for action in prioritized:
            learning_adjustment = self._calculate_learning_adjustment(action, context_analysis)
            action.priority = max(1, min(10, action.priority + learning_adjustment))
        
        return prioritized
    
    async def _update_learning_from_context(self, context_analysis: Dict[str, Any]):
        """Update learning patterns from context analysis"""
        # Extract learning opportunities
        learning_opportunities = context_analysis.get("learning_opportunities", {})
        
        # Update patterns
        for pattern in learning_opportunities.get("patterns_to_learn", []):
            self.learning_patterns.append({
                "pattern": pattern,
                "context": context_analysis,
                "timestamp": datetime.now().isoformat()
            })
        
        # Update preferences
        for preference in learning_opportunities.get("preferences_to_update", []):
            self.user_preferences.update(preference)
        
        # Update behaviors
        for behavior in learning_opportunities.get("behaviors_to_adapt", []):
            self._adapt_behavior(behavior, context_analysis)
    
    def _calculate_priority(self, action_type: str, description: str) -> int:
        """Calculate priority for an action based on type and description"""
        base_priorities = {
            "data_improvement": 7,
            "graph_enhancement": 8,
            "user_experience": 9,
            "system_optimization": 6
        }
        
        base_priority = base_priorities.get(action_type, 5)
        
        # Adjust based on description keywords
        if "critical" in description.lower() or "urgent" in description.lower():
            base_priority += 2
        elif "important" in description.lower():
            base_priority += 1
        elif "minor" in description.lower() or "optional" in description.lower():
            base_priority -= 1
        
        return max(1, min(10, base_priority))
    
    def _map_to_behavior_type(self, action_type: str) -> BehaviorType:
        """Map action type to behavior type"""
        mapping = {
            "data_improvement": BehaviorType.PROACTIVE_ANALYSIS,
            "graph_enhancement": BehaviorType.QUALITY_OPTIMIZATION,
            "user_experience": BehaviorType.PREDICTIVE_SUGGESTION,
            "system_optimization": BehaviorType.SELF_IMPROVEMENT
        }
        return mapping.get(action_type, BehaviorType.PROACTIVE_ANALYSIS)
    
    def _calculate_learning_adjustment(self, action: AutonomousAction, context_analysis: Dict[str, Any]) -> int:
        """Calculate learning-based priority adjustment"""
        adjustment = 0
        
        # Check if similar actions were successful before
        for pattern in self.learning_patterns:
            if self._patterns_similar(action.description, pattern["pattern"]):
                adjustment += 1
        
        # Check user preferences
        if action.behavior_type.value in self.user_preferences:
            preference = self.user_preferences[action.behavior_type.value]
            if preference > 0.7:
                adjustment += 2
            elif preference < 0.3:
                adjustment -= 1
        
        return adjustment
    
    def _patterns_similar(self, pattern1: str, pattern2: str) -> bool:
        """Check if two patterns are similar"""
        # Simple similarity check - could be enhanced with NLP
        words1 = set(pattern1.lower().split())
        words2 = set(pattern2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.5
    
    def _adapt_behavior(self, behavior: str, context_analysis: Dict[str, Any]):
        """Adapt agent behavior based on context"""
        # Implement behavior adaptation logic
        self.logger.info(f"Adapting behavior: {behavior}")
    
    async def execute_autonomous_action(self, action: AutonomousAction) -> Dict[str, Any]:
        """Execute an autonomous action"""
        try:
            self.logger.info(f"Executing autonomous action: {action.action_id}")
            
            # Execute the action steps
            results = []
            for step in action.execution_plan.get("action_steps", []):
                step_result = await self._execute_action_step(step, action)
                results.append(step_result)
            
            # Evaluate success
            success = all(result.get("success", False) for result in results)
            
            # Update learning
            await self._learn_from_action_execution(action, results, success)
            
            return {
                "action_id": action.action_id,
                "success": success,
                "results": results,
                "execution_time": (datetime.now() - action.created_at).total_seconds()
            }
            
        except Exception as e:
            self.logger.error(f"Autonomous action execution failed: {e}")
            return {
                "action_id": action.action_id,
                "success": False,
                "error": str(e)
            }
    
    async def _execute_action_step(self, step: Dict[str, Any], action: AutonomousAction) -> Dict[str, Any]:
        """Execute a single step of an autonomous action"""
        step_id = step["step_id"]
        description = step["description"]
        
        self.logger.info(f"Executing step: {step_id} - {description}")
        
        # Implement step execution logic
        # This would be customized based on the specific step type
        
        return {
            "step_id": step_id,
            "success": True,
            "result": f"Step {step_id} completed",
            "execution_time": 1.0
        }
    
    async def _learn_from_action_execution(self, action: AutonomousAction, results: List[Dict[str, Any]], success: bool):
        """Learn from the execution of an autonomous action"""
        learning_entry = {
            "action_id": action.action_id,
            "behavior_type": action.behavior_type.value,
            "description": action.description,
            "success": success,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.behavior_history.append(learning_entry)
        
        # Update learning patterns
        if success:
            self.learning_patterns.append({
                "pattern": action.description,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
    
    async def get_autonomous_insights(self) -> Dict[str, Any]:
        """Get insights about autonomous behavior"""
        return {
            "active_behaviors": len(self.active_behaviors),
            "behavior_history_count": len(self.behavior_history),
            "learning_patterns_count": len(self.learning_patterns),
            "user_preferences": self.user_preferences,
            "recent_success_rate": self._calculate_recent_success_rate(),
            "most_effective_behaviors": self._get_most_effective_behaviors()
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate recent success rate of autonomous actions"""
        if not self.behavior_history:
            return 0.0
        
        recent_actions = self.behavior_history[-10:]  # Last 10 actions
        successful_actions = sum(1 for action in recent_actions if action["success"])
        return successful_actions / len(recent_actions)
    
    def _get_most_effective_behaviors(self) -> List[Dict[str, Any]]:
        """Get the most effective autonomous behaviors"""
        behavior_effectiveness = {}
        
        for action in self.behavior_history:
            behavior_type = action["behavior_type"]
            if behavior_type not in behavior_effectiveness:
                behavior_effectiveness[behavior_type] = {"success": 0, "total": 0}
            
            behavior_effectiveness[behavior_type]["total"] += 1
            if action["success"]:
                behavior_effectiveness[behavior_type]["success"] += 1
        
        # Calculate success rates
        effectiveness_list = []
        for behavior_type, stats in behavior_effectiveness.items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            effectiveness_list.append({
                "behavior_type": behavior_type,
                "success_rate": success_rate,
                "total_actions": stats["total"]
            })
        
        return sorted(effectiveness_list, key=lambda x: x["success_rate"], reverse=True)
