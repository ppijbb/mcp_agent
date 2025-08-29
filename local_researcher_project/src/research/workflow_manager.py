"""
Research Workflow Manager

This module manages research workflows and coordinates the execution
of research tasks through different stages.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# Try to import asyncio, fallback to synchronous version if not available
try:
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    import time

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStage(Enum):
    """Workflow execution stages."""
    INITIALIZATION = "initialization"
    TOPIC_ANALYSIS = "topic_analysis"
    SOURCE_DISCOVERY = "source_discovery"
    CONTENT_GATHERING = "content_gathering"
    CONTENT_ANALYSIS = "content_analysis"
    REPORT_GENERATION = "report_generation"
    QUALITY_CHECK = "quality_check"
    FINALIZATION = "finalization"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    name: str
    stage: WorkflowStage
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 3
    retry_delay: int = 5
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    output: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchWorkflowManager:
    """Manages research workflows and their execution."""
    
    def __init__(self, config_manager):
        """Initialize the workflow manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, WorkflowResult] = {}
        self.workflow_templates: Dict[str, List[WorkflowStep]] = {}
        
        # Load workflow templates
        self._load_workflow_templates()
        
        logger.info("Research Workflow Manager initialized")
    
    def _load_workflow_templates(self):
        """Load predefined workflow templates."""
        # Basic research workflow
        self.workflow_templates["basic"] = [
            WorkflowStep(
                name="topic_analysis",
                stage=WorkflowStage.TOPIC_ANALYSIS,
                function=self._topic_analysis,
                timeout=60
            ),
            WorkflowStep(
                name="source_discovery",
                stage=WorkflowStage.SOURCE_DISCOVERY,
                function=self._source_discovery,
                dependencies=["topic_analysis"],
                timeout=120
            ),
            WorkflowStep(
                name="content_gathering",
                stage=WorkflowStage.CONTENT_GATHERING,
                function=self._content_gathering,
                dependencies=["source_discovery"],
                timeout=180
            ),
            WorkflowStep(
                name="report_generation",
                stage=WorkflowStage.REPORT_GENERATION,
                function=self._report_generation,
                dependencies=["content_gathering"],
                timeout=120
            )
        ]
        
        # Comprehensive research workflow
        self.workflow_templates["comprehensive"] = [
            WorkflowStep(
                name="topic_analysis",
                stage=WorkflowStage.TOPIC_ANALYSIS,
                function=self._topic_analysis,
                timeout=90
            ),
            WorkflowStep(
                name="source_discovery",
                stage=WorkflowStage.SOURCE_DISCOVERY,
                function=self._source_discovery,
                dependencies=["topic_analysis"],
                timeout=180
            ),
            WorkflowStep(
                name="content_gathering",
                stage=WorkflowStage.CONTENT_GATHERING,
                function=self._content_gathering,
                dependencies=["source_discovery"],
                timeout=300
            ),
            WorkflowStep(
                name="content_analysis",
                stage=WorkflowStage.CONTENT_ANALYSIS,
                function=self._content_analysis,
                dependencies=["content_gathering"],
                timeout=240
            ),
            WorkflowStep(
                name="quality_check",
                stage=WorkflowStage.QUALITY_CHECK,
                function=self._quality_check,
                dependencies=["content_analysis"],
                timeout=120
            ),
            WorkflowStep(
                name="report_generation",
                stage=WorkflowStage.REPORT_GENERATION,
                function=self._report_generation,
                dependencies=["quality_check"],
                timeout=180
            )
        ]
        
        logger.info(f"Loaded {len(self.workflow_templates)} workflow templates")
    
    async def create_workflow(
        self,
        workflow_type: str,
        topic: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new research workflow.
        
        Args:
            workflow_type: Type of workflow (basic, comprehensive)
            topic: Research topic
            parameters: Additional workflow parameters
            
        Returns:
            Workflow ID
        """
        if workflow_type not in self.workflow_templates:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        workflow_id = self._generate_workflow_id(topic)
        
        workflow_data = {
            "type": workflow_type,
            "topic": topic,
            "parameters": parameters or {},
            "template": self.workflow_templates[workflow_type].copy(),
            "created_at": datetime.now(),
            "status": WorkflowStatus.PENDING
        }
        
        self.workflows[workflow_id] = workflow_data
        
        logger.info(f"Created workflow {workflow_id} for topic: {topic}")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> WorkflowResult:
        """Execute a research workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            Workflow execution result
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow_data = self.workflows[workflow_id]
        template = workflow_data["template"]
        
        # Create workflow result
        result = WorkflowResult(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.active_workflows[workflow_id] = result
        
        try:
            logger.info(f"Starting workflow execution: {workflow_id}")
            
            # Execute workflow steps
            await self._execute_workflow_steps(workflow_id, template, result)
            
            result.status = WorkflowStatus.COMPLETED
            result.end_time = datetime.now()
            
            logger.info(f"Workflow completed successfully: {workflow_id}")
            
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = str(e)
            
            logger.error(f"Workflow failed: {workflow_id}, Error: {e}")
        
        return result
    
    async def _execute_workflow_steps(
        self,
        workflow_id: str,
        template: List[WorkflowStep],
        result: WorkflowResult
    ):
        """Execute workflow steps in order.
        
        Args:
            workflow_id: Workflow ID
            template: Workflow template
            result: Workflow result object
        """
        completed_steps = {}
        
        for step in template:
            try:
                # Check dependencies
                if not self._check_dependencies(step, completed_steps):
                    logger.warning(f"Dependencies not met for step: {step.name}")
                    continue
                
                logger.info(f"Executing step: {step.name}")
                
                # Execute step
                step_output = await self._execute_step(step, workflow_id)
                
                # Store step output
                completed_steps[step.name] = step_output
                result.output[step.name] = step_output
                result.steps_completed.append(step.name)
                
                logger.info(f"Step completed: {step.name}")
                
            except Exception as e:
                logger.error(f"Step failed: {step.name}, Error: {e}")
                result.steps_failed.append(step.name)
                
                if step.required:
                    raise e
    
    async def _execute_step(self, step: WorkflowStep, workflow_id: str) -> Any:
        """Execute a single workflow step.
        
        Args:
            step: Workflow step to execute
            workflow_id: Workflow ID
            
        Returns:
            Step execution result
        """
        try:
            # Execute step function with timeout
            result = await asyncio.wait_for(
                step.function(workflow_id),
                timeout=step.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Step {step.name} timed out after {step.timeout} seconds")
        except Exception as e:
            raise e
    
    def _check_dependencies(self, step: WorkflowStep, completed_steps: Dict[str, Any]) -> bool:
        """Check if step dependencies are met.
        
        Args:
            step: Workflow step
            completed_steps: Dictionary of completed steps
            
        Returns:
            True if dependencies are met, False otherwise
        """
        for dependency in step.dependencies:
            if dependency not in completed_steps:
                return False
        return True
    
    def _generate_workflow_id(self, topic: str) -> str:
        """Generate a unique workflow ID.
        
        Args:
            topic: Research topic
            
        Returns:
            Unique workflow ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_")[:20]
        return f"workflow_{topic_slug}_{timestamp}"
    
    # Workflow step implementations
    async def _topic_analysis(self, workflow_id: str) -> Dict[str, Any]:
        """Perform topic analysis step."""
        logger.info(f"Performing topic analysis for workflow: {workflow_id}")
        
        # Mock implementation - replace with actual logic
        if ASYNC_AVAILABLE:
            await asyncio.sleep(2)
        else:
            time.sleep(2)
        
        return {
            "topic_keywords": ["AI", "machine learning", "research"],
            "complexity": "medium",
            "estimated_sources": 15,
            "analysis_complete": True
        }
    
    async def _source_discovery(self, workflow_id: str) -> Dict[str, Any]:
        """Perform source discovery step."""
        logger.info(f"Performing source discovery for workflow: {workflow_id}")
        
        # Mock implementation - replace with actual logic
        if ASYNC_AVAILABLE:
            await asyncio.sleep(3)
        else:
            time.sleep(3)
        
        return {
            "sources_found": 12,
            "source_types": ["academic", "news", "web"],
            "discovery_complete": True
        }
    
    async def _content_gathering(self, workflow_id: str) -> Dict[str, Any]:
        """Perform content gathering step."""
        logger.info(f"Performing content gathering for workflow: {workflow_id}")
        
        # Mock implementation - replace with actual logic
        if ASYNC_AVAILABLE:
            await asyncio.sleep(5)
        else:
            time.sleep(5)
        
        return {
            "content_collected": True,
            "total_content_length": 25000,
            "sources_processed": 12,
            "gathering_complete": True
        }
    
    async def _content_analysis(self, workflow_id: str) -> Dict[str, Any]:
        """Perform content analysis step."""
        logger.info(f"Performing content analysis for workflow: {workflow_id}")
        
        # Mock implementation - replace with actual logic
        if ASYNC_AVAILABLE:
            await asyncio.sleep(4)
        else:
            time.sleep(4)
        
        return {
            "analysis_complete": True,
            "key_insights": ["insight1", "insight2", "insight3"],
            "confidence_score": 0.85
        }
    
    async def _quality_check(self, workflow_id: str) -> Dict[str, Any]:
        """Perform quality check step."""
        logger.info(f"Performing quality check for workflow: {workflow_id}")
        
        # Mock implementation - replace with actual logic
        if ASYNC_AVAILABLE:
            await asyncio.sleep(2)
        else:
            time.sleep(2)
        
        return {
            "quality_score": 0.92,
            "issues_found": 0,
            "quality_check_passed": True
        }
    
    async def _report_generation(self, workflow_id: str) -> Dict[str, Any]:
        """Perform report generation step."""
        logger.info(f"Performing report generation for workflow: {workflow_id}")
        
        # Mock implementation - replace with actual logic
        if ASYNC_AVAILABLE:
            await asyncio.sleep(3)
        else:
            time.sleep(3)
        
        return {
            "report_generated": True,
            "report_path": f"outputs/report_{workflow_id}.md",
            "report_length": 1500,
            "generation_complete": True
        }
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get the status of a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow result or None if not found
        """
        return self.active_workflows.get(workflow_id)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow.
        
        Args:
            workflow_id: Workflow ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if workflow_id in self.active_workflows:
            result = self.active_workflows[workflow_id]
            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.CANCELLED
                result.end_time = datetime.now()
                logger.info(f"Workflow cancelled: {workflow_id}")
                return True
        
        return False
    
    def list_workflows(self, status_filter: Optional[WorkflowStatus] = None) -> List[Dict[str, Any]]:
        """List all workflows with optional status filtering.
        
        Args:
            status_filter: Optional status filter
            
        Returns:
            List of workflow information
        """
        workflows = []
        
        for workflow_id, workflow_data in self.workflows.items():
            if status_filter is None or workflow_data["status"] == status_filter:
                workflow_info = {
                    "id": workflow_id,
                    "type": workflow_data["type"],
                    "topic": workflow_data["topic"],
                    "status": workflow_data["status"],
                    "created_at": workflow_data["created_at"]
                }
                
                if workflow_id in self.active_workflows:
                    result = self.active_workflows[workflow_id]
                    workflow_info.update({
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "steps_completed": len(result.steps_completed),
                        "steps_failed": len(result.steps_failed)
                    })
                
                workflows.append(workflow_info)
        
        return workflows
