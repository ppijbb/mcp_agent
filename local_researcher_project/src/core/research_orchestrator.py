"""
Simple Research Orchestrator
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResearchRequest:
    """Research request data structure."""
    topic: str
    domain: str = "general"
    depth: str = "basic"
    sources: List[str] = None
    output_format: str = "markdown"
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = ["web"]


@dataclass
class ResearchResult:
    """Research result data structure."""
    research_id: str
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None


class ResearchOrchestrator:
    """Simple research orchestrator."""
    
    def __init__(self, config_manager):
        """Initialize the research orchestrator."""
        self.config_manager = config_manager
        self.active_research: Dict[str, ResearchResult] = {}
        self.research_queue: List[str] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        from ..research.workflow_manager import ResearchWorkflowManager
        from ..storage.data_manager import DataManager
        
        self.workflow_manager = ResearchWorkflowManager(config_manager)
        self.data_manager = DataManager(config_manager)
        
        logger.info("Research Orchestrator initialized")
    
    def start_research(self, request: ResearchRequest) -> str:
        """Start a new research project."""
        research_id = self._generate_research_id(request.topic)
        
        # Create research result
        result = ResearchResult(
            research_id=research_id,
            status="pending",
            progress=0.0,
            created_at=datetime.now()
        )
        
        self.active_research[research_id] = result
        self.research_queue.append(research_id)
        
        # Save to database
        self.data_manager.save_research_project(research_id, {
            'topic': request.topic,
            'domain': request.domain,
            'depth': request.depth,
            'sources': request.sources,
            'output_format': request.output_format,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        })
        
        # Start async research process
        asyncio.create_task(self._execute_research(research_id, request))
        
        logger.info(f"Started research: {research_id} for topic: {request.topic}")
        return research_id
    
    async def _execute_research(self, research_id: str, request: ResearchRequest):
        """Execute the research workflow."""
        try:
            result = self.active_research[research_id]
            result.status = "running"
            result.started_at = datetime.now()
            result.progress = 10.0
            result.current_step = "Initializing workflow"
            
            # Step 1: Create and execute workflow
            await self._update_progress(research_id, 20.0, "Creating research workflow...")
            workflow_id = await self.workflow_manager.create_workflow(
                workflow_type=request.depth,
                topic=request.topic,
                parameters={
                    'domain': request.domain,
                    'sources': request.sources,
                    'output_format': request.output_format
                }
            )
            
            # Step 2: Execute workflow
            await self._update_progress(research_id, 30.0, "Executing research workflow...")
            workflow_result = await self.workflow_manager.execute_workflow(workflow_id)
            
            # Step 3: Complete research
            await self._update_progress(research_id, 90.0, "Finalizing research...")
            
            # Get report path from workflow result
            report_path = None
            if workflow_result.status.value == "completed":
                report_generation = workflow_result.output.get('report_generation', {})
                report_path = report_generation.get('report_path')
            
            # Complete research
            result.status = "completed"
            result.progress = 100.0
            result.completed_at = datetime.now()
            result.current_step = "Completed"
            result.report_path = report_path
            
            # Update database
            self.data_manager.update_research_project(research_id, {
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'report_path': report_path
            })
            
            logger.info(f"Research completed: {research_id}")
            
        except Exception as e:
            logger.error(f"Research failed: {research_id} - {str(e)}")
            result = self.active_research[research_id]
            result.status = "failed"
            result.error_message = str(e)
            result.completed_at = datetime.now()
            
            # Update database
            self.data_manager.update_research_project(research_id, {
                'status': 'failed',
                'error_message': str(e),
                'completed_at': datetime.now().isoformat()
            })
    
    async def _update_progress(self, research_id: str, progress: float, message: str):
        """Update research progress."""
        if research_id in self.active_research:
            result = self.active_research[research_id]
            result.progress = progress
            result.current_step = message
            
            # Update database
            self.data_manager.update_research_project(research_id, {
                'progress': progress,
                'current_step': message,
                'status': 'running'
            })
    
    def get_research_status(self, research_id: str) -> Optional[ResearchResult]:
        """Get the status of a research project."""
        return self.active_research.get(research_id)
    
    def get_active_research(self) -> List[ResearchResult]:
        """Get all active research projects."""
        return list(self.active_research.values())
    
    def get_research_list(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of research projects."""
        projects = []
        
        for research_id, result in self.active_research.items():
            if status_filter is None or result.status == status_filter:
                projects.append({
                    'research_id': research_id,
                    'status': result.status,
                    'progress': result.progress,
                    'created_at': result.created_at,
                    'current_step': result.current_step,
                    'report_path': result.report_path
                })
        
        return projects
    
    def cancel_research(self, research_id: str) -> bool:
        """Cancel a research project."""
        if research_id in self.active_research:
            result = self.active_research[research_id]
            if result.status in ["pending", "running"]:
                result.status = "cancelled"
                result.completed_at = datetime.now()
                
                # Update database
                self.data_manager.update_research_project(research_id, {
                    'status': 'cancelled',
                    'completed_at': datetime.now().isoformat()
                })
                
                logger.info(f"Research cancelled: {research_id}")
                return True
        
        return False
    
    def _generate_research_id(self, topic: str) -> str:
        """Generate a unique research ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_")[:20]
        return f"research_{topic_slug}_{timestamp}"