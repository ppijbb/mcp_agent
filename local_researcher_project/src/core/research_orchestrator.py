"""
Local Researcher Core - Research Orchestrator

This module provides the core orchestration logic for integrating
Gemini CLI with Open Deep Research for local research automation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.config_manager import ConfigManager
from ..utils.logger import setup_logger
from ..research.workflow_manager import ResearchWorkflowManager
from ..agents.agent_manager import AgentManager
from ..storage.data_manager import DataManager


@dataclass
class ResearchRequest:
    """Research request data structure."""
    topic: str
    domain: Optional[str] = None
    depth: str = "standard"
    sources: List[str] = None
    output_format: str = "markdown"
    custom_workflow: Optional[str] = None
    priority: int = 1
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.sources is None:
            self.sources = []


class ResearchResult(BaseModel):
    """Research result data structure."""
    request_id: str
    topic: str
    status: str = Field(default="pending")
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    report_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ResearchOrchestrator:
    """
    Main orchestrator for local research operations.
    
    This class coordinates between Gemini CLI commands and Open Deep Research
    to provide a seamless local research experience.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the research orchestrator."""
        self.console = Console()
        self.logger = setup_logger("research_orchestrator")
        
        # Initialize components
        self.config_manager = ConfigManager(config_path)
        self.workflow_manager = ResearchWorkflowManager(self.config_manager)
        self.agent_manager = AgentManager(self.config_manager)
        self.data_manager = DataManager(self.config_manager)
        
        # Research state
        self.active_research: Dict[str, ResearchResult] = {}
        self.research_queue: List[ResearchRequest] = []
        
        self.logger.info("Research Orchestrator initialized")
    
    async def start_research(self, request: ResearchRequest) -> str:
        """
        Start a new research operation.
        
        Args:
            request: Research request object
            
        Returns:
            Research ID for tracking
        """
        research_id = self._generate_research_id(request)
        
        # Create result object
        result = ResearchResult(
            request_id=research_id,
            topic=request.topic,
            status="starting"
        )
        
        self.active_research[research_id] = result
        self.research_queue.append(request)
        
        self.logger.info(f"Started research: {research_id} - {request.topic}")
        
        # Start async research process
        asyncio.create_task(self._execute_research(research_id, request))
        
        return research_id
    
    async def _execute_research(self, research_id: str, request: ResearchRequest):
        """Execute the research workflow."""
        try:
            result = self.active_research[research_id]
            result.status = "running"
            result.progress = 10.0
            
            # Step 1: Initialize workflow
            await self._update_progress(research_id, 20.0, "Initializing research workflow...")
            workflow = await self.workflow_manager.create_workflow(request)
            
            # Step 2: Configure agents
            await self._update_progress(research_id, 30.0, "Configuring research agents...")
            agents = await self.agent_manager.setup_agents(workflow)
            
            # Step 3: Execute research phases
            await self._update_progress(research_id, 40.0, "Starting research phases...")
            
            # Phase 1: Topic analysis
            await self._update_progress(research_id, 50.0, "Analyzing research topic...")
            topic_analysis = await self._analyze_topic(request.topic, agents)
            
            # Phase 2: Source discovery
            await self._update_progress(research_id, 60.0, "Discovering sources...")
            sources = await self._discover_sources(request, agents)
            
            # Phase 3: Content gathering
            await self._update_progress(research_id, 70.0, "Gathering content...")
            content = await self._gather_content(sources, agents)
            
            # Phase 4: Analysis and synthesis
            await self._update_progress(research_id, 80.0, "Analyzing and synthesizing...")
            analysis = await self._analyze_content(content, agents)
            
            # Phase 5: Report generation
            await self._update_progress(research_id, 90.0, "Generating report...")
            report_path = await self._generate_report(request, analysis, agents)
            
            # Complete research
            result.status = "completed"
            result.progress = 100.0
            result.report_path = report_path
            result.completed_at = datetime.now()
            
            self.logger.info(f"Research completed: {research_id}")
            
        except Exception as e:
            self.logger.error(f"Research failed: {research_id} - {str(e)}")
            result = self.active_research[research_id]
            result.status = "failed"
            result.error_message = str(e)
    
    async def _analyze_topic(self, topic: str, agents: Dict) -> Dict[str, Any]:
        """Analyze the research topic."""
        self.logger.info(f"Analyzing topic: {topic}")
        
        # Use topic analysis agent
        topic_agent = agents.get("topic_analyzer")
        if topic_agent:
            analysis = await topic_agent.analyze(topic)
            return analysis
        
        # Fallback analysis
        return {
            "keywords": topic.split(),
            "domain": "general",
            "complexity": "medium",
            "estimated_sources": 10
        }
    
    async def _discover_sources(self, request: ResearchRequest, agents: Dict) -> List[Dict]:
        """Discover relevant sources for research."""
        self.logger.info(f"Discovering sources for: {request.topic}")
        
        # Use source discovery agent
        source_agent = agents.get("source_discoverer")
        if source_agent:
            sources = await source_agent.discover(request)
            return sources
        
        # Fallback source discovery
        return []
    
    async def _gather_content(self, sources: List[Dict], agents: Dict) -> List[Dict]:
        """Gather content from discovered sources."""
        self.logger.info(f"Gathering content from {len(sources)} sources")
        
        # Use content gathering agent
        content_agent = agents.get("content_gatherer")
        if content_agent:
            content = await content_agent.gather(sources)
            return content
        
        # Fallback content gathering
        return []
    
    async def _analyze_content(self, content: List[Dict], agents: Dict) -> Dict[str, Any]:
        """Analyze gathered content."""
        self.logger.info(f"Analyzing {len(content)} content items")
        
        # Use content analysis agent
        analysis_agent = agents.get("content_analyzer")
        if analysis_agent:
            analysis = await analysis_agent.analyze(content)
            return analysis
        
        # Fallback analysis
        return {
            "summary": "Content analysis completed",
            "key_findings": [],
            "insights": []
        }
    
    async def _generate_report(self, request: ResearchRequest, analysis: Dict, agents: Dict) -> str:
        """Generate the final research report."""
        self.logger.info(f"Generating report for: {request.topic}")
        
        # Use report generation agent
        report_agent = agents.get("report_generator")
        if report_agent:
            report_path = await report_agent.generate(request, analysis)
            return report_path
        
        # Fallback report generation
        report_path = self._generate_fallback_report(request, analysis)
        return report_path
    
    def _generate_fallback_report(self, request: ResearchRequest, analysis: Dict) -> str:
        """Generate a fallback report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.md"
        report_path = Path(self.config_manager.get("output_dir")) / filename
        
        # Create basic report
        report_content = f"""# Research Report: {request.topic}

## Overview
This report was generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

## Analysis Summary
{analysis.get('summary', 'No analysis available')}

## Key Findings
{chr(10).join(f"- {finding}" for finding in analysis.get('key_findings', []))}

## Insights
{chr(10).join(f"- {insight}" for insight in analysis.get('insights', []))}

---
*Generated by Local Researcher*
"""
        
        report_path.write_text(report_content, encoding='utf-8')
        return str(report_path)
    
    async def _update_progress(self, research_id: str, progress: float, message: str):
        """Update research progress."""
        if research_id in self.active_research:
            result = self.active_research[research_id]
            result.progress = progress
            self.logger.info(f"Progress {progress}%: {message}")
    
    def get_research_status(self, research_id: str) -> Optional[ResearchResult]:
        """Get the status of a research operation."""
        return self.active_research.get(research_id)
    
    def list_active_research(self) -> List[ResearchResult]:
        """List all active research operations."""
        return list(self.active_research.values())
    
    def cancel_research(self, research_id: str) -> bool:
        """Cancel an active research operation."""
        if research_id in self.active_research:
            result = self.active_research[research_id]
            result.status = "cancelled"
            self.logger.info(f"Research cancelled: {research_id}")
            return True
        return False
    
    def _generate_research_id(self, request: ResearchRequest) -> str:
        """Generate a unique research ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_hash = hash(request.topic) % 10000
        return f"research_{timestamp}_{topic_hash}"
    
    async def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up research orchestrator...")
        await self.workflow_manager.cleanup()
        await self.agent_manager.cleanup()
        await self.data_manager.cleanup() 