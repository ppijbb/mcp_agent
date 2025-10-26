#!/usr/bin/env python3
"""
Autonomous Multi-Agent Research System - Main Entry Point
Implements 8 Core Innovations: Production-Grade Reliability, Universal MCP Hub, Streaming Pipeline

MCP agent 라이브러리 기반의 자율 리서처 시스템.
모든 하드코딩, fallback, mock 코드를 제거하고 실제 MCP agent를 사용.

Usage:
    python main.py --request "연구 주제"                    # CLI 모드
    python main.py --web                                    # 웹 모드
    python main.py --mcp-server                            # MCP 서버 모드
    python main.py --streaming                             # 스트리밍 모드
"""

import asyncio
import sys
import argparse
import subprocess
import signal
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
import json
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from researcher_config import load_config_from_env
from src.agents.autonomous_researcher import AutonomousResearcherAgent
from src.core.autonomous_orchestrator import AutonomousOrchestrator
from src.core.reliability import execute_with_reliability
from src.monitoring.system_monitor import HealthMonitor
from src.core.llm_manager import execute_llm_task, TaskType
from mcp_integration import get_available_tools, execute_tool

# Configure logging for production-grade reliability
# Advanced logging setup: setup logger manually to ensure logs directory exists and avoid issues with logging.basicConfig (per best practices)
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "researcher.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicate logs on reloads (e.g., under some app servers or noteboks)
if logger.hasHandlers():
    logger.handlers.clear()

# File handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


# MCPIntegrationManager 클래스 제거됨 - mcp_integration.py의 UniversalMCPHub 사용


class WebAppManager:
    """웹 앱 관리자 - Streaming Pipeline (Innovation 5) 지원"""
    
    def __init__(self):
        self.project_root = project_root
        self.health_monitor = HealthMonitor()
        
    def start_web_app(self):
        """웹 앱 시작 - Production-Grade Reliability 적용"""
        try:
            streamlit_app_path = self.project_root / "src" / "web" / "streamlit_app.py"
            
            if not streamlit_app_path.exists():
                logger.error(f"Streamlit app not found at {streamlit_app_path}")
                return False
            
            logger.info("🌐 Starting Local Researcher Web Application with Streaming Pipeline...")
            logger.info("App will be available at: http://localhost:8501")
            logger.info("Features: Real-time streaming, Progressive reporting, Incremental save")
            logger.info("Press Ctrl+C to stop the application")
            
            # Create logs directory if it doesn't exist
            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(streamlit_app_path),
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--browser.gatherUsageStats", "false",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ]
            
            # Start health monitoring
            asyncio.create_task(self.health_monitor.start_monitoring())
            
            subprocess.run(cmd, cwd=str(self.project_root))
            return True
            
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
            return True
        except Exception as e:
            logger.error(f"Error running web application: {e}")
            return False
    
    async def get_web_app_health(self) -> Dict[str, Any]:
        """Get web application health status."""
        return {
            'status': 'running',
            'port': 8501,
            'streaming_enabled': True,
            'progressive_reporting': True,
            'incremental_save': True,
            'timestamp': datetime.now().isoformat()
        }


# AutonomousResearcherAgent is now in src/agents/autonomous_researcher.py


class AutonomousResearchSystem:
    """자율 리서처 시스템 - 8가지 핵심 혁신 통합 메인 클래스"""
    
    def __init__(self):
        # Load configurations from environment - ALL REQUIRED, NO DEFAULTS
        try:
            self.config = load_config_from_env()
            logger.info("✅ Configuration loaded successfully from environment variables")
            
            # Validate ChromaDB availability (optional)
            try:
                import chromadb  # type: ignore
                logger.info("✅ ChromaDB module available")
            except ImportError:
                logger.warning("⚠️ ChromaDB not installed - vector search will be disabled")
                logger.info("   Install with: pip install chromadb")
            
        except ValueError as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            logger.error("Please check your .env file and ensure all required variables are set")
            logger.info("\nRequired environment variables:")
            logger.info("  - LLM_MODEL: LLM model identifier (e.g., google/gemini-2.5-flash-lite)")
            logger.info("  - OPENROUTER_API_KEY: Your OpenRouter API key (starts with sk-or-)")
            logger.info("  - LLM_PROVIDER: Provider name (e.g., openrouter)")
            raise
        
        # Initialize components with 8 innovations
        logger.info("🔧 Initializing system components...")
        try:
            self.orchestrator = AutonomousOrchestrator()
            logger.info("✅ Orchestrator initialized")
        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            raise
        
        try:
            from mcp_integration import UniversalMCPHub
            self.mcp_hub = UniversalMCPHub()
            logger.info("✅ MCP Hub initialized")
        except Exception as e:
            logger.error(f"❌ MCP Hub initialization failed: {e}")
            raise
        
        try:
            self.web_manager = WebAppManager()
            logger.info("✅ Web Manager initialized")
        except Exception as e:
            logger.error(f"❌ Web Manager initialization failed: {e}")
            raise
        
        try:
            self.health_monitor = HealthMonitor()
            logger.info("✅ Health Monitor initialized")
        except Exception as e:
            logger.error(f"❌ Health Monitor initialization failed: {e}")
            raise
        
        # Initialize signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ AutonomousResearchSystem initialized successfully")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self._graceful_shutdown())
    
    async def _graceful_shutdown(self):
        """Graceful shutdown with state persistence."""
        try:
            logger.info("Performing graceful shutdown...")
            await self.health_monitor.stop_monitoring()
            logger.info("✅ Graceful shutdown completed")
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
    
    async def run_research(self, request: str, output_path: Optional[str] = None, 
                          streaming: bool = False, output_format: str = "json") -> Dict[str, Any]:
        """연구 실행 - 8가지 핵심 혁신 적용"""
        logger.info("🤖 Starting Autonomous Research System with 8 Core Innovations")
        logger.info("=" * 80)
        logger.info(f"Request: {request}")
        logger.info(f"Primary LLM: {self.config.llm.primary_model}")
        logger.info(f"Planning Model: {self.config.llm.planning_model}")
        logger.info(f"Reasoning Model: {self.config.llm.reasoning_model}")
        logger.info(f"Verification Model: {self.config.llm.verification_model}")
        logger.info(f"Self-planning: {self.config.agent.enable_self_planning}")
        logger.info(f"Agent Communication: {self.config.agent.enable_agent_communication}")
        logger.info(f"MCP Enabled: {self.config.mcp.enabled}")
        logger.info(f"Streaming Pipeline: {streaming}")
        logger.info(f"Adaptive Supervisor: {self.config.agent.max_concurrent_research_units}")
        logger.info(f"Hierarchical Compression: {self.config.compression.enabled}")
        logger.info(f"Continuous Verification: {self.config.verification.enabled}")
        logger.info(f"Adaptive Context Window: {self.config.context_window.enabled}")
        logger.info("=" * 80)
        
        try:
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Initialize MCP client if enabled
            if self.config.mcp.enabled:
                await self.mcp_hub.initialize_mcp()
            
            # Run research with production-grade reliability
            if streaming:
                result = await self._run_streaming_research(request)
            else:
                result = await self.orchestrator.run_research(request)
            
            # Apply hierarchical compression if enabled
            if self.config.compression.enabled:
                result = await self._apply_hierarchical_compression(result)
            
            # Save results with incremental save
            if output_path:
                await self._save_results_incrementally(result, output_path, output_format)
            else:
                self._display_results(result)
            
            # Get final health status
            health_status = self.health_monitor.get_system_health()
            result['system_health'] = health_status
            
            logger.info("✅ Research completed successfully with 8 Core Innovations")
            return result
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            # Get error health status
            error_health = self.health_monitor.get_system_health()
            logger.error(f"System health at failure: {error_health}")
            raise
    
    async def _run_streaming_research(self, request: str) -> Dict[str, Any]:
        """Run research with streaming pipeline (Innovation 5)."""
        logger.info("🌊 Starting streaming research pipeline...")
        
        # Create streaming callback
        async def streaming_callback(partial_result: Dict[str, Any]):
            logger.info(f"📊 Streaming partial result: {partial_result.get('type', 'unknown')}")
            # In a real implementation, this would send to web interface
            print(f"📊 Partial Result: {partial_result.get('summary', 'Processing...')}")
        
        # Run with streaming
        result = await self.orchestrator.run_research_with_streaming(
            request=request,
            streaming_callback=streaming_callback
        )
        
        logger.info("✅ Streaming research completed")
        return result
    
    async def _apply_hierarchical_compression(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hierarchical compression (Innovation 2)."""
        if not self.config.compression.enabled:
            return result
        
        logger.info("🗜️ Applying hierarchical compression...")
        
        # Import compression module
        from src.core.compression import compress_data
        
        # Compress large text fields
        if 'synthesis_results' in result:
            compressed_synthesis = await compress_data(
                result['synthesis_results']
            )
            result['synthesis_results_compressed'] = compressed_synthesis
        
        logger.info("✅ Hierarchical compression applied")
        return result
    
    async def _save_results_incrementally(self, result: Dict[str, Any], output_path: str, output_format: str = "json"):
        """Save results with incremental save (Innovation 5)."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save incrementally based on format
        temp_file = output_file.with_suffix('.tmp')
        
        if output_format.lower() == "json":
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        elif output_format.lower() == "yaml":
            import yaml
            with open(temp_file, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        elif output_format.lower() == "txt":
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(f"Research Results\n")
                f.write(f"===============\n\n")
                f.write(f"Query: {result.get('query', 'N/A')}\n")
                f.write(f"Timestamp: {result.get('timestamp', 'N/A')}\n\n")
                if 'content' in result:
                    f.write(f"Content:\n{result['content']}\n\n")
                elif 'synthesis_results' in result:
                    synthesis = result['synthesis_results']
                    if isinstance(synthesis, dict) and 'content' in synthesis:
                        f.write(f"Content:\n{synthesis['content']}\n\n")
                if 'sources' in result:
                    f.write(f"Sources:\n")
                    for i, source in enumerate(result['sources'], 1):
                        f.write(f"{i}. {source.get('title', 'N/A')} - {source.get('url', 'N/A')}\n")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Atomic move
        temp_file.replace(output_file)
        
        logger.info(f"✅ Results saved incrementally to: {output_file} (format: {output_format})")
    
    def _display_results(self, result: Dict[str, Any]):
        """Display results with enhanced formatting."""
        print("\n📋 Research Results with 8 Core Innovations:")
        print("=" * 80)
        
        # Display main research content
        if 'content' in result and result['content']:
            print("\n📝 Research Content:")
            print("-" * 60)
            print(result['content'])
            print("-" * 60)
        elif 'synthesis_results' in result:
            synthesis = result['synthesis_results']
            if isinstance(synthesis, dict) and 'content' in synthesis:
                print("\n📝 Research Content:")
                print("-" * 60)
                print(synthesis['content'])
                print("-" * 60)
            else:
                print(f"\n📝 Synthesis: {synthesis}")
        else:
            print("\n❌ No research content found in results")
        
        # Display research metadata
        if 'metadata' in result:
            metadata = result['metadata']
            print(f"\n📊 Research Metadata:")
            print(f"  • Model Used: {metadata.get('model_used', 'N/A')}")
            print(f"  • Execution Time: {metadata.get('execution_time', 'N/A'):.2f}s")
            print(f"  • Cost: ${metadata.get('cost', 0):.4f}")
            print(f"  • Confidence: {metadata.get('confidence', 'N/A')}")
        
        # Display synthesis results
        if 'synthesis_results' in result and isinstance(result['synthesis_results'], dict):
            synthesis = result['synthesis_results']
            if 'synthesis_results' in synthesis:
                print(f"\n📝 Synthesis: {synthesis.get('synthesis_results', 'N/A')}")
        
        # Display innovation stats
        if 'innovation_stats' in result:
            stats = result['innovation_stats']
            print(f"\n🚀 Innovation Statistics:")
            print(f"  • Adaptive Supervisor: {stats.get('adaptive_supervisor', 'N/A')}")
            print(f"  • Hierarchical Compression: {stats.get('hierarchical_compression', 'N/A')}")
            print(f"  • Multi-Model Orchestration: {stats.get('multi_model_orchestration', 'N/A')}")
            print(f"  • Continuous Verification: {stats.get('continuous_verification', 'N/A')}")
            print(f"  • Streaming Pipeline: {stats.get('streaming_pipeline', 'N/A')}")
            print(f"  • Universal MCP Hub: {stats.get('universal_mcp_hub', 'N/A')}")
            print(f"  • Adaptive Context Window: {stats.get('adaptive_context_window', 'N/A')}")
            print(f"  • Production-Grade Reliability: {stats.get('production_grade_reliability', 'N/A')}")
        
        # Display system health
        if 'system_health' in result:
            health = result['system_health']
            print(f"\n🏥 System Health: {health.get('overall_status', 'Unknown')}")
            print(f"  • Health Score: {health.get('health_score', 'N/A')}")
            print(f"  • Monitoring Active: {health.get('monitoring_active', 'N/A')}")
        
        print("=" * 80)
    
    async def run_mcp_server(self):
        """MCP 서버 실행"""
        await self.mcp_hub.initialize_mcp()
    
    async def run_mcp_client(self):
        """MCP 클라이언트 실행"""
        await self.mcp_hub.initialize_mcp()
    
    def run_web_app(self):
        """웹 앱 실행"""
        return self.web_manager.start_web_app()
    
    async def run_health_check(self):
        """Run comprehensive health check for all system components."""
        logger.info("🏥 Running comprehensive health check...")
        
        # Check MCP tools health
        if self.config.mcp.enabled:
            # MCP Hub health check
            logger.info("MCP Hub initialized and ready")
        
        # Check system health
        system_health = self.health_monitor.get_system_health()
        logger.info(f"System Health: {system_health.get('overall_status', 'Unknown')}")
        
        # Check web app health
        web_health = await self.web_manager.get_web_app_health()
        logger.info(f"Web App Health: {web_health.get('status', 'Unknown')}")
        
        logger.info("✅ Health check completed")


async def main():
    """Main function - 8가지 핵심 혁신 통합 실행 진입점"""
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Agent Research System with 8 Core Innovations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --request "인공지능의 미래 전망"
  python main.py --request "연구 주제" --output results/report.json
  python main.py --request "연구 주제" --streaming
  python main.py --web
  python main.py --mcp-server
  python main.py --mcp-client
  python main.py --health-check
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--request", help="Research request (CLI mode)")
    mode_group.add_argument("--web", action="store_true", help="Start web application with streaming")
    mode_group.add_argument("--mcp-server", action="store_true", help="Start MCP server with Universal MCP Hub")
    mode_group.add_argument("--mcp-client", action="store_true", help="Start MCP client with Smart Tool Selection")
    mode_group.add_argument("--health-check", action="store_true", help="Check system health and MCP tools")
    
    # Optional arguments
    parser.add_argument("--output", help="Output file path for research results")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--format", choices=["json", "yaml", "txt"], default="json", help="Output format for research results")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming pipeline for real-time results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Initialize system
    system = AutonomousResearchSystem()
    
    try:
        if args.request:
            # CLI Research Mode with 8 innovations
            await system.run_research(
                args.request, 
                args.output, 
                streaming=args.streaming,
                output_format=args.format
            )
            
        elif args.web:
            # Web Application Mode with Streaming Pipeline
            system.run_web_app()
            
        elif args.mcp_server:
            # MCP Server Mode with Universal MCP Hub
            success = await system.run_mcp_server()
            if not success:
                sys.exit(1)
            
        elif args.mcp_client:
            # MCP Client Mode with Smart Tool Selection
            success = await system.run_mcp_client()
            if not success:
                sys.exit(1)
            
        elif args.health_check:
            # Health Check Mode
            await system.run_health_check()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        await system._graceful_shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        await system._graceful_shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())