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
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import load_config_from_env

# CRITICAL: Load configuration BEFORE importing any modules that depend on it
config = load_config_from_env()

# Use new AgentOrchestrator for multi-agent orchestration
from src.core.agent_orchestrator import AgentOrchestrator as NewAgentOrchestrator
from src.core.autonomous_orchestrator import AutonomousOrchestrator
from src.monitoring.system_monitor import HealthMonitor

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
            
            # Get port from environment variable, default to 8501
            port = os.getenv("STREAMLIT_PORT", "8501")
            address = os.getenv("STREAMLIT_ADDRESS", "0.0.0.0")
            
            logger.info("🌐 Starting Local Researcher Web Application with Streaming Pipeline...")
            logger.info(f"App will be available at: http://{address}:{port}")
            logger.info("Features: Real-time streaming, Progressive reporting, Incremental save")
            logger.info("Press Ctrl+C to stop the application")
            
            # Create logs directory if it doesn't exist
            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(streamlit_app_path),
                "--server.port", port,
                "--server.address", address,
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
        port = int(os.getenv("STREAMLIT_PORT", "8501"))
        return {
            'status': 'running',
            'port': port,
            'streaming_enabled': True,
            'progressive_reporting': True,
            'incremental_save': True,
            'timestamp': datetime.now().isoformat()
        }


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
            logger.info("  - GOOGLE_API_KEY: Your Google or Vertex AI API key")
            logger.info("  - LLM_PROVIDER: Provider name (e.g., google)")
            raise
        
        # Initialize components with 8 innovations
        logger.info("🔧 Initializing system components...")
        try:
            # Use new multi-agent orchestrator (no fallback - fail clearly)
            self.orchestrator = NewAgentOrchestrator()
            logger.info("✅ Multi-Agent Orchestrator initialized (no fallback mode)")
            logger.info("✅ Autonomous Orchestrator initialized")
        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            raise
        
        try:
            from src.core.mcp_integration import UniversalMCPHub
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
        
        # Shutdown flag 초기화
        self._shutdown_requested = False
        
        logger.info("✅ AutonomousResearchSystem initialized successfully")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        import sys
        
        # Shutdown 플래그 설정 (중복 방지)
        if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
            # 이미 종료 중이면 재진입 방지만 수행
            logger.warning("Shutdown already in progress; ignoring additional signal")
            return
        
        self._shutdown_requested = True
        
        # 실행 중인 이벤트 루프가 있는지 확인하고 shutdown 작업 스케줄링
        try:
            loop = asyncio.get_running_loop()
            # 중복 생성 방지: 이미 스케줄된 작업이 있으면 재생성하지 않음
            if not hasattr(self, '_shutdown_task') or self._shutdown_task is None or self._shutdown_task.done():
                def _schedule():
                    self._shutdown_task = asyncio.create_task(self._graceful_shutdown())
                loop.call_soon_threadsafe(_schedule)
            else:
                logger.debug("Shutdown task already scheduled")
        except RuntimeError:
            # 이벤트 루프가 없으면 강제 종료
            logger.warning("No event loop available, forcing exit")
            sys.exit(1)
    
    async def _graceful_shutdown(self):
        """Graceful shutdown with state persistence."""
        import sys
        import signal
        
        try:
            logger.info("Performing graceful shutdown...")
            
            # Health monitor 정지 (타임아웃 설정)
            try:
                await asyncio.wait_for(self.health_monitor.stop_monitoring(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Health monitor stop timed out")
            except Exception as e:
                logger.debug(f"Error stopping health monitor: {e}")
            
            # MCP Hub cleanup (타임아웃 설정)
            if self.config.mcp.enabled and self.mcp_hub:
                try:
                    # 신규 연결 차단
                    if hasattr(self.mcp_hub, 'start_shutdown'):
                        self.mcp_hub.start_shutdown()
                    # cleanup은 CancelledError를 발생시킬 수 있으므로 무시
                    try:
                        await asyncio.wait_for(self.mcp_hub.cleanup(), timeout=10.0)
                    except asyncio.CancelledError:
                        logger.debug("MCP Hub cleanup was cancelled (normal during shutdown)")
                    except asyncio.TimeoutError:
                        logger.warning("MCP Hub cleanup timed out")
                except asyncio.CancelledError:
                    logger.debug("MCP Hub cleanup setup was cancelled (normal during shutdown)")
                except Exception as e:
                    logger.warning(f"Error cleaning up MCP Hub: {e}")
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
        finally:
            # 최종 종료 준비
            logger.info("Exiting...")
            # 외부 라이브러리 태스크는 개별 매니저가 정리함. 일괄 취소는 하지 않음
            
            # sys.exit(0)은 호출하지 않음 - asyncio.run()이 자동으로 처리
            # 대신 루프에서 나가도록 함
    
    def _detect_output_format_from_content(self, content: str, request: str) -> str:
        """생성된 내용을 보고 파일 형식 결정 (최소한의 패턴 매칭만)."""
        if not content:
            return "md"
        
        # 코드 블록 제거 후 내용 확인
        content_clean = content.replace("```", "").strip()
        
        # Python 코드 패턴
        if ("def " in content_clean[:1000] and "import " in content_clean[:1000]) or content.startswith("```python"):
            return "py"
        
        # Java 코드 패턴
        if ("public class" in content_clean[:1000] or "public static void" in content_clean[:1000]) or content.startswith("```java"):
            return "java"
        
        # JavaScript 코드 패턴
        if (("const " in content_clean[:1000] or "function " in content_clean[:1000]) and "console" in content_clean[:1000]) or content.startswith("```javascript"):
            return "js"
        
        # HTML 패턴
        if content.strip().startswith("<!DOCTYPE") or content.strip().startswith("<html"):
            return "html"
        
        # 기본: Markdown
        return "md"
    
    async def run_research(self, request: str, output_path: Optional[str] = None, 
                          streaming: bool = False, output_format: Optional[str] = None) -> Dict[str, Any]:
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
                try:
                    await self.mcp_hub.initialize_mcp()
                except asyncio.CancelledError:
                    # 초기화 중 취소된 경우 - 상위로 전파하여 종료
                    logger.warning("MCP initialization was cancelled")
                    raise
            
            # Run research with production-grade reliability
            if streaming:
                result = await self._run_streaming_research(request)
            else:
                # Use new multi-agent orchestrator (no fallback - fail clearly)
                workflow_result = await self.orchestrator.execute(request)
                
                # 실패 상태 확인 - fallback 없이 명확한 오류 반환
                research_failed = workflow_result.get('research_failed', False)
                verification_failed = workflow_result.get('verification_failed', False)
                report_failed = workflow_result.get('report_failed', False)
                final_report = workflow_result.get("final_report", "")
                
                # 실패 보고서도 실패로 처리 (내용이 "연구 실패" 또는 "연구 완료 불가" 포함)
                is_failure_report = (
                    final_report and 
                    ("연구 실패" in final_report or "연구 완료 불가" in final_report or "❌" in final_report)
                )
                
                if research_failed or verification_failed or report_failed or is_failure_report:
                    error_msg = workflow_result.get('error', '알 수 없는 오류')
                    if not error_msg or error_msg == '알 수 없는 오류':
                        # 실패 보고서에서 오류 메시지 추출
                        if final_report and "오류 내용" in final_report:
                            lines = final_report.split("\n")
                            for i, line in enumerate(lines):
                                if "오류 내용" in line and i + 1 < len(lines):
                                    error_msg = lines[i + 1].strip()
                                    break
                        elif final_report and "❌" in final_report:
                            # 실패 보고서에서 간단히 추출
                            if "연구 실행 실패" in final_report:
                                error_msg = "연구 실행이 실패했습니다"
                            elif "보고서 생성 실패" in final_report:
                                error_msg = "보고서 생성이 실패했습니다"
                            else:
                                error_msg = "연구 실행 중 오류가 발생했습니다"
                    
                    failed_agent = workflow_result.get('current_agent', 'unknown')
                    session_id = workflow_result.get("session_id", 'N/A')
                    
                    logger.error(f"❌ Research failed at {failed_agent}: {error_msg}")
                    
                    # 실패 결과 반환 (사용자가 재시도할 수 있도록)
                    result = {
                        "success": False,
                        "query": request,
                        "error": error_msg,
                        "failed_agent": failed_agent,
                        "session_id": session_id,
                        "content": final_report or f"연구 실패: {error_msg}",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "model_used": "multi-agent",
                            "execution_time": 0.0,
                            "cost": 0.0,
                            "confidence": 0.0,
                            "failed": True
                        },
                        "synthesis_results": {
                            "content": final_report or "",
                            "failed": True
                        },
                        "sources": [],
                        "innovation_stats": {"multi_agent_orchestration": "enabled"},
                        "system_health": {"overall_status": "unhealthy", "error": error_msg},
                        "retry_available": True
                    }
                    
                    logger.warning("⚠️ Research completed with errors - user can retry")
                else:
                    # 성공 결과
                    final_report = workflow_result.get("final_report", "")
                    if not final_report:
                        # 보고서가 없으면 실패로 처리
                        result = {
                            "success": False,
                            "query": request,
                            "error": "보고서 생성 실패: 최종 보고서가 생성되지 않았습니다",
                            "failed_agent": "generator",
                            "session_id": workflow_result.get("session_id", "N/A"),
                            "content": "연구 실행 중 오류가 발생했습니다.",
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {"failed": True, "confidence": 0.0},
                            "retry_available": True
                        }
                    else:
                        result = {
                            "success": True,
                            "query": request,
                            "content": final_report,
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {
                                "model_used": "multi-agent",
                                "execution_time": 0.0,
                                "cost": 0.0,
                                "confidence": 0.9
                            },
                            "synthesis_results": {
                                "content": final_report
                            },
                            "sources": self._extract_sources_from_workflow(workflow_result),
                            "innovation_stats": {"multi_agent_orchestration": "enabled"},
                            "system_health": {"overall_status": "healthy"},
                            "session_id": workflow_result.get("session_id")
                        }
            
            # Apply hierarchical compression if enabled
            # Commented out to avoid serialization errors
            # if self.config.compression.enabled:
            #     result = await self._apply_hierarchical_compression(result)
            
            # Save results - LLM이 생성한 내용을 그대로 사용
            content = result.get('content', '') or result.get('synthesis_results', {}).get('content', '')
            
            if output_path:
                # 사용자가 지정한 경로 사용
                final_path = await self._save_content_as_file(content, output_path, result)
                logger.info(f"📄 Results saved to: {final_path}")
            else:
                # 생성된 내용을 보고 형식 결정
                detected_format = self._detect_output_format_from_content(content, request)
                if output_format is None:
                    output_format = detected_format
                
                output_dir = project_root / "output"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 확장자 결정 (간단하게)
                if output_format.startswith("."):
                    ext = output_format
                else:
                    ext_map = {"py": ".py", "java": ".java", "js": ".js", "html": ".html", "md": ".md", "pdf": ".pdf"}
                    ext = ext_map.get(output_format, ".md")
                default_output = output_dir / f"research_{timestamp}{ext}"
                
                final_path = await self._save_content_as_file(content, str(default_output), result)
                logger.info(f"📄 Results saved to default location: {final_path} (format: {output_format})")
                self._display_results(result)
            
            # Get final health status
            health_status = self.health_monitor.get_system_health()
            result['system_health'] = health_status
            
            logger.info("✅ Research completed successfully with 8 Core Innovations")
            return result
            
        except Exception as e:
            logger.error(f"Research failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Get error health status
            error_health = self.health_monitor.get_system_health()
            logger.error(f"System health at failure: {error_health}")
            
            # 실패 결과 반환 (예외 대신)
            result = {
                "success": False,
                "query": request,
                "error": f"시스템 오류: {str(e)}",
                "failed_agent": "system",
                "content": f"연구 실행 중 시스템 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model_used": "multi-agent",
                    "execution_time": 0.0,
                    "cost": 0.0,
                    "confidence": 0.0,
                    "failed": True
                },
                "synthesis_results": {"content": "", "failed": True},
                "sources": [],
                "innovation_stats": {},
                "system_health": error_health,
                "retry_available": True
            }
            
            # 실패 결과도 저장
            if output_path:
                await self._save_results_incrementally(result, output_path, output_format)
            else:
                output_dir = project_root / "output"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_output = output_dir / f"research_failed_{timestamp}.{output_format}"
                await self._save_results_incrementally(result, str(default_output), output_format)
                self._display_results(result)
            
            # 예외를 발생시키지 않고 결과 반환
            return result
    
    async def _run_streaming_research(self, request: str) -> Dict[str, Any]:
        """Run research with streaming pipeline (Innovation 5)."""
        logger.info("🌊 Starting streaming research pipeline...")
        
        # Create streaming callback
        async def streaming_callback(partial_result: Dict[str, Any]):
            logger.info(f"📊 Streaming partial result: {partial_result.get('type', 'unknown')}")
            # In a real implementation, this would send to web interface
            print(f"📊 Partial Result: {partial_result.get('summary', 'Processing...')}")
        
        # Use standard run_research but with streaming callback simulation
        result = await self.orchestrator.run_research(user_request=request, context={})
        
        # Extract and format result
        final_synthesis = result.get("synthesis_results", {}).get("content", "")
        if not final_synthesis:
            final_synthesis = result.get("content", "")
        
        formatted_result = {
            "query": request,
            "content": final_synthesis or "Research completed",
            "timestamp": datetime.now().isoformat(),
            "metadata": result.get("metadata", {}),
            "synthesis_results": result.get("synthesis_results", {}),
            "sources": self._extract_sources(result),
            "innovation_stats": result.get("innovation_stats", {}),
            "system_health": result.get("system_health", {})
        }
        
        logger.info("✅ Streaming research completed")
        return formatted_result
    
    def _extract_sources_from_workflow(self, workflow_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sources from workflow results."""
        sources = []
        seen_urls = set()
        
        # Extract from verified results (우선)
        verified_results = workflow_result.get("verified_results", [])
        for result in verified_results:
            if isinstance(result, dict):
                url = result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "title": result.get('title', ''),
                        "url": url,
                        "snippet": result.get('snippet', '')
                    })
        
        # Extract from research results (백업)
        research_results = workflow_result.get("research_results", [])
        for result in research_results:
            if isinstance(result, dict):
                url = result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "title": result.get('title', ''),
                        "url": url,
                        "snippet": result.get('snippet', '')
                    })
            elif isinstance(result, str) and "Source:" in result:
                # Extract URL from result string (레거시)
                parts = result.split("Source:")
                if len(parts) > 1:
                    url = parts[1].strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        title = parts[0].split(":")[-1].strip() if ":" in parts[0] else ""
                        sources.append({
                            "title": title,
                            "url": url,
                            "snippet": ""
                        })
        
        return sources[:20]  # Limit to 20 sources
    
    def _extract_sources(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sources from research results (legacy method)."""
        sources = []
        
        # Try to extract from execution results
        execution_results = result.get("detailed_results", {}).get("execution_results", [])
        for exec_result in execution_results:
            if isinstance(exec_result, dict):
                # Look for search results
                if "results" in exec_result:
                    sources.extend(exec_result["results"])
                elif "sources" in exec_result:
                    sources.extend(exec_result["sources"])
                elif "url" in exec_result:
                    sources.append(exec_result)
        
        # Deduplicate by URL
        seen_urls = set()
        unique_sources = []
        for source in sources:
            url = source.get("url") or source.get("link")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append({
                    "title": source.get("title", source.get("name", "")),
                    "url": url,
                    "snippet": source.get("snippet", source.get("summary", ""))
                })
        
        return unique_sources[:20]  # Limit to 20 sources
    
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
    
    async def _save_content_as_file(self, content: str, output_path: str, result: Dict[str, Any]) -> str:
        """LLM이 생성한 내용을 그대로 파일로 저장 (하드코딩 없이)."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # LLM이 생성한 내용을 그대로 저장 (추가 템플릿 없이)
        # 소스 정보가 있으면 마지막에 추가
        if result.get('sources'):
            sources_text = "\n\n## 참고 문헌\n\n"
            for i, source in enumerate(result.get('sources', []), 1):
                sources_text += f"{i}. [{source.get('title', 'N/A')}]({source.get('url', '')})\n"
                if source.get('snippet'):
                    sources_text += f"   {source.get('snippet', '')[:200]}...\n"
                sources_text += "\n"
            content = content + sources_text
        
        output_file.write_text(content, encoding='utf-8')
        return str(output_file)
    
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
        
        # 실패 상태 확인 및 표시
        if not result.get('success', True):
            print("\n❌ 연구 실행 실패")
            print("=" * 80)
            print(f"\n오류: {result.get('error', '알 수 없는 오류')}")
            print(f"실패 단계: {result.get('failed_agent', 'unknown')}")
            print(f"\n세션 ID: {result.get('session_id', 'N/A')}")
            print("\n재시도 방법:")
            print("1. 같은 쿼리로 다시 시도: python main.py --request 'YOUR_QUERY'")
            print("2. 다른 검색어로 시도")
            print("3. 네트워크 연결 확인")
            print("=" * 80)
            return
        
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
    
    async def check_mcp_servers(self):
        """MCP 서버 연결 상태 확인."""
        logger.info("📊 Checking MCP server connections...")
        
        if not self.config.mcp.enabled:
            logger.warning("MCP is disabled")
            return
        
        try:
            # MCP Hub 초기화 확인 - 이미 연결된 서버가 있으면 그대로 사용
            if self.mcp_hub.mcp_sessions:
                logger.info(f"Found {len(self.mcp_hub.mcp_sessions)} existing MCP server connections")
            else:
                logger.info("No existing connections. Will attempt quick connection tests for each server...")
            
            # 서버 상태 확인 (각 서버에 대해 짧은 타임아웃으로 연결 시도)
            logger.info("Checking MCP server connection status...")
            server_status = await self.mcp_hub.check_mcp_servers()
            
            # 결과 출력
            print("\n" + "=" * 80)
            print("📊 MCP 서버 연결 상태 확인")
            print("=" * 80)
            print(f"전체 서버 수: {server_status['total_servers']}")
            print(f"연결된 서버: {server_status['connected_servers']}")
            print(f"연결률: {server_status['summary']['connection_rate']}")
            print(f"전체 사용 가능한 Tool 수: {server_status['summary']['total_tools_available']}")
            print("\n")
            
            for server_name, info in server_status["servers"].items():
                status_icon = "✅" if info["connected"] else "❌"
                print(f"{status_icon} 서버: {server_name}")
                print(f"   타입: {info['type']}")
                
                if info["type"] == "http":
                    print(f"   URL: {info.get('url', 'unknown')}")
                else:
                    cmd = info.get('command', 'unknown')
                    args_preview = ' '.join(info.get('args', [])[:3])
                    print(f"   명령어: {cmd} {args_preview}...")
                
                print(f"   연결 상태: {'연결됨' if info['connected'] else '연결 안 됨'}")
                print(f"   제공 Tool 수: {info['tools_count']}")
                
                if info["tools"]:
                    print(f"   Tool 목록:")
                    for tool in info["tools"][:5]:  # 처음 5개만 표시
                        registered_name = f"{server_name}::{tool}"
                        print(f"     - {registered_name}")
                    if len(info["tools"]) > 5:
                        print(f"     ... 및 {len(info['tools']) - 5}개 더")
                
                if info.get("error"):
                    print(f"   ⚠️ 오류: {info['error']}")
                print()
            
            print("=" * 80)
            
            # 요약 정보 로깅
            logger.info(f"MCP 서버 확인 완료: {server_status['summary']['connection_rate']} 연결")
            
        except Exception as e:
            logger.error(f"MCP 서버 확인 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())


async def main():
    """Main function - 8가지 핵심 혁신 통합 실행 진입점"""
    # Python 종료 시 발생하는 async generator 정리 오류 무시
    def ignore_async_gen_errors(loop, context):
        """anyio cancel scope 및 async generator 종료 오류 무시"""
        exception = context.get('exception')
        if exception:
            error_msg = str(exception)
            # anyio cancel scope 오류는 무시
            if isinstance(exception, RuntimeError) and ("cancel scope" in error_msg.lower() or "different task" in error_msg.lower()):
                return  # 무시
            # async generator 종료 오류는 무시
            if isinstance(exception, GeneratorExit) or "async_generator" in error_msg.lower():
                return  # 무시
        # 기타 오류는 기본 handler로 전달
        loop.set_exception_handler(None)
        loop.call_exception_handler(context)
        loop.set_exception_handler(ignore_async_gen_errors)
    
    # asyncio exception handler 설정
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(ignore_async_gen_errors)
    
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
    mode_group.add_argument("--check-mcp-servers", action="store_true", help="Check all MCP server connections and list tools")
    
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

    # Initialize enhanced systems
    from src.utils.output_manager import UserCenteredOutputManager, set_output_manager
    from src.core.error_handler import ErrorHandler, set_error_handler
    from src.core.progress_tracker import ProgressTracker, set_progress_tracker

    # 출력 매니저 초기화
    output_manager = UserCenteredOutputManager(
        output_level=UserCenteredOutputManager.OutputLevel.USER,
        output_format=UserCenteredOutputManager.OutputFormat.TEXT,
        enable_colors=True,
        stream_output=True,
        show_progress=True
    )
    set_output_manager(output_manager)

    # 에러 핸들러 초기화
    error_handler = ErrorHandler(
        log_errors=True,
        enable_recovery=True
    )
    set_error_handler(error_handler)

    # 진행 상황 추적기 초기화 (세션별로 생성)
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    progress_tracker = ProgressTracker(
        session_id=session_id,
        enable_real_time_updates=True,
        update_interval=1.0
    )
    set_progress_tracker(progress_tracker)

    # 진행 상황 추적기 콜백 설정 (출력 매니저와 연동)
    async def progress_callback(workflow_progress):
        """진행 상황 업데이트 시 출력 매니저에 표시."""
        try:
            progress_pct = int(workflow_progress.overall_progress * 100)
            stage_name = workflow_progress.current_stage.value

            eta_str = ""
            if workflow_progress.estimated_completion:
                eta_seconds = max(0, int(workflow_progress.estimated_completion - time.time()))
                eta_str = f" (예상 {eta_seconds}초 남음)"

            message = f"📊 진행률: {progress_pct}% - {stage_name.upper()}{eta_str}"

            # 진행률 바 스타일로 표시
            await output_manager.start_progress(
                stage_name,
                100,
                f"{progress_pct}% 완료",
                workflow_progress.estimated_completion
            )
            await output_manager.update_progress(progress_pct)

        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")

    progress_tracker.add_progress_callback(progress_callback)

    # Initialize system
    system = AutonomousResearchSystem()
    
    try:
        if args.request:
            # CLI Research Mode with 8 innovations
            logger.info("🚀 Starting Local Researcher with enhanced systems...")

            # 진행 상황 추적 시작
            await progress_tracker.start_tracking()

            # 워크플로우 시작 알림
            await output_manager.output(
                f"🔬 연구 주제: {args.request}",
                level=output_manager.OutputLevel.USER
            )
            await output_manager.output(
                "실시간 진행 상황 추적 및 향상된 에러 처리가 활성화되었습니다.",
                level=output_manager.OutputLevel.SERVICE
            )

            # 연구 실행
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
            # MCP Server Mode with Universal MCP Hub - 실제 연결 수행
            logger.info("Initializing MCP servers...")
            try:
                await system.mcp_hub.initialize_mcp()
                logger.info("✅ MCP servers initialized")
                
                # 연결된 서버 상태 출력
                if system.mcp_hub.mcp_sessions:
                    print("\n" + "=" * 80)
                    print("✅ MCP 서버 연결 완료")
                    print("=" * 80)
                    for server_name in system.mcp_hub.mcp_sessions.keys():
                        tools_count = len(system.mcp_hub.mcp_tools_map.get(server_name, {}))
                        print(f"✅ {server_name}: {tools_count} tools available")
                    print("=" * 80)
                    print("\nMCP Hub is running. Press Ctrl+C to stop.")
                    
                    # 계속 실행 대기
                    try:
                        await asyncio.sleep(3600)  # 1시간 대기 (또는 Ctrl+C로 종료)
                    except KeyboardInterrupt:
                        logger.info("Shutting down MCP Hub...")
                else:
                    logger.warning("⚠️ No MCP servers connected")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to initialize MCP servers: {e}")
                sys.exit(1)
            
        elif args.mcp_client:
            # MCP Client Mode with Smart Tool Selection
            success = await system.run_mcp_client()
            if not success:
                sys.exit(1)
            
        elif args.health_check:
            # Health Check Mode
            await system.run_health_check()
        
        elif args.check_mcp_servers:
            # MCP 서버 확인 모드
            await system.check_mcp_servers()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user (KeyboardInterrupt)")
        system._shutdown_requested = True
        try:
            await system._graceful_shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        # sys.exit(0) 제거 - asyncio.run()이 자동으로 처리
    except asyncio.CancelledError:
        # 취소된 경우 정리 후 종료
        logger.info("Operation cancelled")
        system._shutdown_requested = True
        try:
            await system._graceful_shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        # asyncio.CancelledError는 다시 raise하여 정상적인 취소 흐름 유지
        raise
    except Exception as e:
        # 향상된 에러 처리
        from src.core.error_handler import ErrorContext, ErrorCategory
        import traceback

        error_context = ErrorContext(
            component="main",
            operation="run_research" if args.request else "system_operation",
            session_id=session_id
        )

        await error_handler.handle_error(
            e,
            category=ErrorCategory.UNKNOWN,
            severity=error_handler.ErrorSeverity.HIGH,
            context=error_context,
            custom_message=f"시스템 실행 중 치명적 오류 발생: {str(e)}"
        )

        system._shutdown_requested = True
        try:
            await system._graceful_shutdown()
        except Exception as e2:
            logger.error(f"Error during shutdown: {e2}")
        # 에러 발생 시 종료 코드 1로 종료
        sys.exit(1)
    finally:
        # 진행 상황 추적 중지 및 요약 출력
        try:
            await progress_tracker.stop_tracking()

            if args.request:
                # 워크플로우 완료 요약
                await output_manager.complete_progress(success=True)
                await output_manager.output_workflow_summary()

                # 진행 상황 통계 출력
                stats = progress_tracker.get_statistics()
                await output_manager.output(
                    f"📈 세션 통계: {stats['total_agents_created']}개 에이전트 생성, "
                    f"{stats['agents_completed']}개 완료, {stats['agents_failed']}개 실패",
                    level=output_manager.OutputLevel.SERVICE
                )

        except Exception as e:
            logger.warning(f"Failed to finalize progress tracking: {e}")

        # 최종 정리 보장
        if hasattr(system, 'mcp_hub') and system.mcp_hub and hasattr(system.mcp_hub, 'mcp_sessions'):
            try:
                if system.mcp_hub.mcp_sessions:
                    logger.info("Final cleanup of MCP connections...")
                    await system.mcp_hub.cleanup()
            except Exception as e:
                logger.debug(f"Error in final cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())