#!/usr/bin/env python3
"""
Main entry point for Ethereum Trading Multi-Agent System
Enhanced with LangChain-based modular architecture
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Check Python version for LangChain 0.3.0 compatibility
if sys.version_info < (3, 9):
    print("Error: Python 3.9 or higher is required for LangChain 0.3.0")
    print(f"Current Python version: {sys.version}")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Import from new modular structure
from .agents.multi_agent_orchestrator import MultiAgentOrchestrator
from .agents.trading_agent import TradingAgent
from .agents.langchain_agent import TradingAgentChain
from .agents.gemini_agent import GeminiAgent
from .agents.trading_report_agent import TradingReportAgent
from .chains.trading_chain import TradingChain
from .chains.analysis_chain import AnalysisChain
from .memory.trading_memory import TradingMemory
from .prompts import get_prompt, list_available_prompts
from .utils.config import Config
from .utils.database import TradingDatabase
from .utils.mcp_client import MCPClient
from .utils.cron_scheduler import CronScheduler
from .utils.email_service import EmailService
from .utils.trading_monitor import TradingMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ethereum_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class EthereumTradingSystem:
    """Main system orchestrator for Ethereum trading agents"""
    
    def __init__(self):
        self.config = Config()
        self.database = None
        self.mcp_client = None
        self.cron_scheduler = None
        self.trading_memory = None
        self.agents = {}
        self.chains = {}
        self.orchestrator = None
        self.email_service = None
        self.trading_monitor = None
        self.trading_report_agent = None
        
    async def initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Ethereum Trading System...")
            
            # Initialize database
            self.database = TradingDatabase()
            await self.database.connect()
            logger.info("Database initialized successfully")
            
            # Initialize MCP client
            self.mcp_client = MCPClient()
            await self.mcp_client.connect()
            logger.info("MCP client initialized successfully")
            
            # Initialize data collector
            from .utils.data_collector import DataCollector
            self.data_collector = DataCollector()
            await self.data_collector.connect()
            logger.info("Data collector initialized successfully")
            
            # Initialize cron scheduler
            self.cron_scheduler = CronScheduler()
            logger.info("Cron scheduler initialized successfully")
            
            # Initialize trading memory
            self.trading_memory = TradingMemory()
            logger.info("Trading memory initialized successfully")
            
            # Initialize email service
            self.email_service = EmailService()
            logger.info("Email service initialized successfully")
            
            # Initialize trading report agent
            self.trading_report_agent = TradingReportAgent(
                self.mcp_client, 
                self.data_collector, 
                self.email_service
            )
            logger.info("Trading report agent initialized successfully")
            
            # Initialize trading monitor
            self.trading_monitor = TradingMonitor(
                self.mcp_client,
                self.data_collector,
                self.email_service,
                self.trading_report_agent
            )
            logger.info("Trading monitor initialized successfully")
            
            # Initialize agents
            await self._initialize_agents()
            logger.info("Agents initialized successfully")
            
            # Initialize chains
            await self._initialize_chains()
            logger.info("Chains initialized successfully")
            
            # Initialize orchestrator
            await self._initialize_orchestrator()
            logger.info("Orchestrator initialized successfully")
            
            # Start trading monitor
            await self.trading_monitor.start_monitoring()
            logger.info("Trading monitor started successfully")
            
            # Setup cron jobs
            await self._setup_cron_jobs()
            logger.info("Cron jobs setup completed")
            
            logger.info("Ethereum Trading System initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _initialize_agents(self):
        """Initialize all trading agents"""
        try:
            # Initialize traditional trading agents
            self.agents['conservative_trader'] = TradingAgent("conservative_trader")
            self.agents['aggressive_trader'] = TradingAgent("aggressive_trader")
            self.agents['balanced_trader'] = TradingAgent("balanced_trader")
            
            # Initialize LangChain enhanced agents
            self.agents['langchain_conservative'] = TradingAgentChain("langchain_conservative")
            self.agents['langchain_aggressive'] = TradingAgentChain("langchain_aggressive")
            self.agents['langchain_balanced'] = TradingAgentChain("langchain_balanced")
            
            # Initialize Gemini agent
            self.agents['gemini'] = GeminiAgent(
                database=self.database,
                mcp_client=self.mcp_client,
                memory=self.trading_memory
            )
            
            logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def _initialize_chains(self):
        """Initialize LangChain chains"""
        try:
            # Get LLM from LangChain agent
            llm = self.agents['langchain_conservative'].get_llm()
            
            # Initialize trading chain with config
            self.chains['trading'] = TradingChain(
                llm=llm,
                trading_agent=self.agents['conservative_trader'],
                analysis_agent=self.agents['gemini'],
                config=self.config
            )
            
            # Initialize analysis chain
            self.chains['analysis'] = AnalysisChain(llm=llm)
            
            logger.info(f"Initialized {len(self.chains)} chains")
            
        except Exception as e:
            logger.error(f"Failed to initialize chains: {e}")
            raise
    
    async def _initialize_orchestrator(self):
        """Initialize multi-agent orchestrator"""
        try:
            self.orchestrator = MultiAgentOrchestrator()
            
            # Register all agents with orchestrator
            for name, agent in self.agents.items():
                self.orchestrator.register_agent(name, agent)
            
            logger.info("Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def _setup_cron_jobs(self):
        """Setup automated cron jobs"""
        try:
            # Market analysis job (every 15 minutes)
            await self.cron_scheduler.add_job(
                func=self._run_market_analysis,
                trigger="interval",
                minutes=15,
                id="market_analysis"
            )
            
            # Portfolio review job (daily at 9 AM)
            await self.cron_scheduler.add_job(
                func=self._run_portfolio_review,
                trigger="cron",
                hour=9,
                minute=0,
                id="portfolio_review"
            )
            
            # Performance analysis job (weekly on Sunday)
            await self.cron_scheduler.add_job(
                func=self._run_performance_analysis,
                trigger="cron",
                day_of_week="sun",
                hour=10,
                minute=0,
                id="performance_analysis"
            )
            
            # Start the scheduler
            await self.cron_scheduler.start()
            
            logger.info("Cron jobs setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup cron jobs: {e}")
            raise
    
    async def _run_market_analysis(self):
        """Run automated market analysis"""
        try:
            logger.info("Running automated market analysis...")
            
            # Get market data
            market_data = await self._get_market_data()
            
            # Run analysis using analysis chain
            analysis_results = await self.chains['analysis'].execute_comprehensive_analysis(
                market_data=market_data,
                analysis_type="all"
            )
            
            # Store results in memory
            await self.trading_memory.store(
                key="latest_market_analysis",
                value=analysis_results,
                memory_type=self.trading_memory.MemoryType.HISTORICAL_DATA,
                ttl=3600  # 1 hour
            )
            
            logger.info("Market analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
    
    async def _run_portfolio_review(self):
        """Run automated portfolio review"""
        try:
            logger.info("Running automated portfolio review...")
            
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data()
            
            # Use portfolio review prompt
            portfolio_prompt = get_prompt("portfolio_review")
            
            # Run review using orchestrator
            review_results = await self.orchestrator.execute_portfolio_review(
                portfolio_data=portfolio_data
            )
            
            # Store results
            await self.trading_memory.store(
                key="latest_portfolio_review",
                value=review_results,
                memory_type=self.trading_memory.MemoryType.PERFORMANCE,
                ttl=86400  # 24 hours
            )
            
            logger.info("Portfolio review completed successfully")
            
        except Exception as e:
            logger.error(f"Portfolio review failed: {e}")
    
    async def _run_performance_analysis(self):
        """Run automated performance analysis"""
        try:
            logger.info("Running automated performance analysis...")
            
            # Get performance data
            performance_data = await self._get_performance_data()
            
            # Use performance analysis prompt
            performance_prompt = get_prompt("performance_analysis")
            
            # Run analysis using orchestrator
            analysis_results = await self.orchestrator.execute_performance_analysis(
                performance_data=performance_data
            )
            
            # Store results
            await self.trading_memory.store(
                key="latest_performance_analysis",
                value=analysis_results,
                memory_type=self.trading_memory.MemoryType.PERFORMANCE,
                ttl=604800  # 1 week
            )
            
            logger.info("Performance analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data"""
        # This would typically fetch from external APIs
        return {
            "price_data": {},
            "volume_data": {},
            "indicators": {},
            "token_metrics": {},
            "network_stats": {},
            "ecosystem_data": {},
            "social_media": {},
            "news_data": {},
            "community_sentiment": {},
            "historical_patterns": {},
            "current_market": {},
            "timeframe": "1h"
        }
    
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get current portfolio data"""
        # This would fetch from database
        return {
            "portfolio_positions": {},
            "performance_metrics": {},
            "market_outlook": {}
        }
    
    async def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data"""
        # This would fetch from database
        return {
            "trading_history": {},
            "performance_metrics": {},
            "benchmark_data": {}
        }
    
    async def execute_trading_workflow(
        self,
        market_data: Dict[str, Any],
        trading_strategy: str,
        portfolio_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute complete trading workflow"""
        try:
            logger.info("Executing trading workflow...")
            
            # Use trading chain for workflow execution
            workflow_results = await self.chains['trading'].execute_trading_workflow(
                market_data=market_data,
                historical_data={},  # Would fetch from database
                trading_strategy=trading_strategy,
                portfolio_status=portfolio_status
            )
            
            # Store workflow results in memory
            await self.trading_memory.store(
                key=f"workflow_{workflow_results['timestamp']}",
                value=workflow_results,
                memory_type=self.trading_memory.MemoryType.TRADING_CONTEXT,
                ttl=3600  # 1 hour
            )
            
            logger.info("Trading workflow completed successfully")
            return workflow_results
            
        except Exception as e:
            logger.error(f"Trading workflow failed: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            status = {
                "system": "Ethereum Trading System",
                "status": "running",
                "timestamp": self._get_timestamp(),
                "components": {
                    "database": "connected" if self.database else "disconnected",
                    "mcp_client": "connected" if self.mcp_client else "disconnected",
                    "cron_scheduler": "running" if self.cron_scheduler else "stopped",
                    "agents": len(self.agents),
                    "chains": len(self.chains),
                    "orchestrator": "initialized" if self.orchestrator else "not_initialized",
                    "email_service": "initialized" if self.email_service else "not_initialized",
                    "trading_monitor": "running" if self.trading_monitor and self.trading_monitor.monitoring_active else "stopped",
                    "trading_report_agent": "initialized" if self.trading_report_agent else "not_initialized"
                },
                "memory_stats": self.trading_memory.get_memory_stats() if self.trading_memory else {},
                "available_prompts": list_available_prompts()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        try:
            logger.info("Shutting down Ethereum Trading System...")
            
            # Stop trading monitor
            if self.trading_monitor:
                await self.trading_monitor.stop_monitoring()
            
            # Stop cron scheduler
            if self.cron_scheduler:
                await self.cron_scheduler.shutdown()
            
            # Close database connection
            if self.database:
                await self.database.close()
            
            # Close MCP client
            if self.mcp_client:
                await self.mcp_client.close()
            
            # Close data collector
            if self.data_collector:
                await self.data_collector.close()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def main():
    """Main entry point"""
    try:
        # Validate configuration
        Config.validate()
        logger.info("Configuration validated successfully")
        
        # Create and initialize system
        system = EthereumTradingSystem()
        await system.initialize_system()
        
        # Display system status
        status = await system.get_system_status()
        print("=== Ethereum Trading System Status ===")
        print(f"Status: {status['status']}")
        print(f"Components: {status['components']}")
        print(f"Available Prompts: {len(status['available_prompts'])}")
        print("=====================================")
        
        # Start orchestrator
        logger.info("Starting Ethereum trading multi-agent system...")
        await system.orchestrator.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        if 'system' in locals():
            await system.shutdown()
    except Exception as e:
        logger.error(f"System failed to start: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
