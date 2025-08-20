#!/usr/bin/env python3
"""
Main entry point for Ethereum Trading Multi-Agent System
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Check Python version for LangChain 0.3.0 compatibility
if sys.version_info < (3, 9):
    print("Error: Python 3.9 or higher is required for LangChain 0.3.0")
    print(f"Current Python version: {sys.version}")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cron_agents.ethereum_trading_agents.multi_agent_orchestrator import MultiAgentOrchestrator
from cron_agents.ethereum_trading_agents.trading_agent import TradingAgent
from cron_agents.ethereum_trading_agents.langchain_agent import LangChainTradingAgent
from cron_agents.ethereum_trading_agents.config import Config

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

async def main():
    """Main function"""
    try:
        # Validate configuration
        Config.validate()
        logger.info("Configuration validated successfully")
        
        # Create orchestrator
        orchestrator = MultiAgentOrchestrator()
        
        # Create and register trading agents
        # You can create multiple agents with different strategies
        # Traditional agents
        conservative_agent = TradingAgent("conservative_trader")
        aggressive_agent = TradingAgent("aggressive_trader")
        balanced_agent = TradingAgent("balanced_trader")
        
        # LangChain enhanced agents (LangChain 0.3.0 features)
        langchain_conservative = LangChainTradingAgent("langchain_conservative")
        langchain_aggressive = LangChainTradingAgent("langchain_aggressive")
        langchain_balanced = LangChainTradingAgent("langchain_balanced")
        
        # Register traditional agents with orchestrator
        orchestrator.register_agent("conservative_trader", conservative_agent)
        orchestrator.register_agent("aggressive_trader", aggressive_agent)
        orchestrator.register_agent("balanced_trader", balanced_agent)
        
        # Register LangChain enhanced agents
        orchestrator.register_agent("langchain_conservative", langchain_conservative)
        orchestrator.register_agent("langchain_aggressive", langchain_aggressive)
        orchestrator.register_agent("langchain_balanced", langchain_balanced)
        
        logger.info("All agents registered successfully")
        
        # Start orchestrator
        logger.info("Starting Ethereum trading multi-agent system...")
        await orchestrator.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
