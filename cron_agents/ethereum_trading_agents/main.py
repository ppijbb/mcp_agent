#!/usr/bin/env python3
"""
Main entry point for Ethereum Trading Multi-Agent System
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cron_agents.ethereum_trading_agents.multi_agent_orchestrator import MultiAgentOrchestrator
from cron_agents.ethereum_trading_agents.trading_agent import TradingAgent
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
        conservative_agent = TradingAgent("conservative_trader")
        aggressive_agent = TradingAgent("aggressive_trader")
        balanced_agent = TradingAgent("balanced_trader")
        
        # Register agents with orchestrator
        orchestrator.register_agent("conservative_trader", conservative_agent)
        orchestrator.register_agent("aggressive_trader", aggressive_agent)
        orchestrator.register_agent("balanced_trader", balanced_agent)
        
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
