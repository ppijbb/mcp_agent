#!/usr/bin/env python3
"""
Ethereum Trading System Runner

This script starts the Ethereum trading system with monitoring and reporting capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
    """Main entry point for the trading system"""
    try:
        logger.info("🚀 Starting Ethereum Trading System with Monitoring...")

        # Import and initialize system
        from ethereum_trading_agents.main import EthereumTradingSystem

        # Create system instance
        system = EthereumTradingSystem()

        # Initialize system
        logger.info("Initializing system components...")
        await system.initialize_system()

        # Display system status
        status = await system.get_system_status()
        print("\n" + "="*60)
        print("🎯 ETHEREUM TRADING SYSTEM STATUS")
        print("="*60)
        print(f"Status: {status['status']}")
        print(f"Timestamp: {status['timestamp']}")
        print("\n📊 COMPONENTS:")
        for component, status_info in status['components'].items():
            print(f"  • {component}: {status_info}")
        print("\n📧 EMAIL & MONITORING:")
        print(f"  • Email Service: {status['components']['email_service']}")
        print(f"  • Trading Monitor: {status['components']['trading_monitor']}")
        print(f"  • Trading Report Agent: {status['components']['trading_report_agent']}")
        print("="*60)

        # Start the system
        logger.info("Starting Ethereum trading multi-agent system...")
        await system.orchestrator.start()

        # Keep the system running
        logger.info("System is now running. Press Ctrl+C to stop...")

        try:
            # Keep alive
            while True:
                await asyncio.sleep(60)  # Check every minute

                # Display periodic status
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] System running - Monitoring active")

        except KeyboardInterrupt:
            logger.info("Received shutdown signal...")

    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise
    finally:
        # Ensure clean shutdown
        if 'system' in locals():
            logger.info("Shutting down system...")
            await system.shutdown()
            logger.info("System shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
