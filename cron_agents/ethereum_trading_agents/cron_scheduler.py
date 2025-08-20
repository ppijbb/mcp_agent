#!/usr/bin/env python3
"""
Cron Scheduler for Ethereum Trading Agents
Executes agents every 5 minutes
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import signal

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
        logging.FileHandler('ethereum_trading_cron.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class CronScheduler:
    def __init__(self):
        """Initialize cron scheduler"""
        self.config = Config()
        self.orchestrator = MultiAgentOrchestrator()
        self.running = False
        self.execution_interval = timedelta(minutes=self.config.AGENT_EXECUTION_INTERVAL_MINUTES)
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Cron scheduler initialized")
    
    def setup_agents(self):
        """Setup and register trading agents"""
        try:
            # Create trading agents with different strategies
            conservative_agent = TradingAgent("conservative_trader")
            aggressive_agent = TradingAgent("aggressive_trader")
            balanced_agent = TradingAgent("balanced_trader")
            
            # Register agents with orchestrator
            self.orchestrator.register_agent("conservative_trader", conservative_agent)
            self.orchestrator.register_agent("aggressive_trader", aggressive_agent)
            self.orchestrator.register_agent("balanced_trader", balanced_agent)
            
            logger.info("All agents registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup agents: {e}")
            raise
    
    async def start_scheduler(self):
        """Start the cron scheduler"""
        try:
            self.running = True
            logger.info("Starting cron scheduler")
            
            # Setup agents
            self.setup_agents()
            
            # Main scheduling loop
            await self._scheduling_loop()
            
        except Exception as e:
            logger.error(f"Scheduler failed to start: {e}")
            raise
        finally:
            await self.stop_scheduler()
    
    async def stop_scheduler(self):
        """Stop the cron scheduler"""
        try:
            self.running = False
            logger.info("Stopping cron scheduler")
            
            # Stop orchestrator
            await self.orchestrator.stop()
            
            logger.info("Cron scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    async def _scheduling_loop(self):
        """Main scheduling loop"""
        try:
            last_execution = datetime.now()
            
            while self.running:
                try:
                    now = datetime.now()
                    
                    # Check if it's time to execute
                    if now - last_execution >= self.execution_interval:
                        logger.info(f"Executing scheduled trading cycle at {now}")
                        
                        # Execute trading cycle for all agents
                        await self._execute_all_agents()
                        
                        last_execution = now
                        
                        # Log next execution time
                        next_execution = now + self.execution_interval
                        logger.info(f"Next execution scheduled for {next_execution}")
                    
                    # Wait before next check (check every minute)
                    await asyncio.sleep(60)
                    
                except asyncio.CancelledError:
                    logger.info("Scheduling loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in scheduling loop: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
            
        except Exception as e:
            logger.error(f"Fatal error in scheduling loop: {e}")
    
    async def _execute_all_agents(self):
        """Execute trading cycle for all registered agents"""
        try:
            execution_tasks = []
            
            # Create execution tasks for all agents
            for agent_name in self.orchestrator.agents:
                task = asyncio.create_task(
                    self.orchestrator.execute_manual_cycle(agent_name)
                )
                execution_tasks.append(task)
            
            # Execute all agents concurrently
            if execution_tasks:
                logger.info(f"Executing {len(execution_tasks)} agents concurrently")
                
                # Wait for all executions to complete with timeout
                timeout = 300  # 5 minutes timeout
                results = await asyncio.wait_for(
                    asyncio.gather(*execution_tasks, return_exceptions=True),
                    timeout=timeout
                )
                
                # Process results
                for i, result in enumerate(results):
                    agent_name = list(self.orchestrator.agents.keys())[i]
                    if isinstance(result, Exception):
                        logger.error(f"Agent {agent_name} execution failed: {result}")
                    else:
                        if result["status"] == "success":
                            logger.info(f"Agent {agent_name} completed successfully")
                        else:
                            logger.warning(f"Agent {agent_name} completed with warnings: {result.get('message', 'Unknown warning')}")
                
                logger.info("All agent executions completed")
            else:
                logger.warning("No agents registered for execution")
                
        except asyncio.TimeoutError:
            logger.error("Agent execution timeout - some agents may still be running")
        except Exception as e:
            logger.error(f"Failed to execute agents: {e}")
    
    async def get_scheduler_status(self) -> dict:
        """Get current scheduler status"""
        try:
            orchestrator_status = await self.orchestrator.get_orchestrator_status()
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "scheduler_running": self.running,
                "execution_interval_minutes": self.config.AGENT_EXECUTION_INTERVAL_MINUTES,
                "next_execution": datetime.now() + self.execution_interval,
                "orchestrator_status": orchestrator_status
            }
            
        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e)
            }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
        
        # Schedule shutdown
        asyncio.create_task(self.stop_scheduler())
    
    async def graceful_shutdown(self):
        """Perform graceful shutdown"""
        try:
            logger.info("Initiating graceful shutdown")
            
            # Stop accepting new tasks
            self.running = False
            
            # Wait for current executions to complete (with timeout)
            shutdown_timeout = 300  # 5 minutes
            
            # Stop orchestrator
            await asyncio.wait_for(self.orchestrator.stop(), timeout=shutdown_timeout)
            
            logger.info("Graceful shutdown completed")
            
        except asyncio.TimeoutError:
            logger.warning("Shutdown timeout, forcing stop")
            await self.orchestrator.stop()
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            # Force shutdown
            sys.exit(1)

async def main():
    """Main function"""
    try:
        # Validate configuration
        Config.validate()
        logger.info("Configuration validated successfully")
        
        # Create and start scheduler
        scheduler = CronScheduler()
        logger.info("Starting Ethereum trading cron scheduler...")
        await scheduler.start_scheduler()
        
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
