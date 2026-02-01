"""
Multi-Agent Orchestrator for Ethereum Trading
Coordinates multiple trading agents and manages their execution cycles
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

from agents.trading_agent import TradingAgent
from utils.database import TradingDatabase
from utils.config import Config

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    def __init__(self):
        """Initialize multi-agent orchestrator"""
        self.config = Config()
        self.database = TradingDatabase()

        # Agent registry
        self.agents: Dict[str, TradingAgent] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}

        # Execution control
        self.running = False
        self.execution_interval = timedelta(minutes=self.config.AGENT_EXECUTION_INTERVAL_MINUTES)

        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_CONCURRENT_AGENTS)

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Multi-agent orchestrator initialized")

    def register_agent(self, agent_name: str, agent: TradingAgent):
        """Register a trading agent"""
        self.agents[agent_name] = agent
        logger.info(f"Agent {agent_name} registered")

    def unregister_agent(self, agent_name: str):
        """Unregister a trading agent"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Agent {agent_name} unregistered")

    async def start(self):
        """Start the orchestrator and all agents"""
        try:
            self.running = True
            logger.info("Starting multi-agent orchestrator")

            # Start all registered agents
            for agent_name in self.agents:
                await self._start_agent(agent_name)

            # Main orchestration loop
            await self._orchestration_loop()

        except Exception as e:
            logger.error(f"Orchestrator failed to start: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop the orchestrator and all agents"""
        try:
            self.running = False
            logger.info("Stopping multi-agent orchestrator")

            # Stop all agent tasks
            for agent_name, task in self.agent_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Shutdown thread pool
            self.executor.shutdown(wait=True)

            logger.info("Multi-agent orchestrator stopped")

        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")

    async def _start_agent(self, agent_name: str):
        """Start a specific agent"""
        try:
            if agent_name in self.agent_tasks and not self.agent_tasks[agent_name].done():
                logger.warning(f"Agent {agent_name} is already running")
                return

            # Create agent task
            task = asyncio.create_task(self._agent_execution_loop(agent_name))
            self.agent_tasks[agent_name] = task

            logger.info(f"Agent {agent_name} started")

        except Exception as e:
            logger.error(f"Failed to start agent {agent_name}: {e}")

    async def _agent_execution_loop(self, agent_name: str):
        """Main execution loop for a single agent"""
        try:
            agent = self.agents[agent_name]
            last_execution = datetime.now()

            while self.running:
                try:
                    # Check if it's time to execute
                    now = datetime.now()
                    if now - last_execution >= self.execution_interval:
                        logger.info(f"Executing trading cycle for agent {agent_name}")

                        # Execute trading cycle
                        result = await agent.execute_trading_cycle()

                        if result["status"] == "success":
                            logger.info(f"Agent {agent_name} completed trading cycle {result['execution_id']}")
                        else:
                            logger.error(f"Agent {agent_name} failed: {result.get('error_message', 'Unknown error')}")

                        last_execution = now

                    # Wait before next check
                    await asyncio.sleep(60)  # Check every minute

                except asyncio.CancelledError:
                    logger.info(f"Agent {agent_name} execution loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in agent {agent_name} execution loop: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retry

        except Exception as e:
            logger.error(f"Fatal error in agent {agent_name} execution loop: {e}")

    async def _orchestration_loop(self):
        """Main orchestration loop"""
        try:
            while self.running:
                try:
                    # Monitor agent health
                    await self._monitor_agent_health()

                    # Check system resources
                    await self._check_system_resources()

                    # Wait before next check
                    await asyncio.sleep(300)  # Check every 5 minutes

                except asyncio.CancelledError:
                    logger.info("Orchestration loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in orchestration loop: {e}")
                    await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Fatal error in orchestration loop: {e}")

    async def _monitor_agent_health(self):
        """Monitor health of all agents"""
        try:
            for agent_name, agent in self.agents.items():
                try:
                    # Get agent status
                    status = await agent.get_agent_status()

                    if status["status"] != "success":
                        logger.warning(f"Agent {agent_name} health check failed: {status.get('error_message', 'Unknown error')}")

                        # Attempt to restart agent if needed
                        if self._should_restart_agent(agent_name, status):
                            logger.info(f"Restarting agent {agent_name}")
                            await self._restart_agent(agent_name)

                except Exception as e:
                    logger.error(f"Failed to check health of agent {agent_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to monitor agent health: {e}")

    def _should_restart_agent(self, agent_name: str, status: Dict) -> bool:
        """Determine if an agent should be restarted"""
        try:
            # Check if agent has been failing repeatedly
            last_execution = status.get("last_execution")
            if not last_execution:
                return True

            # Check if last execution was more than 2 intervals ago
            last_execution_time = datetime.fromisoformat(last_execution["execution_time"])
            if datetime.now() - last_execution_time > self.execution_interval * 2:
                return True

            return False

        except Exception as e:
            logger.error(f"Error determining if agent {agent_name} should restart: {e}")
            return False

    async def _restart_agent(self, agent_name: str):
        """Restart a specific agent"""
        try:
            # Stop current task
            if agent_name in self.agent_tasks:
                task = self.agent_tasks[agent_name]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Wait a moment before restarting
            await asyncio.sleep(10)

            # Start agent again
            await self._start_agent(agent_name)

        except Exception as e:
            logger.error(f"Failed to restart agent {agent_name}: {e}")

    async def _check_system_resources(self):
        """Check system resources and adjust if needed"""
        try:
            # Get daily trading summary
            daily_summary = self.database.get_daily_trading_summary()

            # Check if we're approaching daily limits
            if daily_summary["total_trades"] >= self.config.MAX_DAILY_TRADES * 0.8:
                logger.warning(f"Approaching daily trade limit: {daily_summary['total_trades']}/{self.config.MAX_DAILY_TRADES}")

                # Could implement additional risk management here
                # For example, reduce trade sizes or increase stop-loss levels

        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")

    async def execute_manual_cycle(self, agent_name: str) -> Dict[str, Any]:
        """Manually execute a trading cycle for a specific agent"""
        try:
            if agent_name not in self.agents:
                return {
                    "status": "error",
                    "message": f"Agent {agent_name} not found"
                }

            agent = self.agents[agent_name]
            result = await agent.execute_trading_cycle()

            return {
                "status": "success",
                "agent_name": agent_name,
                "result": result
            }

        except Exception as e:
            logger.error(f"Failed to execute manual cycle for agent {agent_name}: {e}")
            return {
                "status": "error",
                "agent_name": agent_name,
                "message": str(e)
            }

    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get overall orchestrator status"""
        try:
            agent_statuses = {}

            # Get status of all agents
            for agent_name, agent in self.agents.items():
                try:
                    status = await agent.get_agent_status()
                    agent_statuses[agent_name] = status
                except Exception as e:
                    agent_statuses[agent_name] = {
                        "status": "error",
                        "error_message": str(e)
                    }

            # Get system-wide statistics
            daily_summary = self.database.get_daily_trading_summary()

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "running": self.running,
                "execution_interval_minutes": self.config.AGENT_EXECUTION_INTERVAL_MINUTES,
                "registered_agents": list(self.agents.keys()),
                "agent_statuses": agent_statuses,
                "daily_summary": daily_summary,
                "config": {
                    "max_concurrent_agents": self.config.MAX_CONCURRENT_AGENTS,
                    "max_daily_trades": self.config.MAX_DAILY_TRADES,
                    "max_daily_loss_eth": self.config.MAX_DAILY_LOSS_ETH
                }
            }

        except Exception as e:
            logger.error(f"Failed to get orchestrator status: {e}")
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
        asyncio.create_task(self.stop())

    async def graceful_shutdown(self):
        """Perform graceful shutdown"""
        try:
            logger.info("Initiating graceful shutdown")

            # Stop accepting new tasks
            self.running = False

            # Wait for current executions to complete (with timeout)
            shutdown_timeout = 300  # 5 minutes

            for agent_name, task in self.agent_tasks.items():
                if not task.done():
                    logger.info(f"Waiting for agent {agent_name} to complete...")
                    try:
                        await asyncio.wait_for(task, timeout=shutdown_timeout)
                    except asyncio.TimeoutError:
                        logger.warning(f"Agent {agent_name} did not complete within timeout, cancelling")
                        task.cancel()

            # Final cleanup
            await self.stop()

            logger.info("Graceful shutdown completed")

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            # Force shutdown
            sys.exit(1)
