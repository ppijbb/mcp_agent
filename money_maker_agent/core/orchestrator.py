"""
Unified Orchestrator for Money Maker Agent System

Manages all agents simultaneously, handles scheduling, resource sharing, and error recovery.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

from .config_manager import ConfigManager, AgentsConfigManager
from .ledger import Ledger
from .payout_manager import PayoutManager

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all money-making agents."""
    
    def __init__(self, name: str, config: Dict[str, Any], ledger: Ledger):
        """
        Initialize agent.
        
        Args:
            name: Agent name
            config: Agent configuration
            ledger: Ledger instance for recording transactions
        """
        self.name = name
        self.config = config
        self.ledger = ledger
        self.enabled = config.get('enabled', True)
        self.priority = config.get('priority', 999)
        self._running = False
    
    async def initialize(self) -> bool:
        """Initialize agent. Override in subclasses."""
        return True
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute agent's main logic. Override in subclasses.
        
        Returns:
            Execution result dictionary
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    async def shutdown(self):
        """Shutdown agent. Override in subclasses if needed."""
        self._running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'priority': self.priority,
            'running': self._running
        }


class AgentOrchestrator:
    """
    Unified orchestrator for all money-making agents.
    
    Features:
    - Simultaneous execution of all agents
    - Priority-based scheduling
    - Resource sharing and conflict prevention
    - Automatic restart and error recovery
    - Real-time configuration updates
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        agents_config_manager: AgentsConfigManager,
        ledger: Ledger,
        payout_manager: PayoutManager
    ):
        """
        Initialize orchestrator.
        
        Args:
            config_manager: Main configuration manager
            agents_config_manager: Agents configuration manager
            ledger: Ledger instance
            payout_manager: Payout manager instance
        """
        self.config_manager = config_manager
        self.agents_config_manager = agents_config_manager
        self.ledger = ledger
        self.payout_manager = payout_manager
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        
        # Execution control
        self.running = False
        self.execution_interval = timedelta(minutes=5)  # Default 5 minutes
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup config change callbacks
        self.config_manager.on_reload(self._on_config_reload)
        
        logger.info("Agent orchestrator initialized")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent."""
        self.agents[agent.name] = agent
        logger.info(f"Agent registered: {agent.name} (priority: {agent.priority})")
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            if agent_name in self.agent_tasks:
                self.agent_tasks[agent_name].cancel()
                del self.agent_tasks[agent_name]
            logger.info(f"Agent unregistered: {agent_name}")
    
    def _on_config_reload(self, new_config: Dict[str, Any]):
        """Handle configuration reload."""
        logger.info("Configuration reloaded, updating agents...")
        
        # Update payout manager config
        payout_config = new_config.get('payout', {})
        if payout_config:
            self.payout_manager.set_config(
                threshold=payout_config.get('threshold'),
                schedule=payout_config.get('schedule'),
                payout_time=payout_config.get('time'),
                enabled=payout_config.get('enabled', True)
            )
        
        # Update agent configurations
        agents_config = new_config.get('agents', {})
        for agent_name, agent_config in agents_config.items():
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                agent.config = agent_config
                agent.enabled = agent_config.get('enabled', True)
                agent.priority = agent_config.get('priority', 999)
                
                # Restart agent if configuration changed significantly
                if not agent.enabled and agent_name in self.agent_tasks:
                    # Stop disabled agent
                    self.agent_tasks[agent_name].cancel()
                    del self.agent_tasks[agent_name]
                elif agent.enabled and agent_name not in self.agent_tasks:
                    # Start newly enabled agent
                    asyncio.create_task(self._start_agent(agent_name))
    
    async def start(self):
        """Start the orchestrator and all enabled agents."""
        try:
            self.running = True
            logger.info("Starting agent orchestrator")
            
            # Initialize all agents
            for agent in self.agents.values():
                if agent.enabled:
                    try:
                        success = await agent.initialize()
                        if not success:
                            logger.error(f"Failed to initialize agent: {agent.name}")
                            continue
                    except Exception as e:
                        logger.error(f"Error initializing agent {agent.name}: {e}")
                        continue
            
            # Start all enabled agents
            enabled_agents = [
                agent for agent in self.agents.values()
                if agent.enabled
            ]
            
            # Sort by priority
            enabled_agents.sort(key=lambda a: a.priority)
            
            for agent in enabled_agents:
                await self._start_agent(agent.name)
            
            # Start main orchestration loops
            await asyncio.gather(
                self._orchestration_loop(),
                self._payout_check_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Orchestrator failed to start: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the orchestrator and all agents."""
        try:
            self.running = False
            logger.info("Stopping agent orchestrator")
            
            # Stop all agent tasks
            for agent_name, task in self.agent_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown all agents
            for agent in self.agents.values():
                try:
                    await agent.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down agent {agent.name}: {e}")
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            logger.info("Agent orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")
    
    async def _start_agent(self, agent_name: str):
        """Start a specific agent."""
        if agent_name not in self.agents:
            logger.warning(f"Agent not found: {agent_name}")
            return
        
        agent = self.agents[agent_name]
        
        if not agent.enabled:
            logger.debug(f"Agent {agent_name} is disabled")
            return
        
        if agent_name in self.agent_tasks and not self.agent_tasks[agent_name].done():
            logger.warning(f"Agent {agent_name} is already running")
            return
        
        try:
            # Create agent execution task
            task = asyncio.create_task(self._agent_execution_loop(agent_name))
            self.agent_tasks[agent_name] = task
            logger.info(f"Agent {agent_name} started")
        except Exception as e:
            logger.error(f"Failed to start agent {agent_name}: {e}")
    
    async def _agent_execution_loop(self, agent_name: str):
        """Main execution loop for a single agent."""
        agent = self.agents[agent_name]
        last_execution = datetime.now()
        
        try:
            while self.running and agent.enabled:
                try:
                    # Check if it's time to execute
                    now = datetime.now()
                    if now - last_execution >= self.execution_interval:
                        logger.debug(f"Executing agent: {agent_name}")
                        
                        try:
                            # Execute agent
                            result = await agent.execute()
                            
                            # Record results
                            if result.get('success', False):
                                # Record income if any
                                if 'income' in result:
                                    self.ledger.record_transaction(
                                        agent_name=agent_name,
                                        transaction_type='income',
                                        amount=result['income'],
                                        description=result.get('description', 'Agent income'),
                                        metadata=result.get('metadata')
                                    )
                                
                                # Record expenses if any
                                if 'expenses' in result:
                                    self.ledger.record_transaction(
                                        agent_name=agent_name,
                                        transaction_type='expense',
                                        amount=result['expenses'],
                                        description=result.get('expense_description', 'Agent expenses'),
                                        metadata=result.get('expense_metadata')
                                    )
                                
                                logger.info(
                                    f"Agent {agent_name} completed successfully: "
                                    f"income=${result.get('income', 0):.2f}, "
                                    f"expenses=${result.get('expenses', 0):.2f}"
                                )
                            else:
                                error_msg = result.get('error', 'Unknown error')
                                logger.warning(f"Agent {agent_name} execution failed: {error_msg}")
                                
                                # Record error as expense if applicable
                                if 'expenses' in result:
                                    self.ledger.record_transaction(
                                        agent_name=agent_name,
                                        transaction_type='expense',
                                        amount=result['expenses'],
                                        description=f"Error: {error_msg}",
                                        metadata={'error': True, **result.get('metadata', {})}
                                    )
                            
                            last_execution = now
                            
                        except Exception as e:
                            logger.error(f"Error executing agent {agent_name}: {e}", exc_info=True)
                            last_execution = now  # Prevent rapid retries
                    
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
        finally:
            agent._running = False
    
    async def _orchestration_loop(self):
        """Main orchestration loop for monitoring and management."""
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
    
    async def _payout_check_loop(self):
        """Loop for checking and executing payouts."""
        try:
            while self.running:
                try:
                    # Check if payout is due
                    should_pay, balance = self.payout_manager.should_payout()
                    
                    if should_pay:
                        logger.info(f"Payout due: ${balance:.2f} available")
                        success, message = await self.payout_manager.execute_payout()
                        
                        if success:
                            logger.info(f"Payout successful: {message}")
                        else:
                            logger.warning(f"Payout failed: {message}")
                    
                    # Wait before next check (check every hour)
                    await asyncio.sleep(3600)
                    
                except asyncio.CancelledError:
                    logger.info("Payout check loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in payout check loop: {e}")
                    await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Fatal error in payout check loop: {e}")
    
    async def _monitor_agent_health(self):
        """Monitor health of all agents."""
        try:
            for agent_name, agent in self.agents.items():
                if not agent.enabled:
                    continue
                
                # Check if agent task is still running
                if agent_name in self.agent_tasks:
                    task = self.agent_tasks[agent_name]
                    if task.done():
                        # Task completed or failed, restart if enabled
                        if agent.enabled:
                            logger.warning(f"Agent {agent_name} task completed, restarting...")
                            await self._start_agent(agent_name)
                else:
                    # Task not running but agent is enabled, start it
                    if agent.enabled:
                        logger.warning(f"Agent {agent_name} not running but enabled, starting...")
                        await self._start_agent(agent_name)
            
        except Exception as e:
            logger.error(f"Failed to monitor agent health: {e}")
    
    async def _check_system_resources(self):
        """Check system resources and adjust if needed."""
        try:
            # Get daily summary
            daily_summary = self.ledger.get_daily_summary()
            
            logger.debug(
                f"Daily summary: income=${daily_summary['total_income']:.2f}, "
                f"expenses=${daily_summary['total_expenses']:.2f}, "
                f"net=${daily_summary['net_profit']:.2f}"
            )
            
            # Check total assets
            assets = self.ledger.get_total_assets()
            total_assets = assets.get('USD', 0.0)
            
            if total_assets < 0:
                logger.warning(f"Negative assets detected: ${total_assets:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        agent_statuses = {}
        for agent_name, agent in self.agents.items():
            agent_statuses[agent_name] = {
                **agent.get_status(),
                'task_running': agent_name in self.agent_tasks and not self.agent_tasks[agent_name].done()
            }
        
        return {
            'running': self.running,
            'agents': agent_statuses,
            'total_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents.values() if a.enabled),
            'assets': self.ledger.get_total_assets()
        }

