#!/usr/bin/env python3
"""
Money Maker Agent - Main Entry Point

24/7 automated money-making system that runs all agents simultaneously
and automatically transfers profits to your bank account.
"""

import asyncio
import logging
import sys
from pathlib import Path
import signal

from core.config_manager import ConfigManager, AgentsConfigManager
from core.ledger import Ledger
from core.account_manager import AccountManager
from core.payout_manager import PayoutManager
from core.orchestrator import AgentOrchestrator

from agents.debt_management.debt_agent import DebtManagementAgent
from agents.coupon_discount.coupon_agent import CouponDiscountAgent
from agents.content_generation.content_agent import ContentGenerationAgent
from agents.data_collection.data_agent import DataCollectionAgent
from agents.portfolio_investment.portfolio_agent import PortfolioInvestmentAgent
from agents.competitor_monitoring.competitor_agent import CompetitorMonitoringAgent
from agents.dropshipping.dropshipping_agent import DropshippingAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('money_maker_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class MoneyMakerAgent:
    """
    Main Money Maker Agent System
    
    Coordinates all agents and manages the entire money-making operation.
    """
    
    def __init__(self, config_dir: Path = None, data_dir: Path = None):
        """
        Initialize Money Maker Agent system.
        
        Args:
            config_dir: Configuration directory (default: ./config)
            data_dir: Data directory (default: ./data)
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.config_manager = None
        self.agents_config_manager = None
        self.ledger = None
        self.account_manager = None
        self.payout_manager = None
        self.orchestrator = None
        
        self.running = False
    
    async def initialize(self):
        """Initialize all components."""
        try:
            logger.info("Initializing Money Maker Agent System...")
            
            # Load main configuration
            config_file = self.config_dir / "config.yaml"
            self.config_manager = ConfigManager(config_file, watch=True)
            self.config_manager.start_watching()
            
            # Validate configuration
            is_valid, error = self.config_manager.validate()
            if not is_valid:
                raise ValueError(f"Invalid configuration: {error}")
            
            # Load agents configuration
            self.agents_config_manager = AgentsConfigManager(self.config_dir)
            
            # Initialize ledger
            ledger_db = self.data_dir / "ledger.db"
            self.ledger = Ledger(ledger_db)
            
            # Initialize account manager
            accounts_file = self.config_dir / "accounts.json"
            encryption_key = None  # Will use environment variable or generate
            self.account_manager = AccountManager(accounts_file, encryption_key)
            
            # Initialize payout manager
            payout_config = self.config_manager.get('payout', {})
            self.payout_manager = PayoutManager(
                ledger=self.ledger,
                account_manager=self.account_manager,
                threshold=payout_config.get('threshold', 100.0),
                schedule=payout_config.get('schedule', 'daily'),
                payout_time=payout_config.get('time', '23:00')
            )
            
            # Initialize orchestrator
            self.orchestrator = AgentOrchestrator(
                config_manager=self.config_manager,
                agents_config_manager=self.agents_config_manager,
                ledger=self.ledger,
                payout_manager=self.payout_manager
            )
            
            # Register agents
            await self._register_agents()
            
            logger.info("Money Maker Agent System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}", exc_info=True)
            raise
    
    async def _register_agents(self):
        """Register all enabled agents."""
        agents_config = self.config_manager.get('agents', {})
        
        # Debt Management Agent
        if agents_config.get('debt_management', {}).get('enabled', False):
            debt_config = self.agents_config_manager.get_agent_config('debt_management')
            debt_agent = DebtManagementAgent(
                name='debt_management',
                config=debt_config,
                ledger=self.ledger
            )
            self.orchestrator.register_agent(debt_agent)
            logger.info("Debt Management Agent registered")
        
        # Coupon/Discount Agent
        if agents_config.get('coupon_discount', {}).get('enabled', False):
            coupon_config = self.agents_config_manager.get_agent_config('coupon_discount')
            coupon_agent = CouponDiscountAgent(
                name='coupon_discount',
                config=coupon_config,
                ledger=self.ledger
            )
            self.orchestrator.register_agent(coupon_agent)
            logger.info("Coupon/Discount Agent registered")
        
        # Content Generation Agent (Phase 2)
        if agents_config.get('content_generation', {}).get('enabled', False):
            content_config = self.agents_config_manager.get_agent_config('content_generation')
            content_agent = ContentGenerationAgent(
                name='content_generation',
                config=content_config,
                ledger=self.ledger
            )
            self.orchestrator.register_agent(content_agent)
            logger.info("Content Generation Agent registered")
        
        # Data Collection Agent (Phase 2)
        if agents_config.get('data_collection', {}).get('enabled', False):
            data_config = self.agents_config_manager.get_agent_config('data_collection')
            data_agent = DataCollectionAgent(
                name='data_collection',
                config=data_config,
                ledger=self.ledger
            )
            self.orchestrator.register_agent(data_agent)
            logger.info("Data Collection Agent registered")
        
        # Portfolio Investment Agent (Phase 3)
        if agents_config.get('portfolio_investment', {}).get('enabled', False):
            portfolio_config = self.agents_config_manager.get_agent_config('portfolio_investment')
            portfolio_agent = PortfolioInvestmentAgent(
                name='portfolio_investment',
                config=portfolio_config,
                ledger=self.ledger
            )
            self.orchestrator.register_agent(portfolio_agent)
            logger.info("Portfolio Investment Agent registered")
        
        # Competitor Monitoring Agent (Phase 3)
        if agents_config.get('competitor_monitoring', {}).get('enabled', False):
            competitor_config = self.agents_config_manager.get_agent_config('competitor_monitoring')
            competitor_agent = CompetitorMonitoringAgent(
                name='competitor_monitoring',
                config=competitor_config,
                ledger=self.ledger
            )
            self.orchestrator.register_agent(competitor_agent)
            logger.info("Competitor Monitoring Agent registered")
        
        # Dropshipping Agent (Phase 3)
        if agents_config.get('dropshipping', {}).get('enabled', False):
            dropshipping_config = self.agents_config_manager.get_agent_config('dropshipping')
            dropshipping_agent = DropshippingAgent(
                name='dropshipping',
                config=dropshipping_config,
                ledger=self.ledger
            )
            self.orchestrator.register_agent(dropshipping_agent)
            logger.info("Dropshipping Agent registered")
    
    async def start(self):
        """Start the Money Maker Agent system."""
        try:
            self.running = True
            logger.info("=" * 60)
            logger.info("Starting Money Maker Agent System")
            logger.info("=" * 60)
            
            # Display system status
            status = self.orchestrator.get_status()
            logger.info(f"Total agents: {status['total_agents']}")
            logger.info(f"Active agents: {status['active_agents']}")
            logger.info(f"Current assets: ${status['assets'].get('USD', 0.0):.2f}")
            
            # Start orchestrator (this will run all agents)
            await self.orchestrator.start()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the system gracefully."""
        try:
            self.running = False
            logger.info("Shutting down Money Maker Agent System...")
            
            if self.orchestrator:
                await self.orchestrator.stop()
            
            if self.config_manager:
                self.config_manager.stop_watching()
            
            logger.info("Shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point."""
    # Get base directory
    base_dir = Path(__file__).parent
    
    # Create system instance
    system = MoneyMakerAgent(
        config_dir=base_dir / "config",
        data_dir=base_dir / "data"
    )
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        await system.initialize()
        
        # Start system (runs until interrupted)
        await system.start()
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

