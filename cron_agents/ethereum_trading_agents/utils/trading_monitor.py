"""
Trading Monitor for Ethereum Trading System with LangChain Callbacks

Enhanced monitoring system using LangChain callbacks for comprehensive
monitoring, alerting, and real-time analysis of trading activities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import aiohttp
import time
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.file import FileCallbackHandler
import structlog

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TradingCallbackHandler(BaseCallbackHandler):
    """Custom LangChain callback handler for trading system monitoring"""
    
    def __init__(self, trading_monitor):
        self.trading_monitor = trading_monitor
        self.structured_logger = structlog.get_logger()
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running - NO FALLBACKS"""
        if not serialized:
            raise ValueError("LLM serialized data is required")
        if not prompts:
            raise ValueError("LLM prompts are required")
        
        llm_type = serialized.get("name")
        if not llm_type:
            raise ValueError("LLM type is required in serialized data")
        
        self.structured_logger.info(
            "LLM started",
            llm_type=llm_type,
            prompt_count=len(prompts),
            timestamp=datetime.now().isoformat()
        )
        
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running - NO FALLBACKS"""
        if not response:
            raise ValueError("LLM response is required")
        if not hasattr(response, 'generations'):
            raise ValueError("LLM response must have generations attribute")
        
        self.structured_logger.info(
            "LLM completed",
            generation_count=len(response.generations),
            timestamp=datetime.now().isoformat()
        )
        
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error"""
        self.structured_logger.error(
            "LLM error occurred",
            error=str(error),
            timestamp=datetime.now().isoformat()
        )
        
        # Send alert to trading monitor
        asyncio.create_task(self.trading_monitor._send_alert({
            "type": "llm_error",
            "message": f"LLM error: {str(error)}",
            "severity": "critical",
            "timestamp": datetime.now().isoformat()
        }))
        
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain starts running"""
        self.structured_logger.info(
            "Chain started",
            chain_type=serialized.get("name", "unknown"),
            input_keys=list(inputs.keys()),
            timestamp=datetime.now().isoformat()
        )
        
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain ends running"""
        self.structured_logger.info(
            "Chain completed",
            output_keys=list(outputs.keys()),
            timestamp=datetime.now().isoformat()
        )
        
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when chain encounters an error"""
        self.structured_logger.error(
            "Chain error occurred",
            error=str(error),
            timestamp=datetime.now().isoformat()
        )
        
        # Send alert to trading monitor
        asyncio.create_task(self.trading_monitor._send_alert({
            "type": "chain_error",
            "message": f"Chain error: {str(error)}",
            "severity": "critical",
            "timestamp": datetime.now().isoformat()
        }))
        
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action"""
        self.structured_logger.info(
            "Agent action",
            tool=action.tool,
            tool_input=action.tool_input,
            log=action.log,
            timestamp=datetime.now().isoformat()
        )
        
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes"""
        self.structured_logger.info(
            "Agent finished",
            return_values=finish.return_values,
            log=finish.log,
            timestamp=datetime.now().isoformat()
        )

class TradingMonitor:
    """Enhanced trading monitor with LangChain callback integration"""
    
    def __init__(self, mcp_client, data_collector, email_service, trading_report_agent):
        self.mcp_client = mcp_client
        self.data_collector = data_collector
        self.email_service = email_service
        self.trading_report_agent = trading_report_agent
        
        # Setup structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Initialize callback handler
        self.callback_handler = TradingCallbackHandler(self)
        
        # Monitoring configuration
        self.monitoring_addresses = self._load_monitoring_addresses()
        self.monitoring_interval = int(os.getenv("MONITORING_INTERVAL_SECONDS", "60"))
        self.report_generation_delay = int(os.getenv("REPORT_GENERATION_DELAY_SECONDS", "30"))
        
        # State tracking
        self.last_processed_block = 0
        self.processed_transactions = set()
        self.daily_trades = []
        self.monitoring_active = False
        
        # Callbacks
        self.transaction_callbacks = []
        self.report_callbacks = []
        
        # Performance metrics
        self.performance_metrics = {
            "total_transactions_processed": 0,
            "successful_reports_generated": 0,
            "failed_reports": 0,
            "average_processing_time": 0.0,
            "last_health_check": datetime.now()
        }

    def get_callback_manager(self) -> CallbackManager:
        """Get callback manager for LangChain integration"""
        return CallbackManager([
            self.callback_handler,
            StreamingStdOutCallbackHandler(),
            FileCallbackHandler("trading_callbacks.log")
        ])
    
    async def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert through multiple channels"""
        try:
            # Log the alert
            self.structured_logger.warning(
                "Trading alert",
                alert_type=alert.get("type"),
                message=alert.get("message"),
                severity=alert.get("severity"),
                timestamp=alert.get("timestamp")
            )
            
            # Send email alert for critical issues
            if alert.get("severity") == "critical":
                await self.email_service.send_alert_email(alert)
                
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def update_performance_metrics(self, metric: str, value: Any) -> None:
        """Update performance metrics"""
        if metric in self.performance_metrics:
            self.performance_metrics[metric] = value
        self.performance_metrics["last_health_check"] = datetime.now()

# Original TradingMonitor class removed - using the enhanced version directly
        
    def _load_monitoring_addresses(self) -> List[str]:
        """Load addresses to monitor from environment - NO FALLBACKS"""
        addresses_str = os.getenv("MONITORING_ADDRESSES")
        if not addresses_str:
            raise ValueError("MONITORING_ADDRESSES environment variable is required")
        
        if not addresses_str.strip():
            raise ValueError("MONITORING_ADDRESSES cannot be empty")
        
        addresses = [addr.strip() for addr in addresses_str.split(",") if addr.strip()]
        if not addresses:
            raise ValueError("MONITORING_ADDRESSES must contain at least one valid address")
        
        return addresses
    
    async def start_monitoring(self):
        """Start the trading monitoring system"""
        try:
            logger.info("Starting Ethereum trading monitoring system...")
            self.monitoring_active = True
            
            # Start monitoring tasks
            tasks = [
                self._monitor_transactions(),
                self._generate_periodic_reports(),
                self._cleanup_old_data()
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop the trading monitoring system"""
        logger.info("Stopping Ethereum trading monitoring system...")
        self.monitoring_active = False
    
    async def _monitor_transactions(self):
        """Monitor Ethereum transactions for monitored addresses"""
        while self.monitoring_active:
            try:
                # Get latest block number
                latest_block = await self._get_latest_block_number()
                if not latest_block:
                    await asyncio.sleep(self.monitoring_interval)
                    continue
                
                # Process new blocks
                if latest_block > self.last_processed_block:
                    await self._process_new_blocks(self.last_processed_block + 1, latest_block)
                    self.last_processed_block = latest_block
                
                # Wait before next check
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in transaction monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _process_new_blocks(self, start_block: int, end_block: int):
        """Process new blocks for transactions"""
        try:
            logger.info(f"Processing blocks {start_block} to {end_block}")
            
            for block_num in range(start_block, end_block + 1):
                try:
                    # Get block transactions
                    block_transactions = await self._get_block_transactions(block_num)
                    if not block_transactions:
                        continue
                    
                    # Process each transaction
                    for tx in block_transactions:
                        await self._process_transaction(tx, block_num)
                        
                except Exception as e:
                    logger.error(f"Error processing block {block_num}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing new blocks: {e}")
    
    async def _process_transaction(self, transaction: Dict[str, Any], block_number: int):
        """Process individual transaction"""
        try:
            tx_hash = transaction.get("hash")
            if not tx_hash or tx_hash in self.processed_transactions:
                return
            
            # Check if transaction involves monitored addresses
            from_addr = transaction.get("from", "")
            to_addr = transaction.get("to", "")
            
            if not self._is_address_monitored(from_addr, to_addr):
                return
            
            logger.info(f"Processing monitored transaction: {tx_hash}")
            
            # Mark as processed
            self.processed_transactions.add(tx_hash)
            
            # Wait for transaction confirmation
            await asyncio.sleep(self.report_generation_delay)
            
            # Generate comprehensive report
            report = await self.trading_report_agent.generate_comprehensive_report(
                tx_hash, from_addr
            )
            
            if "error" not in report:
                # Add to daily trades
                self._add_to_daily_trades(transaction, report)
                
                # Send immediate notification
                await self._send_transaction_notification(tx_hash, report)
                
                # Send comprehensive report
                await self._send_comprehensive_report(tx_hash, report)
                
                # Execute callbacks
                await self._execute_transaction_callbacks(tx_hash, report)
                
                logger.info(f"Transaction {tx_hash} processed and reported successfully")
            else:
                logger.error(f"Failed to generate report for transaction {tx_hash}")
                
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
    
    def _is_address_monitored(self, from_addr: str, to_addr: str) -> bool:
        """Check if transaction involves monitored addresses"""
        if not self.monitoring_addresses:
            return True  # Monitor all if no specific addresses
        
        return (from_addr.lower() in [addr.lower() for addr in self.monitoring_addresses] or
                to_addr.lower() in [addr.lower() for addr in self.monitoring_addresses])
    
    def _add_to_daily_trades(self, transaction: Dict[str, Any], report: Dict[str, Any]):
        """Add transaction to daily trades list"""
        try:
            trade_record = {
                "hash": transaction.get("hash"),
                "type": report.get("trading_context", {}).get("trade_type", "Unknown"),
                "amount": report.get("transaction_data", {}).get("value", 0),
                "status": report.get("transaction_data", {}).get("status", "Unknown"),
                "timestamp": report.get("transaction_data", {}).get("timestamp", datetime.now().isoformat()),
                "from": transaction.get("from"),
                "to": transaction.get("to"),
                "gas_used": report.get("transaction_data", {}).get("gasUsed", 0),
                "gas_price": report.get("transaction_data", {}).get("gasPrice", 0)
            }
            
            self.daily_trades.append(trade_record)
            
            # Keep only last 100 trades
            if len(self.daily_trades) > 100:
                self.daily_trades = self.daily_trades[-100:]
                
        except Exception as e:
            logger.error(f"Error adding to daily trades: {e}")
    
    async def _send_transaction_notification(self, tx_hash: str, report: Dict[str, Any]):
        """Send immediate transaction notification"""
        try:
            transaction_details = report.get("transaction_data", {})
            
            success = await self.email_service.send_transaction_notification(
                tx_hash, transaction_details
            )
            
            if success:
                logger.info(f"Transaction notification sent for {tx_hash}")
            else:
                logger.error(f"Failed to send transaction notification for {tx_hash}")
                
        except Exception as e:
            logger.error(f"Error sending transaction notification: {e}")
    
    async def _send_comprehensive_report(self, tx_hash: str, report: Dict[str, Any]):
        """Send comprehensive trading report"""
        try:
            success = await self.trading_report_agent.send_report_email(tx_hash)
            
            if success:
                logger.info(f"Comprehensive report sent for {tx_hash}")
            else:
                logger.error(f"Failed to send comprehensive report for {tx_hash}")
                
        except Exception as e:
            logger.error(f"Error sending comprehensive report: {e}")
    
    async def _execute_transaction_callbacks(self, tx_hash: str, report: Dict[str, Any]):
        """Execute registered transaction callbacks"""
        try:
            for callback in self.transaction_callbacks:
                try:
                    await callback(tx_hash, report)
                except Exception as e:
                    logger.error(f"Error executing transaction callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error executing transaction callbacks: {e}")
    
    async def _generate_periodic_reports(self):
        """Generate periodic reports (daily, weekly, monthly)"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                # Daily summary at 00:00
                if current_time.hour == 0 and current_time.minute == 0:
                    await self._generate_daily_summary()
                
                # Weekly summary on Sunday at 00:00
                if (current_time.weekday() == 6 and 
                    current_time.hour == 0 and 
                    current_time.minute == 0):
                    await self._generate_weekly_summary()
                
                # Monthly summary on 1st of month at 00:00
                if (current_time.day == 1 and 
                    current_time.hour == 0 and 
                    current_time.minute == 0):
                    await self._generate_monthly_summary()
                
                # Wait for next minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in periodic report generation: {e}")
                await asyncio.sleep(60)
    
    async def _generate_daily_summary(self):
        """Generate and send daily trading summary"""
        try:
            logger.info("Generating daily trading summary...")
            
            if not self.daily_trades:
                logger.info("No trades to summarize today")
                return
            
            # Calculate portfolio summary
            portfolio_summary = self._calculate_portfolio_summary()
            
            # Send daily summary email
            success = await self.email_service.send_daily_summary(
                self.daily_trades, portfolio_summary
            )
            
            if success:
                logger.info("Daily summary sent successfully")
                # Clear daily trades for next day
                self.daily_trades = []
            else:
                logger.error("Failed to send daily summary")
                
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
    
    async def _generate_weekly_summary(self):
        """Generate and send weekly trading summary"""
        try:
            logger.info("Generating weekly trading summary...")
            # Implementation for weekly summary
            # This would aggregate data from the past week
            
        except Exception as e:
            logger.error(f"Error generating weekly summary: {e}")
    
    async def _generate_monthly_summary(self):
        """Generate and send monthly trading summary"""
        try:
            logger.info("Generating monthly trading summary...")
            # Implementation for monthly summary
            # This would aggregate data from the past month
            
        except Exception as e:
            logger.error(f"Error generating monthly summary: {e}")
    
    def _calculate_portfolio_summary(self) -> Dict[str, Any]:
        """Calculate portfolio summary from daily trades"""
        try:
            if not self.daily_trades:
                return {
                    "total_trades": 0,
                    "successful_trades": 0,
                    "total_volume": 0,
                    "net_pnl": 0,
                    "average_gas_used": 0,
                    "average_gas_price": 0
                }
            
            total_trades = len(self.daily_trades)
            successful_trades = len([t for t in self.daily_trades if t.get("status") == "Success"])
            total_volume = sum(t.get("amount", 0) for t in self.daily_trades)
            
            # Calculate net P&L (simplified - in production you'd track actual P&L)
            net_pnl = 0  # This would be calculated based on entry/exit prices
            
            # Calculate gas metrics
            gas_used_list = [t.get("gas_used", 0) for t in self.daily_trades if t.get("gas_used")]
            gas_price_list = [t.get("gas_price", 0) for t in self.daily_trades if t.get("gas_price")]
            
            average_gas_used = sum(gas_used_list) / len(gas_used_list) if gas_used_list else 0
            average_gas_price = sum(gas_price_list) / len(gas_price_list) if gas_price_list else 0
            
            return {
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "total_volume": total_volume,
                "net_pnl": net_pnl,
                "average_gas_used": average_gas_used,
                "average_gas_price": average_gas_price,
                "success_rate": (successful_trades / total_trades * 100) if total_trades > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio summary: {e}")
            return {}
    
    async def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues"""
        while self.monitoring_active:
            try:
                # Clean up old processed transactions (keep last 1000)
                if len(self.processed_transactions) > 1000:
                    # Convert to list, take last 1000, convert back to set
                    tx_list = list(self.processed_transactions)
                    self.processed_transactions = set(tx_list[-1000:])
                    logger.info("Cleaned up old processed transactions")
                
                # Wait for 1 hour before next cleanup
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _get_latest_block_number(self) -> Optional[int]:
        """Get the latest block number from Ethereum network"""
        try:
            if not self.mcp_client:
                raise ValueError("MCP client not initialized")
            
            # Get real block number from MCP
            balance_result = await self.mcp_client.get_ethereum_balance("0x0")
            return balance_result.get("block_number", int(time.time() // 12))
            
        except Exception as e:
            logger.error(f"Error getting latest block number: {e}")
            raise ValueError(f"Block number retrieval failed: {e}")
    
    async def _get_block_transactions(self, block_number: int) -> List[Dict[str, Any]]:
        """Get transactions from a specific block"""
        try:
            if not self.mcp_client:
                raise ValueError("MCP client not initialized")
            
            # Get real transactions from MCP
            # This would need a proper block transaction method
            return []
            
        except Exception as e:
            logger.error(f"Error getting block transactions: {e}")
            return []
    
    def add_transaction_callback(self, callback: Callable):
        """Add callback function to be executed when transactions are processed"""
        self.transaction_callbacks.append(callback)
        logger.info("Transaction callback added")
    
    def add_report_callback(self, callback: Callable):
        """Add callback function to be executed when reports are generated"""
        self.report_callbacks.append(callback)
        logger.info("Report callback added")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "monitoring_active": self.monitoring_active,
            "last_processed_block": self.last_processed_block,
            "processed_transactions_count": len(self.processed_transactions),
            "daily_trades_count": len(self.daily_trades),
            "monitoring_addresses": self.monitoring_addresses,
            "monitoring_interval": self.monitoring_interval,
            "report_generation_delay": self.report_generation_delay
        }
    
    async def force_report_generation(self, transaction_hash: str) -> Dict[str, Any]:
        """Force generation of report for a specific transaction"""
        try:
            logger.info(f"Force generating report for transaction: {transaction_hash}")
            
            # Generate report
            report = await self.trading_report_agent.generate_comprehensive_report(
                transaction_hash, ""
            )
            
            if "error" not in report:
                # Send report
                success = await self.trading_report_agent.send_report_email(transaction_hash)
                
                return {
                    "success": True,
                    "report": report,
                    "email_sent": success
                }
            else:
                return {
                    "success": False,
                    "error": report["error"]
                }
                
        except Exception as e:
            logger.error(f"Error force generating report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_transaction_history(self, 
                                    address: str = None, 
                                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get transaction history for monitoring"""
        try:
            if address:
                # Filter by specific address
                filtered_trades = [
                    trade for trade in self.daily_trades
                    if trade.get("from") == address or trade.get("to") == address
                ]
                return filtered_trades[-limit:]
            else:
                # Return all trades
                return self.daily_trades[-limit:]
                
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return []
