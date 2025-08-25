"""
Test suite for Trading Report System

Tests the email service, trading report agent, and trading monitor functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.email_service import EmailService
from agents.trading_report_agent import TradingReportAgent
from utils.trading_monitor import TradingMonitor


class TestEmailService:
    """Test EmailService functionality"""
    
    @pytest.fixture
    def email_service(self):
        """Create EmailService instance for testing"""
        with patch.dict(os.environ, {
            'EMAIL_ADDRESS': 'test@example.com',
            'EMAIL_PASSWORD': 'test_password',
            'SMTP_SERVER': 'smtp.gmail.com',
            'SMTP_PORT': '587',
            'SENDER_NAME': 'Test Trading Agent'
        }):
            return EmailService()
    
    @pytest.mark.asyncio
    async def test_email_service_initialization(self, email_service):
        """Test EmailService initialization"""
        assert email_service.email_address == 'test@example.com'
        assert email_service.email_password == 'test_password'
        assert email_service.smtp_server == 'smtp.gmail.com'
        assert email_service.smtp_port == 587
        assert email_service.sender_name == 'Test Trading Agent'
    
    @pytest.mark.asyncio
    async def test_load_recipients(self, email_service):
        """Test recipient loading from environment"""
        with patch.dict(os.environ, {'EMAIL_RECIPIENTS': 'user1@test.com,user2@test.com'}):
            service = EmailService()
            assert service.default_recipients == ['user1@test.com', 'user2@test.com']
    
    @pytest.mark.asyncio
    async def test_generate_trading_report_html(self, email_service):
        """Test HTML report generation"""
        transaction_data = {
            'hash': '0x1234567890abcdef',
            'blockNumber': 12345,
            'from': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
            'to': '0x1234567890123456789012345678901234567890',
            'value': 1.5,
            'gasUsed': 21000,
            'gasPrice': 20,
            'status': 'Success'
        }
        
        market_analysis = {
            'current_price': 2850,
            'price_change_24h': 3.2,
            'sentiment': {'overall_sentiment': 'Bullish'},
            'technical_indicators': {'rsi': 45.2, 'macd': 'Bullish'}
        }
        
        html_content = email_service._generate_trading_report_html(transaction_data, market_analysis)
        
        assert '0x1234567890abcdef' in html_content
        assert '1.5 ETH' in html_content
        assert '2850' in html_content
        assert 'Bullish' in html_content
        assert '<!DOCTYPE html>' in html_content
    
    @pytest.mark.asyncio
    async def test_generate_trading_report_text(self, email_service):
        """Test text report generation"""
        transaction_data = {
            'hash': '0x1234567890abcdef',
            'value': 1.5,
            'status': 'Success'
        }
        
        market_analysis = {
            'current_price': 2850,
            'price_change_24h': 3.2
        }
        
        text_content = email_service._generate_trading_report_text(transaction_data, market_analysis)
        
        assert '0x1234567890abcdef' in text_content
        assert '1.5 ETH' in text_content
        assert '2850' in text_content
        assert 'Ethereum Trading Report' in text_content


class TestTradingReportAgent:
    """Test TradingReportAgent functionality"""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client"""
        client = Mock()
        client.get_transaction_status = AsyncMock(return_value={
            'blockNumber': 12345,
            'gasUsed': 21000,
            'status': 1,
            'contractAddress': None,
            'logs': [],
            'cumulativeGasUsed': 21000,
            'effectiveGasPrice': '20000000000'
        })
        client.get_ethereum_price = AsyncMock(return_value={
            'price': 2850,
            'price_change_24h': 3.2,
            'market_cap': 1000000000,
            'volume_24h': 50000000
        })
        client.get_market_trends = AsyncMock(return_value={
            'trend': 'bullish',
            'strength': 'medium'
        })
        return client
    
    @pytest.fixture
    def mock_data_collector(self):
        """Create mock data collector"""
        collector = Mock()
        collector.collect_comprehensive_data = AsyncMock(return_value={
            'news_data': {'positive_news_count': 5, 'negative_news_count': 2},
            'social_sentiment': {'overall_sentiment_score': 0.3},
            'technical_data': {'rsi': 45.2},
            'onchain_data': {'active_addresses': 1000000},
            'expert_opinions': {'bullish_count': 3, 'bearish_count': 1}
        })
        return collector
    
    @pytest.fixture
    def mock_email_service(self):
        """Create mock email service"""
        service = Mock()
        service.send_trading_report = AsyncMock(return_value=True)
        return service
    
    @pytest.fixture
    def trading_report_agent(self, mock_mcp_client, mock_data_collector, mock_email_service):
        """Create TradingReportAgent instance for testing"""
        return TradingReportAgent(mock_mcp_client, mock_data_collector, mock_email_service)
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(self, trading_report_agent):
        """Test comprehensive report generation"""
        tx_hash = "0x1234567890abcdef"
        address = "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"
        
        report = await trading_report_agent.generate_comprehensive_report(tx_hash, address)
        
        assert "error" not in report
        assert "transaction_data" in report
        assert "market_analysis" in report
        assert "trading_context" in report
        assert report["transaction_data"]["hash"] == tx_hash
    
    @pytest.mark.asyncio
    async def test_determine_trade_type(self, trading_report_agent):
        """Test trade type determination"""
        # ETH Transfer
        eth_transfer = {'to': '0x1234', 'contractAddress': None, 'logs': []}
        assert trading_report_agent._determine_trade_type(eth_transfer) == "ETH Transfer"
        
        # Smart Contract Interaction
        contract_interaction = {'contractAddress': '0x5678', 'logs': []}
        assert trading_report_agent._determine_trade_type(contract_interaction) == "Smart Contract Interaction"
        
        # Token Transfer
        token_transfer = {'to': '0x1234', 'contractAddress': None, 'logs': [{'data': 'transfer'}]}
        assert trading_report_agent._determine_trade_type(token_transfer) == "Token Transfer"
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, trading_report_agent):
        """Test risk assessment functionality"""
        transaction_data = {
            'gasPrice': 150,  # High gas price
            'value': 15  # Large transaction
        }
        
        market_analysis = {
            'price_change_24h': 8,  # High volatility
            'sentiment': {'overall_sentiment': 'Very Bullish'}
        }
        
        risk_assessment = trading_report_agent._assess_trade_risk(transaction_data, market_analysis)
        
        assert risk_assessment["risk_level"] in ["Medium", "High", "Very High"]
        assert len(risk_assessment["risk_factors"]) > 0
        assert len(risk_assessment["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_send_report_email(self, trading_report_agent):
        """Test report email sending"""
        tx_hash = "0x1234567890abcdef"
        
        # Mock report generation
        trading_report_agent.report_cache[tx_hash] = {
            "transaction_data": {"hash": tx_hash},
            "market_analysis": {"current_price": 2850}
        }
        
        success = await trading_report_agent.send_report_email(tx_hash)
        assert success is True


class TestTradingMonitor:
    """Test TradingMonitor functionality"""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Create mock MCP client"""
        client = Mock()
        client.get_transaction_status = AsyncMock(return_value={'status': 1})
        return client
    
    @pytest.fixture
    def mock_data_collector(self):
        """Create mock data collector"""
        collector = Mock()
        collector.collect_comprehensive_data = AsyncMock(return_value={})
        return collector
    
    @pytest.fixture
    def mock_email_service(self):
        """Create mock email service"""
        service = Mock()
        service.send_transaction_notification = AsyncMock(return_value=True)
        service.send_daily_summary = AsyncMock(return_value=True)
        return service
    
    @pytest.fixture
    def mock_trading_report_agent(self):
        """Create mock trading report agent"""
        agent = Mock()
        agent.generate_comprehensive_report = AsyncMock(return_value={
            "transaction_data": {"hash": "0x1234", "value": 1.0, "status": "Success"},
            "trading_context": {"trade_type": "ETH Transfer"}
        })
        agent.send_report_email = AsyncMock(return_value=True)
        return agent
    
    @pytest.fixture
    def trading_monitor(self, mock_mcp_client, mock_data_collector, mock_email_service, mock_trading_report_agent):
        """Create TradingMonitor instance for testing"""
        with patch.dict(os.environ, {'MONITORING_INTERVAL_SECONDS': '60'}):
            return TradingMonitor(mock_mcp_client, mock_data_collector, mock_email_service, mock_trading_report_agent)
    
    def test_monitoring_addresses_loading(self, trading_monitor):
        """Test monitoring addresses loading from environment"""
        with patch.dict(os.environ, {'MONITORING_ADDRESSES': '0x1234,0x5678'}):
            monitor = TradingMonitor(Mock(), Mock(), Mock(), Mock())
            assert monitor.monitoring_addresses == ['0x1234', '0x5678']
    
    def test_is_address_monitored(self, trading_monitor):
        """Test address monitoring logic"""
        # Test with specific addresses
        trading_monitor.monitoring_addresses = ['0x1234', '0x5678']
        
        assert trading_monitor._is_address_monitored('0x1234', '0x9999') is True
        assert trading_monitor._is_address_monitored('0x9999', '0x5678') is True
        assert trading_monitor._is_address_monitored('0x9999', '0x8888') is False
        
        # Test with no specific addresses (monitor all)
        trading_monitor.monitoring_addresses = []
        assert trading_monitor._is_address_monitored('0x9999', '0x8888') is True
    
    def test_add_to_daily_trades(self, trading_monitor):
        """Test adding transactions to daily trades"""
        transaction = {'hash': '0x1234', 'from': '0x1234', 'to': '0x5678'}
        report = {
            'trading_context': {'trade_type': 'ETH Transfer'},
            'transaction_data': {'value': 1.0, 'status': 'Success', 'timestamp': '2024-01-15T10:00:00'}
        }
        
        trading_monitor._add_to_daily_trades(transaction, report)
        
        assert len(trading_monitor.daily_trades) == 1
        assert trading_monitor.daily_trades[0]['hash'] == '0x1234'
        assert trading_monitor.daily_trades[0]['type'] == 'ETH Transfer'
        assert trading_monitor.daily_trades[0]['amount'] == 1.0
    
    def test_calculate_portfolio_summary(self, trading_monitor):
        """Test portfolio summary calculation"""
        # Add some test trades
        trading_monitor.daily_trades = [
            {'status': 'Success', 'amount': 1.0, 'gas_used': 21000, 'gas_price': 20},
            {'status': 'Success', 'amount': 2.0, 'gas_used': 21000, 'gas_price': 25},
            {'status': 'Failed', 'amount': 0.5, 'gas_used': 21000, 'gas_price': 20}
        ]
        
        summary = trading_monitor._calculate_portfolio_summary()
        
        assert summary['total_trades'] == 3
        assert summary['successful_trades'] == 2
        assert summary['total_volume'] == 3.5
        assert summary['success_rate'] == 66.66666666666666
    
    @pytest.mark.asyncio
    async def test_force_report_generation(self, trading_monitor):
        """Test forced report generation"""
        tx_hash = "0x1234567890abcdef"
        
        result = await trading_monitor.force_report_generation(tx_hash)
        
        assert result["success"] is True
        assert "report" in result
        assert result["email_sent"] is True
    
    def test_get_monitoring_status(self, trading_monitor):
        """Test monitoring status retrieval"""
        status = trading_monitor.get_monitoring_status()
        
        assert "monitoring_active" in status
        assert "last_processed_block" in status
        assert "processed_transactions_count" in status
        assert "daily_trades_count" in status
        assert "monitoring_addresses" in status
        assert "monitoring_interval" in status
        assert "report_generation_delay" in status


@pytest.mark.asyncio
async def test_integration_email_and_report():
    """Integration test for email service and report generation"""
    # This test would require actual email configuration
    # For now, we'll test the integration without sending actual emails
    
    with patch.dict(os.environ, {
        'EMAIL_ADDRESS': 'test@example.com',
        'EMAIL_PASSWORD': 'test_password',
        'SMTP_SERVER': 'smtp.gmail.com',
        'SMTP_PORT': '587'
    }):
        email_service = EmailService()
        
        # Test report generation
        transaction_data = {
            'hash': '0x1234567890abcdef',
            'blockNumber': 12345,
            'from': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
            'to': '0x1234567890123456789012345678901234567890',
            'value': 1.5,
            'gasUsed': 21000,
            'gasPrice': 20,
            'status': 'Success'
        }
        
        market_analysis = {
            'current_price': 2850,
            'price_change_24h': 3.2,
            'sentiment': {'overall_sentiment': 'Bullish'},
            'technical_indicators': {'rsi': 45.2, 'macd': 'Bullish'}
        }
        
        # Generate HTML and text content
        html_content = email_service._generate_trading_report_html(transaction_data, market_analysis)
        text_content = email_service._generate_trading_report_text(transaction_data, market_analysis)
        
        # Verify content generation
        assert len(html_content) > 0
        assert len(text_content) > 0
        assert '0x1234567890abcdef' in html_content
        assert '0x1234567890abcdef' in text_content
        assert '1.5 ETH' in html_content
        assert '1.5 ETH' in text_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
