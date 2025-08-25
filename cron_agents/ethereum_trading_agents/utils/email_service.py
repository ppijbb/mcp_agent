"""
Professional Email Service for Ethereum Trading Reports

This module provides comprehensive email functionality for sending trading reports,
transaction notifications, and market analysis summaries.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
import json

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EmailService:
    """Professional email service for trading reports"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_address = os.getenv("EMAIL_ADDRESS")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.sender_name = os.getenv("SENDER_NAME", "Ethereum Trading Agent")
        
        # MCP Email Server (if available)
        self.mcp_email_url = os.getenv("MCP_EMAIL_URL")
        self.mcp_api_key = os.getenv("MCP_EMAIL_API_KEY")
        
        # Recipients
        self.default_recipients = self._load_recipients()
        
    def _load_recipients(self) -> List[str]:
        """Load email recipients from environment"""
        recipients_str = os.getenv("EMAIL_RECIPIENTS", "")
        if recipients_str:
            return [email.strip() for email in recipients_str.split(",")]
        return []
    
    async def send_trading_report(self, 
                                 transaction_data: Dict[str, Any],
                                 market_analysis: Dict[str, Any],
                                 recipients: Optional[List[str]] = None) -> bool:
        """Send comprehensive trading report email"""
        try:
            subject = f"🚀 Ethereum Trading Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Generate HTML content
            html_content = self._generate_trading_report_html(transaction_data, market_analysis)
            
            # Generate plain text content
            text_content = self._generate_trading_report_text(transaction_data, market_analysis)
            
            # Try MCP first, fallback to SMTP
            if self.mcp_email_url and self.mcp_api_key:
                success = await self._send_via_mcp(subject, html_content, text_content, recipients)
                if success:
                    logger.info("Trading report sent successfully via MCP")
                    return True
            
            # Fallback to SMTP
            success = await self._send_via_smtp(subject, html_content, text_content, recipients)
            if success:
                logger.info("Trading report sent successfully via SMTP")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to send trading report: {e}")
            return False
    
    async def send_transaction_notification(self, 
                                          transaction_hash: str,
                                          transaction_details: Dict[str, Any],
                                          recipients: Optional[List[str]] = None) -> bool:
        """Send immediate transaction notification email"""
        try:
            subject = f"⚡ Ethereum Transaction Executed - {transaction_hash[:10]}..."
            
            # Generate HTML content
            html_content = self._generate_transaction_notification_html(transaction_hash, transaction_details)
            
            # Generate plain text content
            text_content = self._generate_transaction_notification_text(transaction_hash, transaction_details)
            
            # Try MCP first, fallback to SMTP
            if self.mcp_email_url and self.mcp_api_key:
                success = await self._send_via_mcp(subject, html_content, text_content, recipients)
                if success:
                    logger.info("Transaction notification sent successfully via MCP")
                    return True
            
            # Fallback to SMTP
            success = await self._send_via_smtp(subject, html_content, text_content, recipients)
            if success:
                logger.info("Transaction notification sent successfully via SMTP")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to send transaction notification: {e}")
            return False
    
    async def send_daily_summary(self, 
                                daily_trades: List[Dict[str, Any]],
                                portfolio_summary: Dict[str, Any],
                                recipients: Optional[List[str]] = None) -> bool:
        """Send daily trading summary email"""
        try:
            subject = f"📊 Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Generate HTML content
            html_content = self._generate_daily_summary_html(daily_trades, portfolio_summary)
            
            # Generate plain text content
            text_content = self._generate_daily_summary_text(daily_trades, portfolio_summary)
            
            # Try MCP first, fallback to SMTP
            if self.mcp_email_url and self.mcp_api_key:
                success = await self._send_via_mcp(subject, html_content, text_content, recipients)
                if success:
                    logger.info("Daily summary sent successfully via MCP")
                    return True
            
            # Fallback to SMTP
            success = await self._send_via_smtp(subject, html_content, text_content, recipients)
            if success:
                logger.info("Daily summary sent successfully via SMTP")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False
    
    async def _send_via_mcp(self, subject: str, html_content: str, text_content: str, 
                           recipients: Optional[List[str]] = None) -> bool:
        """Send email via MCP server"""
        try:
            if not self.mcp_email_url or not self.mcp_api_key:
                return False
                
            recipients = recipients or self.default_recipients
            if not recipients:
                logger.error("No recipients specified for MCP email")
                return False
            
            payload = {
                "api_key": self.mcp_api_key,
                "from": self.email_address,
                "from_name": self.sender_name,
                "to": recipients,
                "subject": subject,
                "html_content": html_content,
                "text_content": text_content
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.mcp_email_url}/send-email", 
                                      json=payload, 
                                      timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("success", False)
                    else:
                        logger.error(f"MCP email server returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send email via MCP: {e}")
            return False
    
    async def _send_via_smtp(self, subject: str, html_content: str, text_content: str,
                            recipients: Optional[List[str]] = None) -> bool:
        """Send email via SMTP"""
        try:
            if not all([self.email_address, self.email_password]):
                logger.error("SMTP credentials not configured")
                return False
                
            recipients = recipients or self.default_recipients
            if not recipients:
                logger.error("No recipients specified for SMTP email")
                return False
            
            # Create message
            message = MIMEMultipart("alternative")
            message["From"] = f"{self.sender_name} <{self.email_address}>"
            message["To"] = ", ".join(recipients)
            message["Subject"] = subject
            
            # Add both HTML and text parts
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")
            
            message.attach(text_part)
            message.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_address, self.email_password)
                server.sendmail(self.email_address, recipients, message.as_string())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return False
    
    def _generate_trading_report_html(self, transaction_data: Dict[str, Any], 
                                    market_analysis: Dict[str, Any]) -> str:
        """Generate HTML content for trading report"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ethereum Trading Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; }}
                .section h3 {{ color: #667eea; margin-top: 0; }}
                .transaction-details {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .market-analysis {{ background: #e8f5e8; padding: 15px; border-radius: 5px; }}
                .highlight {{ background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Ethereum Trading Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h3>📊 Transaction Summary</h3>
                    <div class="transaction-details">
                        <p><strong>Transaction Hash:</strong> {transaction_data.get('hash', 'N/A')}</p>
                        <p><strong>Block Number:</strong> {transaction_data.get('blockNumber', 'N/A')}</p>
                        <p><strong>From Address:</strong> {transaction_data.get('from', 'N/A')}</p>
                        <p><strong>To Address:</strong> {transaction_data.get('to', 'N/A')}</p>
                        <p><strong>Value:</strong> {transaction_data.get('value', 'N/A')} ETH</p>
                        <p><strong>Gas Used:</strong> {transaction_data.get('gasUsed', 'N/A')}</p>
                        <p><strong>Gas Price:</strong> {transaction_data.get('gasPrice', 'N/A')} Gwei</p>
                        <p><strong>Status:</strong> {transaction_data.get('status', 'N/A')}</p>
                    </div>
                </div>
                
                <div class="section">
                    <h3>🔍 Market Analysis</h3>
                    <div class="market-analysis">
                        <p><strong>Current ETH Price:</strong> ${market_analysis.get('current_price', 'N/A')}</p>
                        <p><strong>24h Change:</strong> {market_analysis.get('price_change_24h', 'N/A')}%</p>
                        <p><strong>Market Sentiment:</strong> {market_analysis.get('sentiment', 'N/A')}</p>
                        <p><strong>Technical Indicators:</strong> {market_analysis.get('technical_indicators', 'N/A')}</p>
                    </div>
                </div>
                
                <div class="section">
                    <h3>💡 Trading Insights</h3>
                    <div class="highlight">
                        <p><strong>Why this trade was executed:</strong></p>
                        <p>{transaction_data.get('reason', 'Based on market analysis and trading strategy')}</p>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This report was automatically generated by the Ethereum Trading Agent System</p>
                    <p>For questions or support, please contact the system administrator</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _generate_trading_report_text(self, transaction_data: Dict[str, Any], 
                                    market_analysis: Dict[str, Any]) -> str:
        """Generate plain text content for trading report"""
        return f"""
Ethereum Trading Report
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRANSACTION SUMMARY:
Transaction Hash: {transaction_data.get('hash', 'N/A')}
Block Number: {transaction_data.get('blockNumber', 'N/A')}
From Address: {transaction_data.get('from', 'N/A')}
To Address: {transaction_data.get('to', 'N/A')}
Value: {transaction_data.get('value', 'N/A')} ETH
Gas Used: {transaction_data.get('gasUsed', 'N/A')}
Gas Price: {transaction_data.get('gasPrice', 'N/A')} Gwei
Status: {transaction_data.get('status', 'N/A')}

MARKET ANALYSIS:
Current ETH Price: ${market_analysis.get('current_price', 'N/A')}
24h Change: {market_analysis.get('price_change_24h', 'N/A')}%
Market Sentiment: {market_analysis.get('sentiment', 'N/A')}
Technical Indicators: {market_analysis.get('technical_indicators', 'N/A')}

TRADING INSIGHTS:
Why this trade was executed: {transaction_data.get('reason', 'Based on market analysis and trading strategy')}

---
This report was automatically generated by the Ethereum Trading Agent System
For questions or support, please contact the system administrator
        """
    
    def _generate_transaction_notification_html(self, transaction_hash: str, 
                                             transaction_details: Dict[str, Any]) -> str:
        """Generate HTML content for transaction notification"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Transaction Notification</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #28a745; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .details {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>⚡ Transaction Executed</h2>
                    <p>Your Ethereum transaction has been processed</p>
                </div>
                
                <div class="details">
                    <p><strong>Transaction Hash:</strong> {transaction_hash}</p>
                    <p><strong>Status:</strong> {transaction_details.get('status', 'Confirmed')}</p>
                    <p><strong>Amount:</strong> {transaction_details.get('value', 'N/A')} ETH</p>
                    <p><strong>Gas Used:</strong> {transaction_details.get('gasUsed', 'N/A')}</p>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="footer">
                    <p>Ethereum Trading Agent System</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _generate_transaction_notification_text(self, transaction_hash: str, 
                                             transaction_details: Dict[str, Any]) -> str:
        """Generate plain text content for transaction notification"""
        return f"""
Transaction Executed

Your Ethereum transaction has been processed successfully.

Transaction Hash: {transaction_hash}
Status: {transaction_details.get('status', 'Confirmed')}
Amount: {transaction_details.get('value', 'N/A')} ETH
Gas Used: {transaction_details.get('gasUsed', 'N/A')}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Ethereum Trading Agent System
        """
    
    def _generate_daily_summary_html(self, daily_trades: List[Dict[str, Any]], 
                                   portfolio_summary: Dict[str, Any]) -> str:
        """Generate HTML content for daily summary"""
        trades_html = ""
        for trade in daily_trades:
            trades_html += f"""
                <tr>
                    <td>{trade.get('hash', 'N/A')[:10]}...</td>
                    <td>{trade.get('type', 'N/A')}</td>
                    <td>{trade.get('amount', 'N/A')} ETH</td>
                    <td>{trade.get('status', 'N/A')}</td>
                    <td>{trade.get('timestamp', 'N/A')}</td>
                </tr>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Daily Trading Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; }}
                .section h3 {{ color: #667eea; margin-top: 0; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .summary-card h4 {{ margin: 0 0 10px 0; color: #667eea; }}
                .trades-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .trades-table th, .trades-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                .trades-table th {{ background: #f8f9fa; font-weight: bold; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Daily Trading Summary</h1>
                    <p>{datetime.now().strftime('%Y-%m-%d')}</p>
                </div>
                
                <div class="section">
                    <h3>📈 Portfolio Summary</h3>
                    <div class="summary-grid">
                        <div class="summary-card">
                            <h4>Total Trades</h4>
                            <p>{portfolio_summary.get('total_trades', 0)}</p>
                        </div>
                        <div class="summary-card">
                            <h4>Successful Trades</h4>
                            <p>{portfolio_summary.get('successful_trades', 0)}</p>
                        </div>
                        <div class="summary-card">
                            <h4>Total Volume</h4>
                            <p>{portfolio_summary.get('total_volume', 0)} ETH</p>
                        </div>
                        <div class="summary-card">
                            <h4>Net P&L</h4>
                            <p>{portfolio_summary.get('net_pnl', 0)} ETH</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h3>🔄 Today's Trades</h3>
                    <table class="trades-table">
                        <thead>
                            <tr>
                                <th>Hash</th>
                                <th>Type</th>
                                <th>Amount</th>
                                <th>Status</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {trades_html}
                        </tbody>
                    </table>
                </div>
                
                <div class="footer">
                    <p>This summary was automatically generated by the Ethereum Trading Agent System</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _generate_daily_summary_text(self, daily_trades: List[Dict[str, Any]], 
                                   portfolio_summary: Dict[str, Any]) -> str:
        """Generate plain text content for daily summary"""
        trades_text = ""
        for trade in daily_trades:
            trades_text += f"""
Hash: {trade.get('hash', 'N/A')[:10]}...
Type: {trade.get('type', 'N/A')}
Amount: {trade.get('amount', 'N/A')} ETH
Status: {trade.get('status', 'N/A')}
Time: {trade.get('timestamp', 'N/A')}
---"""
        
        return f"""
Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}

PORTFOLIO SUMMARY:
Total Trades: {portfolio_summary.get('total_trades', 0)}
Successful Trades: {portfolio_summary.get('successful_trades', 0)}
Total Volume: {portfolio_summary.get('total_volume', 0)} ETH
Net P&L: {portfolio_summary.get('net_pnl', 0)} ETH

TODAY'S TRADES:
{trades_text}

---
This summary was automatically generated by the Ethereum Trading Agent System
        """
