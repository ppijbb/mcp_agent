"""
Competitor Monitoring Agent

Automatically monitors competitors' prices, products, and marketing strategies.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import re

import aiohttp
from bs4 import BeautifulSoup

from ...core.orchestrator import BaseAgent
from ...core.ledger import Ledger

logger = logging.getLogger(__name__)


class CompetitorMonitoringAgent(BaseAgent):
    """
    Competitor Monitoring Agent
    
    Automatically:
    - Monitors competitor prices
    - Detects new product launches
    - Analyzes marketing strategies
    - Generates competitive intelligence reports
    - Identifies opportunities
    """
    
    def __init__(self, name: str, config: Dict[str, Any], ledger: Ledger):
        super().__init__(name, config, ledger)
        self.config_detail = config.get('config', {})
        self.competitors = self.config_detail.get('competitors', [])
        self.monitoring_interval = self.config_detail.get('monitoring_interval', 3600)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.competitor_data: Dict[str, Dict[str, Any]] = {}
        self.previous_snapshots: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize competitor monitoring agent."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Load previous snapshots
            await self._load_previous_snapshots()
            
            logger.info(f"Competitor Monitoring Agent initialized with {len(self.competitors)} competitors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Competitor Monitoring Agent: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown agent."""
        await super().shutdown()
        if self.session:
            await self.session.close()
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute competitor monitoring cycle.
        
        Returns:
            Execution result with insights value
        """
        try:
            self._running = True
            
            # Monitor each competitor
            insights = []
            for competitor in self.competitors:
                try:
                    competitor_insights = await self._monitor_competitor(competitor)
                    insights.extend(competitor_insights)
                except Exception as e:
                    logger.error(f"Failed to monitor competitor {competitor}: {e}")
            
            # Generate report
            report = self._generate_report(insights)
            
            # Estimate value of insights
            insights_value = self._estimate_insights_value(insights)
            
            # Record value as income
            if insights_value > 0:
                self.ledger.record_transaction(
                    agent_name=self.name,
                    transaction_type='income',
                    amount=insights_value,
                    description=f"Competitive intelligence: {len(insights)} insights",
                    metadata={
                        'competitors_monitored': len(self.competitors),
                        'insights_count': len(insights),
                        'report_generated': True
                    }
                )
            
            result = {
                'success': True,
                'income': insights_value,
                'description': f"Competitor Monitoring: {len(insights)} insights, ${insights_value:.2f} value",
                'metadata': {
                    'competitors_monitored': len(self.competitors),
                    'insights_count': len(insights),
                    'report': report
                }
            }
            
            logger.info(
                f"Competitor Monitoring Agent executed: "
                f"{len(insights)} insights, ${insights_value:.2f} value"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing Competitor Monitoring Agent: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'income': 0.0
            }
        finally:
            self._running = False
    
    async def _monitor_competitor(self, competitor: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Monitor a single competitor.
        
        Args:
            competitor: Competitor configuration
            
        Returns:
            List of insights
        """
        insights = []
        url = competitor.get('url', '')
        name = competitor.get('name', 'Unknown')
        
        if not url:
            return insights
        
        try:
            # Fetch competitor website
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return insights
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract current snapshot
                snapshot = {
                    'url': url,
                    'name': name,
                    'timestamp': datetime.now().isoformat(),
                    'prices': await self._extract_prices(soup, competitor),
                    'products': await self._extract_products(soup, competitor),
                    'promotions': await self._extract_promotions(soup, competitor)
                }
                
                # Compare with previous snapshot
                previous = self.previous_snapshots.get(url, {})
                if previous:
                    insights.extend(self._compare_snapshots(previous, snapshot))
                
                # Save current snapshot
                self.previous_snapshots[url] = snapshot
                self.competitor_data[name] = snapshot
                
        except Exception as e:
            logger.error(f"Error monitoring competitor {name}: {e}")
        
        return insights
    
    async def _extract_prices(self, soup: BeautifulSoup, competitor: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract prices from competitor website."""
        prices = []
        
        # Look for price elements
        price_selectors = competitor.get('price_selectors', ['.price', '[class*="price"]', '[data-price]'])
        
        for selector in price_selectors:
            price_elements = soup.select(selector)
            for elem in price_elements[:20]:  # Limit to 20
                try:
                    price_text = elem.get_text(strip=True)
                    # Extract numeric price
                    price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                    if price_match:
                        price = float(price_match.group().replace(',', ''))
                        product_name = elem.find_previous(['h1', 'h2', 'h3', 'a'])
                        product_name = product_name.get_text(strip=True) if product_name else "Unknown"
                        
                        prices.append({
                            'product': product_name[:100],
                            'price': price,
                            'currency': 'USD'
                        })
                except Exception as e:
                    logger.debug(f"Error extracting price: {e}")
                    continue
        
        return prices
    
    async def _extract_products(self, soup: BeautifulSoup, competitor: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract product information."""
        products = []
        
        # Look for product elements
        product_selectors = competitor.get('product_selectors', ['.product', '[class*="product"]', 'article'])
        
        for selector in product_selectors:
            product_elements = soup.select(selector)
            for elem in product_elements[:20]:  # Limit to 20
                try:
                    title_elem = elem.find(['h1', 'h2', 'h3', 'a', 'span'], class_=re.compile(r'title|name', re.I))
                    title = title_elem.get_text(strip=True) if title_elem else "Unknown Product"
                    
                    link_elem = elem.find('a', href=True)
                    link = link_elem['href'] if link_elem else ''
                    
                    products.append({
                        'title': title[:200],
                        'url': link,
                        'extracted_at': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.debug(f"Error extracting product: {e}")
                    continue
        
        return products
    
    async def _extract_promotions(self, soup: BeautifulSoup, competitor: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract promotions and marketing messages."""
        promotions = []
        
        # Look for promotion elements
        promo_keywords = ['sale', 'discount', 'promo', 'offer', 'deal', 'coupon']
        promo_elements = soup.find_all(string=re.compile('|'.join(promo_keywords), re.I))
        
        for elem in promo_elements[:10]:  # Limit to 10
            try:
                parent = elem.find_parent(['div', 'section', 'article'])
                if parent:
                    promo_text = parent.get_text(strip=True)[:200]
                    promotions.append({
                        'text': promo_text,
                        'extracted_at': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.debug(f"Error extracting promotion: {e}")
                continue
        
        return promotions
    
    def _compare_snapshots(
        self,
        previous: Dict[str, Any],
        current: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Compare previous and current snapshots to find changes.
        
        Returns:
            List of insights
        """
        insights = []
        
        # Price changes
        previous_prices = {p['product']: p['price'] for p in previous.get('prices', [])}
        current_prices = {p['product']: p['price'] for p in current.get('prices', [])}
        
        for product, current_price in current_prices.items():
            if product in previous_prices:
                previous_price = previous_prices[product]
                if abs(current_price - previous_price) / previous_price > 0.05:  # 5% change
                    change_pct = ((current_price - previous_price) / previous_price) * 100
                    insights.append({
                        'type': 'price_change',
                        'product': product,
                        'previous_price': previous_price,
                        'current_price': current_price,
                        'change_percentage': change_pct,
                        'timestamp': current['timestamp']
                    })
            else:
                # New product
                insights.append({
                    'type': 'new_product',
                    'product': product,
                    'price': current_price,
                    'timestamp': current['timestamp']
                })
        
        # New products
        previous_products = {p['title'] for p in previous.get('products', [])}
        current_products = {p['title'] for p in current.get('products', [])}
        new_products = current_products - previous_products
        
        for product_title in new_products:
            insights.append({
                'type': 'product_launch',
                'product': product_title,
                'timestamp': current['timestamp']
            })
        
        return insights
    
    def _generate_report(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate competitive intelligence report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_insights': len(insights),
            'insights_by_type': {},
            'summary': ''
        }
        
        # Group insights by type
        for insight in insights:
            insight_type = insight.get('type', 'unknown')
            if insight_type not in report['insights_by_type']:
                report['insights_by_type'][insight_type] = []
            report['insights_by_type'][insight_type].append(insight)
        
        # Generate summary
        summary_parts = []
        for insight_type, type_insights in report['insights_by_type'].items():
            summary_parts.append(f"{len(type_insights)} {insight_type.replace('_', ' ')}")
        
        report['summary'] = f"Found {len(insights)} insights: {', '.join(summary_parts)}"
        
        return report
    
    async def _load_previous_snapshots(self):
        """Load previous monitoring snapshots."""
        # TODO: Load from persistent storage
        self.previous_snapshots = {}
    
    def _estimate_insights_value(self, insights: List[Dict[str, Any]]) -> float:
        """
        Estimate value of competitive insights.
        
        Args:
            insights: List of insights
            
        Returns:
            Estimated value in USD
        """
        if not insights:
            return 0.0
        
        # Simple estimation:
        # - Price change insight: $5-20 value
        # - New product insight: $10-50 value
        # - Marketing insight: $2-10 value
        # - Average: $10 per insight
        
        value_per_insight = 10.0
        total_value = len(insights) * value_per_insight
        
        # Convert to daily estimate
        daily_value = total_value / 30.0  # Assume insights are valuable for a month
        
        return daily_value



