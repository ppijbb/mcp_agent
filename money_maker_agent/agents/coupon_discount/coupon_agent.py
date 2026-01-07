"""
Coupon/Discount Agent

Automatically scrapes coupons, creates content, and monetizes through affiliate links.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
import re

from bs4 import BeautifulSoup
import aiohttp

from ...core.orchestrator import BaseAgent
from ...core.ledger import Ledger

logger = logging.getLogger(__name__)


class Coupon:
    """Represents a coupon or discount."""
    
    def __init__(
        self,
        title: str,
        description: str,
        discount: str,
        url: str,
        expiry: Optional[str] = None,
        category: Optional[str] = None
    ):
        self.title = title
        self.description = description
        self.discount = discount
        self.url = url
        self.expiry = expiry
        self.category = category
        self.affiliate_link: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'description': self.description,
            'discount': self.discount,
            'url': self.url,
            'expiry': self.expiry,
            'category': self.category,
            'affiliate_link': self.affiliate_link
        }


class CouponDiscountAgent(BaseAgent):
    """
    Coupon/Discount Agent
    
    Automatically:
    - Scrapes coupons from websites
    - Creates blog content
    - Generates affiliate links
    - Tracks revenue from commissions
    """
    
    def __init__(self, name: str, config: Dict[str, Any], ledger: Ledger):
        super().__init__(name, config, ledger)
        self.config_detail = config.get('config', {})
        self.scraping_config = self.config_detail.get('scraping', {})
        self.affiliate_config = self.config_detail.get('affiliate', {})
        self.content_config = self.config_detail.get('content', {})
        self.revenue_config = self.config_detail.get('revenue', {})
        
        self.coupons: List[Coupon] = []
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> bool:
        """Initialize coupon discount agent."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            logger.info("Coupon/Discount Agent initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Coupon/Discount Agent: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown agent."""
        await super().shutdown()
        if self.session:
            await self.session.close()
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute coupon discount cycle.
        
        Returns:
            Execution result with estimated revenue
        """
        try:
            self._running = True
            
            # Scrape coupons
            if self.scraping_config.get('enabled', True):
                await self._scrape_coupons()
            
            # Generate affiliate links
            if self.affiliate_config:
                await self._generate_affiliate_links()
            
            # Create content (if enabled)
            if self.content_config.get('blog_enabled', False):
                await self._create_content()
            
            # Estimate revenue
            estimated_revenue = self._estimate_revenue()
            
            # Record revenue (estimated, will be confirmed when actual sales occur)
            if estimated_revenue > 0:
                self.ledger.record_transaction(
                    agent_name=self.name,
                    transaction_type='income',
                    amount=estimated_revenue,
                    description=f"Estimated affiliate revenue from {len(self.coupons)} coupons",
                    metadata={
                        'coupons_count': len(self.coupons),
                        'estimated': True
                    }
                )
            
            result = {
                'success': True,
                'income': estimated_revenue,
                'description': f"Coupon agent: {len(self.coupons)} coupons processed, ${estimated_revenue:.2f} estimated revenue",
                'metadata': {
                    'coupons_found': len(self.coupons),
                    'content_created': self.content_config.get('blog_enabled', False)
                }
            }
            
            logger.info(
                f"Coupon/Discount Agent executed: "
                f"{len(self.coupons)} coupons, ${estimated_revenue:.2f} estimated revenue"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing Coupon/Discount Agent: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'income': 0.0
            }
        finally:
            self._running = False
    
    async def _scrape_coupons(self):
        """Scrape coupons from configured websites."""
        websites = self.scraping_config.get('websites', [])
        
        if not websites:
            logger.debug("No websites configured for scraping")
            return
        
        for website_config in websites:
            if isinstance(website_config, dict):
                url = website_config.get('url')
                categories = website_config.get('categories', [])
            else:
                url = website_config
                categories = []
            
            if not url:
                continue
            
            try:
                coupons = await self._scrape_website(url, categories)
                self.coupons.extend(coupons)
                logger.info(f"Scraped {len(coupons)} coupons from {url}")
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
    
    async def _scrape_website(self, url: str, categories: List[str]) -> List[Coupon]:
        """
        Scrape coupons from a single website.
        
        Args:
            url: Website URL
            categories: Categories to filter
        
        Returns:
            List of Coupon objects
        """
        if not self.session:
            return []
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Generic coupon extraction (adapt based on website structure)
                coupons = []
                
                # Look for common coupon patterns
                # This is a generic implementation - real scraping would be site-specific
                coupon_elements = soup.find_all(['div', 'article', 'section'], class_=re.compile(r'coupon|deal|discount', re.I))
                
                for element in coupon_elements[:10]:  # Limit to 10 per site
                    try:
                        title_elem = element.find(['h1', 'h2', 'h3', 'h4', 'span', 'a'], class_=re.compile(r'title|name', re.I))
                        title = title_elem.get_text(strip=True) if title_elem else "Coupon"
                        
                        desc_elem = element.find(['p', 'div'], class_=re.compile(r'desc|text', re.I))
                        description = desc_elem.get_text(strip=True) if desc_elem else ""
                        
                        discount_elem = element.find(['span', 'div'], class_=re.compile(r'discount|save|off', re.I))
                        discount = discount_elem.get_text(strip=True) if discount_elem else "Discount available"
                        
                        link_elem = element.find('a', href=True)
                        coupon_url = link_elem['href'] if link_elem else url
                        if not coupon_url.startswith('http'):
                            coupon_url = urljoin(url, coupon_url)
                        
                        coupon = Coupon(
                            title=title[:100],  # Limit length
                            description=description[:200],
                            discount=discount,
                            url=coupon_url,
                            category=categories[0] if categories else None
                        )
                        coupons.append(coupon)
                    except Exception as e:
                        logger.debug(f"Error parsing coupon element: {e}")
                        continue
                
                return coupons
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []
    
    async def _generate_affiliate_links(self):
        """Generate affiliate links for coupons."""
        # Amazon Associates
        if self.affiliate_config.get('amazon_associates', {}).get('enabled', False):
            tag = self.affiliate_config['amazon_associates'].get('tag', '')
            if tag:
                for coupon in self.coupons:
                    # Check if URL is Amazon
                    if 'amazon.com' in coupon.url or 'amzn.to' in coupon.url:
                        # Add affiliate tag
                        if '?' in coupon.url:
                            coupon.affiliate_link = f"{coupon.url}&tag={tag}"
                        else:
                            coupon.affiliate_link = f"{coupon.url}?tag={tag}"
        
        # TODO: Implement other affiliate networks (Rakuten, etc.)
    
    async def _create_content(self):
        """Create blog content from coupons."""
        if not self.coupons:
            return
        
        # Group coupons by category
        coupons_by_category = {}
        for coupon in self.coupons:
            category = coupon.category or 'General'
            if category not in coupons_by_category:
                coupons_by_category[category] = []
            coupons_by_category[category].append(coupon)
        
        # Create blog post content (simplified)
        # In production, this would use LLM to generate SEO-optimized content
        content = f"# Best Deals and Coupons - {datetime.now().strftime('%B %Y')}\n\n"
        
        for category, category_coupons in coupons_by_category.items():
            content += f"## {category}\n\n"
            for coupon in category_coupons[:5]:  # Top 5 per category
                link = coupon.affiliate_link or coupon.url
                content += f"- **{coupon.title}**: {coupon.discount} - [Get Deal]({link})\n"
            content += "\n"
        
        # TODO: Publish to blog platform (WordPress, Medium, etc.)
        logger.info(f"Generated blog content: {len(content)} characters")
    
    def _estimate_revenue(self) -> float:
        """
        Estimate revenue from coupons.
        
        Returns:
            Estimated revenue in USD
        """
        if not self.coupons:
            return 0.0
        
        # Simple estimation: assume 1% click-through rate, 5% conversion, average $50 order
        # with 5% commission
        clicks_per_coupon = 100  # Estimated
        conversion_rate = self.revenue_config.get('commission_rate', 0.05)  # 5%
        average_order_value = 50.0
        commission_rate = self.revenue_config.get('commission_rate', 0.05)  # 5%
        
        total_clicks = len(self.coupons) * clicks_per_coupon
        conversions = total_clicks * 0.01 * conversion_rate  # 1% CTR, then conversion rate
        revenue = conversions * average_order_value * commission_rate
        
        return revenue

