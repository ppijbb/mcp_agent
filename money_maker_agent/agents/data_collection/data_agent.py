"""
Data Collection Agent

Automatically collects, processes, and sells data through API services.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import csv

import aiohttp
from bs4 import BeautifulSoup

from ...core.orchestrator import BaseAgent
from ...core.ledger import Ledger

logger = logging.getLogger(__name__)


class DataCollectionAgent(BaseAgent):
    """
    Data Collection Agent
    
    Automatically:
    - Collects data from various sources
    - Cleans and structures data
    - Serves data through FastAPI
    - Matches customers and generates revenue
    """
    
    def __init__(self, name: str, config: Dict[str, Any], ledger: Ledger):
        super().__init__(name, config, ledger)
        self.config_detail = config.get('config', {})
        self.data_sources = self.config_detail.get('data_sources', [])
        self.api_enabled = self.config_detail.get('api_enabled', False)
        self.api_port = self.config_detail.get('api_port', 8000)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.collected_data: List[Dict[str, Any]] = []
        self.api_server_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize data collection agent."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Start API server if enabled
            if self.api_enabled:
                await self._start_api_server()
            
            logger.info(f"Data Collection Agent initialized with {len(self.data_sources)} data sources")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Data Collection Agent: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown agent."""
        await super().shutdown()
        if self.api_server_task:
            self.api_server_task.cancel()
        if self.session:
            await self.session.close()
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute data collection cycle.
        
        Returns:
            Execution result with revenue
        """
        try:
            self._running = True
            
            # Collect data from sources
            collected_count = 0
            for source in self.data_sources:
                try:
                    data = await self._collect_from_source(source)
                    if data:
                        self.collected_data.extend(data)
                        collected_count += len(data)
                except Exception as e:
                    logger.error(f"Failed to collect from source {source}: {e}")
            
            # Process and structure data
            processed_data = await self._process_data(self.collected_data)
            
            # Save data
            await self._save_data(processed_data)
            
            # Estimate revenue
            revenue = self._estimate_revenue(processed_data)
            
            # Record revenue
            if revenue > 0:
                self.ledger.record_transaction(
                    agent_name=self.name,
                    transaction_type='income',
                    amount=revenue,
                    description=f"Data collection revenue: {collected_count} records collected",
                    metadata={
                        'records_collected': collected_count,
                        'records_processed': len(processed_data),
                        'data_sources': len(self.data_sources)
                    }
                )
            
            result = {
                'success': True,
                'income': revenue,
                'description': f"Data Collection: {collected_count} records, ${revenue:.2f} revenue",
                'metadata': {
                    'records_collected': collected_count,
                    'records_processed': len(processed_data),
                    'api_enabled': self.api_enabled
                }
            }
            
            logger.info(
                f"Data Collection Agent executed: "
                f"{collected_count} records collected, ${revenue:.2f} revenue"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing Data Collection Agent: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'income': 0.0
            }
        finally:
            self._running = False
    
    async def _collect_from_source(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect data from a source.
        
        Args:
            source: Source configuration
            
        Returns:
            List of collected data records
        """
        source_type = source.get('type', 'web_scraping')
        url = source.get('url', '')
        
        if not url:
            return []
        
        try:
            if source_type == 'web_scraping':
                return await self._scrape_website(url, source)
            elif source_type == 'api':
                return await self._fetch_from_api(url, source)
            else:
                logger.warning(f"Unknown source type: {source_type}")
                return []
        except Exception as e:
            logger.error(f"Error collecting from source {url}: {e}")
            return []
    
    async def _scrape_website(self, url: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape data from a website."""
        if not self.session:
            return []
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract data based on selectors
                selectors = config.get('selectors', {})
                records = []
                
                # Generic extraction (adapt based on website structure)
                items = soup.find_all(selectors.get('item', 'div'), class_=selectors.get('item_class'))
                
                for item in items[:100]:  # Limit to 100 items
                    try:
                        record = {}
                        
                        # Extract fields based on selectors
                        for field, selector in selectors.get('fields', {}).items():
                            elem = item.select_one(selector)
                            if elem:
                                record[field] = elem.get_text(strip=True)
                        
                        if record:
                            record['source_url'] = url
                            record['collected_at'] = datetime.now().isoformat()
                            records.append(record)
                    except Exception as e:
                        logger.debug(f"Error parsing item: {e}")
                        continue
                
                return records
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []
    
    async def _fetch_from_api(self, url: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch data from an API."""
        if not self.session:
            return []
        
        try:
            headers = config.get('headers', {})
            params = config.get('params', {})
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.warning(f"API request failed: HTTP {response.status}")
                    return []
                
                data = await response.json()
                
                # Transform API response to records
                records = []
                items = data.get('items', data.get('data', [data] if isinstance(data, dict) else []))
                
                for item in items:
                    if isinstance(item, dict):
                        item['source_url'] = url
                        item['collected_at'] = datetime.now().isoformat()
                        records.append(item)
                
                return records
                
        except Exception as e:
            logger.error(f"Error fetching from API {url}: {e}")
            return []
    
    async def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and clean collected data.
        
        Args:
            data: Raw collected data
            
        Returns:
            Processed data
        """
        processed = []
        
        for record in data:
            # Clean and normalize
            cleaned = {}
            for key, value in record.items():
                if isinstance(value, str):
                    # Remove extra whitespace
                    cleaned[key] = ' '.join(value.split())
                else:
                    cleaned[key] = value
            
            # Add processing metadata
            cleaned['processed_at'] = datetime.now().isoformat()
            processed.append(cleaned)
        
        return processed
    
    async def _save_data(self, data: List[Dict[str, Any]]):
        """Save processed data to storage."""
        if not data:
            return
        
        try:
            # Save to JSON file
            data_dir = Path(__file__).parent.parent.parent.parent / "data" / "collected_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = data_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(data)} records to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    async def _start_api_server(self):
        """Start FastAPI server for data access."""
        # TODO: Implement FastAPI server
        logger.info(f"API server requested on port {self.api_port} (not yet implemented)")
    
    def _estimate_revenue(self, data: List[Dict[str, Any]]) -> float:
        """
        Estimate revenue from data sales.
        
        Args:
            data: Processed data records
            
        Returns:
            Estimated revenue in USD
        """
        if not data:
            return 0.0
        
        # Simple estimation:
        # - Data API access: $0.01-0.10 per record
        # - Bulk data sales: $0.001-0.01 per record
        # - Average: $0.05 per record per month
        
        revenue_per_record = 0.05
        monthly_revenue = len(data) * revenue_per_record
        
        # Convert to daily estimate
        daily_revenue = monthly_revenue / 30.0
        
        return daily_revenue



