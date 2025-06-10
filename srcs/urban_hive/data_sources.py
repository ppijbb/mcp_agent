"""
External Data Sources Manager

This module provides interfaces to external data sources (APIs, databases, files)
to completely eliminate hardcoded data from the Urban Hive system.
"""

import json
import asyncio
import aiofiles
import httpx
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
# Removed database imports - external sources only


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    source_type: str  # "api", "database", "file", "web"
    endpoint: Optional[str] = None
    file_path: Optional[str] = None
    cache_duration: int = 3600  # seconds
    headers: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def fetch_data(self, data_type: str, **kwargs) -> Any:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the data source is available."""
        pass


class APIDataSource(DataSource):
    """Data source that fetches from external APIs."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.base_url = config.endpoint
        self.headers = config.headers or {}
        if config.auth_token:
            self.headers["Authorization"] = f"Bearer {config.auth_token}"
    
    async def fetch_data(self, data_type: str, **kwargs) -> Any:
        """Fetch data from API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.base_url}/{data_type}"
                if kwargs:
                    url += "?" + "&".join([f"{k}={v}" for k, v in kwargs.items()])
                
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"API fetch error for {data_type}: {e}")
            return None
    
    async def is_available(self) -> bool:
        """Check API availability."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health", headers=self.headers)
                return response.status_code == 200
        except:
            return False


# DatabaseDataSource removed - external sources only


class FileDataSource(DataSource):
    """Data source that fetches from JSON/YAML files."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.data_dir = Path(config.file_path or "data")
        self.cache = {}
        
    async def fetch_data(self, data_type: str, **kwargs) -> Any:
        """Fetch data from file."""
        try:
            file_path = self.data_dir / f"{data_type}.json"
            
            if not file_path.exists():
                print(f"Data file not found: {file_path}")
                return None
            
            # Check cache first
            if data_type in self.cache:
                return self._filter_data(self.cache[data_type], **kwargs)
            
            # Load from file
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                self.cache[data_type] = data
                return self._filter_data(data, **kwargs)
                
        except Exception as e:
            print(f"File fetch error for {data_type}: {e}")
            return None
    
    def _filter_data(self, data: Any, **kwargs) -> Any:
        """Filter data based on kwargs."""
        if not kwargs or not isinstance(data, list):
            return data
        
        filtered = []
        for item in data:
            if isinstance(item, dict):
                match = True
                for key, value in kwargs.items():
                    if key in item and item[key] != value:
                        match = False
                        break
                if match:
                    filtered.append(item)
        
        return filtered if filtered else data
    
    async def is_available(self) -> bool:
        """Check file availability."""
        return self.data_dir.exists()


class WebScrapingDataSource(DataSource):
    """Data source that scrapes data from web pages."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.base_url = config.endpoint
        
    async def fetch_data(self, data_type: str, **kwargs) -> Any:
        """Fetch data by web scraping."""
        try:
            # This would implement web scraping for real-time data
            # For now, we'll simulate with placeholder
            scraped_data = {
                "real_time_districts": await self._scrape_district_data(),
                "trending_activities": await self._scrape_trending_activities(),
                "current_resource_prices": await self._scrape_market_prices()
            }
            return scraped_data.get(data_type)
        except Exception as e:
            print(f"Web scraping error for {data_type}: {e}")
            return None
    
    async def _scrape_district_data(self) -> List[Dict]:
        """Scrape real-time district data from government websites."""
        # Simulate scraping Seoul district data
        return [
            {"name": "강남구", "population": 542000, "area": 39.5, "status": "active"},
            {"name": "서초구", "population": 445000, "area": 47.0, "status": "active"},
            # This would be real scraped data
        ]
    
    async def _scrape_trending_activities(self) -> List[Dict]:
        """Scrape trending activities from social media or community sites."""
        # Simulate scraping trending activities
        return [
            {"activity": "비건 요리", "popularity": 95, "season": "current"},
            {"activity": "도시농업", "popularity": 87, "season": "spring"},
        ]
    
    async def _scrape_market_prices(self) -> Dict[str, int]:
        """Scrape current market prices for resources."""
        return {
            "전동드릴": 8500,
            "자전거": 15000,
            "청소기": 12000
        }
    
    async def is_available(self) -> bool:
        """Check web source availability."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.base_url)
                return response.status_code == 200
        except:
            return False


class DataSourceManager:
    """Manages multiple data sources with fallback mechanisms."""
    
    def __init__(self):
        self.sources: Dict[str, List[DataSource]] = {}
        self.cache = {}
        self.cache_timestamps = {}
        
    def register_source(self, data_type: str, source: DataSource, priority: int = 0):
        """Register a data source for a specific data type."""
        if data_type not in self.sources:
            self.sources[data_type] = []
        
        # Insert based on priority (higher priority first)
        inserted = False
        for i, (existing_priority, existing_source) in enumerate(self.sources[data_type]):
            if priority > existing_priority:
                self.sources[data_type].insert(i, (priority, source))
                inserted = True
                break
        
        if not inserted:
            self.sources[data_type].append((priority, source))
    
    async def get_data(self, data_type: str, use_cache: bool = True, **kwargs) -> Any:
        """Get data from the best available source."""
        cache_key = f"{data_type}_{hash(str(sorted(kwargs.items())))}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
        
        # Try sources in priority order
        if data_type not in self.sources:
            print(f"No data sources registered for {data_type}")
            return None
        
        for priority, source in self.sources[data_type]:
            try:
                if await source.is_available():
                    data = await source.fetch_data(data_type, **kwargs)
                    if data is not None:
                        # Cache successful result
                        self.cache[cache_key] = data
                        self.cache_timestamps[cache_key] = asyncio.get_event_loop().time()
                        return data
                else:
                    print(f"Data source unavailable for {data_type}")
            except Exception as e:
                print(f"Error fetching {data_type} from source: {e}")
                continue
        
        print(f"All data sources failed for {data_type}")
        return None
    
    def _is_cache_valid(self, cache_key: str, max_age: int = 3600) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        age = asyncio.get_event_loop().time() - self.cache_timestamps[cache_key]
        return age < max_age
    
    async def get_districts(self, region: str = "seoul") -> List[str]:
        """Get districts from data sources."""
        data = await self.get_data("districts", region=region)
        if data and isinstance(data, list):
            return [item.get("name", item) if isinstance(item, dict) else item for item in data]
        return []
    
    async def get_names(self, name_type: str = "family") -> List[str]:
        """Get names from data sources."""
        data = await self.get_data("names", type=name_type)
        if data and isinstance(data, list):
            return [item.get("name", item) if isinstance(item, dict) else item for item in data]
        return []
    
    async def get_activities(self, category: str = None) -> List[str]:
        """Get activities from data sources."""
        kwargs = {"category": category} if category else {}
        data = await self.get_data("activities", **kwargs)
        if data and isinstance(data, list):
            return [item.get("name", item) if isinstance(item, dict) else item for item in data]
        return []
    
    async def get_resources(self, category: str = None) -> List[Dict]:
        """Get resources from data sources."""
        kwargs = {"category": category} if category else {}
        data = await self.get_data("resources", **kwargs)
        return data if isinstance(data, list) else []
    
    async def get_templates(self, template_type: str) -> List[str]:
        """Get text templates from data sources."""
        data = await self.get_data("templates", type=template_type)
        if data and isinstance(data, list):
            return [item.get("template", item) if isinstance(item, dict) else item for item in data]
        return []
    
    async def get_locations(self, location_type: str) -> List[str]:
        """Get locations from data sources."""
        data = await self.get_data("locations", type=location_type)
        if data and isinstance(data, list):
            return [item.get("name", item) if isinstance(item, dict) else item for item in data]
        return []


# Initialize data source manager
data_source_manager = DataSourceManager()


async def init_data_sources():
    """Initialize all data sources with fallbacks."""
    
    # Primary: API sources (highest priority)
    api_config = DataSourceConfig(
        source_type="api",
        endpoint="https://api.korea-data.go.kr",
        auth_token="your_api_token"
    )
    api_source = APIDataSource(api_config)
    
    # Database sources removed - external sources only
    
    # Tertiary: File sources
    file_config = DataSourceConfig(
        source_type="file",
        file_path="data"
    )
    file_source = FileDataSource(file_config)
    
    # Quaternary: Web scraping sources
    web_config = DataSourceConfig(
        source_type="web",
        endpoint="https://www.seoul.go.kr"
    )
    web_source = WebScrapingDataSource(web_config)
    
    # Register sources for each data type with priorities
    data_types = ["districts", "names", "activities", "resources", "templates", "locations"]
    
    for data_type in data_types:
        data_source_manager.register_source(data_type, api_source, priority=100)
        # Database sources removed - external sources only
        data_source_manager.register_source(data_type, file_source, priority=60)
        data_source_manager.register_source(data_type, web_source, priority=40)


# Convenience functions for backward compatibility
async def get_external_districts(region: str = "seoul") -> List[str]:
    """Get districts from external data sources."""
    return await data_source_manager.get_districts(region)


async def get_external_names(name_type: str = "family") -> List[str]:
    """Get names from external data sources."""
    return await data_source_manager.get_names(name_type)


async def get_external_activities(category: str = None) -> List[str]:
    """Get activities from external data sources."""
    return await data_source_manager.get_activities(category)


async def get_external_resources(category: str = None) -> List[Dict]:
    """Get resources from external data sources."""
    return await data_source_manager.get_resources(category)


async def get_external_templates(template_type: str) -> List[str]:
    """Get text templates from external data sources."""
    return await data_source_manager.get_templates(template_type)


async def get_external_locations(location_type: str) -> List[str]:
    """Get locations from external data sources."""
    return await data_source_manager.get_locations(location_type)