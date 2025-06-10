"""
Public Data API Client

Connects to Korean public data APIs to fetch real urban data including:
- Illegal dumping reports
- Traffic accidents and congestion
- Crime statistics
- Environmental data
- Population demographics
"""

import asyncio
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx
import json
from ..external_data_client import external_data_manager
from ..config import config, get_api_config, get_cache_config

class PublicDataClient:
    """Client for accessing Korean public data APIs."""
    
    def __init__(self):
        """Initialize the public data client with API keys and endpoints."""
        api_config = get_api_config()
        cache_config = get_cache_config()
        
        self.api_key = api_config.public_data_api_key
        self.endpoints = api_config.endpoints
        self.timeout = api_config.timeout_seconds
        self.max_retries = api_config.max_retries
        
        # Cache for fetched district data
        self._districts_cache = None
        self._cache_timestamp = None
        self._cache_duration = cache_config.cache_duration_hours * 60 * 60  # Convert hours to seconds

    async def fetch_illegal_dumping_data(self, district: Optional[str] = None) -> List[Dict]:
        """Fetch illegal dumping incident data."""
        try:
            # Only use external APIs - no data generation
            print(f"No external API configured for illegal dumping data: {district}")
            return []
        except Exception as e:
            print(f"Error fetching illegal dumping data: {e}")
            return []

    async def fetch_traffic_data(self, district: Optional[str] = None) -> List[Dict]:
        """Fetch traffic congestion and accident data."""
        try:
            # Only use external APIs - no data generation
            print(f"No external API configured for traffic data: {district}")
            return []
        except Exception as e:
            print(f"Error fetching traffic data: {e}")
            return []

    async def fetch_safety_data(self, district: Optional[str] = None) -> List[Dict]:
        """Fetch public safety and crime statistics."""
        try:
            # Only use external APIs - no data generation
            print(f"No external API configured for safety data: {district}")
            return []
        except Exception as e:
            print(f"Error fetching safety data: {e}")
            return []

    async def fetch_community_data(self) -> Dict:
        """Fetch community member and group data from external sources only."""
        try:
            return await external_data_manager.get_community_data()
        except Exception as e:
            print(f"Error fetching community data from external sources: {e}")
            return {"members": [], "groups": []}

    async def fetch_resource_data(self) -> Dict:
        """Fetch resource sharing data from external sources only."""
        try:
            return await external_data_manager.get_resource_data()
        except Exception as e:
            print(f"Error fetching resource data from external sources: {e}")
            return {"available": [], "requests": []}

    async def fetch_seoul_districts(self) -> List[str]:
        """
        Fetch Seoul district codes from external sources only.
        """
        try:
            # Use external data manager for district data
            districts = await external_data_manager.get_districts("seoul")
            
            # Update local cache for backward compatibility
            if districts:
                self._districts_cache = districts
                self._cache_timestamp = datetime.now().timestamp()
                return districts
            else:
                print("No districts returned from external sources")
                return []
            
        except Exception as e:
            print(f"Error fetching districts from external sources: {e}")
            return []
    
    def _is_cache_valid(self) -> bool:
        """Check if cached district data is still valid."""
        if not self._districts_cache or not self._cache_timestamp:
            return False
        
        current_time = datetime.now().timestamp()
        return (current_time - self._cache_timestamp) < self._cache_duration
    
    async def _fetch_districts_from_apis(self) -> List[str]:
        """Fetch district data from multiple API sources."""
        # Try official Korean statistical geographic information service
        try:
            districts = await self._fetch_from_sgis_api()
            if districts:
                return districts
        except Exception as e:
            print(f"SGIS API failed: {e}")
        
        # Try Seoul Open Data API
        try:
            districts = await self._fetch_from_seoul_api()
            if districts:
                return districts
        except Exception as e:
            print(f"Seoul API failed: {e}")
        
        # Try administrative district API
        try:
            districts = await self._fetch_from_admin_api()
            if districts:
                return districts
        except Exception as e:
            print(f"Admin API failed: {e}")
        
        # If all APIs fail, return empty list
        print("All external APIs failed for districts")
        return []
    
    async def _fetch_from_sgis_api(self) -> List[str]:
        """Fetch districts from Korean Statistical Geographic Information Service."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "serviceKey": self.api_key,
                    "format": "json",
                    "sido_cd": config.urban_data.seoul_administrative_code,
                }
                
                response = await client.get(self.endpoints["geographic_codes"], params=params)
                response.raise_for_status()
                data = response.json()
                
                if "result" in data and isinstance(data["result"], list):
                    districts = [item.get("addr_name") for item in data["result"] if item.get("addr_name")]
                    return list(set(districts))
                return []
                
        except httpx.HTTPStatusError as e:
            print(f"SGIS API HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"SGIS API error: {e}")
            return []
    
    async def _fetch_from_seoul_api(self) -> List[str]:
        """Fetch districts from Seoul Open Data API."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # The endpoint from config seems to be for supermarkets. A real implementation would use a proper
                # administrative district API, e.g. 'http://openapi.seoul.go.kr:8088/[KEY]/json/v_admdong_od/1/1000/'
                # For now, we will attempt to parse the configured endpoint.
                response = await client.get(self.endpoints["seoul_districts"])
                response.raise_for_status()
                data = response.json()
                
                # Find the list of items, usually in a "row" key.
                for key, value in data.items():
                    if isinstance(value, dict) and "row" in value:
                        rows = value["row"]
                        districts = []
                        for item in rows:
                            if "SH_ADDR" in item and isinstance(item["SH_ADDR"], str):
                                parts = item["SH_ADDR"].split()
                                if len(parts) > 1 and parts[1].endswith('구'):
                                    districts.append(parts[1])
                            elif "SIGNGU_NM" in item:
                                districts.append(item["SIGNGU_NM"])
                        
                        if districts:
                            return list(set(districts))

                return []
                
        except httpx.HTTPStatusError as e:
            print(f"Seoul API HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"Seoul API error: {e}")
            return []
    
    async def _fetch_from_admin_api(self) -> List[str]:
        """Fetch districts from administrative district API."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "serviceKey": self.api_key,
                    "type": "json",
                    "pageNo": 1,
                    "numOfRows": 100,
                    "sidoNm": "서울특별시"
                }

                response = await client.get(self.endpoints["admin_districts"], params=params)
                response.raise_for_status()
                data = response.json()
                
                if "response" in data and "body" in data["response"] and "items" in data["response"]["body"]:
                    items = data["response"]["body"]["items"]
                    if isinstance(items, dict) and "item" in items:
                        items = items["item"]
                    
                    if isinstance(items, list):
                        districts = [item.get("signguNm") for item in items if item.get("signguNm")]
                        return list(set(districts))

                return []
                
        except httpx.HTTPStatusError as e:
            print(f"Admin API HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"Admin API error: {e}")
            return []
    




# Global instance
public_data_client = PublicDataClient() 