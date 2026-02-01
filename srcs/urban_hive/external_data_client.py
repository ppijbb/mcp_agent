"""
External Data Client

This module provides interfaces to fetch real data from external sources only.
No data generation or simulation - only fetching existing data from APIs and MCP servers.
"""

import httpx
from typing import Dict, List, Any
import os
from datetime import datetime

# Raise explicit error when data unavailable
from .exceptions import ExternalDataUnavailableError


class ExternalAPIClient:
    """Client for fetching data from external APIs only."""

    def __init__(self):
        self.timeout = 30.0
        self.api_keys = {
            "korea_data": os.getenv("KOREA_DATA_API_KEY"),
            "seoul_open": os.getenv("SEOUL_OPEN_API_KEY"),
            "statistics": os.getenv("STATISTICS_API_KEY")
        }

        # Real API endpoints - no simulation
        self.endpoints = {
            "districts": "https://api.data.go.kr/openapi/district-service",
            "population": "https://kosis.kr/openapi/statisticsData.do",
            "activities": "https://api.seoul.go.kr/cultural-activities",
            "community": "https://api.seoul.go.kr/community-data",
            "resources": "https://api.seoul.go.kr/sharing-economy"
        }

    async def fetch_districts(self, region: str = "seoul") -> List[str]:
        """Fetch real district data from Korean government APIs."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "serviceKey": self.api_keys["korea_data"],
                    "sido": region,
                    "format": "json"
                }

                response = await client.get(self.endpoints["districts"], params=params)
                response.raise_for_status()

                data = response.json()
                if "result" in data and isinstance(data["result"], list):
                    return [item.get("name", "") for item in data["result"] if item.get("name")]

                return []

        except Exception as e:
            raise ExternalDataUnavailableError(
                f"Failed to fetch districts from external API: {e}"
            )

    async def fetch_community_data(self) -> Dict[str, List]:
        """Fetch real community data from Seoul Open Data API."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "KEY": self.api_keys["seoul_open"],
                    "TYPE": "json",
                    "SERVICE": "CommunityData"
                }

                response = await client.get(self.endpoints["community"], params=params)
                response.raise_for_status()

                data = response.json()
                return {
                    "members": data.get("members", []),
                    "groups": data.get("groups", [])
                }

        except Exception as e:
            raise ExternalDataUnavailableError(f"Failed to fetch community data: {e}")

    async def fetch_resource_data(self) -> Dict[str, List]:
        """Fetch real resource sharing data from Seoul APIs."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "KEY": self.api_keys["seoul_open"],
                    "TYPE": "json",
                    "SERVICE": "SharingEconomy"
                }

                response = await client.get(self.endpoints["resources"], params=params)
                response.raise_for_status()

                data = response.json()
                return {
                    "available": data.get("available_resources", []),
                    "requests": data.get("resource_requests", [])
                }

        except Exception as e:
            raise ExternalDataUnavailableError(f"Failed to fetch resource data: {e}")

    async def fetch_activity_data(self, category: str = None) -> List[Dict]:
        """Fetch real activity data from cultural APIs."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "KEY": self.api_keys["seoul_open"],
                    "TYPE": "json",
                    "SERVICE": "CulturalActivities"
                }

                if category:
                    params["CATEGORY"] = category

                response = await client.get(self.endpoints["activities"], params=params)
                response.raise_for_status()

                data = response.json()
                return data.get("activities", [])

        except Exception as e:
            raise ExternalDataUnavailableError(f"Failed to fetch activity data: {e}")


class MCPDataClient:
    """Client for fetching data from MCP servers only."""

    def __init__(self):
        self.mcp_endpoints = {
            "urban_hive": "mcp://urban-hive-server",
            "seoul_data": "mcp://seoul-data-server",
            "community": "mcp://community-server"
        }

    async def fetch_from_mcp(self, server: str, resource: str, **params) -> Any:
        """Fetch data from MCP server."""
        try:
            # This would use actual MCP protocol
            # For now, return None since we only want external data
            print(f"MCP fetch from {server}/{resource} with params: {params}")
            return None

        except Exception as e:
            print(f"Failed to fetch from MCP {server}/{resource}: {e}")
            return None

    async def get_districts_from_mcp(self, region: str = "seoul") -> List[str]:
        """Get districts from MCP server."""
        result = await self.fetch_from_mcp("seoul_data", "districts", region=region)
        return result if isinstance(result, list) else []

    async def get_community_from_mcp(self) -> Dict:
        """Get community data from MCP server."""
        result = await self.fetch_from_mcp("community", "all")
        return result if isinstance(result, dict) else {"members": [], "groups": []}

    async def get_resources_from_mcp(self) -> Dict:
        """Get resource data from MCP server."""
        result = await self.fetch_from_mcp("urban_hive", "resources")
        return result if isinstance(result, dict) else {"available": [], "requests": []}


class ExternalDataManager:
    """Manager for fetching data from external sources only."""

    def __init__(self):
        self.api_client = ExternalAPIClient()
        self.mcp_client = MCPDataClient()
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 3600  # 1 hour

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache_timestamps:
            return False

        age = datetime.now().timestamp() - self.cache_timestamps[key]
        return age < self.cache_duration

    def _update_cache(self, key: str, data: Any):
        """Update cache with fetched data."""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now().timestamp()

    async def get_districts(self, region: str = "seoul") -> List[str]:
        """Get districts from external sources only."""
        cache_key = f"districts_{region}"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        # Try API first
        try:
            districts = await self.api_client.fetch_districts(region)
            if districts:
                self._update_cache(cache_key, districts)
                return districts
        except ExternalDataUnavailableError as e:
            self.logger.warning(f"API failed for districts in {region}: {e}")

        # Try MCP as fallback
        try:
            districts = await self.mcp_client.get_districts_from_mcp(region)
            if districts:
                self._update_cache(cache_key, districts)
                return districts
        except Exception as e:
            self.logger.warning(f"MCP failed for districts in {region}: {e}")

        # No data available from any source
        raise ExternalDataUnavailableError(
            f"No external data available for districts in {region} from any source"
        )

    async def get_community_data(self) -> Dict[str, List]:
        """Get community data from external sources only."""
        cache_key = "community_data"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        # Try API first
        try:
            data = await self.api_client.fetch_community_data()
            if data.get("members") or data.get("groups"):
                self._update_cache(cache_key, data)
                return data
        except ExternalDataUnavailableError as e:
            self.logger.warning(f"API failed for community data: {e}")

        # Try MCP as fallback
        try:
            mcp_data = await self.mcp_client.get_community_from_mcp()
            if mcp_data and (mcp_data.get("members") or mcp_data.get("groups")):
                self._update_cache(cache_key, mcp_data)
                return mcp_data
        except Exception as e:
            self.logger.warning(f"MCP failed for community data: {e}")

        # No data available from any source
        raise ExternalDataUnavailableError("No external community data available from any source")

    async def get_resource_data(self) -> Dict[str, List]:
        """Get resource data from external sources only."""
        cache_key = "resource_data"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        # Try API first
        try:
            data = await self.api_client.fetch_resource_data()
            if data.get("available") or data.get("requests"):
                self._update_cache(cache_key, data)
                return data
        except ExternalDataUnavailableError as e:
            self.logger.warning(f"API failed for resource data: {e}")

        # Try MCP as fallback
        try:
            mcp_data = await self.mcp_client.get_resources_from_mcp()
            if mcp_data and (mcp_data.get("available") or mcp_data.get("requests")):
                self._update_cache(cache_key, mcp_data)
                return mcp_data
        except Exception as e:
            self.logger.warning(f"MCP failed for resource data: {e}")

        # No data available from any source
        raise ExternalDataUnavailableError("No external resource data available from any source")

    async def get_activity_data(self, category: str = None) -> List[Dict]:
        """Get activity data from external sources only."""
        cache_key = f"activities_{category or 'all'}"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        # Only try API - no MCP equivalent for activities
        try:
            activities = await self.api_client.fetch_activity_data(category)
            if activities:
                self._update_cache(cache_key, activities)
                return activities
        except ExternalDataUnavailableError as e:
            self.logger.warning(f"API failed for activity data (category: {category}): {e}")

        # No data available
        raise ExternalDataUnavailableError(
            f"No external activity data available for category: {category}"
        )

    async def health_check(self) -> Dict[str, bool]:
        """Check health of external data sources."""
        results = {}

        # Check API endpoints
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for name, endpoint in self.api_client.endpoints.items():
                    try:
                        response = await client.get(endpoint, timeout=5.0)
                        results[f"api_{name}"] = response.status_code < 500
                    except:
                        results[f"api_{name}"] = False
        except:
            pass

        # Check MCP servers (would be actual MCP health checks)
        for name in self.mcp_client.mcp_endpoints.keys():
            results[f"mcp_{name}"] = False  # Cannot check without real MCP implementation

        return results


# Global instance
external_data_manager = ExternalDataManager()


# Convenience functions
async def get_external_districts(region: str = "seoul") -> List[str]:
    """Get districts from external sources only."""
    return await external_data_manager.get_districts(region)


async def get_external_community_data() -> Dict[str, List]:
    """Get community data from external sources only."""
    return await external_data_manager.get_community_data()


async def get_external_resource_data() -> Dict[str, List]:
    """Get resource data from external sources only."""
    return await external_data_manager.get_resource_data()


async def get_external_activity_data(category: str = None) -> List[Dict]:
    """Get activity data from external sources only."""
    return await external_data_manager.get_activity_data(category)


async def check_external_data_health() -> Dict[str, bool]:
    """Check health of all external data sources."""
    return await external_data_manager.health_check()
