"""
Dynamic Data Agent for Urban Hive

This agent fetches real data from external sources only (APIs, MCP servers).
No data generation or simulation - only retrieval of existing data.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from .external_data_client import external_data_manager


class DynamicDataAgent:
    """
    Agent that fetches data from external sources only.
    No data generation - purely external data retrieval.
    """

    def __init__(self):
        """Initialize the Dynamic Data Agent."""
        self.external_manager = external_data_manager
        self.cache_duration = 3600  # 1 hour cache
    
    async def get_dynamic_districts(self, region: str = "seoul") -> List[str]:
        """Get districts from external sources only."""
        return await self.external_manager.get_districts(region)
    
    async def get_dynamic_community_members(self, count: int = None) -> List[Dict]:
        """Get community members from external sources only."""
        data = await self.external_manager.get_community_data()
        members = data.get("members", [])
        
        # Limit count if specified
        if count and len(members) > count:
            members = members[:count]
        
        return members
    
    async def get_dynamic_community_groups(self, count: int = None) -> List[Dict]:
        """Get community groups from external sources only."""
        data = await self.external_manager.get_community_data()
        groups = data.get("groups", [])
        
        # Limit count if specified
        if count and len(groups) > count:
            groups = groups[:count]
        
        return groups
    
    async def get_dynamic_resources(self, resource_type: str = "available", count: int = None) -> List[Dict]:
        """Get resources from external sources only."""
        data = await self.external_manager.get_resource_data()
        
        if resource_type == "available":
            resources = data.get("available", [])
        elif resource_type == "requests":
            resources = data.get("requests", [])
        else:
            # Return both types combined
            available = data.get("available", [])
            requests = data.get("requests", [])
            resources = available + requests
        
        # Limit count if specified
        if count and len(resources) > count:
            resources = resources[:count]
        
        return resources
    
    async def get_district_characteristics_dynamic(self, district: str) -> Dict[str, Any]:
        """Get district characteristics - return empty dict since no external source."""
        print(f"No external source for district characteristics: {district}")
        return {}
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of external data sources."""
        return await self.external_manager.health_check()


# Global instance
dynamic_data_agent = DynamicDataAgent()


# Convenience functions for backward compatibility
async def get_dynamic_seoul_districts() -> List[str]:
    """Get Seoul districts from external sources only."""
    return await dynamic_data_agent.get_dynamic_districts("seoul")


async def get_dynamic_community_data() -> Dict[str, List[Dict]]:
    """Get community data from external sources only."""
    members = await dynamic_data_agent.get_dynamic_community_members()
    groups = await dynamic_data_agent.get_dynamic_community_groups()
    
    return {
        "members": members,
        "groups": groups
    }


async def get_dynamic_resource_data() -> Dict[str, List[Dict]]:
    """Get resource data from external sources only."""
    available = await dynamic_data_agent.get_dynamic_resources("available")
    requests = await dynamic_data_agent.get_dynamic_resources("requests")
    
    return {
        "available": available,
        "requests": requests
    }


async def get_district_characteristics(district: str) -> Dict[str, Any]:
    """Get district characteristics from external sources (returns empty if none)."""
    return await dynamic_data_agent.get_district_characteristics_dynamic(district)


async def check_data_agent_health() -> Dict[str, bool]:
    """Check health of data agent's external sources."""
    return await dynamic_data_agent.health_check()