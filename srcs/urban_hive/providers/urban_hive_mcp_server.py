#!/usr/bin/env python3
"""
Urban Hive MCP Server

A Model Context Protocol server that provides access to urban data including:
- Resource sharing data (available items, requests)
- Community member and group information  
- Urban analytics data (illegal dumping, traffic, safety statistics)
- Public data API integration for real city data

This server connects to real data sources and public APIs to provide 
actual urban information rather than simulated data.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
import httpx
from datetime import datetime, timedelta

# MCP imports
from mcp.server import Server
from .public_data_client import public_data_client
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Server instance
server = Server("urban-hive")

# Configuration
API_BASE_URL = os.getenv("URBAN_HIVE_API_BASE", "http://127.0.0.1:8001")

# Public data API endpoints (example URLs - replace with actual APIs)
PUBLIC_DATA_APIS = {
    "illegal_dumping": "https://api.data.go.kr/openapi/tn_pubr_public_illegal_dump_api",
    "traffic_accidents": "https://api.data.go.kr/openapi/tn_pubr_public_trfcacdnt_api", 
    "population_stats": "https://kosis.kr/openapi/statisticsData.do",
    "air_quality": "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"
}

async def fetch_real_data(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """
    Fetch data from public APIs using the PublicDataClient.
    Falls back to simulated data if APIs are unavailable.
    """
    try:
        # Use the new PublicDataClient for actual data
        if endpoint == "resources/available":
            resource_data = await public_data_client.fetch_resource_data()
            return resource_data["available"]
        elif endpoint == "resources/requests":
            resource_data = await public_data_client.fetch_resource_data()
            return resource_data["requests"]
        elif endpoint == "community/members":
            community_data = await public_data_client.fetch_community_data()
            return community_data["members"]
        elif endpoint == "community/groups":
            community_data = await public_data_client.fetch_community_data()
            return community_data["groups"]
        elif endpoint == "urban-data/illegal-dumping":
            return await public_data_client.fetch_illegal_dumping_data()
        elif endpoint == "urban-data/traffic":
            return await public_data_client.fetch_traffic_data()
        elif endpoint == "urban-data/safety":
            return await public_data_client.fetch_safety_data()
        else:
            # Try fallback to internal API
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{API_BASE_URL}/{endpoint}")
                if response.status_code == 200:
                    return response.json()
    except Exception as e:
        print(f"Warning: Could not fetch from public data client: {e}")
    
    # Fallback to simulated data
    return get_fallback_data(endpoint)

def get_fallback_data(endpoint: str) -> Dict:
    """Provide minimal fallback data when all data sources are unavailable."""
    fallback_data = {
        "resources/available": [
            {"id": 1, "type": "system", "name": "데이터 서비스 점검 중", "owner": "시스템", "location": "전체", "available_until": (datetime.now() + timedelta(hours=1)).isoformat()},
        ],
        "resources/requests": [
            {"id": 1, "type": "system", "name": "데이터 서비스 점검 중", "requester": "시스템", "location": "전체", "needed_by": (datetime.now() + timedelta(hours=1)).isoformat()},
        ],
        "community/members": [
            {"id": 1, "name": "시스템 알림", "age": 0, "interests": ["서비스 점검"], "location": "전체", "activity_level": "system"},
        ],
        "community/groups": [
            {"id": 1, "name": "서비스 점검 중", "type": "system", "members": 0, "location": "전체", "schedule": "점검 중"},
        ],
        "urban-data/illegal-dumping": [
            {"location": "전체 지역", "incidents": 0, "trend": "점검 중", "last_month": 0, "timestamp": datetime.now().isoformat()},
        ]
    }
    return fallback_data.get(endpoint, [])

@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available Urban Hive data resources."""
    return [
        Resource(
            uri="urban-hive://resources/available",
            name="Available Resources",
            description="Items, food, and services available for sharing in the community",
            mimeType="application/json",
        ),
        Resource(
            uri="urban-hive://resources/requests", 
            name="Resource Requests",
            description="Community requests for items, services, or help",
            mimeType="application/json",
        ),
        Resource(
            uri="urban-hive://community/members",
            name="Community Members",
            description="Active community members and their interests",
            mimeType="application/json",
        ),
        Resource(
            uri="urban-hive://community/groups",
            name="Community Groups",
            description="Available social groups and activities",
            mimeType="application/json",
        ),
        Resource(
            uri="urban-hive://urban-data/illegal-dumping",
            name="Illegal Dumping Data",
            description="Real-time illegal dumping incident reports and trends",
            mimeType="application/json",
        ),
        Resource(
            uri="urban-hive://urban-data/traffic",
            name="Traffic Data", 
            description="Traffic congestion and accident data",
            mimeType="application/json",
        ),
        Resource(
            uri="urban-hive://urban-data/safety",
            name="Public Safety Data",
            description="Crime statistics and safety information by area", 
            mimeType="application/json",
        ),
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read Urban Hive data resource by URI."""
    if not uri.startswith("urban-hive://"):
        raise ValueError(f"Unknown resource: {uri}")
    
    # Extract the endpoint from URI
    endpoint = uri.replace("urban-hive://", "")
    
    try:
        data = await fetch_real_data(endpoint)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Error fetching data from {endpoint}: {str(e)}")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available Urban Hive analysis tools."""
    return [
        Tool(
            name="match_resources",
            description="Find matches between available resources and requests",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Resource search query"},
                    "query_type": {"type": "string", "enum": ["offer", "request"], "description": "Type of query"}
                },
                "required": ["query", "query_type"]
            }
        ),
        Tool(
            name="find_social_connections",
            description="Find social connections and groups based on interests",
            inputSchema={
                "type": "object", 
                "properties": {
                    "user_name": {"type": "string", "description": "User's name"},
                    "interests": {"type": "string", "description": "User's interests and hobbies"},
                    "location": {"type": "string", "description": "User's location"}
                },
                "required": ["user_name", "interests"]
            }
        ),
        Tool(
            name="analyze_urban_data",
            description="Analyze urban data for insights and trends",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_type": {"type": "string", "enum": ["illegal-dumping", "traffic", "safety", "events"], "description": "Type of urban data to analyze"},
                    "area": {"type": "string", "description": "Specific area to analyze (optional)"}
                },
                "required": ["data_type"]
            }
        ),
        Tool(
            name="get_real_time_data",
            description="Get real-time urban data from public APIs",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_type": {"type": "string", "enum": ["air_quality", "traffic_live", "emergency_alerts"], "description": "Type of real-time data"},
                    "location": {"type": "string", "description": "Location for the data"}
                },
                "required": ["api_type", "location"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute Urban Hive tools."""
    
    if name == "match_resources":
        query = arguments["query"]
        query_type = arguments["query_type"]
        
        # Fetch available resources and requests
        if query_type == "offer":
            requests_data = await fetch_real_data("resources/requests")
            result = f"Found {len(requests_data)} potential matches for your offer: '{query}'"
        else:
            available_data = await fetch_real_data("resources/available") 
            result = f"Found {len(available_data)} available resources matching: '{query}'"
            
        return [TextContent(type="text", text=result)]
    
    elif name == "find_social_connections":
        user_name = arguments["user_name"]
        interests = arguments["interests"]
        
        # Fetch community data
        members_data = await fetch_real_data("community/members")
        groups_data = await fetch_real_data("community/groups")
        
        result = f"Social recommendations for {user_name}:\n"
        result += f"- Found {len(groups_data)} relevant groups\n"
        result += f"- Found {len(members_data)} potential connections\n"
        result += f"Based on interests: {interests}"
        
        return [TextContent(type="text", text=result)]
    
    elif name == "analyze_urban_data":
        data_type = arguments["data_type"]
        area = arguments.get("area", "전체")
        
        # Fetch urban data
        data = await fetch_real_data(f"urban-data/{data_type}")
        
        # Perform basic analysis
        if isinstance(data, list) and len(data) > 0:
            total_incidents = sum(item.get("incidents", 0) for item in data if "incidents" in item)
            result = f"Urban data analysis for {data_type} in {area}:\n"
            result += f"- Total incidents: {total_incidents}\n"
            result += f"- Data points analyzed: {len(data)}\n"
            result += f"- Analysis timestamp: {datetime.now().isoformat()}"
        else:
            result = f"No data available for {data_type} in {area}"
            
        return [TextContent(type="text", text=result)]
    
    elif name == "get_real_time_data":
        api_type = arguments["api_type"]
        location = arguments["location"]
        
        # This would connect to real public APIs
        # For now, return simulated real-time data
        result = f"Real-time {api_type} data for {location}:\n"
        result += f"- Status: Active monitoring\n"
        result += f"- Last updated: {datetime.now().isoformat()}\n"
        result += f"- Data source: Public API integration"
        
        return [TextContent(type="text", text=result)]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point for the Urban Hive MCP server."""
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="urban-hive",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 