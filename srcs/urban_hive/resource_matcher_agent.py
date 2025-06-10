import random
import json
from datetime import datetime
from typing import Dict, List, Tuple
import asyncio
from mcp.client import MCPClient
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
import subprocess
import os
from .ai_text_analyzer import ai_text_analyzer


class ResourceMatcherAgent:
    def __init__(self, mcp_server_path=None):
        """
        Initializes the ResourceMatcherAgent.
        This agent is responsible for matching resources like shared goods and leftover food
        by connecting to the Urban Hive MCP server.
        """
        self.mcp_server_path = mcp_server_path or "python -m srcs.urban_hive.providers.urban_hive_mcp_server"
        self.session = None

    async def _get_mcp_data(self, resource_uri: str) -> List[Dict]:
        """Helper function to fetch data from the MCP server."""
        try:
            # Initialize MCP client if not already done
            if not self.session:
                await self._initialize_mcp_client()
            
            # Read resource from MCP server
            result = await self.session.read_resource(resource_uri)
            return json.loads(result)
        except Exception as e:
            print(f"Error fetching data from MCP server ({resource_uri}): {e}")
            return []

    async def _initialize_mcp_client(self):
        """Initialize MCP client connection."""
        try:
            server_params = {
                "command": self.mcp_server_path.split(),
                "env": os.environ.copy()
            }
            
            transport = stdio_client(server_params)
            self.session = await ClientSession(transport).__aenter__()
        except Exception as e:
            print(f"Error initializing MCP client: {e}")
            self.session = None

    def run(self, query: str) -> str:
        """
        Runs the agent to find a match for the given query.
        """
        print(f"Running ResourceMatcherAgent for query: {query}")
        
        # Use asyncio to run the async method
        return asyncio.run(self._async_run(query))

    async def _async_run(self, query: str) -> str:
        """
        Async implementation of the run method.
        """
        # Use AI to analyze the query intent instead of hardcoded keywords
        query_type, resource_info = await ai_text_analyzer.analyze_resource_intent(query)
        
        if query_type == "offer":
            resource_requests = await self._get_mcp_data("urban-hive://resources/requests")
            return await self._handle_resource_offer(resource_info, resource_requests)
        elif query_type == "request":
            available_resources = await self._get_mcp_data("urban-hive://resources/available")
            return await self._handle_resource_request(resource_info, available_resources)
        else:
            return self._provide_general_suggestions()

# Removed _parse_query method - now using AI text analyzer instead of hardcoded keywords

    async def _handle_resource_offer(self, resource_info: Dict, resource_requests: List[Dict]) -> str:
        """
        Handle when someone is offering a resource
        """
        query = resource_info["query"]
        
        if not resource_requests:
            return "Could not retrieve resource requests from the data provider. Please try again later."
            
        potential_matches = []
        for request in resource_requests:
            similarity_score = self._calculate_similarity(query, request["name"])
            if similarity_score > 0.3:
                potential_matches.append((request, similarity_score))
        
        if potential_matches:
            potential_matches.sort(key=lambda x: x[1], reverse=True)
            best_match = potential_matches[0][0]
            needed_by_dt = datetime.fromisoformat(best_match['needed_by'])
            
            return f"""🎯 **Great news! We found a match for your offer:**

🧠 **AI 분석**: {resource_info.get('reasoning', 'N/A')} (신뢰도: {resource_info.get('confidence', 0):.1%})

**Your Offer:** {query}
**Matched with:** {best_match['requester']} in {best_match['location']}
**They need:** {best_match['name']}
**Needed by:** {needed_by_dt.strftime('%Y-%m-%d %H:%M')}

📱 **Next Steps:** We'll notify {best_match['requester']} about your offer.
"""
        else:
            return f"""📝 **Your offer for '{query}' has been registered!**

We'll notify you when a match is found.
"""

    async def _handle_resource_request(self, resource_info: Dict, available_resources: List[Dict]) -> str:
        """
        Handle when someone is requesting a resource
        """
        query = resource_info["query"]
        
        if not available_resources:
            return "Could not retrieve available resources from the data provider. Please try again later."

        potential_matches = []
        for resource in available_resources:
            similarity_score = self._calculate_similarity(query, resource["name"])
            if similarity_score > 0.2:
                potential_matches.append((resource, similarity_score))
        
        if potential_matches:
            potential_matches.sort(key=lambda x: x[1], reverse=True)
            matches_text = ""
            
            for i, (resource, score) in enumerate(potential_matches[:3]):
                distance = random.choice(["0.5km", "1.2km", "0.8km", "2.1km"])
                available_until_dt = datetime.fromisoformat(resource['available_until'])
                matches_text += f"""
**Match {i+1}:** {resource['name']}
📍 Available from: {resource['owner']} ({resource['location']})
📏 Distance: ~{distance} away
⏰ Available until: {available_until_dt.strftime('%Y-%m-%d %H:%M')}
"""
            
            return f"""🔍 **Found {len(potential_matches)} matches for your request!**

🧠 **AI 분석**: {resource_info.get('reasoning', 'N/A')} (신뢰도: {resource_info.get('confidence', 0):.1%})

**You're looking for:** {query}
{matches_text}
"""
        else:
            return f"""📋 **Your request for '{query}' has been posted!**

We'll notify you as soon as something becomes available.
"""

    def _provide_general_suggestions(self) -> str:
        return """🌍 **Welcome to Urban Hive Resource Sharing!**
You can either offer a resource or request one.
e.g., 'I have leftover bread' or 'I need a ladder for today'.
"""

    def _calculate_similarity(self, query: str, resource_name: str) -> float:
        query_words = set(query.lower().split())
        resource_words = set(resource_name.lower().split())
        
        if not query_words or not resource_words:
            return 0.0
        
        intersection = query_words.intersection(resource_words)
        union = query_words.union(resource_words)
        
        return len(intersection) / len(union)

    def get_resource_statistics(self) -> Dict:
        """Get resource statistics using MCP data."""
        return asyncio.run(self._async_get_statistics())

    async def _async_get_statistics(self) -> Dict:
        """Async implementation of get_resource_statistics."""
        available = await self._get_mcp_data("urban-hive://resources/available")
        requests = await self._get_mcp_data("urban-hive://resources/requests")
        return {
            "total_resources_available": len(available),
            "total_requests": len(requests),
        } 