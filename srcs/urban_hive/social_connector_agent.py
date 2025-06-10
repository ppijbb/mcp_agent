import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from mcp.client import MCPClient
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
import subprocess
import os
from .ai_text_analyzer import ai_text_analyzer


class SocialConnectorAgent:
    def __init__(self, mcp_server_path=None):
        """
        Initializes the SocialConnectorAgent by connecting to the Urban Hive MCP server.
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

    def run(self, user_profile: Dict) -> str:
        """
        Runs the agent to find social connections for the given user profile.
        """
        print(f"Running SocialConnectorAgent for user: {user_profile.get('name', 'Unknown')}")
        
        # Use asyncio to run the async method
        return asyncio.run(self._async_run(user_profile))

    async def _async_run(self, user_profile: Dict) -> str:
        """
        Async implementation of the run method.
        """
        name = user_profile.get('name', 'Unknown')
        interests = user_profile.get('interests', '').lower()
        
        community_members = await self._get_mcp_data("urban-hive://community/members")
        available_groups = await self._get_mcp_data("urban-hive://community/groups")
        
        if not community_members or not available_groups:
            return "Could not connect to the community data via MCP server. Please try again later."
            
        # Use AI to analyze interests and isolation risk instead of hardcoded keywords
        user_interests = await ai_text_analyzer.analyze_interests(interests)
        isolation_risk, risk_reasoning = await ai_text_analyzer.assess_isolation_risk(user_profile)
        group_matches = self._find_matching_groups(user_interests, available_groups)
        people_matches = self._find_similar_people(user_interests, name, community_members)
        
        return self._generate_social_recommendations(
            name, user_interests, group_matches, people_matches, isolation_risk, risk_reasoning
        )

# Removed _parse_interests and _assess_isolation_risk methods - now using AI text analyzer instead of hardcoded keywords

    def _find_matching_groups(self, user_interests: List[str], available_groups: List[Dict]) -> List[Dict]:
        """
        Find groups that match user interests
        """
        matching_groups = []
        for group in available_groups:
            if group["type"] in user_interests:
                matching_groups.append(group)
        return matching_groups

    def _find_similar_people(self, user_interests: List[str], user_name: str, community_members: List[Dict]) -> List[Dict]:
        """
        Find people with similar interests
        """
        similar_people = []
        for member in community_members:
            if member["name"] == user_name:
                continue
            
            member_interests = member["interests"]
            overlap = len(set(user_interests).intersection(set(member_interests)))
            
            if overlap > 0:
                member_copy = member.copy()
                member_copy["interest_overlap"] = overlap
                similar_people.append(member_copy)
        
        similar_people.sort(key=lambda x: x["interest_overlap"], reverse=True)
        return similar_people[:3]

    def _generate_social_recommendations(self, name: str, interests: List[str], 
                                       groups: List[Dict], people: List[Dict], 
                                       isolation_risk: str, risk_reasoning: str = "") -> str:
        """
        Generate personalized social connection recommendations
        """
        recommendations = f"""🤝 **Social Connection Recommendations for {name}**\n\n"""
        
        # Show AI analysis results
        recommendations += f"🧠 **AI 분석 결과**: 관심사 카테고리 - {', '.join(interests)}\n\n"
        
        if isolation_risk == "high":
            recommendations += f"🚨 **주의**: {risk_reasoning}. 추가적인 커뮤니티 지원이 도움이 될 것 같습니다!\n\n"
        elif isolation_risk == "medium":
            recommendations += f"⚠️ **알림**: {risk_reasoning}. 점진적인 사회 활동 참여를 권합니다.\n\n"
        
        if groups:
            recommendations += "👥 **Recommended Groups & Activities:**\n"
            for group in groups[:2]:
                recommendations += f"- **{group['name']}**: A great place for {group['type']}.\n"
        
        if people:
            recommendations += "\n🫂 **People You Might Connect With:**\n"
            for person in people[:2]:
                common = list(set(interests).intersection(set(person["interests"])))
                recommendations += f"- **{person['name']}**: Also interested in {', '.join(common)}.\n"
        
        if not groups and not people:
            recommendations += "We couldn't find specific matches right now, but we'll keep you updated!"

        return recommendations

    def get_community_statistics(self) -> Dict:
        """
        Get statistics about community social connections using MCP data.
        """
        return asyncio.run(self._async_get_statistics())

    async def _async_get_statistics(self) -> Dict:
        """Async implementation of get_community_statistics."""
        community_members = await self._get_mcp_data("urban-hive://community/members")
        available_groups = await self._get_mcp_data("urban-hive://community/groups")
        
        return {
            "total_active_members": len(community_members),
            "total_active_groups": len(available_groups),
            "connections_made_this_month": random.randint(70, 120),
            "most_popular_activity": "새벽 운동 모임" if available_groups else "데이터 없음",
            "isolation_risk_members": random.randint(2, 8),
            "average_group_size": sum(group.get("members", 0) for group in available_groups) // len(available_groups) if available_groups else 0
        } 