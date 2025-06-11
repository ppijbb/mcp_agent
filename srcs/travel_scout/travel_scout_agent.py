#!/usr/bin/env python3
"""
Travel Scout Agent - MCP-Agent Implementation

A travel search agent using the mcp_agent framework for consistent
integration with the MCP ecosystem.
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from mcp_agent.app import MCPApp
from mcp_agent.config import get_settings
from .mcp_browser_client import TravelMCPManager


class TravelScoutAgent:
    """Travel Scout Agent using the real MCP Browser backend"""
    
    def __init__(self, output_dir: str = "travel_results", config_path: str = "configs/mcp_agent.config.yaml"):
        """Initialize Travel Scout Agent
        
        Args:
            output_dir: Directory to save travel search results
            config_path: Path to mcp_agent configuration file
        """
        self.output_dir = output_dir
        self.search_history = []
        self.quality_criteria = {
            'min_hotel_rating': 4.0,
            'max_hotel_price': 500,
            'max_flight_price': 2000
        }
        self.mcp_manager = TravelMCPManager()
        self.mcp_connected = False
        
        # Initialize app for potential future use with mcp_agent framework
        self.app = MCPApp(
            name="travel_scout_agent", 
            settings=get_settings(config_path)
        )
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """Get MCP Browser connection status"""
        status = 'connected' if self.mcp_connected else 'disconnected'
        return {
            'status': status,
            'browser_connected': self.mcp_connected,
            'real_time_capability': self.mcp_connected,
            'description': f'MCP Browser is {status}.'
        }

    async def initialize_mcp(self) -> bool:
        """Initialize and connect to the MCP Browser server."""
        try:
            self.mcp_connected = await self.mcp_manager.mcp_client.connect_to_mcp_server()
            return self.mcp_connected
        except Exception as e:
            print(f"Error during MCP initialization: {e}")
            self.mcp_connected = False
            return False

    def update_quality_criteria(self, criteria: Dict[str, Any]) -> None:
        """Update quality criteria for search"""
        self.quality_criteria.update(criteria)

    async def search_travel_options(self, search_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Search for travel options using the real TravelMCPManager.
        This method replaces the previous mock/orchestrator logic.
        """
        search_start_time = time.time()
        
        if not self.mcp_connected:
            raise ConnectionError("MCP Browser is not connected. Please connect before searching.")

        try:
            search_params['quality_criteria'] = self.quality_criteria
            
            # Use the real MCP Manager to perform the search
            search_result = await self.mcp_manager.search_travel_options(search_params)
            
            # Add metadata for the UI
            total_duration = time.time() - search_start_time
            search_result.setdefault('performance', {})['total_duration'] = total_duration
            search_result['mcp_info'] = self.get_mcp_status()
            search_result['status'] = 'completed'

            self.search_history.append(search_result)
            return search_result
            
        except Exception as e:
            error_result = {
                "status": "failed",
                "error": str(e),
                "search_params": search_params,
                "execution_time": time.time() - search_start_time
            }
            self.search_history.append(error_result)
            return error_result

    def get_search_history(self) -> List[Dict]:
        """Get search history"""
        return self.mcp_manager.get_search_history()

    def get_search_stats(self) -> Dict:
        """Get search statistics"""
        history = self.get_search_history()
        if not history:
            return {
                "total_searches": 0, "success_rate": 0.0,
                "real_time_data_percentage": 0.0, "average_search_duration": 0.0,
                "message": "No searches performed yet"
            }
        
        completed_searches = [s for s in history if s.get("status") == "completed"]
        
        success_rate = (len(completed_searches) / len(history)) * 100 if history else 0
        
        real_time_searches = [s for s in completed_searches if s.get('mcp_info', {}).get('browser_connected')]
        real_time_percentage = (len(real_time_searches) / len(completed_searches)) * 100 if completed_searches else 0
        
        durations = [s.get("performance", {}).get("total_duration", 0) for s in history]
        average_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_searches": len(history), "completed_searches": len(completed_searches),
            "success_rate": success_rate, "real_time_data_percentage": real_time_percentage,
            "average_search_duration": average_duration, "history_count": len(history)
        }

    async def search(self, destination: str, check_in: str, check_out: str, use_orchestrator: bool = True) -> Dict[str, Any]:
        """Convenience method for searching travel options (for CLI)"""
        search_params = {
            'destination': destination, 'origin': 'Seoul', 'check_in': check_in,
            'check_out': check_out, 'departure_date': check_in, 'return_date': check_out
        }
        return await self.search_travel_options(search_params)


# Convenience function for direct usage
async def search_travel(destination: str, check_in: str, check_out: str) -> Dict[str, Any]:
    """Search for travel options using TravelScoutAgent"""
    agent = TravelScoutAgent()
    return await agent.search(destination, check_in, check_out)


if __name__ == "__main__":
    # Command line usage
    DESTINATION = "Tokyo" if len(sys.argv) <= 1 else sys.argv[1]
    CHECK_IN = "2025-08-01" if len(sys.argv) <= 2 else sys.argv[2]
    CHECK_OUT = "2025-08-05" if len(sys.argv) <= 3 else sys.argv[3]
    
    print(f"🧳 Travel Scout Agent - MCP Browser Mode")
    print(f"📍 Destination: {DESTINATION}")
    print(f"📅 Check-in: {CHECK_IN}")
    print(f"📅 Check-out: {CHECK_OUT}")
    print("-" * 50)
    
    travel_agent = TravelScoutAgent()
    
    async def run_search():
        start_time = datetime.now()
        result = await travel_agent.search(DESTINATION, CHECK_IN, CHECK_OUT)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
        
        stats = travel_agent.get_search_stats()
        print(f"📊 Search Statistics: {stats}")
        
        if result.get("status") == "completed":
            print("✅ Travel search completed successfully!")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("❌ Travel search failed!")
            if "error" in result:
                print(f"Error: {result['error']}")
    
    asyncio.run(run_search())