#!/usr/bin/env python3
"""
Enhanced Drone Scout - Enterprise Drone Control Agent with MCP Integration

A comprehensive MCP-integrated system for autonomous drone fleet management
with enhanced real-time monitoring, data processing, and safety systems.

Features:
- 🚁 Multi-drone fleet coordination with MCP integration
- 🎯 Natural language task definition (Korean/English)
- 📊 Real-time progress monitoring via MCP servers
- 🛡️ Advanced safety systems with MCP validation
- 🔌 Multi-provider hardware support
- 🌐 Web-based research and data collection
- 📡 Real-time weather and environmental data
- 🔍 Enhanced search and analysis capabilities
"""

import asyncio
import os
import json
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import re
import uuid
import httpx

# MCP imports for enhanced functionality
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP packages not available. Running in limited mode.")

# Import our models
from models.drone_data import (
    DronePosition, DroneStatus, DroneTask, TaskResult, RealTimeReport,
    WeatherData, DroneCapability, DroneFleet, DroneAlert, FlightPlan
)
from models.task_types import (
    TaskType, TaskPriority, SensorType, DroneOperationalStatus,
    KOREAN_TASK_KEYWORDS, TASK_SENSOR_REQUIREMENTS, TASK_DURATION_ESTIMATES
)

# Configuration
OUTPUT_DIR = "drone_scout_reports"
DEFAULT_FLEET_SIZE = 5
MISSION_CONTROL_CENTER = "Seoul Drone Operations Center"

class EnhancedDroneControlAgent:
    """
    Enhanced Drone Control Agent with comprehensive MCP integration.
    """
    
    def __init__(self, 
                 output_dir: str = "drone_scout_reports",
                 enable_mcp: bool = True,
                 mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the Enhanced Drone Control Agent
        
        Args:
            output_dir: Directory to save drone reports and data
            enable_mcp: Whether to enable MCP functionality
            mcp_servers: Dictionary of MCP server configurations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_mcp = enable_mcp and MCP_AVAILABLE
        self.mcp_servers = mcp_servers or {}
        
        # MCP client sessions
        self.filesystem_session: Optional[ClientSession] = None
        self.weather_session: Optional[ClientSession] = None
        self.search_session: Optional[ClientSession] = None
        self.browser_session: Optional[ClientSession] = None
        self.gis_session: Optional[ClientSession] = None
        
        # Drone fleet management
        self.drone_fleet: Dict[str, DroneStatus] = {}
        self.active_missions: Dict[str, Dict[str, Any]] = {}
        self.mission_history: List[Dict[str, Any]] = []
        self.flight_plans: Dict[str, FlightPlan] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize MCP connections if enabled
        if self.enable_mcp:
            asyncio.create_task(self._initialize_mcp_connections())
    
    async def _initialize_mcp_connections(self):
        """Initialize connections to MCP servers"""
        try:
            # Initialize filesystem MCP server
            if "filesystem" in self.mcp_servers:
                await self._connect_filesystem_server()
            
            # Initialize weather MCP server
            if "weather" in self.mcp_servers:
                await self._connect_weather_server()
                
            # Initialize search MCP server
            if "search" in self.mcp_servers:
                await self._connect_search_server()
                
            # Initialize browser MCP server
            if "browser" in self.mcp_servers:
                await self._connect_browser_server()
                
            # Initialize GIS MCP server
            if "gis" in self.mcp_servers:
                await self._connect_gis_server()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP connections: {e}")
    
    async def _connect_filesystem_server(self):
        """Connect to filesystem MCP server"""
        try:
            server_config = self.mcp_servers["filesystem"]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )
            
            context = stdio_client(server_params)
            receive_stream, write_stream = await context.__aenter__()
            self.filesystem_session = ClientSession(receive_stream, write_stream)
            
            self.logger.info("✅ Connected to filesystem MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to filesystem server: {e}")
    
    async def _connect_weather_server(self):
        """Connect to weather MCP server"""
        try:
            server_config = self.mcp_servers["weather"]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )
            
            context = stdio_client(server_params)
            receive_stream, write_stream = await context.__aenter__()
            self.weather_session = ClientSession(receive_stream, write_stream)
            
            self.logger.info("✅ Connected to weather MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to weather server: {e}")
    
    async def _connect_search_server(self):
        """Connect to search MCP server"""
        try:
            server_config = self.mcp_servers["search"]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )
            
            context = stdio_client(server_params)
            receive_stream, write_stream = await context.__aenter__()
            self.search_session = ClientSession(receive_stream, write_stream)
            
            self.logger.info("✅ Connected to search MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to search server: {e}")
    
    async def _connect_browser_server(self):
        """Connect to browser MCP server"""
        try:
            server_config = self.mcp_servers["browser"]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )
            
            context = stdio_client(server_params)
            receive_stream, write_stream = await context.__aenter__()
            self.browser_session = ClientSession(receive_stream, write_stream)
            
            self.logger.info("✅ Connected to browser MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to browser server: {e}")
    
    async def _connect_gis_server(self):
        """Connect to GIS MCP server"""
        try:
            server_config = self.mcp_servers["gis"]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )
            
            context = stdio_client(server_params)
            receive_stream, write_stream = await context.__aenter__()
            self.gis_session = ClientSession(receive_stream, write_stream)
            
            self.logger.info("✅ Connected to GIS MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to GIS server: {e}")
    
    async def _call_mcp_tool(self, session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result"""
        try:
            # List available tools first
            tools_response = await session.call_tool("list_tools", {})
            tools = tools_response.content[0].text if tools_response.content else "[]"
            
            # Call the specific tool
            result = await session.call_tool(tool_name, arguments)
            return json.loads(result.content[0].text) if result.content else {}
        except Exception as e:
            self.logger.error(f"Failed to call MCP tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def research_mission_context(self, mission: str) -> Dict[str, Any]:
        """Research mission context using MCP search and browser tools"""
        research_data = {
            "mission": mission,
            "search_results": [],
            "weather_data": [],
            "gis_data": [],
            "web_data": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Use search MCP server if available
            if self.search_session:
                search_result = await self._call_mcp_tool(
                    self.search_session, 
                    "search_web", 
                    {"query": f"{mission} drone operations safety regulations", "count": 5}
                )
                if "error" not in search_result:
                    research_data["search_results"] = search_result.get("results", [])
            
            # Use weather MCP server if available
            if self.weather_session:
                # Get current weather for Seoul area
                weather_result = await self._call_mcp_tool(
                    self.weather_session,
                    "get_current_weather",
                    {"location": "Seoul, South Korea"}
                )
                if "error" not in weather_result:
                    research_data["weather_data"].append(weather_result)
            
            # Use GIS MCP server if available
            if self.gis_session:
                # Get geographic information for the mission area
                gis_result = await self._call_mcp_tool(
                    self.gis_session,
                    "get_area_info",
                    {"query": mission}
                )
                if "error" not in gis_result:
                    research_data["gis_data"].append(gis_result)
            
            # Use browser MCP server if available
            if self.browser_session:
                # Navigate to relevant research sites
                browser_result = await self._call_mcp_tool(
                    self.browser_session,
                    "navigate",
                    {"url": "https://www.google.com/search?q=" + mission.replace(" ", "+")}
                )
                if "error" not in browser_result:
                    research_data["web_data"].append(browser_result)
                    
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            research_data["error"] = str(e)
        
        return research_data
    
    async def get_real_time_weather(self, location: str = "Seoul") -> Optional[WeatherData]:
        """Get real-time weather data using MCP weather server"""
        if not self.weather_session or not self.enable_mcp:
            return None
        
        try:
            weather_result = await self._call_mcp_tool(
                self.weather_session,
                "get_current_weather",
                {"location": location}
            )
            
            if "error" not in weather_result:
                # Convert MCP response to WeatherData model
                weather_data = WeatherData(
                    temperature=weather_result.get("temperature", 20.0),
                    humidity=weather_result.get("humidity", 50.0),
                    wind_speed=weather_result.get("wind_speed", 5.0),
                    wind_direction=weather_result.get("wind_direction", 0.0),
                    visibility=weather_result.get("visibility", 10.0),
                    precipitation=weather_result.get("precipitation", False),
                    pressure=weather_result.get("pressure", 1013.25),
                    condition=weather_result.get("condition", "good"),
                    flight_safe=weather_result.get("flight_safe", True)
                )
                return weather_data
                
        except Exception as e:
            self.logger.error(f"Failed to get weather data: {e}")
        
        return None
    
    async def analyze_flight_safety(self, mission: str, weather_data: Optional[WeatherData] = None) -> Dict[str, Any]:
        """Analyze flight safety using MCP tools and data"""
        safety_analysis = {
            "mission": mission,
            "weather_safe": True,
            "regulatory_compliant": True,
            "risk_factors": [],
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Weather safety check
            if weather_data:
                if weather_data.wind_speed > 15:
                    safety_analysis["weather_safe"] = False
                    safety_analysis["risk_factors"].append("High wind speed")
                    safety_analysis["recommendations"].append("Delay mission until wind conditions improve")
                
                if weather_data.visibility < 1:
                    safety_analysis["weather_safe"] = False
                    safety_analysis["risk_factors"].append("Low visibility")
                    safety_analysis["recommendations"].append("Wait for better visibility conditions")
                
                if weather_data.precipitation:
                    safety_analysis["weather_safe"] = False
                    safety_analysis["risk_factors"].append("Precipitation")
                    safety_analysis["recommendations"].append("Avoid flying in precipitation")
            
            # Regulatory compliance check using search MCP
            if self.search_session:
                compliance_result = await self._call_mcp_tool(
                    self.search_session,
                    "search_web",
                    {"query": "drone regulations South Korea flight restrictions", "count": 3}
                )
                
                if "error" not in compliance_result:
                    # Analyze search results for compliance issues
                    results = compliance_result.get("results", [])
                    for result in results:
                        if "restricted" in result.get("title", "").lower() or "no-fly" in result.get("title", "").lower():
                            safety_analysis["regulatory_compliant"] = False
                            safety_analysis["risk_factors"].append("Potential regulatory restrictions")
                            safety_analysis["recommendations"].append("Verify flight permissions for target area")
            
            # GIS-based safety check
            if self.gis_session:
                gis_result = await self._call_mcp_tool(
                    self.gis_session,
                    "check_flight_restrictions",
                    {"area": mission}
                )
                
                if "error" not in gis_result:
                    restrictions = gis_result.get("restrictions", [])
                    if restrictions:
                        safety_analysis["regulatory_compliant"] = False
                        safety_analysis["risk_factors"].extend(restrictions)
                        safety_analysis["recommendations"].append("Review GIS data for flight restrictions")
                        
        except Exception as e:
            self.logger.error(f"Safety analysis failed: {e}")
            safety_analysis["error"] = str(e)
        
        return safety_analysis
    
    async def create_enhanced_flight_plan(self, mission: str, target_area: List[DronePosition]) -> FlightPlan:
        """Create an enhanced flight plan using MCP tools"""
        try:
            # Get weather data for planning
            weather_data = await self.get_real_time_weather()
            
            # Analyze safety
            safety_analysis = await self.analyze_flight_safety(mission, weather_data)
            
            # Create optimized waypoints using GIS data if available
            waypoints = target_area
            if self.gis_session:
                gis_result = await self._call_mcp_tool(
                    self.gis_session,
                    "optimize_route",
                    {"waypoints": [{"lat": pos.latitude, "lon": pos.longitude, "alt": pos.altitude} for pos in target_area]}
                )
                
                if "error" not in gis_result:
                    optimized_waypoints = gis_result.get("optimized_route", [])
                    if optimized_waypoints:
                        waypoints = [
                            DronePosition(
                                latitude=wp["lat"],
                                longitude=wp["lon"],
                                altitude=wp["alt"]
                            ) for wp in optimized_waypoints
                        ]
            
            # Calculate flight metrics
            total_distance = self._calculate_total_distance(waypoints)
            estimated_duration = self._estimate_flight_duration(total_distance, weather_data)
            
            # Create flight plan
            flight_plan = FlightPlan(
                waypoints=waypoints,
                total_distance=total_distance,
                estimated_duration=estimated_duration,
                takeoff_point=waypoints[0] if waypoints else DronePosition(latitude=37.5665, longitude=126.9780, altitude=0),
                landing_point=waypoints[-1] if waypoints else DronePosition(latitude=37.5665, longitude=126.9780, altitude=0),
                emergency_landing_points=self._generate_emergency_landing_points(waypoints),
                flight_restrictions=safety_analysis.get("risk_factors", [])
            )
            
            return flight_plan
            
        except Exception as e:
            self.logger.error(f"Failed to create flight plan: {e}")
            # Return basic flight plan as fallback
            return FlightPlan(
                waypoints=target_area,
                total_distance=1000.0,
                estimated_duration=30.0,
                takeoff_point=target_area[0] if target_area else DronePosition(latitude=37.5665, longitude=126.9780, altitude=0),
                landing_point=target_area[-1] if target_area else DronePosition(latitude=37.5665, longitude=126.9780, altitude=0),
                emergency_landing_points=[],
                flight_restrictions=[]
            )
    
    def _calculate_total_distance(self, waypoints: List[DronePosition]) -> float:
        """Calculate total flight distance"""
        if len(waypoints) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            pos1 = waypoints[i]
            pos2 = waypoints[i + 1]
            
            # Simple Euclidean distance (can be enhanced with proper geodesic calculations)
            lat_diff = pos2.latitude - pos1.latitude
            lon_diff = pos2.longitude - pos1.longitude
            alt_diff = pos2.altitude - pos1.altitude
            
            distance = (lat_diff**2 + lon_diff**2 + alt_diff**2)**0.5
            total_distance += distance * 111000  # Rough conversion to meters
        
        return total_distance
    
    def _estimate_flight_duration(self, distance: float, weather_data: Optional[WeatherData]) -> float:
        """Estimate flight duration based on distance and weather"""
        base_speed = 10.0  # m/s base speed
        
        # Adjust speed based on weather conditions
        if weather_data:
            if weather_data.wind_speed > 10:
                base_speed *= 0.8  # Reduce speed in high winds
            if weather_data.visibility < 5:
                base_speed *= 0.7  # Reduce speed in low visibility
        
        duration_minutes = (distance / base_speed) / 60.0
        return max(duration_minutes, 5.0)  # Minimum 5 minutes
    
    def _generate_emergency_landing_points(self, waypoints: List[DronePosition]) -> List[DronePosition]:
        """Generate emergency landing points along the route"""
        emergency_points = []
        
        if len(waypoints) < 2:
            return emergency_points
        
        # Add emergency landing points every 500m along the route
        for i in range(0, len(waypoints) - 1, 2):
            pos1 = waypoints[i]
            pos2 = waypoints[i + 1]
            
            # Midpoint between waypoints
            mid_lat = (pos1.latitude + pos2.latitude) / 2
            mid_lon = (pos1.longitude + pos2.longitude) / 2
            mid_alt = max(0, (pos1.altitude + pos2.altitude) / 2 - 10)  # Lower altitude for emergency landing
            
            emergency_points.append(DronePosition(
                latitude=mid_lat,
                longitude=mid_lon,
                altitude=mid_alt
            ))
        
        return emergency_points
    
    async def execute_mission(self, mission: str, target_area: List[DronePosition]) -> Dict[str, Any]:
        """Execute a drone mission with comprehensive MCP integration"""
        mission_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        mission_result = {
            "mission_id": mission_id,
            "mission": mission,
            "start_time": start_time.isoformat(),
            "status": "executing",
            "progress": 0.0,
            "weather_data": None,
            "safety_analysis": None,
            "flight_plan": None,
            "execution_log": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # Research mission context
            self.logger.info(f"🔍 Researching mission context: {mission}")
            research_data = await self.research_mission_context(mission)
            mission_result["research_data"] = research_data
            
            # Get real-time weather
            self.logger.info("🌤️ Getting real-time weather data")
            weather_data = await self.get_real_time_weather()
            mission_result["weather_data"] = weather_data.dict() if weather_data else None
            
            # Analyze flight safety
            self.logger.info("🛡️ Analyzing flight safety")
            safety_analysis = await self.analyze_flight_safety(mission, weather_data)
            mission_result["safety_analysis"] = safety_analysis
            
            # Check if mission is safe to proceed
            if not safety_analysis.get("weather_safe", True) or not safety_analysis.get("regulatory_compliant", True):
                mission_result["status"] = "cancelled"
                mission_result["errors"].append("Mission cancelled due to safety concerns")
                return mission_result
            
            # Create flight plan
            self.logger.info("✈️ Creating enhanced flight plan")
            flight_plan = await self.create_enhanced_flight_plan(mission, target_area)
            mission_result["flight_plan"] = flight_plan.dict()
            
            # Simulate mission execution
            self.logger.info("🚁 Executing mission")
            await self._simulate_mission_execution(mission_result, flight_plan)
            
            # Mission completed successfully
            mission_result["status"] = "completed"
            mission_result["progress"] = 100.0
            mission_result["completion_time"] = datetime.now().isoformat()
            
            # Save mission result
            await self._save_mission_result(mission_result)
            
            self.logger.info(f"✅ Mission {mission_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Mission {mission_id} failed: {e}")
            mission_result["status"] = "failed"
            mission_result["errors"].append(str(e))
            mission_result["completion_time"] = datetime.now().isoformat()
        
        return mission_result
    
    async def _simulate_mission_execution(self, mission_result: Dict[str, Any], flight_plan: FlightPlan):
        """Simulate mission execution with progress updates"""
        waypoints = flight_plan.waypoints
        total_waypoints = len(waypoints)
        
        for i, waypoint in enumerate(waypoints):
            # Update progress
            progress = (i / total_waypoints) * 100
            mission_result["progress"] = progress
            
            # Log waypoint reached
            mission_result["execution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "action": f"Reached waypoint {i+1}/{total_waypoints}",
                "position": waypoint.dict(),
                "progress": progress
            })
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Check for weather changes every few waypoints
            if i % 3 == 0 and self.weather_session:
                weather_update = await self.get_real_time_weather()
                if weather_update and not weather_update.flight_safe:
                    mission_result["warnings"].append(f"Weather conditions deteriorated at waypoint {i+1}")
    
    async def _save_mission_result(self, mission_result: Dict[str, Any]) -> str:
        """Save mission result using MCP filesystem server or local filesystem"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mission_result_{mission_result['mission_id'][:8]}_{timestamp}.json"
        
        file_path = self.output_dir / filename
        
        # Convert datetime objects to strings for JSON serialization
        serializable_result = self._make_json_serializable(mission_result)
        
        try:
            # Try to use MCP filesystem server if available
            if self.filesystem_session and self.enable_mcp:
                result = await self._call_mcp_tool(
                    self.filesystem_session,
                    "write_file",
                    {
                        "file_path": str(file_path),
                        "content": json.dumps(serializable_result, ensure_ascii=False, indent=2),
                        "encoding": "utf-8"
                    }
                )
                if "error" not in result:
                    self.logger.info(f"✅ Mission result saved via MCP filesystem: {file_path}")
                    return str(file_path)
            
            # Fallback to local filesystem
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Mission result saved locally: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save mission result: {e}")
            raise
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):  # Other objects with dict
            return obj.__dict__
        else:
            return obj
    
    async def get_mission_status(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a mission"""
        # Check active missions
        if mission_id in self.active_missions:
            return self.active_missions[mission_id]
        
        # Check mission history
        for mission in self.mission_history:
            if mission.get("mission_id") == mission_id:
                return mission
        
        return None
    
    async def list_all_missions(self) -> List[Dict[str, Any]]:
        """List all missions (active and completed)"""
        all_missions = []
        
        # Add active missions
        for mission_id, mission_data in self.active_missions.items():
            all_missions.append({
                "mission_id": mission_id,
                "status": "active",
                **mission_data
            })
        
        # Add completed missions
        all_missions.extend(self.mission_history)
        
        return all_missions
    
    async def cleanup(self):
        """Clean up MCP connections and resources"""
        try:
            # Clean up MCP sessions safely
            sessions = [
                ("filesystem", self.filesystem_session),
                ("weather", self.weather_session),
                ("search", self.search_session),
                ("browser", self.browser_session),
                ("gis", self.gis_session)
            ]
            
            for name, session in sessions:
                if session:
                    try:
                        # Check if session has close method
                        if hasattr(session, 'close'):
                            await session.close()
                        elif hasattr(session, '__aexit__'):
                            await session.__aexit__(None, None, None)
                        self.logger.info(f"✅ {name} session cleaned up")
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup {name} session: {e}")
            
            self.logger.info("✅ MCP connections cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup MCP connections: {e}")


async def main(mission: str, result_json_path: str):
    """
    Enhanced Drone Scout - Enterprise Drone Control Agent System
    
    This function uses the enhanced MCP-integrated drone control agent.
    """
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # MCP server configurations
    mcp_servers = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "env": {"ALLOWED_PATHS": f"{OUTPUT_DIR},reports/,data/"}
        },
        "weather": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-weather"],
            "env": {}
        },
        "search": {
            "command": "npx", 
            "args": ["-y", "g-search-mcp"],
            "env": {}
        },
        "browser": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
            "env": {}
        },
        "gis": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-gis"],
            "env": {}
        }
    }
    
    # Initialize enhanced drone control agent
    agent = EnhancedDroneControlAgent(
        output_dir=OUTPUT_DIR,
        enable_mcp=True,
        mcp_servers=mcp_servers
    )
    
    try:
        print(f"🚁 Enhanced Drone Scout Agent System Initialized for mission: {mission}")
        print("🔧 MCP integration enabled - using enhanced capabilities")
        
        # Create sample target area (Seoul Forest Park)
        target_area = [
            DronePosition(latitude=37.5665, longitude=126.9780, altitude=50),  # Seoul Forest
            DronePosition(latitude=37.5675, longitude=126.9790, altitude=50),  # North
            DronePosition(latitude=37.5655, longitude=126.9790, altitude=50),  # South
            DronePosition(latitude=37.5665, longitude=126.9770, altitude=50),  # West
            DronePosition(latitude=37.5665, longitude=126.9780, altitude=50),  # Return to start
        ]
        
        # Execute mission
        print("🎯 Executing enhanced mission...")
        mission_result = await agent.execute_mission(mission, target_area)
        
        # Prepare result for UI
        flight_plan = mission_result.get("flight_plan", {})
        waypoints = flight_plan.get("waypoints", [])
        
        # Convert waypoints to trajectory format
        trajectory = []
        for wp in waypoints:
            if hasattr(wp, 'longitude') and hasattr(wp, 'latitude') and hasattr(wp, 'altitude'):
                # Pydantic model
                trajectory.append([wp.longitude, wp.latitude, wp.altitude])
            elif isinstance(wp, dict):
                # Dictionary format
                trajectory.append([
                    wp.get('longitude', 0.0),
                    wp.get('latitude', 0.0),
                    wp.get('altitude', 0.0)
                ])
            else:
                # Fallback
                trajectory.append([0.0, 0.0, 0.0])
        
        final_result = {
            "success": mission_result["status"] == "completed",
            "summary": {
                "mission": mission,
                "status": mission_result["status"],
                "output_files_location": f"{OUTPUT_DIR}/",
                "mission_id": mission_result["mission_id"],
                "execution_time": mission_result.get("completion_time", "N/A")
            },
            "trajectory": trajectory,
            "enhanced_data": {
                "weather_data": mission_result.get("weather_data"),
                "safety_analysis": mission_result.get("safety_analysis"),
                "research_data": mission_result.get("research_data"),
                "execution_log": mission_result.get("execution_log", [])
            }
        }
        
        # Make the final result JSON serializable
        def make_json_serializable(obj):
            """Convert objects to JSON serializable format"""
            if isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'dict'):  # Pydantic models
                return obj.dict()
            elif hasattr(obj, '__dict__'):  # Other objects with dict
                return obj.__dict__
            else:
                return obj
        
        serializable_final_result = make_json_serializable(final_result)
        
        # Save the processed, UI-friendly JSON
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_final_result, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Enhanced mission completed and results saved to {result_json_path}")
        
    except Exception as e:
        print(f"❌ Enhanced mission failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error result
        error_result = {
            "success": False,
            "error": str(e),
            "summary": {
                "mission": mission,
                "status": "failed",
                "error_time": datetime.now().isoformat()
            }
        }
        
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, ensure_ascii=False, indent=4)
    
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    # Example of how to run this directly for testing
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_mission = "Fly over Seoul Forest park at 50m altitude and check for unusual activity."
        test_result_path = "test_enhanced_drone_result.json"
        print(f"--- RUNNING ENHANCED TEST ---")
        print(f"Mission: {test_mission}")
        print(f"Result Path: {test_result_path}")
        asyncio.run(main(mission=test_mission, result_json_path=test_result_path))
        print(f"--- ENHANCED TEST COMPLETE ---")
    else:
        print("This script is meant to be called by the runner. To test, run with `python ... enhanced_drone_control_agent.py test`")
