"""
Drone Control Agent - Enterprise Edition
엔터프라이즈급 드론 제어 에이전트

Task-driven autonomous drone control with real-time reporting:
- Natural language task definition
- Automatic drone deployment and control
- Real-time progress monitoring and reporting
- Multi-drone coordination
- Emergency response capabilities

Features:
- Task-Driven Architecture: Users define tasks in natural language
- Real-time Reporting: Live progress updates and findings
- Multi-Provider Support: Simulation, DJI, Parrot, etc.
- Enterprise-grade Security and Reliability
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from srcs.enterprise_agents.models.drone_data import (
    DroneStatus, DronePosition, DroneTask, TaskResult, RealTimeReport,
    SensorReading, WeatherCondition, DroneCapability,
    TaskType, SensorType, TaskPriority
)

from srcs.enterprise_agents.providers.drone_providers import (
    DroneProvider, SimulationDroneProvider, DJIDroneProvider
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DroneControlAgent:
    """Enterprise-grade task-driven drone control agent with real-time reporting"""
    
    def __init__(self):
        self.agent_name = "Drone Control Agent"
        self.version = "1.0.0"
        
        # Initialize drone providers
        self.providers: Dict[str, DroneProvider] = {
            "simulation": SimulationDroneProvider(),
            "dji": DJIDroneProvider()
        }
        
        # Active tasks and drones
        self.active_tasks: Dict[str, DroneTask] = {}
        self.connected_drones: Dict[str, str] = {}  # drone_id -> provider_name
        self.task_reports: Dict[str, List[RealTimeReport]] = {}
        
        # Task parser for natural language
        self.task_templates = {
            "작물": TaskType.CROP_INSPECTION,
            "농장": TaskType.CROP_INSPECTION,
            "감시": TaskType.SURVEILLANCE,
            "보안": TaskType.SURVEILLANCE,
            "배송": TaskType.DELIVERY,
            "지도": TaskType.MAPPING,
            "매핑": TaskType.MAPPING,
            "수색": TaskType.SEARCH_RESCUE,
            "구조": TaskType.SEARCH_RESCUE,
            "촬영": TaskType.PHOTOGRAPHY,
            "사진": TaskType.PHOTOGRAPHY,
            "점검": TaskType.INSPECTION
        }
        
        logger.info(f"✅ {self.agent_name} v{self.version} initialized")
        logger.info(f"🚁 Available providers: {list(self.providers.keys())}")
    
    async def execute_natural_language_task(self, task_description: str, user_requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Main entry point: Execute drone task from natural language description
        
        Example:
        - "농장 A구역 작물 상태 점검해줘"
        - "회사 주변 보안 감시 시작해"
        - "신축 건물 3D 매핑 작업 진행"
        """
        logger.info(f"🎯 Task Request: {task_description}")
        
        try:
            # 1. Parse natural language to structured task
            task = await self._parse_natural_language_task(task_description, user_requirements or {})
            
            # 2. Find suitable drone and provider
            drone_id, provider = await self._find_suitable_drone(task)
            
            # 3. Connect to drone if not already connected
            if drone_id not in self.connected_drones:
                success = await provider.connect_drone(drone_id)
                if not success:
                    return f"❌ Failed to connect to drone {drone_id}"
                # Store provider key to match with self.providers
                for key, prov in self.providers.items():
                    if prov == provider:
                        self.connected_drones[drone_id] = key
                        break
            
            # 4. Execute task
            success = await provider.execute_task(drone_id, task)
            if not success:
                return f"❌ Failed to start task execution"
            
            # 5. Store active task
            self.active_tasks[task.task_id] = task
            self.task_reports[task.task_id] = []
            
            # 6. Start real-time monitoring
            asyncio.create_task(self._monitor_task_progress(task.task_id, provider))
            
            return f"✅ Task '{task.title}' started successfully! Task ID: {task.task_id}"
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            return f"❌ Task execution failed: {str(e)}"
    
    async def _parse_natural_language_task(self, description: str, requirements: Dict[str, Any]) -> DroneTask:
        """Parse natural language description into structured drone task"""
        
        # Determine task type from keywords
        task_type = TaskType.SURVEILLANCE  # Default
        for keyword, ttype in self.task_templates.items():
            if keyword in description:
                task_type = ttype
                break
        
        # Extract area information (simplified - in production would use NLP)
        area_name = "Unknown Area"
        if "A구역" in description:
            area_name = "농장 A구역"
        elif "주변" in description:
            area_name = "회사 주변"
        elif "건물" in description:
            area_name = "신축 건물"
        
        # Generate target area (simplified polygon around Seoul)
        base_lat, base_lon = 37.5665, 126.9780
        target_area = [
            DronePosition(latitude=base_lat + 0.001, longitude=base_lon + 0.001, altitude=100),
            DronePosition(latitude=base_lat + 0.001, longitude=base_lon - 0.001, altitude=100),
            DronePosition(latitude=base_lat - 0.001, longitude=base_lon - 0.001, altitude=100),
            DronePosition(latitude=base_lat - 0.001, longitude=base_lon + 0.001, altitude=100)
        ]
        
        # Determine priority
        priority = TaskPriority.MEDIUM
        if "긴급" in description or "urgent" in description.lower():
            priority = TaskPriority.URGENT
        elif "중요" in description or "important" in description.lower():
            priority = TaskPriority.HIGH
        
        # Required sensors based on task type
        required_sensors = [SensorType.CAMERA, SensorType.GPS]
        if task_type == TaskType.CROP_INSPECTION:
            required_sensors.extend([SensorType.MULTISPECTRAL, SensorType.THERMAL])
        elif task_type == TaskType.SURVEILLANCE:
            required_sensors.extend([SensorType.THERMAL])
        elif task_type == TaskType.MAPPING:
            required_sensors.extend([SensorType.LIDAR])
        
        # Estimate duration based on task type
        duration_map = {
            TaskType.CROP_INSPECTION: 15.0,
            TaskType.SURVEILLANCE: 30.0,
            TaskType.DELIVERY: 10.0,
            TaskType.MAPPING: 20.0,
            TaskType.SEARCH_RESCUE: 45.0,
            TaskType.PHOTOGRAPHY: 8.0,
            TaskType.INSPECTION: 12.0
        }
        
        return DroneTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            priority=priority,
            title=f"{task_type.value.replace('_', ' ').title()} - {area_name}",
            description=description,
            target_area=target_area,
            max_altitude=requirements.get('max_altitude', 200.0),
            required_sensors=required_sensors,
            estimated_duration=duration_map.get(task_type, 15.0),
            weather_constraints=requirements.get('weather_constraints', {}),
            scheduled_start=datetime.now(),
            deadline=requirements.get('deadline')
        )
    
    async def _find_suitable_drone(self, task: DroneTask) -> tuple[str, DroneProvider]:
        """Find the most suitable drone and provider for the task"""
        
        # For now, prefer simulation provider for demo
        # In production, would check drone availability, capabilities, location, etc.
        provider = self.providers["simulation"]
        drone_id = f"drone_{task.task_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Check if provider supports required sensors
        capabilities = await provider.get_capabilities(drone_id)
        required_sensors = set(task.required_sensors)
        available_sensors = set(capabilities.available_sensors)
        
        if not required_sensors.issubset(available_sensors):
            missing_sensors = required_sensors - available_sensors
            logger.warning(f"⚠️ Missing sensors: {missing_sensors}, proceeding anyway...")
        
        logger.info(f"🎯 Selected drone {drone_id} with provider {provider.name}")
        return drone_id, provider
    
    async def _monitor_task_progress(self, task_id: str, provider: DroneProvider):
        """Monitor task progress and generate real-time reports"""
        logger.info(f"📊 Starting real-time monitoring for task {task_id}")
        
        while task_id in self.active_tasks:
            try:
                # Get real-time report
                report = await provider.get_real_time_report(task_id)
                
                # Store report
                if task_id not in self.task_reports:
                    self.task_reports[task_id] = []
                self.task_reports[task_id].append(report)
                
                # Log progress
                logger.info(f"📍 Task {task_id}: {report.progress_percentage:.1f}% - {report.current_action}")
                
                # Check if task completed
                if report.progress_percentage >= 100:
                    logger.info(f"✅ Task {task_id} completed successfully!")
                    break
                
                # Wait before next update
                await asyncio.sleep(3)  # Update every 3 seconds
                
            except Exception as e:
                logger.error(f"❌ Monitoring error for task {task_id}: {e}")
                break
        
        # Task finished, remove from active tasks
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a specific task"""
        if task_id not in self.task_reports:
            return {"error": f"Task {task_id} not found"}
        
        reports = self.task_reports[task_id]
        if not reports:
            return {"error": f"No reports available for task {task_id}"}
        
        latest_report = reports[-1]
        task = self.active_tasks.get(task_id)
        
        return {
            "task_id": task_id,
            "task_info": {
                "title": task.title if task else "Unknown",
                "type": task.task_type.value if task else "Unknown",
                "priority": task.priority.value if task else "Unknown"
            },
            "current_status": {
                "progress": latest_report.progress_percentage,
                "current_action": latest_report.current_action,
                "position": {
                    "latitude": latest_report.current_position.latitude,
                    "longitude": latest_report.current_position.longitude,
                    "altitude": latest_report.current_position.altitude
                },
                "battery": latest_report.battery_remaining,
                "estimated_completion": latest_report.estimated_completion.isoformat() if latest_report.estimated_completion else None
            },
            "findings": latest_report.recent_findings,
            "alerts": latest_report.alerts,
            "data_collected": {
                "sensor_readings": latest_report.data_collected,
                "images_captured": latest_report.images_taken
            },
            "total_reports": len(reports)
        }
    
    async def get_all_active_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all currently active tasks"""
        active_statuses = []
        
        for task_id in self.active_tasks.keys():
            status = await self.get_task_status(task_id)
            if "error" not in status:
                active_statuses.append(status)
        
        return active_statuses
    
    async def emergency_stop_task(self, task_id: str) -> str:
        """Emergency stop for a specific task"""
        if task_id not in self.active_tasks:
            return f"❌ Task {task_id} not found or already completed"
        
        task = self.active_tasks[task_id]
        
        # Find the drone executing this task
        drone_id = None
        provider_name = None
        
        for did, pname in self.connected_drones.items():
            if task_id in self.task_reports:
                reports = self.task_reports[task_id]
                if reports and reports[-1].drone_id == did:
                    drone_id = did
                    provider_name = pname
                    break
        
        if not drone_id or not provider_name:
            return f"❌ Could not find drone for task {task_id}"
        
        provider = self.providers[provider_name]
        success = await provider.emergency_stop(drone_id)
        
        if success:
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            logger.warning(f"🚨 Emergency stop completed for task {task_id}")
            return f"✅ Emergency stop completed for task {task_id}"
        else:
            return f"❌ Emergency stop failed for task {task_id}"
    
    async def get_drone_fleet_status(self) -> Dict[str, Any]:
        """Get status of entire drone fleet"""
        fleet_status = {
            "total_drones": len(self.connected_drones),
            "active_tasks": len(self.active_tasks),
            "providers": list(self.providers.keys()),
            "drones": {}
        }
        
        for drone_id, provider_name in self.connected_drones.items():
            provider = self.providers[provider_name]
            try:
                status = await provider.get_drone_status(drone_id)
                capabilities = await provider.get_capabilities(drone_id)
                
                fleet_status["drones"][drone_id] = {
                    "provider": provider_name,
                    "status": status.status,
                    "battery": status.battery_level,
                    "position": {
                        "lat": status.position.latitude,
                        "lon": status.position.longitude,
                        "alt": status.position.altitude
                    },
                    "capabilities": {
                        "model": capabilities.drone_model,
                        "max_flight_time": capabilities.max_flight_time,
                        "sensors": [s.value for s in capabilities.available_sensors]
                    }
                }
            except Exception as e:
                fleet_status["drones"][drone_id] = {"error": str(e)}
        
        return fleet_status


async def main():
    """Demo the drone control agent"""
    agent = DroneControlAgent()
    
    print("🚁 Drone Control Agent Demo")
    print("=" * 50)
    
    # Example 1: Crop inspection task
    print("\n📋 Example 1: 농장 작물 점검")
    task_result = await agent.execute_natural_language_task(
        "농장 A구역 작물 상태 점검해줘",
        {"max_altitude": 150}
    )
    print(f"Result: {task_result}")
    
    # Wait a bit and check status
    await asyncio.sleep(5)
    
    # Get all active tasks
    active_tasks = await agent.get_all_active_tasks()
    if active_tasks:
        task_id = active_tasks[0]["task_id"]
        print(f"\n📊 Task Status Update:")
        status = await agent.get_task_status(task_id)
        print(f"Progress: {status['current_status']['progress']:.1f}%")
        print(f"Action: {status['current_status']['current_action']}")
        print(f"Findings: {status['findings']}")
        
        # Wait more to see progress
        print("\n⏳ Waiting for more progress...")
        await asyncio.sleep(10)
        
        # Updated status
        status = await agent.get_task_status(task_id)
        print(f"\n📊 Updated Status:")
        print(f"Progress: {status['current_status']['progress']:.1f}%")
        print(f"Action: {status['current_status']['current_action']}")
        print(f"Findings: {status['findings']}")
        if status['alerts']:
            print(f"Alerts: {status['alerts']}")
    
    # Example 2: Surveillance task
    print("\n📋 Example 2: 보안 감시")
    task_result2 = await agent.execute_natural_language_task(
        "회사 주변 보안 감시 시작해",
        {"priority": "high"}
    )
    print(f"Result: {task_result2}")
    
    # Fleet status
    await asyncio.sleep(3)
    fleet_status = await agent.get_drone_fleet_status()
    print(f"\n🚁 Fleet Status:")
    print(f"Total Drones: {fleet_status['total_drones']}")
    print(f"Active Tasks: {fleet_status['active_tasks']}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(main()) 