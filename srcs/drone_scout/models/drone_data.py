"""
Drone Scout Data Models

Pydantic-based data models for enterprise drone operations with comprehensive validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from .task_types import (
    TaskType, TaskPriority, SensorType, DroneOperationalStatus,
    FlightMode, WeatherCondition
)


class DronePosition(BaseModel):
    """3D position data for drone with enhanced validation"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    altitude: float = Field(..., ge=0, le=10000, description="Altitude in meters (max 10km)")
    heading: Optional[float] = Field(None, ge=0, le=360, description="Heading in degrees")
    speed: Optional[float] = Field(None, ge=0, description="Speed in m/s")
    timestamp: datetime = Field(default_factory=datetime.now, description="Position timestamp")

    @field_validator('altitude')
    @classmethod
    def validate_altitude(cls, v):
        if v > 150:  # Standard legal drone limit in most countries
            print(f"Warning: Altitude {v}m exceeds typical legal limits (150m)")
        return v


class DroneStatus(BaseModel):
    """Real-time drone status information"""
    drone_id: str = Field(..., description="Unique drone identifier")
    status: DroneOperationalStatus = Field(..., description="Current operational status")
    flight_mode: FlightMode = Field(FlightMode.AUTO, description="Current flight mode")
    position: DronePosition = Field(..., description="Current 3D position")
    battery_level: float = Field(..., ge=0, le=100, description="Battery percentage")
    signal_strength: float = Field(..., ge=0, le=100, description="Communication signal strength")
    flight_time: Optional[float] = Field(None, ge=0, description="Current flight time in minutes")
    max_flight_time: Optional[float] = Field(None, ge=0, description="Maximum flight time in minutes")
    last_update: datetime = Field(default_factory=datetime.now, description="Last status update")
    errors: List[str] = Field(default_factory=list, description="Current error messages")
    warnings: List[str] = Field(default_factory=list, description="Current warnings")


class SensorReading(BaseModel):
    """Individual sensor data reading with metadata"""
    sensor_type: SensorType = Field(..., description="Type of sensor")
    value: Union[float, int, str, Dict[str, Any]] = Field(..., description="Sensor reading value")
    unit: str = Field(..., description="Unit of measurement")
    quality: float = Field(..., ge=0, le=1, description="Data quality score (0-1)")
    position: DronePosition = Field(..., description="Position where reading was taken")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional sensor metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Reading timestamp")


class WeatherData(BaseModel):
    """Weather information affecting drone operations"""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    wind_direction: float = Field(..., ge=0, le=360, description="Wind direction in degrees")
    visibility: float = Field(..., ge=0, description="Visibility in kilometers")
    precipitation: bool = Field(False, description="Whether it's raining/snowing")
    pressure: Optional[float] = Field(None, description="Atmospheric pressure in hPa")
    condition: WeatherCondition = Field(..., description="Overall weather condition")
    flight_safe: bool = Field(True, description="Whether conditions are safe for flight")
    timestamp: datetime = Field(default_factory=datetime.now, description="Weather reading time")

    @field_validator('flight_safe')
    @classmethod
    def check_flight_safety(cls, v, info):
        """Automatically determine flight safety based on weather conditions"""
        if info.data.get('wind_speed', 0) > 15:  # 15 m/s wind limit
            return False
        if info.data.get('visibility', 10) < 1:  # 1km visibility limit
            return False
        if info.data.get('precipitation', False):
            return False
        return v


class DroneTask(BaseModel):
    """Comprehensive drone mission/task definition"""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    title: str = Field(..., description="Human-readable task title")
    description: str = Field(..., description="Detailed task description")

    # Geographic area
    target_area: List[DronePosition] = Field(..., min_items=3, description="Polygon defining target area")
    max_altitude: float = Field(150.0, ge=0, le=400, description="Maximum allowed altitude")
    min_altitude: float = Field(10.0, ge=0, description="Minimum altitude for task")

    # Task requirements
    required_sensors: List[SensorType] = Field(default_factory=list, description="Required sensors")
    estimated_duration: float = Field(..., gt=0, description="Estimated duration in minutes")
    max_duration: Optional[float] = Field(None, description="Maximum allowed duration")

    # Environmental constraints
    weather_constraints: Dict[str, Any] = Field(default_factory=dict, description="Weather limitations")
    no_fly_zones: List[List[DronePosition]] = Field(default_factory=list, description="Prohibited areas")

    # Scheduling
    scheduled_start: Optional[datetime] = Field(None, description="Scheduled start time")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    repeat_schedule: Optional[str] = Field(None, description="Cron-like repeat schedule")

    # Progress tracking
    status: str = Field("pending", description="Current task status")
    progress: float = Field(0, ge=0, le=100, description="Task completion percentage")
    assigned_drone_id: Optional[str] = Field(None, description="Assigned drone ID")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation time")
    started_at: Optional[datetime] = Field(None, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")

    # Additional parameters
    custom_parameters: Dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")

    @field_validator('target_area')
    @classmethod
    def validate_target_area(cls, v):
        if len(v) < 3:
            raise ValueError("Target area must have at least 3 points to form a polygon")
        return v

    @field_validator('deadline')
    @classmethod
    def validate_deadline(cls, v, info):
        if v and info.data.get('scheduled_start'):
            if v <= info.data['scheduled_start']:
                raise ValueError("Deadline must be after scheduled start time")
        return v


class TaskResult(BaseModel):
    """Comprehensive results from completed drone task"""
    task_id: str = Field(..., description="Associated task identifier")
    drone_id: str = Field(..., description="Executing drone identifier")
    completion_status: str = Field(..., description="Task completion status")
    completion_time: datetime = Field(default_factory=datetime.now, description="Task completion time")

    # Execution summary
    actual_duration: float = Field(..., description="Actual execution time in minutes")
    area_covered: float = Field(0, description="Area covered in square meters")
    distance_traveled: float = Field(0, description="Total distance traveled in meters")

    # Data collected
    sensor_data: List[SensorReading] = Field(default_factory=list, description="Collected sensor readings")
    images_captured: int = Field(0, description="Number of images captured")
    videos_recorded: int = Field(0, description="Number of videos recorded")
    data_points_collected: int = Field(0, description="Total data points collected")

    # Analysis results
    findings: List[str] = Field(default_factory=list, description="Key findings from analysis")
    anomalies_detected: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    alerts_generated: List[str] = Field(default_factory=list, description="Alerts generated during task")

    # Quality metrics
    data_quality_score: float = Field(0, ge=0, le=1, description="Overall data quality")
    coverage_percentage: float = Field(0, ge=0, le=100, description="Area coverage percentage")
    mission_success_rate: float = Field(0, ge=0, le=1, description="Mission success rate")

    # Files and reports
    report_files: List[str] = Field(default_factory=list, description="Generated report file paths")
    raw_data_files: List[str] = Field(default_factory=list, description="Raw data file paths")
    processed_data_files: List[str] = Field(default_factory=list, description="Processed data file paths")

    # Performance metrics
    battery_consumed: float = Field(0, ge=0, le=100, description="Battery percentage consumed")
    fuel_consumed: Optional[float] = Field(None, description="Fuel consumed (for gas drones)")
    weather_conditions: Optional[WeatherData] = Field(None, description="Weather during execution")


class RealTimeReport(BaseModel):
    """Real-time progress report during task execution"""
    task_id: str = Field(..., description="Associated task identifier")
    drone_id: str = Field(..., description="Executing drone identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Report timestamp")

    # Current status
    current_action: str = Field(..., description="Current action being performed")
    progress_percentage: float = Field(..., ge=0, le=100, description="Overall progress")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    time_remaining: Optional[float] = Field(None, description="Estimated time remaining in minutes")

    # Live data
    current_position: DronePosition = Field(..., description="Current drone position")
    current_status: DroneStatus = Field(..., description="Current drone status")
    recent_findings: List[str] = Field(default_factory=list, description="Recent discoveries")
    alerts: List[str] = Field(default_factory=list, description="Current alerts or warnings")

    # Performance metrics
    data_collected: int = Field(0, description="Number of data points collected so far")
    images_taken: int = Field(0, description="Number of images captured so far")
    area_covered_so_far: float = Field(0, description="Area covered so far in square meters")
    battery_remaining: float = Field(..., ge=0, le=100, description="Remaining battery percentage")

    # Next actions
    next_waypoint: Optional[DronePosition] = Field(None, description="Next planned position")
    estimated_eta: Optional[float] = Field(None, description="ETA to next waypoint in minutes")
    upcoming_actions: List[str] = Field(default_factory=list, description="Planned upcoming actions")


class DroneCapability(BaseModel):
    """Comprehensive drone capabilities and specifications"""
    drone_model: str = Field(..., description="Drone model name")
    manufacturer: str = Field(..., description="Drone manufacturer")

    # Flight capabilities
    max_flight_time: float = Field(..., description="Maximum flight time in minutes")
    max_range: float = Field(..., description="Maximum range in meters")
    max_altitude: float = Field(..., description="Maximum altitude in meters")
    max_speed: float = Field(..., description="Maximum speed in m/s")
    cruise_speed: float = Field(..., description="Cruise speed in m/s")

    # Physical specifications
    weight: float = Field(..., description="Drone weight in kg")
    payload_capacity: float = Field(..., description="Maximum payload in kg")
    dimensions: Dict[str, float] = Field(..., description="Drone dimensions (length, width, height)")

    # Sensors and equipment
    available_sensors: List[SensorType] = Field(default_factory=list, description="Available sensors")
    camera_specs: Optional[Dict[str, Any]] = Field(None, description="Camera specifications")

    # Environmental capabilities
    weather_resistance: Dict[str, Any] = Field(default_factory=dict, description="Weather resistance specs")
    operating_temperature: Dict[str, float] = Field(..., description="Operating temperature range")
    wind_resistance: float = Field(..., description="Maximum wind speed resistance in m/s")

    # Advanced features
    autonomous_features: List[str] = Field(default_factory=list, description="Autonomous capabilities")
    safety_features: List[str] = Field(default_factory=list, description="Safety features")

    # Connectivity
    communication_protocols: List[str] = Field(default_factory=list, description="Supported protocols")
    control_range: float = Field(..., description="Maximum control range in meters")

    class Config:
        use_enum_values = True


# Utility Models

class FlightPlan(BaseModel):
    """Detailed flight plan for drone mission"""
    waypoints: List[DronePosition] = Field(..., description="Ordered list of waypoints")
    total_distance: float = Field(..., description="Total planned distance in meters")
    estimated_duration: float = Field(..., description="Estimated flight duration in minutes")
    takeoff_point: DronePosition = Field(..., description="Takeoff location")
    landing_point: DronePosition = Field(..., description="Landing location")
    emergency_landing_points: List[DronePosition] = Field(default_factory=list, description="Emergency landing options")
    flight_restrictions: List[str] = Field(default_factory=list, description="Flight restrictions to observe")


class DroneFleet(BaseModel):
    """Fleet management data structure"""
    fleet_id: str = Field(..., description="Fleet identifier")
    drones: List[str] = Field(..., description="List of drone IDs in fleet")
    active_tasks: List[str] = Field(default_factory=list, description="Currently active task IDs")
    total_drones: int = Field(..., description="Total number of drones")
    operational_drones: int = Field(..., description="Number of operational drones")
    fleet_status: str = Field("operational", description="Overall fleet status")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last fleet update")


from enum import Enum


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DroneAlert(BaseModel):
    """Alert/notification system for drone operations"""
    alert_id: str = Field(..., description="Unique alert identifier")
    level: AlertLevel = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Detailed alert message")
    drone_id: Optional[str] = Field(None, description="Associated drone ID")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert timestamp")
    acknowledged: bool = Field(False, description="Whether alert has been acknowledged")
    resolved: bool = Field(False, description="Whether alert has been resolved")
    auto_resolve: bool = Field(False, description="Whether alert can auto-resolve")
    actions_required: List[str] = Field(default_factory=list, description="Required actions")
