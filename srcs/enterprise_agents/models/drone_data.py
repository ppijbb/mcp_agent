"""
Drone Data Models
드론 데이터 모델

Pydantic models for drone control, missions, and sensor data
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum

class DroneOperationalStatus(str, Enum):
    """Drone operational status"""
    IDLE = "idle"
    READY = "ready"
    FLYING = "flying"
    LANDING = "landing"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"

class TaskType(str, Enum):
    """Available drone task types"""
    SURVEILLANCE = "surveillance"
    CROP_INSPECTION = "crop_inspection"
    DELIVERY = "delivery"
    MAPPING = "mapping"
    SEARCH_RESCUE = "search_rescue"
    PHOTOGRAPHY = "photography"
    INSPECTION = "inspection"

class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class SensorType(str, Enum):
    """Available sensor types"""
    CAMERA = "camera"
    THERMAL = "thermal"
    LIDAR = "lidar"
    MULTISPECTRAL = "multispectral"
    GPS = "gps"
    ALTIMETER = "altimeter"
    GYROSCOPE = "gyroscope"

class DronePosition(BaseModel):
    """3D position data for drone"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    altitude: float = Field(..., ge=0, description="Altitude in meters")
    heading: Optional[float] = Field(None, ge=0, le=360, description="Heading in degrees")
    timestamp: datetime = Field(default_factory=datetime.now, description="Position timestamp")

class DroneStatus(BaseModel):
    """Real-time drone status information"""
    drone_id: str = Field(..., description="Unique drone identifier")
    status: DroneOperationalStatus = Field(..., description="Current operational status")
    position: DronePosition = Field(..., description="Current 3D position")
    battery_level: float = Field(..., ge=0, le=100, description="Battery percentage")
    signal_strength: float = Field(..., ge=0, le=100, description="Communication signal strength")
    flight_time: Optional[float] = Field(None, ge=0, description="Current flight time in minutes")
    speed: Optional[float] = Field(None, ge=0, description="Current speed in m/s")
    last_update: datetime = Field(default_factory=datetime.now, description="Last status update")

class SensorReading(BaseModel):
    """Individual sensor data reading"""
    sensor_type: SensorType = Field(..., description="Type of sensor")
    value: Any = Field(..., description="Sensor reading value")
    unit: str = Field(..., description="Unit of measurement")
    quality: float = Field(..., ge=0, le=1, description="Data quality score")
    position: DronePosition = Field(..., description="Position where reading was taken")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional sensor metadata")

class DroneTask(BaseModel):
    """Drone mission/task definition"""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    title: str = Field(..., description="Human-readable task title")
    description: str = Field(..., description="Detailed task description")
    
    # Geographic area
    target_area: List[DronePosition] = Field(..., description="Polygon defining target area")
    max_altitude: Optional[float] = Field(None, description="Maximum allowed altitude")
    
    # Task requirements
    required_sensors: List[SensorType] = Field(default_factory=list, description="Required sensors")
    estimated_duration: Optional[float] = Field(None, description="Estimated duration in minutes")
    weather_constraints: Dict[str, Any] = Field(default_factory=dict, description="Weather limitations")
    
    # Scheduling
    scheduled_start: Optional[datetime] = Field(None, description="Scheduled start time")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    
    # Status tracking
    status: str = Field("pending", description="Current task status")
    progress: float = Field(0, ge=0, le=100, description="Task completion percentage")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation time")
    
    @validator('target_area')
    def validate_target_area(cls, v):
        if len(v) < 3:
            raise ValueError('Target area must have at least 3 points to form a polygon')
        return v

class TaskResult(BaseModel):
    """Results and findings from completed drone task"""
    task_id: str = Field(..., description="Associated task identifier")
    completion_status: str = Field(..., description="Task completion status")
    completion_time: datetime = Field(default_factory=datetime.now, description="Task completion time")
    
    # Data collected
    sensor_data: List[SensorReading] = Field(default_factory=list, description="Collected sensor readings")
    images_captured: int = Field(0, description="Number of images captured")
    area_covered: float = Field(0, description="Area covered in square meters")
    
    # Analysis results
    findings: List[str] = Field(default_factory=list, description="Key findings from analysis")
    anomalies_detected: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    
    # Quality metrics
    data_quality_score: float = Field(0, ge=0, le=1, description="Overall data quality")
    coverage_percentage: float = Field(0, ge=0, le=100, description="Area coverage percentage")
    
    # Files and reports
    report_files: List[str] = Field(default_factory=list, description="Generated report file paths")
    raw_data_files: List[str] = Field(default_factory=list, description="Raw data file paths")

class RealTimeReport(BaseModel):
    """Real-time progress report during task execution"""
    task_id: str = Field(..., description="Associated task identifier")
    drone_id: str = Field(..., description="Executing drone identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Report timestamp")
    
    # Current status
    current_action: str = Field(..., description="Current action being performed")
    progress_percentage: float = Field(..., ge=0, le=100, description="Overall progress")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Live data
    current_position: DronePosition = Field(..., description="Current drone position")
    recent_findings: List[str] = Field(default_factory=list, description="Recent discoveries")
    alerts: List[str] = Field(default_factory=list, description="Current alerts or warnings")
    
    # Performance metrics
    data_collected: int = Field(0, description="Number of data points collected")
    images_taken: int = Field(0, description="Number of images captured")
    battery_remaining: float = Field(..., ge=0, le=100, description="Remaining battery percentage")
    
    # Next actions
    next_waypoint: Optional[DronePosition] = Field(None, description="Next planned position")
    estimated_eta: Optional[float] = Field(None, description="ETA to next waypoint in minutes")

class WeatherCondition(BaseModel):
    """Current weather conditions affecting drone operations"""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    wind_direction: float = Field(..., ge=0, le=360, description="Wind direction in degrees")
    visibility: float = Field(..., ge=0, description="Visibility in kilometers")
    precipitation: bool = Field(False, description="Whether it's raining/snowing")
    flight_safe: bool = Field(True, description="Whether conditions are safe for flight")
    timestamp: datetime = Field(default_factory=datetime.now, description="Weather reading time")

class DroneCapability(BaseModel):
    """Drone capabilities and specifications"""
    drone_model: str = Field(..., description="Drone model name")
    max_flight_time: float = Field(..., description="Maximum flight time in minutes")
    max_range: float = Field(..., description="Maximum range in meters")
    max_altitude: float = Field(..., description="Maximum altitude in meters")
    max_speed: float = Field(..., description="Maximum speed in m/s")
    payload_capacity: float = Field(..., description="Maximum payload in kg")
    available_sensors: List[SensorType] = Field(default_factory=list, description="Available sensors")
    weather_resistance: Dict[str, Any] = Field(default_factory=dict, description="Weather resistance specs")
    
    class Config:
        use_enum_values = True 