"""
Drone Scout Configuration

Centralized configuration management for drone operations.
All hardcoded values are moved to this configuration file.
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class DroneConfig:
    """Drone-specific configuration settings."""

    # Output and file settings
    output_dir: str = os.getenv("DRONE_OUTPUT_DIR", "drone_scout_reports")
    default_fleet_size: int = int(os.getenv("DRONE_FLEET_SIZE", "5"))
    mission_control_center: str = os.getenv("MISSION_CONTROL", "Seoul Drone Operations Center")

    # Flight parameters
    max_altitude: float = float(os.getenv("MAX_ALTITUDE", "120.0"))
    default_altitude: float = float(os.getenv("DEFAULT_ALTITUDE", "50.0"))
    min_altitude: float = float(os.getenv("MIN_ALTITUDE", "10.0"))

    # Safety settings
    emergency_landing_altitude: float = float(os.getenv("EMERGENCY_LANDING_ALTITUDE", "5.0"))
    battery_warning_threshold: float = float(os.getenv("BATTERY_WARNING_THRESHOLD", "20.0"))
    battery_critical_threshold: float = float(os.getenv("BATTERY_CRITICAL_THRESHOLD", "10.0"))

    # Weather thresholds
    max_wind_speed: float = float(os.getenv("MAX_WIND_SPEED", "15.0"))
    max_rain_intensity: float = float(os.getenv("MAX_RAIN_INTENSITY", "2.0"))
    min_visibility: float = float(os.getenv("MIN_VISIBILITY", "1000.0"))

    # Mission settings
    default_mission_duration: int = int(os.getenv("DEFAULT_MISSION_DURATION", "30"))
    max_mission_duration: int = int(os.getenv("MAX_MISSION_DURATION", "120"))
    auto_return_enabled: bool = os.getenv("AUTO_RETURN_ENABLED", "true").lower() == "true"

    # MCP server settings
    mcp_servers: List[str] = None

    def __post_init__(self):
        if self.mcp_servers is None:
            self.mcp_servers = [
                "filesystem",
                "weather",
                "g-search",
                "browser",
                "gis"
            ]


@dataclass
class DroneCapabilityConfig:
    """Drone capability configuration."""

    # Camera settings
    camera_resolution: str = os.getenv("CAMERA_RESOLUTION", "4K")
    camera_fov: float = float(os.getenv("CAMERA_FOV", "84.0"))
    night_vision: bool = os.getenv("NIGHT_VISION", "true").lower() == "true"

    # Sensor settings
    gps_precision: float = float(os.getenv("GPS_PRECISION", "1.0"))
    obstacle_detection_range: float = float(os.getenv("OBSTACLE_DETECTION_RANGE", "30.0"))
    temperature_sensor: bool = os.getenv("TEMPERATURE_SENSOR", "true").lower() == "true"
    humidity_sensor: bool = os.getenv("HUMIDITY_SENSOR", "true").lower() == "true"

    # Flight capabilities
    max_speed: float = float(os.getenv("MAX_SPEED", "20.0"))
    max_flight_time: int = int(os.getenv("MAX_FLIGHT_TIME", "30"))
    hover_precision: float = float(os.getenv("HOVER_PRECISION", "0.5"))


@dataclass
class DroneTaskConfig:
    """Task-specific configuration."""

    # Task priorities
    emergency_priority: int = int(os.getenv("EMERGENCY_PRIORITY", "1"))
    high_priority: int = int(os.getenv("HIGH_PRIORITY", "2"))
    normal_priority: int = int(os.getenv("NORMAL_PRIORITY", "3"))
    low_priority: int = int(os.getenv("LOW_PRIORITY", "4"))

    # Task timeouts
    task_timeout: int = int(os.getenv("TASK_TIMEOUT", "300"))
    emergency_timeout: int = int(os.getenv("EMERGENCY_TIMEOUT", "60"))

    # Quality thresholds
    min_quality_rating: str = os.getenv("MIN_QUALITY_RATING", "GOOD")
    auto_retry_failed_tasks: bool = os.getenv("AUTO_RETRY_FAILED_TASKS", "true").lower() == "true"
    max_retry_attempts: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))


@dataclass
class DroneSystemConfig:
    """Overall drone system configuration."""

    drone: DroneConfig = None
    capability: DroneCapabilityConfig = None
    task: DroneTaskConfig = None

    def __post_init__(self):
        if self.drone is None:
            self.drone = DroneConfig()
        if self.capability is None:
            self.capability = DroneCapabilityConfig()
        if self.task is None:
            self.task = DroneTaskConfig()


# Global configuration instance
config = DroneSystemConfig()


def get_drone_config() -> DroneConfig:
    """Get drone configuration."""
    return config.drone


def get_capability_config() -> DroneCapabilityConfig:
    """Get drone capability configuration."""
    return config.capability


def get_task_config() -> DroneTaskConfig:
    """Get task configuration."""
    return config.task


def update_config_from_env():
    """Update configuration from environment variables."""
    # Drone settings
    if os.getenv("DRONE_OUTPUT_DIR"):
        config.drone.output_dir = os.getenv("DRONE_OUTPUT_DIR")
    if os.getenv("DRONE_FLEET_SIZE"):
        config.drone.default_fleet_size = int(os.getenv("DRONE_FLEET_SIZE"))
    if os.getenv("MISSION_CONTROL"):
        config.drone.mission_control_center = os.getenv("MISSION_CONTROL")

    # Flight parameters
    if os.getenv("MAX_ALTITUDE"):
        config.drone.max_altitude = float(os.getenv("MAX_ALTITUDE"))
    if os.getenv("DEFAULT_ALTITUDE"):
        config.drone.default_altitude = float(os.getenv("DEFAULT_ALTITUDE"))
    if os.getenv("MIN_ALTITUDE"):
        config.drone.min_altitude = float(os.getenv("MIN_ALTITUDE"))

    # Safety settings
    if os.getenv("EMERGENCY_LANDING_ALTITUDE"):
        config.drone.emergency_landing_altitude = float(os.getenv("EMERGENCY_LANDING_ALTITUDE"))
    if os.getenv("BATTERY_WARNING_THRESHOLD"):
        config.drone.battery_warning_threshold = float(os.getenv("BATTERY_WARNING_THRESHOLD"))
    if os.getenv("BATTERY_CRITICAL_THRESHOLD"):
        config.drone.battery_critical_threshold = float(os.getenv("BATTERY_CRITICAL_THRESHOLD"))

    # Weather thresholds
    if os.getenv("MAX_WIND_SPEED"):
        config.drone.max_wind_speed = float(os.getenv("MAX_WIND_SPEED"))
    if os.getenv("MAX_RAIN_INTENSITY"):
        config.drone.max_rain_intensity = float(os.getenv("MAX_RAIN_INTENSITY"))
    if os.getenv("MIN_VISIBILITY"):
        config.drone.min_visibility = float(os.getenv("MIN_VISIBILITY"))

    # Mission settings
    if os.getenv("DEFAULT_MISSION_DURATION"):
        config.drone.default_mission_duration = int(os.getenv("DEFAULT_MISSION_DURATION"))
    if os.getenv("MAX_MISSION_DURATION"):
        config.drone.max_mission_duration = int(os.getenv("MAX_MISSION_DURATION"))
    if os.getenv("AUTO_RETURN_ENABLED"):
        config.drone.auto_return_enabled = os.getenv("AUTO_RETURN_ENABLED").lower() == "true"

    # Capability settings
    if os.getenv("CAMERA_RESOLUTION"):
        config.capability.camera_resolution = os.getenv("CAMERA_RESOLUTION")
    if os.getenv("CAMERA_FOV"):
        config.capability.camera_fov = float(os.getenv("CAMERA_FOV"))
    if os.getenv("NIGHT_VISION"):
        config.capability.night_vision = os.getenv("NIGHT_VISION").lower() == "true"

    # Task settings
    if os.getenv("MIN_QUALITY_RATING"):
        config.task.min_quality_rating = os.getenv("MIN_QUALITY_RATING")
    if os.getenv("AUTO_RETRY_FAILED_TASKS"):
        config.task.auto_retry_failed_tasks = os.getenv("AUTO_RETRY_FAILED_TASKS").lower() == "true"
    if os.getenv("MAX_RETRY_ATTEMPTS"):
        config.task.max_retry_attempts = int(os.getenv("MAX_RETRY_ATTEMPTS"))


# Initialize configuration from environment
update_config_from_env()
