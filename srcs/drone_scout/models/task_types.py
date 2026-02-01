"""
Drone Scout Task Types and Enumerations

Defines task types, priorities, sensor types, and operational status for drone operations.
"""

from enum import Enum


class TaskType(str, Enum):
    """Available drone task types with enhanced categorization"""

    # 농업 (Agriculture)
    CROP_INSPECTION = "crop_inspection"
    PEST_MONITORING = "pest_monitoring"
    GROWTH_ANALYSIS = "growth_analysis"
    IRRIGATION_MONITORING = "irrigation_monitoring"
    FIELD_MAPPING = "field_mapping"

    # 보안 (Security)
    PERIMETER_PATROL = "perimeter_patrol"
    INTRUSION_DETECTION = "intrusion_detection"
    SURVEILLANCE = "surveillance"
    ASSET_MONITORING = "asset_monitoring"

    # 건설 (Construction)
    CONSTRUCTION_MONITORING = "construction_monitoring"
    SAFETY_INSPECTION = "safety_inspection"
    SITE_MAPPING = "site_mapping"
    PROGRESS_TRACKING = "progress_tracking"

    # 물류 (Logistics)
    DELIVERY = "delivery"
    INVENTORY_CHECK = "inventory_check"
    WAREHOUSE_MONITORING = "warehouse_monitoring"

    # 응급 (Emergency)
    SEARCH_RESCUE = "search_rescue"
    EMERGENCY_RESPONSE = "emergency_response"
    DISASTER_ASSESSMENT = "disaster_assessment"

    # 환경 (Environmental)
    ENVIRONMENTAL_MONITORING = "environmental_monitoring"
    WILDLIFE_TRACKING = "wildlife_tracking"
    POLLUTION_DETECTION = "pollution_detection"

    # 일반 (General)
    PHOTOGRAPHY = "photography"
    VIDEOGRAPHY = "videography"
    GENERAL_INSPECTION = "general_inspection"


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class SensorType(str, Enum):
    """Available sensor types on drones"""

    # 기본 센서
    CAMERA = "camera"
    GPS = "gps"
    ALTIMETER = "altimeter"
    GYROSCOPE = "gyroscope"
    ACCELEROMETER = "accelerometer"
    MAGNETOMETER = "magnetometer"

    # 고급 센서
    THERMAL = "thermal"
    LIDAR = "lidar"
    RADAR = "radar"
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"

    # 환경 센서
    WEATHER_STATION = "weather_station"
    AIR_QUALITY = "air_quality"
    HUMIDITY = "humidity"
    TEMPERATURE = "temperature"

    # 특수 센서
    GAS_DETECTOR = "gas_detector"
    RADIATION_DETECTOR = "radiation_detector"
    SOUND_RECORDER = "sound_recorder"


class DroneOperationalStatus(str, Enum):
    """Drone operational status"""
    IDLE = "idle"
    READY = "ready"
    PREFLIGHT_CHECK = "preflight_check"
    TAKING_OFF = "taking_off"
    FLYING = "flying"
    EXECUTING_TASK = "executing_task"
    RETURNING_HOME = "returning_home"
    LANDING = "landing"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class FlightMode(str, Enum):
    """Drone flight modes"""
    MANUAL = "manual"
    STABILIZED = "stabilized"
    AUTO = "auto"
    GUIDED = "guided"
    LOITER = "loiter"
    RTL = "rtl"  # Return to Launch
    LAND = "land"
    EMERGENCY = "emergency"


class WeatherCondition(str, Enum):
    """Weather condition categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DANGEROUS = "dangerous"


# Task Type to Sensor Mapping
TASK_SENSOR_REQUIREMENTS = {
    # 농업
    TaskType.CROP_INSPECTION: [SensorType.CAMERA, SensorType.MULTISPECTRAL, SensorType.GPS],
    TaskType.PEST_MONITORING: [SensorType.CAMERA, SensorType.THERMAL, SensorType.GPS],
    TaskType.GROWTH_ANALYSIS: [SensorType.MULTISPECTRAL, SensorType.HYPERSPECTRAL, SensorType.GPS],
    TaskType.IRRIGATION_MONITORING: [SensorType.THERMAL, SensorType.HUMIDITY, SensorType.GPS],

    # 보안
    TaskType.PERIMETER_PATROL: [SensorType.CAMERA, SensorType.THERMAL, SensorType.GPS],
    TaskType.INTRUSION_DETECTION: [SensorType.CAMERA, SensorType.THERMAL, SensorType.SOUND_RECORDER],
    TaskType.SURVEILLANCE: [SensorType.CAMERA, SensorType.THERMAL, SensorType.GPS],

    # 건설
    TaskType.CONSTRUCTION_MONITORING: [SensorType.CAMERA, SensorType.LIDAR, SensorType.GPS],
    TaskType.SAFETY_INSPECTION: [SensorType.CAMERA, SensorType.GAS_DETECTOR, SensorType.GPS],
    TaskType.SITE_MAPPING: [SensorType.LIDAR, SensorType.CAMERA, SensorType.GPS],

    # 물류
    TaskType.DELIVERY: [SensorType.GPS, SensorType.CAMERA, SensorType.ALTIMETER],
    TaskType.INVENTORY_CHECK: [SensorType.CAMERA, SensorType.LIDAR, SensorType.GPS],

    # 응급
    TaskType.SEARCH_RESCUE: [SensorType.THERMAL, SensorType.CAMERA, SensorType.GPS],
    TaskType.EMERGENCY_RESPONSE: [SensorType.CAMERA, SensorType.THERMAL, SensorType.GPS],
    TaskType.DISASTER_ASSESSMENT: [SensorType.CAMERA, SensorType.LIDAR, SensorType.GPS],

    # 환경
    TaskType.ENVIRONMENTAL_MONITORING: [SensorType.AIR_QUALITY, SensorType.TEMPERATURE, SensorType.GPS],
    TaskType.WILDLIFE_TRACKING: [SensorType.CAMERA, SensorType.THERMAL, SensorType.GPS],
    TaskType.POLLUTION_DETECTION: [SensorType.GAS_DETECTOR, SensorType.AIR_QUALITY, SensorType.GPS],

    # 일반
    TaskType.PHOTOGRAPHY: [SensorType.CAMERA, SensorType.GPS],
    TaskType.VIDEOGRAPHY: [SensorType.CAMERA, SensorType.GPS],
    TaskType.GENERAL_INSPECTION: [SensorType.CAMERA, SensorType.GPS],
}

# Task Duration Estimates (minutes)
TASK_DURATION_ESTIMATES = {
    TaskType.CROP_INSPECTION: 20,
    TaskType.PEST_MONITORING: 15,
    TaskType.GROWTH_ANALYSIS: 25,
    TaskType.IRRIGATION_MONITORING: 18,
    TaskType.FIELD_MAPPING: 30,

    TaskType.PERIMETER_PATROL: 35,
    TaskType.INTRUSION_DETECTION: 45,
    TaskType.SURVEILLANCE: 40,
    TaskType.ASSET_MONITORING: 25,

    TaskType.CONSTRUCTION_MONITORING: 30,
    TaskType.SAFETY_INSPECTION: 20,
    TaskType.SITE_MAPPING: 35,
    TaskType.PROGRESS_TRACKING: 25,

    TaskType.DELIVERY: 15,
    TaskType.INVENTORY_CHECK: 20,
    TaskType.WAREHOUSE_MONITORING: 30,

    TaskType.SEARCH_RESCUE: 60,
    TaskType.EMERGENCY_RESPONSE: 45,
    TaskType.DISASTER_ASSESSMENT: 40,

    TaskType.ENVIRONMENTAL_MONITORING: 30,
    TaskType.WILDLIFE_TRACKING: 50,
    TaskType.POLLUTION_DETECTION: 25,

    TaskType.PHOTOGRAPHY: 10,
    TaskType.VIDEOGRAPHY: 15,
    TaskType.GENERAL_INSPECTION: 20,
}

# Korean Task Keywords Mapping
KOREAN_TASK_KEYWORDS = {
    # 농업
    "작물": TaskType.CROP_INSPECTION,
    "농장": TaskType.CROP_INSPECTION,
    "농작물": TaskType.CROP_INSPECTION,
    "병해충": TaskType.PEST_MONITORING,
    "해충": TaskType.PEST_MONITORING,
    "생육": TaskType.GROWTH_ANALYSIS,
    "성장": TaskType.GROWTH_ANALYSIS,
    "관개": TaskType.IRRIGATION_MONITORING,
    "물주기": TaskType.IRRIGATION_MONITORING,

    # 보안
    "감시": TaskType.SURVEILLANCE,
    "보안": TaskType.SURVEILLANCE,
    "순찰": TaskType.PERIMETER_PATROL,
    "침입": TaskType.INTRUSION_DETECTION,
    "자산": TaskType.ASSET_MONITORING,

    # 건설
    "건설": TaskType.CONSTRUCTION_MONITORING,
    "공사": TaskType.CONSTRUCTION_MONITORING,
    "안전": TaskType.SAFETY_INSPECTION,
    "점검": TaskType.SAFETY_INSPECTION,
    "측량": TaskType.SITE_MAPPING,
    "지도": TaskType.SITE_MAPPING,
    "매핑": TaskType.SITE_MAPPING,

    # 물류
    "배송": TaskType.DELIVERY,
    "운송": TaskType.DELIVERY,
    "재고": TaskType.INVENTORY_CHECK,
    "창고": TaskType.WAREHOUSE_MONITORING,

    # 응급
    "수색": TaskType.SEARCH_RESCUE,
    "구조": TaskType.SEARCH_RESCUE,
    "응급": TaskType.EMERGENCY_RESPONSE,
    "재해": TaskType.DISASTER_ASSESSMENT,
    "재난": TaskType.DISASTER_ASSESSMENT,

    # 환경
    "환경": TaskType.ENVIRONMENTAL_MONITORING,
    "야생동물": TaskType.WILDLIFE_TRACKING,
    "동물": TaskType.WILDLIFE_TRACKING,
    "오염": TaskType.POLLUTION_DETECTION,

    # 일반
    "촬영": TaskType.PHOTOGRAPHY,
    "사진": TaskType.PHOTOGRAPHY,
    "영상": TaskType.VIDEOGRAPHY,
    "비디오": TaskType.VIDEOGRAPHY,
}
