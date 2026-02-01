"""
Drone Physics Simulator

물리 법칙 기반 드론 비행 시뮬레이션.
하드코딩 없이 물리 엔진을 사용하여 실제 드론 비행과 유사한 시뮬레이션을 생성합니다.
"""

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

from srcs.common.simulation_utils import (
    TimeSeriesGenerator, NoiseGenerator, ProbabilityDistributions
)
from models.drone_data import (
    DronePosition, WeatherData, SensorReading, SensorType
)


class DronePhysicsSimulator:
    """물리 법칙 기반 드론 비행 시뮬레이터"""

    # 물리 상수
    GRAVITY = 9.81  # m/s²
    AIR_DENSITY = 1.225  # kg/m³ (해수면 기준)
    DRAG_COEFFICIENT = 0.3  # 드론 항력 계수

    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self.time_series_gen = TimeSeriesGenerator(seed)
        self.noise_gen = NoiseGenerator(seed)
        self.prob_dist = ProbabilityDistributions(seed)

    def generate_flight_trajectory(
        self,
        start_pos: DronePosition,
        waypoints: List[DronePosition],
        cruise_speed: float = 10.0,  # m/s
        max_acceleration: float = 2.0,  # m/s²
        update_interval: float = 0.5  # seconds
    ) -> List[Dict[str, Any]]:
        """
        물리 기반 비행 경로 생성

        Args:
            start_pos: 시작 위치
            waypoints: 경유지 목록
            cruise_speed: 순항 속도 (m/s)
            max_acceleration: 최대 가속도 (m/s²)
            update_interval: 위치 업데이트 간격 (초)

        Returns:
            비행 경로 데이터 [{"lon": ..., "lat": ..., "alt": ..., "timestamp": ...}, ...]
        """
        trajectory = []
        current_pos = start_pos
        current_speed = 0.0
        current_time = datetime.now()

        all_points = [start_pos] + waypoints

        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]

            # 거리 계산 (Haversine 공식)
            distance = self._calculate_distance(
                start.latitude, start.longitude,
                end.latitude, end.longitude
            )

            # 고도 차이
            altitude_diff = end.altitude - start.altitude

            # 3D 거리
            total_distance = math.sqrt(distance**2 + altitude_diff**2)

            # 필요한 시간 계산 (가속/감속 고려)
            if total_distance > 0:
                # 가속 구간
                accel_time = cruise_speed / max_acceleration
                accel_distance = 0.5 * max_acceleration * accel_time**2

                if total_distance < 2 * accel_distance:
                    # 거리가 짧아서 가속/감속만
                    time_needed = 2 * math.sqrt(total_distance / max_acceleration)
                else:
                    # 가속 + 순항 + 감속
                    cruise_distance = total_distance - 2 * accel_distance
                    cruise_time = cruise_distance / cruise_speed
                    time_needed = 2 * accel_time + cruise_time
            else:
                time_needed = 0.1

            # 경로 생성
            num_steps = max(1, int(time_needed / update_interval))

            for step in range(num_steps):
                t = step / num_steps if num_steps > 1 else 1.0

                # 베지어 곡선을 사용한 부드러운 경로
                if num_steps > 2:
                    # 중간 제어점 추가 (부드러운 곡선)
                    mid_lat = (start.latitude + end.latitude) / 2
                    mid_lon = (start.longitude + end.longitude) / 2
                    mid_alt = (start.altitude + end.altitude) / 2

                    # 베지어 곡선 계산
                    lat = (1-t)**2 * start.latitude + 2*(1-t)*t * mid_lat + t**2 * end.latitude
                    lon = (1-t)**2 * start.longitude + 2*(1-t)*t * mid_lon + t**2 * end.longitude
                    alt = (1-t)**2 * start.altitude + 2*(1-t)*t * mid_alt + t**2 * end.altitude
                else:
                    # 선형 보간
                    lat = start.latitude + (end.latitude - start.latitude) * t
                    lon = start.longitude + (end.longitude - start.longitude) * t
                    alt = start.altitude + (end.altitude - start.altitude) * t

                # 속도 계산
                if t < 0.5:
                    # 가속 구간
                    current_speed = max_acceleration * (t * time_needed)
                else:
                    # 감속 구간
                    current_speed = cruise_speed - max_acceleration * ((t - 0.5) * time_needed)

                current_speed = min(current_speed, cruise_speed)

                # 방향 계산
                if step < num_steps - 1:
                    next_t = (step + 1) / num_steps
                    next_lat = start.latitude + (end.latitude - start.latitude) * next_t
                    next_lon = start.longitude + (end.longitude - start.longitude) * next_t
                    heading = self._calculate_heading(lat, lon, next_lat, next_lon)
                else:
                    heading = self._calculate_heading(
                        lat, lon, end.latitude, end.longitude
                    )

                # GPS 노이즈 추가 (실제 GPS 정확도 모방)
                lat = self.noise_gen.add_gaussian_noise(lat, 0.00001)  # ~1m 오차
                lon = self.noise_gen.add_gaussian_noise(lon, 0.00001)
                alt = self.noise_gen.add_gaussian_noise(alt, 0.5)  # 고도는 더 큰 오차

                timestamp = current_time + timedelta(seconds=step * update_interval)

                trajectory.append({
                    "lon": round(lon, 6),
                    "lat": round(lat, 6),
                    "alt": round(alt, 2),
                    "heading": round(heading, 1),
                    "speed": round(current_speed, 2),
                    "timestamp": timestamp.isoformat()
                })

            current_time += timedelta(seconds=time_needed)
            current_pos = end

        return trajectory

    def simulate_sensor_readings(
        self,
        position: DronePosition,
        weather: WeatherData,
        sensor_types: List[SensorType],
        duration_seconds: float = 1.0,
        interval: float = 0.1
    ) -> List[SensorReading]:
        """
        센서 데이터 시뮬레이션 (노이즈 포함)

        Args:
            position: 드론 위치
            weather: 날씨 데이터
            sensor_types: 센서 타입 목록
            duration_seconds: 측정 지속 시간
            interval: 측정 간격
        """
        readings = []
        num_readings = int(duration_seconds / interval)

        for i in range(num_readings):
            timestamp = datetime.now() + timedelta(seconds=i * interval)

            for sensor_type in sensor_types:
                if sensor_type == SensorType.TEMPERATURE:
                    # 온도: 고도에 따른 변화 + 날씨 영향 + 노이즈
                    base_temp = weather.temperature
                    altitude_factor = -0.0065 * position.altitude  # 고도 1m당 -0.0065°C
                    temp = base_temp + altitude_factor
                    temp = self.noise_gen.add_gaussian_noise(temp, 0.5)
                    value = temp
                    unit = "°C"

                elif sensor_type == SensorType.HUMIDITY:
                    # 습도: 고도에 따른 변화 + 노이즈
                    base_humidity = weather.humidity
                    altitude_factor = -0.1 * (position.altitude / 100)  # 고도에 따라 감소
                    humidity = max(0, min(100, base_humidity + altitude_factor))
                    humidity = self.noise_gen.add_gaussian_noise(humidity, 2.0)
                    value = humidity
                    unit = "%"

                elif sensor_type == SensorType.WIND_SPEED:
                    # 풍속: 고도에 따른 변화 + 노이즈
                    base_wind = weather.wind_speed
                    altitude_factor = 1.0 + (position.altitude / 100) * 0.1  # 고도에 따라 증가
                    wind = base_wind * altitude_factor
                    wind = self.noise_gen.add_gaussian_noise(wind, 0.5)
                    value = max(0, wind)
                    unit = "m/s"

                elif sensor_type == SensorType.PRESSURE:
                    # 기압: 고도에 따른 변화 (기압 고도 공식)
                    sea_level_pressure = 1013.25  # hPa
                    altitude_m = position.altitude
                    pressure = sea_level_pressure * (1 - (0.0065 * altitude_m) / 288.15) ** 5.256
                    pressure = self.noise_gen.add_gaussian_noise(pressure, 0.5)
                    value = pressure
                    unit = "hPa"

                elif sensor_type == SensorType.ALTITUDE:
                    # 고도: GPS 기반 + 기압계 보정
                    gps_alt = position.altitude
                    baro_correction = self.noise_gen.add_gaussian_noise(0, 1.0)
                    alt = gps_alt + baro_correction
                    value = max(0, alt)
                    unit = "m"

                else:
                    # 기본 센서 값
                    value = self.rng.uniform(0, 100)
                    unit = "unit"

                # 데이터 품질 계산 (신호 강도, 환경 조건 기반)
                quality = 0.9 - (position.altitude / 1000) * 0.1  # 고도가 높을수록 품질 감소
                quality = max(0.5, min(1.0, quality))
                quality = self.noise_gen.add_gaussian_noise(quality, 0.05)
                quality = max(0, min(1, quality))

                reading = SensorReading(
                    sensor_type=sensor_type,
                    value=round(value, 2),
                    unit=unit,
                    quality=round(quality, 3),
                    position=position,
                    timestamp=timestamp,
                    metadata={
                        "simulated": True,
                        "noise_level": "normal"
                    }
                )
                readings.append(reading)

        return readings

    def calculate_battery_consumption(
        self,
        flight_time_minutes: float,
        altitude: float,
        wind_speed: float,
        wind_direction: float,
        drone_heading: float,
        payload_weight: float = 0.0,
        base_consumption_rate: float = 2.0  # % per minute
    ) -> float:
        """
        배터리 소모 계산 (물리 모델 기반)

        Args:
            flight_time_minutes: 비행 시간 (분)
            altitude: 고도 (m)
            wind_speed: 풍속 (m/s)
            wind_direction: 풍향 (도)
            drone_heading: 드론 방향 (도)
            payload_weight: 페이로드 무게 (kg)
            base_consumption_rate: 기본 소모율 (%/분)
        """
        # 기본 소모
        base_consumption = base_consumption_rate * flight_time_minutes

        # 고도 영향 (고도가 높을수록 공기 밀도 감소, 효율 향상)
        altitude_factor = 1.0 - (altitude / 10000) * 0.1
        base_consumption *= altitude_factor

        # 바람 영향
        wind_angle_diff = abs(wind_direction - drone_heading)
        if wind_angle_diff > 180:
            wind_angle_diff = 360 - wind_angle_diff

        # 역풍/순풍 계산
        if wind_angle_diff < 45 or wind_angle_diff > 135:
            # 역풍 또는 측풍
            wind_factor = 1.0 + (wind_speed / 10) * 0.2
        else:
            # 순풍
            wind_factor = 1.0 - (wind_speed / 10) * 0.1

        base_consumption *= wind_factor

        # 페이로드 영향
        payload_factor = 1.0 + (payload_weight / 2.0) * 0.15
        base_consumption *= payload_factor

        # 랜덤 변동 (배터리 상태, 온도 등)
        variation = self.prob_dist.sample(
            ProbabilityDistributions.DistributionType.NORMAL,
            mean=0, std=base_consumption * 0.05
        )

        total_consumption = base_consumption + variation
        return max(0, min(100, total_consumption))

    def apply_weather_effects(
        self,
        base_position: DronePosition,
        weather: WeatherData,
        time_elapsed: float
    ) -> DronePosition:
        """
        날씨 영향 적용 (바람에 의한 위치 변화)

        Args:
            base_position: 기본 위치
            weather: 날씨 데이터
            time_elapsed: 경과 시간 (초)
        """
        # 바람에 의한 표류 계산
        wind_speed = weather.wind_speed
        wind_direction_rad = math.radians(weather.wind_direction)

        # 바람에 의한 이동 거리
        drift_distance = wind_speed * time_elapsed

        # 위도/경도 변화 (간단한 근사)
        lat_offset = drift_distance * math.cos(wind_direction_rad) / 111000  # 1도 ≈ 111km
        lon_offset = drift_distance * math.sin(wind_direction_rad) / (111000 * math.cos(math.radians(base_position.latitude)))

        new_lat = base_position.latitude + lat_offset
        new_lon = base_position.longitude + lon_offset

        # 노이즈 추가
        new_lat = self.noise_gen.add_gaussian_noise(new_lat, 0.00001)
        new_lon = self.noise_gen.add_gaussian_noise(new_lon, 0.00001)

        return DronePosition(
            latitude=new_lat,
            longitude=new_lon,
            altitude=base_position.altitude,
            heading=base_position.heading,
            speed=base_position.speed,
            timestamp=base_position.timestamp + timedelta(seconds=time_elapsed)
        )

    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Haversine 공식을 사용한 거리 계산 (km)"""
        R = 6371.0  # 지구 반경 (km)

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2)**2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c * 1000  # 미터로 변환
        return distance

    def _calculate_heading(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """두 지점 간 방향 계산 (도)"""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        y = math.sin(dlon) * math.cos(math.radians(lat2))
        x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
             math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))

        heading = math.degrees(math.atan2(y, x))
        heading = (heading + 360) % 360
        return heading
