"""
반려동물 관련 도구

반려동물 프로필 관리, 행동 기록, 활동 추적
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PetProfileInput(BaseModel):
    """반려동물 프로필 입력 스키마"""
    pet_id: str = Field(description="반려동물 ID")
    name: Optional[str] = Field(default=None, description="반려동물 이름")
    species: Optional[str] = Field(default=None, description="종류 (dog, cat, etc.)")
    breed: Optional[str] = Field(default=None, description="품종")
    age: Optional[int] = Field(default=None, description="나이 (개월)")


class BehaviorRecordInput(BaseModel):
    """행동 기록 입력 스키마"""
    pet_id: str = Field(description="반려동물 ID")
    behavior_type: str = Field(description="행동 타입 (eating, sleeping, playing, etc.)")
    duration: Optional[int] = Field(default=None, description="지속 시간 (분)")
    location: Optional[str] = Field(default=None, description="위치")
    notes: Optional[str] = Field(default=None, description="추가 메모")


class ActivityTrackingInput(BaseModel):
    """활동 추적 입력 스키마"""
    pet_id: str = Field(description="반려동물 ID")
    activity_type: str = Field(description="활동 타입 (walk, play, rest)")
    start_time: Optional[str] = Field(default=None, description="시작 시간")
    end_time: Optional[str] = Field(default=None, description="종료 시간")


class PetTools:
    """
    반려동물 관련 도구 모음
    
    반려동물 프로필 관리, 행동 기록, 활동 추적
    """
    
    def __init__(self, data_dir: str = "petcare_data"):
        """
        PetTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_file = self.data_dir / "pet_profiles.json"
        self.behaviors_file = self.data_dir / "pet_behaviors.json"
        self.activities_file = self.data_dir / "pet_activities.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.profiles_file.exists():
            with open(self.profiles_file, 'r', encoding='utf-8') as f:
                self.profiles = json.load(f)
        else:
            self.profiles = {}
        
        if self.behaviors_file.exists():
            with open(self.behaviors_file, 'r', encoding='utf-8') as f:
                self.behaviors = json.load(f)
        else:
            self.behaviors = {}
        
        if self.activities_file.exists():
            with open(self.activities_file, 'r', encoding='utf-8') as f:
                self.activities = json.load(f)
        else:
            self.activities = {}
    
    def _save_profiles(self):
        """프로필 저장"""
        with open(self.profiles_file, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)
    
    def _save_behaviors(self):
        """행동 기록 저장"""
        with open(self.behaviors_file, 'w', encoding='utf-8') as f:
            json.dump(self.behaviors, f, indent=2, ensure_ascii=False)
    
    def _save_activities(self):
        """활동 기록 저장"""
        with open(self.activities_file, 'w', encoding='utf-8') as f:
            json.dump(self.activities, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """반려동물 도구 초기화"""
        self.tools.append(self._create_get_pet_profile_tool())
        self.tools.append(self._create_update_pet_profile_tool())
        self.tools.append(self._create_record_behavior_tool())
        self.tools.append(self._create_track_activity_tool())
        self.tools.append(self._create_get_behavior_history_tool())
        logger.info(f"Initialized {len(self.tools)} pet tools")
    
    def _create_get_pet_profile_tool(self) -> BaseTool:
        @tool("pet_get_profile", args_schema=PetProfileInput)
        def get_pet_profile(pet_id: str, name: Optional[str] = None, species: Optional[str] = None, breed: Optional[str] = None, age: Optional[int] = None) -> str:
            """
            반려동물 프로필을 조회합니다.
            Args:
                pet_id: 반려동물 ID
                name: 반려동물 이름 (선택)
                species: 종류 (선택)
                breed: 품종 (선택)
                age: 나이 (선택)
            Returns:
                반려동물 프로필 정보 (JSON 문자열) 또는 오류 메시지
            """
            logger.info(f"Getting profile for pet '{pet_id}'")
            if pet_id in self.profiles:
                return json.dumps(self.profiles[pet_id], ensure_ascii=False, indent=2)
            return f"Error: Pet profile not found for '{pet_id}'"
        return get_pet_profile
    
    def _create_update_pet_profile_tool(self) -> BaseTool:
        @tool("pet_update_profile", args_schema=PetProfileInput)
        def update_pet_profile(pet_id: str, name: Optional[str] = None, species: Optional[str] = None, breed: Optional[str] = None, age: Optional[int] = None) -> str:
            """
            반려동물 프로필을 업데이트합니다.
            Args:
                pet_id: 반려동물 ID
                name: 반려동물 이름
                species: 종류
                breed: 품종
                age: 나이
            Returns:
                업데이트 결과 메시지
            """
            logger.info(f"Updating profile for pet '{pet_id}'")
            if pet_id not in self.profiles:
                self.profiles[pet_id] = {"pet_id": pet_id}
            
            if name:
                self.profiles[pet_id]["name"] = name
            if species:
                self.profiles[pet_id]["species"] = species
            if breed:
                self.profiles[pet_id]["breed"] = breed
            if age:
                self.profiles[pet_id]["age"] = age
            
            self.profiles[pet_id]["updated_at"] = datetime.now().isoformat()
            self._save_profiles()
            return f"Pet profile updated for '{pet_id}'"
        return update_pet_profile
    
    def _create_record_behavior_tool(self) -> BaseTool:
        @tool("pet_record_behavior", args_schema=BehaviorRecordInput)
        def record_behavior(pet_id: str, behavior_type: str, duration: Optional[int] = None, location: Optional[str] = None, notes: Optional[str] = None) -> str:
            """
            반려동물 행동을 기록합니다.
            Args:
                pet_id: 반려동물 ID
                behavior_type: 행동 타입
                duration: 지속 시간 (분)
                location: 위치
                notes: 추가 메모
            Returns:
                기록 결과 메시지
            """
            logger.info(f"Recording behavior for pet '{pet_id}': type='{behavior_type}'")
            if pet_id not in self.behaviors:
                self.behaviors[pet_id] = []
            
            behavior_record = {
                "pet_id": pet_id,
                "behavior_type": behavior_type,
                "duration": duration,
                "location": location,
                "notes": notes,
                "timestamp": datetime.now().isoformat(),
            }
            self.behaviors[pet_id].append(behavior_record)
            self._save_behaviors()
            return f"Behavior recorded for pet '{pet_id}': {behavior_type}"
        return record_behavior
    
    def _create_track_activity_tool(self) -> BaseTool:
        @tool("pet_track_activity", args_schema=ActivityTrackingInput)
        def track_activity(pet_id: str, activity_type: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> str:
            """
            반려동물 활동을 추적합니다.
            Args:
                pet_id: 반려동물 ID
                activity_type: 활동 타입
                start_time: 시작 시간
                end_time: 종료 시간
            Returns:
                추적 결과 메시지
            """
            logger.info(f"Tracking activity for pet '{pet_id}': type='{activity_type}'")
            if pet_id not in self.activities:
                self.activities[pet_id] = []
            
            activity_record = {
                "pet_id": pet_id,
                "activity_type": activity_type,
                "start_time": start_time or datetime.now().isoformat(),
                "end_time": end_time,
                "timestamp": datetime.now().isoformat(),
            }
            self.activities[pet_id].append(activity_record)
            self._save_activities()
            return f"Activity tracked for pet '{pet_id}': {activity_type}"
        return track_activity
    
    def _create_get_behavior_history_tool(self) -> BaseTool:
        @tool("pet_get_behavior_history", args_schema=PetProfileInput)
        def get_behavior_history(pet_id: str, name: Optional[str] = None, species: Optional[str] = None, breed: Optional[str] = None, age: Optional[int] = None) -> str:
            """
            반려동물 행동 이력을 조회합니다.
            Args:
                pet_id: 반려동물 ID
            Returns:
                행동 이력 (JSON 문자열) 또는 오류 메시지
            """
            logger.info(f"Getting behavior history for pet '{pet_id}'")
            if pet_id in self.behaviors:
                return json.dumps(self.behaviors[pet_id], ensure_ascii=False, indent=2)
            return f"Error: Behavior history not found for '{pet_id}'"
        return get_behavior_history
    
    def get_tools(self) -> List[BaseTool]:
        """모든 반려동물 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 반려동물 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

