# Urban Hive Dynamic Data Agent

## 개요

Urban Hive Dynamic Data Agent는 기존 시스템의 하드코딩된 데이터를 지능적이고 동적인 데이터 생성으로 대체하는 핵심 컴포넌트입니다. 이 에이전트는 컨텍스트 인식, 시간 기반 변화, 계절적 요인을 고려하여 현실적인 도시 데이터를 생성합니다.

## 🎯 주요 기능

### 1. 동적 지역구 관리
- **지능적 지역구 생성**: 서울 25개 구를 포함한 동적 지역구 목록
- **지역별 특성 분석**: 인구밀도, 경제수준, 교통수준, 안전수준 등
- **실시간 특성 계산**: 시간대와 계절에 따른 동적 특성 변화

### 2. 커뮤니티 데이터 생성
- **동적 멤버 생성**: 현실적인 이름, 나이, 관심사, 전문분야
- **지능적 그룹 생성**: 계절과 트렌드를 반영한 커뮤니티 그룹
- **활동 패턴 분석**: 활동 레벨, 가용성, 참여 빈도

### 3. 리소스 관리
- **동적 리소스 생성**: 카테고리별 현실적인 아이템과 서비스
- **가격 책정 모델**: 카테고리와 상태에 따른 지능적 가격 설정
- **수요-공급 분석**: 긴급도와 지역별 수요 패턴

### 4. 시간 기반 지능
- **계절적 요인**: 계절에 따른 활동 패턴과 선호도 변화
- **시간대 분석**: 러시아워, 주말, 야간 등 시간대별 특성
- **트렌드 반영**: 현재 시점의 사회적 트렌드 반영

## 🏗️ 아키텍처

```
Dynamic Data Agent
├── DataGenerationConfig     # 데이터 생성 설정
├── DistrictCharacteristics  # 지역구 특성 모델
├── DynamicDataAgent        # 핵심 에이전트 클래스
├── Cache Management        # 지능적 캐싱 시스템
└── Integration Layer       # 기존 시스템과의 통합
```

## 📊 해결된 하드코딩 문제들

### Before (하드코딩)
```python
# 하드코딩된 서울 구 목록
districts = ["강남구", "서초구", "송파구", ...]

# 하드코딩된 커뮤니티 멤버
names = ["김영수", "이민정", "박철수", ...]

# 하드코딩된 리소스 타입
resources = ["전동 드릴", "사다리", "김치", ...]
```

### After (동적 생성)
```python
# 동적 지역구 생성
districts = await dynamic_data_agent.get_dynamic_districts("seoul")

# 동적 커뮤니티 멤버 생성
members = await dynamic_data_agent.get_dynamic_community_members()

# 동적 리소스 생성
resources = await dynamic_data_agent.get_dynamic_resources("available")
```

## 🚀 사용법

### 기본 사용법

```python
from urban_hive.dynamic_data_agent import dynamic_data_agent

# 서울 지역구 목록 가져오기
districts = await dynamic_data_agent.get_dynamic_districts("seoul")

# 커뮤니티 멤버 생성 (8명)
members = await dynamic_data_agent.get_dynamic_community_members(count=8)

# 커뮤니티 그룹 생성 (6개)
groups = await dynamic_data_agent.get_dynamic_community_groups(count=6)

# 사용 가능한 리소스 생성
available_resources = await dynamic_data_agent.get_dynamic_resources("available")

# 리소스 요청 생성
resource_requests = await dynamic_data_agent.get_dynamic_resources("requests")
```

### 지역구 특성 분석

```python
from urban_hive.dynamic_data_agent import get_district_characteristics

# 강남구 특성 분석
characteristics = await get_district_characteristics("강남구")

print(f"인구밀도: {characteristics['population_density']}")
print(f"현재 교통 혼잡도: {characteristics['current_traffic_level']}%")
print(f"현재 범죄율: {characteristics['current_crime_rate']}")
```

### 설정 커스터마이징

```python
from urban_hive.dynamic_data_agent import DynamicDataAgent, DataGenerationConfig

# 커스텀 설정으로 에이전트 생성
config = DataGenerationConfig(
    locale="ko_KR",
    region="seoul",
    data_freshness_hours=12,
    randomization_seed=42
)

custom_agent = DynamicDataAgent(config)
```

## ⚙️ 설정 관리

### 환경 변수 설정

```bash
# API 설정
export PUBLIC_DATA_API_KEY="your_api_key"
export API_TIMEOUT=15
export CACHE_DURATION_HOURS=24

# 데이터 생성 설정
export DEFAULT_REGION="seoul"
export RANDOMIZATION_SEED=42

# 디버그 설정
export DEBUG=true
export LOG_LEVEL=DEBUG
```

### 설정 파일 사용

```python
from urban_hive.config import config, update_config_from_env

# 환경 변수에서 설정 업데이트
update_config_from_env()

# 설정 값 확인
print(f"API Key: {config.api.public_data_api_key}")
print(f"Cache Duration: {config.cache.cache_duration_hours}시간")
```

## 🧪 테스트

### 테스트 실행

```bash
cd srcs/urban_hive
python test_dynamic_agent.py
```

### 테스트 항목
- ✅ 동적 지역구 생성
- ✅ 커뮤니티 데이터 생성
- ✅ 리소스 데이터 생성
- ✅ 지역구 특성 분석
- ✅ Public Data Client 통합
- ✅ 계절/시간 기반 요인

## 📈 성능 최적화

### 캐싱 전략
- **지능적 캐싱**: 데이터 유형별 차별화된 캐시 지속 시간
- **메모리 관리**: 최대 캐시 크기 제한으로 메모리 사용량 최적화
- **캐시 무효화**: 시간 기반 자동 캐시 갱신

### 데이터 생성 최적화
- **지연 로딩**: 필요할 때만 데이터 생성
- **배치 처리**: 여러 데이터를 한 번에 생성하여 효율성 향상
- **컨텍스트 인식**: 이전 생성 결과를 활용한 일관성 있는 데이터

## 🔧 확장성

### 새로운 지역 추가

```python
# 부산 지역구 생성
busan_districts = await dynamic_data_agent.get_dynamic_districts("busan")

# 지역별 설정 가져오기
from urban_hive.config import get_region_specific_config
busan_config = get_region_specific_config("busan")
```

### 새로운 데이터 타입 추가

```python
class DynamicDataAgent:
    async def get_dynamic_events(self, event_type: str = "cultural") -> List[Dict]:
        """새로운 이벤트 데이터 생성"""
        # 구현 로직
        pass
```

## 🎯 주요 이점

### 1. 유지보수성 향상
- **중앙화된 설정**: 모든 설정이 한 곳에서 관리
- **환경별 설정**: 개발/테스트/운영 환경별 다른 설정
- **버전 관리**: 설정 변경 이력 추적 가능

### 2. 현실성 증대
- **컨텍스트 인식**: 지역별, 시간별 특성 반영
- **동적 변화**: 실시간 상황 변화 반영
- **트렌드 반영**: 최신 사회적 트렌드 적용

### 3. 확장성 확보
- **모듈화**: 독립적인 컴포넌트로 설계
- **플러그인 구조**: 새로운 데이터 타입 쉽게 추가
- **API 호환성**: 기존 인터페이스 유지

### 4. 성능 최적화
- **지능적 캐싱**: 불필요한 재계산 방지
- **비동기 처리**: 높은 동시성 지원
- **메모리 효율성**: 최적화된 메모리 사용

## 🔮 향후 계획

### Phase 1: 기본 기능 완성 ✅
- 동적 데이터 생성 엔진
- 기본 캐싱 시스템
- 설정 관리 시스템

### Phase 2: 지능화 (진행 중)
- 머신러닝 기반 패턴 학습
- 사용자 행동 분석
- 예측 모델링

### Phase 3: 고도화 (계획)
- 실시간 외부 데이터 연동
- A/B 테스트 지원
- 성능 모니터링 대시보드

## 📚 참고 자료

- [Urban Hive 전체 아키텍처](../README.md)
- [Public Data Client 문서](providers/README.md)
- [설정 관리 가이드](config.py)
- [테스트 가이드](test_dynamic_agent.py)

## 🤝 기여하기

1. 이슈 생성 또는 기능 제안
2. 포크 및 브랜치 생성
3. 코드 작성 및 테스트
4. 풀 리퀘스트 제출

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 