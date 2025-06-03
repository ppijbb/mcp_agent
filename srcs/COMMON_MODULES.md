# Common Modules Usage Guide

공통 모듈을 사용하여 새로운 agent를 효율적으로 개발하는 방법을 설명합니다.

## 📁 공통 모듈 구조

```
common/
├── __init__.py          # 통합 모듈 진입점
├── imports.py           # 표준화된 imports 
├── config.py           # 공통 설정과 상수
├── utils.py            # 공통 유틸리티 함수
└── templates.py        # Agent 베이스 템플릿
```

## 🚀 빠른 시작

### 1. 기본 Agent 생성

```python
from common import BasicAgentTemplate

class MyResearchAgent(BasicAgentTemplate):
    def __init__(self, topic="AI trends"):
        super().__init__(
            agent_name="my_research",
            task_description=f"Research comprehensive information about {topic}"
        )
        self.topic = topic
    
    def create_agents(self):
        return [
            Agent(
                name="trend_analyzer",
                instruction=f"Analyze current trends in {self.topic}",
                server_names=DEFAULT_SERVERS,
            )
        ]

# 실행
async def main():
    agent = MyResearchAgent("Machine Learning")
    await agent.run()
```

### 2. Enterprise Agent 생성

```python
from common import EnterpriseAgentTemplate

class MyEnterpriseAgent(EnterpriseAgentTemplate):
    def __init__(self):
        super().__init__(
            agent_name="business_optimization",
            business_scope="Global Operations"
        )
    
    def create_agents(self):
        return [
            Agent(
                name="strategy_analyzer",
                instruction=f"Analyze business strategy for {self.company_name}",
                server_names=DEFAULT_SERVERS,
            ),
            Agent(
                name="performance_optimizer", 
                instruction=f"Optimize performance metrics for {self.business_scope}",
                server_names=DEFAULT_SERVERS,
            )
        ]
    
    def create_evaluator(self):
        criteria = [
            ("Business Impact", 40, "ROI and value creation metrics"),
            ("Implementation Feasibility", 30, "Resource requirements and timeline"),
            ("Strategic Alignment", 30, "Alignment with company objectives")
        ]
        return self.create_standard_evaluator(criteria)
```

## 📦 모듈별 상세 설명

### `common/imports.py` - 표준화된 Imports

모든 agent에서 공통으로 사용하는 imports를 한 곳에서 관리:

```python
from common.imports import (
    asyncio, os, json, datetime, 
    MCPApp, Agent, Orchestrator, 
    RequestParams, OpenAIAugmentedLLM
)
```

**장점:**
- Import 일관성 보장
- 중복 코드 제거
- 버전 관리 용이

### `common/config.py` - 공통 설정

설정값과 상수를 중앙에서 관리:

```python
from common.config import (
    DEFAULT_COMPANY_NAME,    # "TechCorp Inc."
    DEFAULT_SERVERS,         # ["filesystem", "g-search", "fetch"]
    COMPLIANCE_FRAMEWORKS,   # ["GDPR", "SOX", "HIPAA", ...]
    get_timestamp,           # 표준화된 타임스탬프
    get_output_dir          # 표준화된 출력 디렉토리명
)
```

**주요 함수:**
- `get_timestamp()`: `20241201_143022` 형식의 타임스탬프
- `get_output_dir(type, name)`: `agent_name_reports` 형식의 디렉토리명
- `get_app_config(name)`: 표준화된 MCP 앱 설정

### `common/utils.py` - 공통 유틸리티

반복적인 작업을 자동화하는 유틸리티 함수들:

```python
from common.utils import (
    setup_agent_app,           # MCP 앱 초기화
    ensure_output_directory,   # 출력 디렉토리 생성
    configure_filesystem_server, # 파일시스템 서버 설정
    create_executive_summary,  # 임원진 요약 보고서 생성
    create_kpi_template       # KPI 템플릿 생성
)
```

**Executive Summary 생성 예시:**
```python
summary_data = {
    "title": "My Agent Analysis",
    "overview": {
        "title": "Analysis Overview", 
        "content": "Comprehensive analysis completed..."
    },
    "impact_metrics": {
        "Efficiency Improvement": "30-50%",
        "Cost Reduction": "20-35%"
    },
    "initiatives": {
        "Phase 1": "Initial assessment and planning",
        "Phase 2": "Implementation and deployment"
    },
    "action_items": [
        "Review analysis findings",
        "Approve implementation plan"
    ],
    "next_steps": [
        "Executive review meeting",
        "Resource allocation planning"
    ]
}

create_executive_summary(
    output_dir="my_reports",
    agent_name="my_agent", 
    **summary_data
)
```

### `common/templates.py` - Agent 템플릿

기본 구조와 패턴을 제공하는 베이스 클래스들:

#### `BasicAgentTemplate`
간단한 agent를 위한 기본 템플릿:

```python
class MyAgent(BasicAgentTemplate):
    def create_agents(self):
        # 전문화된 agent들 리스트 반환
    
    def create_evaluator(self): 
        # 품질 평가 agent 반환
    
    def define_task(self):
        # orchestrator 실행 태스크 정의
```

#### `EnterpriseAgentTemplate`
기업급 agent를 위한 고급 템플릿:

```python
class MyEnterpriseAgent(EnterpriseAgentTemplate):
    def create_agents(self):
        # 복잡한 비즈니스 로직 agent들
    
    def create_evaluator(self):
        # 표준화된 평가 기준 사용
        criteria = [
            ("Category", weight, "description"),
            # ...
        ]
        return self.create_standard_evaluator(criteria)
    
    def create_summary(self):
        # 기업용 요약 보고서 생성
        return self.create_enterprise_summary(summary_data)
    
    def create_kpis(self):
        # 기업용 KPI 템플릿 생성
        return self.create_enterprise_kpis(kpi_structure)
```

## 🎯 사용 예시

### 완전한 Agent 예시 (researcher_v2.py)

```python
from common import *

class ResearcherAgent(BasicAgentTemplate):
    def __init__(self, research_topic="AI trends"):
        super().__init__(
            agent_name="researcher_v2",
            task_description=f"Research {research_topic}"
        )
        self.research_topic = research_topic
    
    def create_agents(self):
        return [
            Agent(
                name="trend_researcher",
                instruction=f"Research trends in {self.research_topic}",
                server_names=DEFAULT_SERVERS,
            ),
            Agent(
                name="competitive_researcher",
                instruction=f"Research competitors in {self.research_topic}",
                server_names=DEFAULT_SERVERS,
            )
        ]
    
    def create_evaluator(self):
        return Agent(
            name="research_evaluator",
            instruction="Evaluate research quality and comprehensiveness..."
        )
    
    def define_task(self):
        return f"Execute comprehensive research on {self.research_topic}..."
    
    def create_summary(self):
        summary_data = {
            "impact_metrics": {
                "Research Depth": "95%+ comprehensive",
                "Source Quality": "High-reliability sources"
            },
            "action_items": [
                "Review research findings",
                "Develop action plan"
            ]
        }
        return create_executive_summary(
            output_dir=self.output_dir,
            agent_name="research",
            **summary_data
        )

# 실행
async def main():
    researcher = ResearcherAgent("Quantum Computing")
    await researcher.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## 💡 장점 및 효과

### 1. **코드 재사용성**
- 공통 로직 중복 제거
- 표준화된 패턴 적용
- 유지보수 비용 절감

### 2. **개발 속도 향상**
- 템플릿 기반 빠른 개발
- 사전 검증된 패턴 사용
- 보일러플레이트 코드 최소화

### 3. **품질 일관성**
- 표준화된 구조와 형식
- 일관된 에러 처리
- 통일된 로깅과 리포팅

### 4. **확장성**
- 새로운 agent 유형 쉽게 추가
- 모듈식 설계로 기능 확장 용이
- 플러그인 형태의 컴포넌트 추가

## 🔄 마이그레이션 가이드

기존 agent를 공통 모듈로 마이그레이션하는 단계:

### 1단계: Import 정리
```python
# Before
import asyncio
import os
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
# ... 많은 imports

# After  
from common import *
```

### 2단계: 템플릿 적용
```python
# Before
async def main():
    app = MCPApp(...)
    # ... 복잡한 설정

# After
class MyAgent(BasicAgentTemplate):
    async def run(self):
        # 템플릿이 모든 설정 자동 처리
```

### 3단계: 공통 함수 활용
```python
# Before
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# ... 반복적인 코드

# After
ensure_output_directory(self.output_dir)
timestamp = get_timestamp()
create_executive_summary(...)
```

## 🛠️ 개발 도구

### 실행 예시 확인
```bash
# 공통 모듈 데모 실행
python run_agent.py --dev common_demo

# 템플릿 예시 확인  
python run_agent.py --dev template_basic
python run_agent.py --dev template_enterprise

# 실제 예시 실행
python run_agent.py --basic researcher_v2
```

### 새 Agent 개발 체크리스트
- [ ] 적절한 템플릿 선택 (Basic vs Enterprise)
- [ ] 공통 모듈에서 imports 가져오기
- [ ] 표준 설정과 상수 활용
- [ ] 공통 유틸리티 함수 활용
- [ ] 일관된 로깅과 에러 처리
- [ ] 표준화된 출력 형식 적용

공통 모듈을 활용하면 개발 시간을 50-70% 단축하고 코드 품질을 크게 향상시킬 수 있습니다! 