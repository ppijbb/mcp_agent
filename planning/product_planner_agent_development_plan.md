# 프로덕트 기획자 Agent 개발 계획서

**작성일**: 2025년 1월 15일  
**프로젝트명**: Product Planner Agent  
**목표**: Figma와 Notion 연동 프로덕트 기획 자동화 Agent 개발  

---

## 📋 프로젝트 개요

### 1.1 Agent 개요
- **Agent 명**: `product_planner_agent`
- **분류**: Enterprise Agent (기업급)
- **목적**: Figma 디자인과 Notion 문서를 연동하여 프로덕트 기획 업무를 자동화하는 AI Agent
- **기반 프레임워크**: mcp-agent 라이브러리 활용
- **표준 준수**: MCP (Model Context Protocol) 표준 준수

### 1.2 핵심 기능
- ✅ Figma에서 디자인 분석 및 메타데이터 추출
- ✅ Notion에서 기획 문서 생성 및 관리
- ✅ 디자인-기획 간 연관성 분석 및 추적
- ✅ 프로덕트 로드맵 및 요구사항 문서 자동 생성
- ✅ 실시간 디자인 변경사항 동기화
- ✅ 협업 워크플로우 최적화

---

## 🏗️ 기술 아키텍처

### 2.1 프로젝트 구조
```
srcs/product_planner_agent/
├── __init__.py
├── product_planner_agent.py    # 메인 Agent 클래스 (EnterpriseAgentTemplate 상속)
├── config.py                   # 설정 및 상수 정의
├── integrations/
│   ├── __init__.py
│   ├── figma_integration.py    # Figma MCP 서버 연동
│   └── notion_integration.py   # Notion MCP 서버 연동
├── processors/
│   ├── __init__.py
│   ├── design_analyzer.py      # 디자인 분석 엔진
│   ├── requirement_generator.py # 요구사항 생성기
│   └── roadmap_builder.py      # 로드맵 생성기
├── templates/
│   ├── __init__.py
│   ├── prd_template.py         # PRD(Product Requirements Document) 템플릿
│   ├── spec_template.py        # 기술 스펙 템플릿
│   └── roadmap_template.py     # 로드맵 템플릿
└── utils/
    ├── __init__.py
    ├── helpers.py              # 공통 유틸리티 함수
    └── validators.py           # 데이터 검증 로직
```

### 2.2 기술 스택
- **Backend**: Python 3.8+ with mcp-agent library
- **MCP 서버**: Figma Dev Mode MCP, Notion MCP
- **API**: Figma API, Notion API
- **데이터 처리**: Pandas, JSON, YAML
- **템플릿 엔진**: Jinja2
- **검증**: Pydantic v2

---

## 🔌 MCP 서버 연동 계획

### 3.1 Figma MCP 서버

#### 옵션 1: 공식 Figma Dev Mode MCP Server (권장)
- **설명**: Figma 공식 베타 서버
- **설정**: 로컬 실행 (`http://127.0.0.1:3845/sse`)
- **기능**: 
  - 디자인 메타데이터 추출
  - 컴포넌트 정보 분석
  - 코드 생성 지원
  - 변수 및 스타일 정보
- **요구사항**: Figma Desktop App 필요

#### 옵션 2: Framelink Figma MCP Server (대안)
- **GitHub**: [GLips/Figma-Context-MCP](https://github.com/GLips/Figma-Context-MCP)
- **설치**: `npx -y figma-developer-mcp`
- **장점**: API 키만으로 사용 가능
- **기능**: 레이아웃 정보, 디자인 컨텍스트 제공

### 3.2 Notion MCP 서버

#### 옵션 1: 공식 Notion MCP Server (권장)
- **패키지**: `@notionhq/notion-mcp-server`
- **설치**: `npx -y @notionhq/notion-mcp-server`
- **기능**: 기본 CRUD 작업
- **공식 지원**: Notion 개발자 문서 제공

#### 옵션 2: 고급 기능 서버 (확장)
- **GitHub**: [awkoy/notion-mcp-server](https://github.com/awkoy/notion-mcp-server)
- **추가 기능**: 
  - 데이터베이스 고급 조작
  - 블록 관리
  - 댓글 시스템
  - 사용자 관리

---

## 💼 상세 기능 명세

### 4.1 핵심 Agent 클래스

```python
class ProductPlannerAgent(EnterpriseAgentTemplate):
    """프로덕트 기획자 Agent - 기업급 템플릿 상속"""
    
    def __init__(self):
        super().__init__(
            agent_name="product_planner",
            business_scope="Product Planning & Design Integration"
        )
        self.figma_client = None
        self.notion_client = None
        
    def create_agents(self):
        """전문화된 서브 Agent들 생성"""
        return [
            Agent(
                name="design_analyzer",
                instruction="Figma 디자인 분석 및 프로덕트 요구사항 추출",
                server_names=["figma-dev-mode", "filesystem"]
            ),
            Agent(
                name="requirement_synthesizer", 
                instruction="디자인 분석으로부터 PRD 및 기술 스펙 생성",
                server_names=["notion-api", "filesystem"]
            ),
            Agent(
                name="roadmap_planner",
                instruction="프로덕트 로드맵 및 마일스톤 추적 생성",
                server_names=["notion-api", "filesystem"]
            ),
            Agent(
                name="design_notion_connector",
                instruction="디자인 변경사항과 Notion 문서 동기화",
                server_names=["figma-dev-mode", "notion-api"]
            )
        ]
    
    def create_evaluator(self):
        """기업급 품질 평가자 생성"""
        evaluation_criteria = [
            ("Product Feasibility", 30, "제품 실현 가능성 및 기술적 타당성"),
            ("Market Alignment", 25, "시장 요구사항 및 사용자 니즈 부합도"),
            ("Design Consistency", 20, "디자인 시스템 일관성 및 UX 품질"),
            ("Documentation Quality", 15, "문서화 완성도 및 명확성"),
            ("Timeline Realism", 10, "개발 일정의 현실성 및 리스크 고려")
        ]
        return self.create_standard_evaluator(evaluation_criteria)
```

### 4.2 Figma 통합 모듈

```python
class FigmaIntegration:
    """Figma API 및 MCP 서버 통합"""
    
    async def analyze_design(self, figma_url: str) -> Dict:
        """디자인 종합 분석"""
        # 1. 디자인 컴포넌트 추출
        # 2. 레이아웃 구조 분석
        # 3. 사용자 플로우 파악
        # 4. 기능 요구사항 도출
        # 5. 디자인 시스템 준수도 검사
        
    async def extract_design_specs(self, node_id: str) -> Dict:
        """상세 디자인 스펙 추출"""
        # 1. 디자인 시스템 정보 (색상, 폰트, 스타일)
        # 2. 반응형 규칙 및 브레이크포인트
        # 3. 접근성 요구사항 분석
        # 4. 인터랙션 및 애니메이션 정의
        
    async def monitor_design_changes(self, file_id: str) -> List[Dict]:
        """디자인 변경사항 모니터링"""
        # 1. 버전 비교 분석
        # 2. 변경 영향도 평가
        # 3. 관련 문서 업데이트 필요성 판단
```

### 4.3 Notion 통합 모듈

```python
class NotionIntegration:
    """Notion API 및 MCP 서버 통합"""
    
    async def create_prd(self, design_analysis: Dict) -> str:
        """PRD 자동 생성"""
        # 1. PRD 페이지 생성
        # 2. 요구사항 데이터베이스 연동
        # 3. 우선순위 매트릭스 생성
        # 4. 이해관계자 태그 및 할당
        
    async def update_roadmap(self, milestones: List[Dict]) -> str:
        """로드맵 업데이트"""
        # 1. 로드맵 데이터베이스 업데이트
        # 2. 진행 상황 추적 설정
        # 3. 의존성 관계 매핑
        # 4. 알림 및 리마인더 설정
        
    async def sync_design_changes(self, changes: Dict) -> List[str]:
        """디자인 변경사항 동기화"""
        # 1. 변경사항 문서화
        # 2. 이해관계자 자동 알림
        # 3. 영향도 분석 보고서 생성
        # 4. 후속 액션 아이템 생성
```

---

## 🚀 개발 로드맵

### Phase 1: 기본 인프라 구축 (1-2주)
- [ ] **MCP 서버 환경 설정**
  - Figma Dev Mode MCP Server 설치 및 테스트
  - Notion MCP Server 설치 및 API 연동 확인
  - 로컬 개발 환경 구축

- [ ] **기본 Agent 클래스 구현**
  - `EnterpriseAgentTemplate` 상속 구조 구현
  - 공통 모듈 적용 (`common/` 디렉토리 활용)
  - 기본 설정 파일 생성 (`config.py`, MCP 설정)

- [ ] **API 연결 검증**
  - Figma API 키 발급 및 테스트
  - Notion 통합 계정 설정 및 권한 확인
  - 기본 CRUD 작업 테스트

### Phase 2: 핵심 기능 개발 (2-3주)
- [ ] **디자인 분석 엔진 구현**
  - Figma 컴포넌트 파싱 로직
  - 사용자 플로우 자동 감지
  - 디자인 패턴 분석 알고리즘

- [ ] **PRD 자동 생성 기능**
  - 템플릿 기반 문서 생성
  - 요구사항 자동 추출 및 분류
  - 우선순위 매트릭스 생성

- [ ] **Notion 템플릿 개발**
  - PRD 템플릿 구조 설계
  - 기술 스펙 템플릿 개발
  - 로드맵 데이터베이스 스키마 정의

- [ ] **기본 워크플로우 구현**
  - 디자인 → 분석 → 문서화 파이프라인
  - 오류 처리 및 예외 상황 대응
  - 기본 로깅 및 모니터링

### Phase 3: 고급 기능 추가 (2-3주)
- [ ] **로드맵 자동 생성**
  - 복잡도 기반 일정 추정
  - 의존성 분석 및 순서 최적화
  - 리스크 요인 식별 및 완충 시간 계산

- [ ] **실시간 싱크 기능**
  - 디자인 변경 감지 시스템
  - 자동 문서 업데이트 메커니즘
  - 변경사항 알림 시스템

- [ ] **영향도 분석 도구**
  - 디자인 변경의 개발 영향도 평가
  - 관련 문서 자동 식별
  - 업데이트 우선순위 결정

- [ ] **협업 기능 강화**
  - 이해관계자 자동 태그
  - 코멘트 및 피드백 수집
  - 승인 워크플로우 통합

### Phase 4: 최적화 및 배포 (1-2주)
- [ ] **성능 최적화**
  - API 호출 최적화 (배치 처리, 캐싱)
  - 메모리 사용량 최적화
  - 처리 속도 개선

- [ ] **오류 처리 강화**
  - 포괄적인 예외 처리
  - 자동 재시도 메커니즘
  - 장애 복구 시나리오

- [ ] **문서화 완성**
  - API 문서 자동 생성
  - 사용자 가이드 작성
  - 트러블슈팅 가이드

- [ ] **테스트 케이스 작성**
  - 단위 테스트 (Unit Test)
  - 통합 테스트 (Integration Test)
  - 엔드투엔드 테스트 (E2E Test)

---

## 📦 개발 자원 명세

### 5.1 의존성 패키지
```toml
# pyproject.toml
[project]
dependencies = [
    "mcp-agent>=0.0.21",
    "figma-api>=1.0.0",
    "notion-client>=2.3.0", 
    "jinja2>=3.1.0",
    "pandas>=2.0.0",
    "pydantic>=2.0.0",
    "httpx>=0.25.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0"
]
```

### 5.2 설정 파일

#### MCP 서버 설정 (`mcp_agent.config.yaml`)
```yaml
mcp:
  servers:
    figma-dev-mode:
      command: "figma-dev-mode-server"
      transport: "sse" 
      url: "http://127.0.0.1:3845/sse"
      description: "Figma 디자인 분석 및 메타데이터 추출"
      
    notion-api:
      command: "npx"
      args: ["-y", "@notionhq/notion-mcp-server"]
      env:
        OPENAPI_MCP_HEADERS: '{"Authorization": "Bearer ${NOTION_API_KEY}", "Notion-Version": "2022-06-28"}'
      description: "Notion 문서 생성 및 관리"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
execution:
  timeout: 300  # 5분
  max_retries: 3
  batch_size: 10
```

#### 시크릿 설정 (`mcp_agent.secrets.yaml`) 
```yaml
# 이 파일은 .gitignore에 포함되어야 함
figma:
  api_key: "fig_****"
  access_token: "****"
  personal_access_token: "****"
  
notion:
  api_key: "ntn_****"
  integration_secret: "****"
  workspace_id: "****"

# 선택적 설정
openai:
  api_key: "${OPENAI_API_KEY}"  # LLM 기능 사용시 - 환경변수에서 로드
```

### 5.3 Agent 전용 설정 (`config.py`)
```python
from typing import List, Dict
from common.config import DEFAULT_SERVERS, DEFAULT_COMPANY_NAME

# Product Planner Agent 전용 설정
PRODUCT_PLANNER_SERVERS = [
    "figma-dev-mode",
    "notion-api", 
    "filesystem"
]

# 템플릿 설정
PRD_TEMPLATE_CONFIG = {
    "sections": [
        "executive_summary",
        "problem_statement", 
        "solution_overview",
        "user_stories",
        "technical_requirements",
        "success_metrics",
        "timeline_milestones"
    ],
    "required_fields": [
        "product_name",
        "target_audience",
        "key_features",
        "success_criteria"
    ]
}

ROADMAP_CONFIG = {
    "phases": ["discovery", "design", "development", "testing", "launch"],
    "estimation_factors": {
        "complexity_multiplier": 1.5,
        "risk_buffer": 0.2,
        "integration_overhead": 0.3
    }
}

# Notion 데이터베이스 스키마
NOTION_DATABASE_SCHEMAS = {
    "requirements": {
        "Name": {"type": "title"},
        "Priority": {"type": "select", "options": ["High", "Medium", "Low"]},
        "Status": {"type": "select", "options": ["New", "In Progress", "Done"]},
        "Assignee": {"type": "people"},
        "Due Date": {"type": "date"},
        "Figma Link": {"type": "url"}
    },
    "roadmap": {
        "Milestone": {"type": "title"},
        "Phase": {"type": "select", "options": ["Discovery", "Design", "Development", "Testing", "Launch"]},
        "Start Date": {"type": "date"},
        "End Date": {"type": "date"},
        "Dependencies": {"type": "relation"},
        "Progress": {"type": "number"}
    }
}
```

---

## 📊 성공 지표 및 KPI

### 6.1 기능적 지표
- **Figma 디자인 분석 정확도**: >90%
  - 컴포넌트 인식률
  - 플로우 파악 정확도
  - 요구사항 추출 완성도

- **PRD 자동 생성 완성도**: >85%
  - 필수 섹션 포함률
  - 내용 관련성 점수
  - 수정 필요 빈도

- **Notion 동기화 성공률**: >95%
  - API 호출 성공률
  - 데이터 일관성 유지
  - 실시간 업데이트 지연시간

- **처리 성능**: 중간 복잡도 디자인 기준
  - 디자인 분석: <30초
  - PRD 생성: <60초
  - 전체 워크플로우: <5분

### 6.2 비즈니스 지표
- **업무 효율성 개선**
  - 기획 문서 작성 시간 단축: 70%
  - 디자인-기획 간 일관성 향상: 80%
  - 협업 효율성 증대: 60%

- **품질 개선**
  - 요구사항 누락율 감소: 50%
  - 디자인-개발 간 괴리 감소: 60%
  - 문서 업데이트 지연 감소: 80%

### 6.3 기술적 지표
- **시스템 안정성**
  - 가동 시간(Uptime): >99%
  - 오류 발생률: <1%
  - 평균 복구 시간(MTTR): <10분

- **확장성**
  - 동시 사용자 지원: 10명+
  - 프로젝트 수 처리: 100개+
  - 월간 처리량: 1000건+

---

## 🎯 다음 단계 액션 아이템

### 즉시 실행 (오늘 내)
- [ ] Figma 개발자 계정 생성 및 API 키 발급
- [ ] Notion 통합 계정 설정 및 워크스페이스 권한 확인
- [ ] 기존 srcs 코드베이스 분석 및 패턴 학습

### 1주 내 완료
- [ ] 로컬 개발 환경에 MCP 서버 설치 및 테스트
- [ ] `product_planner_agent/` 폴더 구조 생성
- [ ] 기본 Agent 클래스 스켈레톤 코드 작성
- [ ] 샘플 Figma 디자인 및 Notion 페이지로 POC 구현

### 2주 내 완료
- [ ] 첫 번째 워킹 프로토타입 완성
  - 기본 디자인 분석 기능
  - 간단한 PRD 생성 기능
  - Notion 페이지 생성 기능
- [ ] 공통 모듈 적용 및 기존 패턴 준수 확인
- [ ] 기본 테스트 케이스 작성

### 1개월 내 완료
- [ ] 핵심 기능 완전 구현
- [ ] 실제 프로젝트 데이터로 테스트
- [ ] 사용자 피드백 수집 및 개선
- [ ] 문서화 및 배포 준비

---

## 📝 참고 자료

### 기술 문서
- [mcp-agent GitHub Repository](https://github.com/lastmile-ai/mcp-agent)
- [Figma Dev Mode MCP Server Guide](https://help.figma.com/hc/en-us/articles/32132100833559-Guide-to-the-Dev-Mode-MCP-Server)
- [Notion MCP Server Documentation](https://developers.notion.com/docs/mcp)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)

### 참조 구현체
- `srcs/business_strategy_agents/notion_integration.py` - Notion 통합 참조
- `srcs/common/templates.py` - Agent 템플릿 패턴
- `srcs/basic_agents/researcher_v2.py` - 공통 모듈 활용 예시

### 외부 리소스
- [Framelink Figma MCP Server](https://github.com/GLips/Figma-Context-MCP)
- [Community Notion MCP Servers](https://github.com/ccabanillas/notion-mcp)
- [Figma API Documentation](https://www.figma.com/developers/api)
- [Notion API Documentation](https://developers.notion.com/)

---

**문서 버전**: v1.0  
**최종 수정**: 2025년 1월 15일  
**작성자**: MCP Agent Development Team  
**검토자**: Enterprise Architecture Team  

> 이 문서는 살아있는 문서(Living Document)로서 프로젝트 진행에 따라 지속적으로 업데이트됩니다. 