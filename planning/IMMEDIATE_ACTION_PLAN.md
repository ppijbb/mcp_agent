# 🚨 MCP Agent 프로젝트 즉시 행동 계획

**📅 작성 일자**: 2024년 12월 18일  
**⏰ 긴급도**: CRITICAL  
**🎯 목표**: 실제 사용 가능한 Agent 완성에 집중

---

## 🔍 **현재 상황 재정의**

### **✅ 확인된 기본 인프라**
1. **Python 환경**: ✅ 설치 완료 (conda agent)
2. **Streamlit**: ✅ 버전 1.45.1 설치됨
3. **주요 Agent Import**: ✅ Product Planner, Urban Hive, SEO Doctor 성공
4. **MockDecisionAgent**: ✅ 완전 삭제됨

### **🚨 실제 문제 (MCP 인프라 제외)**
- **Mock/Fallback 코드 대량 잔존**: 50+ 파일에서 실제 기능 대신 임시 코드 사용
- **완성된 기능 부족**: 사용자가 실제로 사용할 수 있는 Agent 기능 없음
- **UI/UX 미완성**: Streamlit 페이지들이 실제로 작동하는지 불명
- **에러 처리 부족**: Agent 실행 시 에러 발생 가능성 높음

---

## 🎯 **우선순위별 즉시 행동 계획**

### **Phase 1: 1개 Agent라도 완전 작동하게 (오늘)**

#### **1.1 Product Planner Agent 완성 집중 (2시간)**

**왜 Product Planner인가?**
- ✅ ReAct 패턴 이미 구현됨
- ✅ Import 성공 확인됨  
- ✅ 상대적으로 완성도 높음

**구체적 작업:**
```bash
# 1. Streamlit 페이지 실제 테스트
streamlit run pages/product_planner.py --server.headless=true

# 2. Agent 기본 기능 테스트
python3 -c "
import sys
sys.path.append('srcs')
from product_planner_agent.agents.coordinator_agent import CoordinatorAgent
from common.config import get_orchestrator
orchestrator = get_orchestrator()
agent = CoordinatorAgent(orchestrator)
print('✅ Agent 생성 성공')
"
```

#### **1.2 Mock 코드 집중 제거 (1시간)**

**Product Planner 관련 Mock 코드만 우선 제거:**
```bash
# Mock/Fallback 코드 찾기
grep -r "fallback\|sample.*=\|mock" srcs/product_planner_agent/

# coordinator_agent.py의 샘플 URL 교체
# "https://www.figma.com/design/sample" → 실제 처리 로직
```

#### **1.3 실제 사용자 시나리오 테스트 (30분)**

**테스트 시나리오:**
1. Streamlit 페이지 접속
2. "모바일 앱 MVP 기획" 입력
3. Agent 실행 및 결과 확인
4. PRD 파일 생성 확인

### **Phase 2: Urban Hive Agent 안정화 (내일)**

#### **2.1 Mock 데이터 실제 데이터로 교체**
- `urban_hive/ai_text_analyzer.py`: Fallback 메소드 제거
- `urban_hive/config.py`: Sample 키 → 실제 설정
- `urban_hive/data_sources.py`: Fallback 메커니즘 정리

#### **2.2 실제 도시 분석 기능 구현**
- 서울시 공공데이터 API 연동
- 최소 1개 실제 데이터소스 연결
- 사용자 입력 → 실제 분석 결과 출력

### **Phase 3: 나머지 Agent 순차 안정화 (이번주)**

#### **3.1 SEO Doctor Agent**
- Lighthouse 연동 (로컬에서 가능한 방법)
- 실제 웹사이트 분석 기능
- Mock 분석 결과 제거

#### **3.2 Decision Agent**
- Sample 데이터 제거
- 실제 의사결정 로직 구현
- 사용자 프로필 기반 분석

---

## 📋 **구체적 작업 체크리스트**

### **오늘 (2024년 12월 18일) 필수 완료 항목**

#### **Product Planner 완성**
- [ ] `coordinator_agent.py` Mock URL 제거
- [ ] Streamlit 페이지 실제 실행 테스트
- [ ] ReAct 패턴 1회 실제 실행 성공
- [ ] PRD 파일 생성 기능 검증
- [ ] 에러 발생 시 적절한 처리 확인

#### **Mock 코드 제거 (우선순위)**
- [ ] `pages/decision_agent.py`: sample_history 제거
- [ ] `pages/finance_health.py`: 시뮬레이션 함수 정리
- [ ] `coordinator_agent.py`: 기본 URL 하드코딩 제거

#### **실제 테스트**
- [ ] Streamlit 앱 3개 페이지 로드 테스트
- [ ] 각 Agent 기본 기능 실행 테스트
- [ ] 에러 로그 수집 및 분석

### **내일 (2024년 12월 19일) 목표**
- [ ] Urban Hive Agent 완전 작동
- [ ] SEO Doctor Agent 기본 기능 구현
- [ ] 2-3개 Agent에서 실제 사용자 시나리오 성공

### **이번 주말 (2024년 12월 22일) 목표**
- [ ] 5개 핵심 Agent 모두 기본 사용 가능
- [ ] Mock/Fallback 코드 80% 이상 제거
- [ ] 실제 Demo 가능한 상태 달성

---

## 🔧 **실제 작업 순서**

### **Step 1: Product Planner 집중 완성 (지금 시작)**
```bash
# 1. 현재 상태 확인
cd /home/user/workspace/mcp_agent
python3 -c "
import sys
sys.path.append('srcs')
try:
    from product_planner_agent.agents.coordinator_agent import CoordinatorAgent
    print('✅ CoordinatorAgent import 성공')
except Exception as e:
    print(f'❌ Import 실패: {e}')
"

# 2. Streamlit 테스트
streamlit run pages/product_planner.py --server.headless=true --server.port=8501 &
sleep 5
curl -I http://localhost:8501 || echo "Streamlit 실행 실패"
```

### **Step 2: Mock 코드 우선순위별 제거**
```bash
# 가장 쉽게 제거 가능한 Mock 코드부터
grep -n "sample.*=" srcs/product_planner_agent/agents/coordinator_agent.py
grep -n "fallback" srcs/product_planner_agent/agents/
```

### **Step 3: 실제 기능 테스트**
- Product Planner에서 실제 작업 입력
- Agent 실행 과정 로그 확인
- 결과 파일 생성 여부 확인

---

## ⚠️ **실패 시 대안 계획**

### **Plan B: Streamlit 실행 실패 시**
- 직접 Python 스크립트로 Agent 테스트
- 최소한 Agent 클래스 실행은 성공시키기
- 결과를 텍스트 파일로 저장

### **Plan C: Agent 실행 실패 시**
- Import 에러부터 하나씩 해결
- 가장 단순한 기능부터 테스트
- Mock 코드라도 일단 실행되게 만들기

---

## 💡 **성공 기준 재정의**

### **최소 성공 기준 (오늘)**
- ✅ Product Planner Streamlit 페이지 에러 없이 로드
- ✅ CoordinatorAgent 기본 기능 1회 실행 성공
- ✅ 결과 출력 (파일 또는 콘솔) 확인

### **이상적 성공 기준 (내일)**
- ✅ Product Planner에서 실제 사용자 입력 처리
- ✅ Urban Hive에서 실제 도시 데이터 분석
- ✅ 2개 Agent가 실제 사용 가능한 수준 달성

### **완전 성공 기준 (주말)**
- ✅ 5개 Agent 모두 기본 Demo 가능
- ✅ Mock 코드 대신 실제 기능 구현
- ✅ 사용자가 실제로 가치를 느낄 수 있는 결과 제공

---

## 🚀 **지금 당장 시작할 작업**

1. **Product Planner Streamlit 테스트**
2. **Mock URL 제거 및 실제 로직 구현**
3. **ReAct 패턴 실제 실행 검증**

**다음 명령어부터 시작:**
```bash
cd /home/user/workspace/mcp_agent
streamlit run pages/product_planner.py --server.headless=true --server.port=8501
``` 