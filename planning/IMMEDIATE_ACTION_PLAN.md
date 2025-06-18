# 🚨 MCP Agent 프로젝트 즉시 행동 계획

**📅 작성 일자**: 2025년 06월 18일  
**⏰ 긴급도**: CRITICAL  
**🎯 목표**: 최소 1개 Agent라도 실제 사용 가능하게 만들기

---

## 🔍 **현재 상황 요약**

### **✅ 확인된 사실들**
1. **MockDecisionAgent 삭제 완료**: ✅ 가짜 Agent 제거됨
2. **Streamlit 설치됨**: ✅ 버전 1.45.1 확인
3. **uvx 설치됨**: ✅ 버전 0.7.13 확인  
4. **Node.js 미설치**: ❌ MCP 서버들 실행 불가

### **🚨 핵심 문제**
- **MCP 서버 인프라 완전 부재**: Node.js 미설치로 모든 MCP 서버 사용 불가
- **실제 사용 가능한 Agent 0개**: 사용자가 실제로 사용할 수 있는 기능 없음
- **Mock/Fallback 코드 50+ 파일**: 실제 기능 대신 임시 코드 의존

---

## 🎯 **즉시 실행 계획 (오늘)**

### **Phase 1: 기본 인프라 구축 (1시간)**

#### **1.1 Node.js 설치 (20분)**
```bash
# Ubuntu에서 Node.js 설치
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 설치 확인
node --version
npm --version
```

#### **1.2 핵심 MCP 서버 설치 (30분)**
```bash
# 파일시스템 MCP 서버 (가장 기본)
npm install -g @modelcontextprotocol/server-filesystem

# Google 검색 MCP 서버
npm install -g g-search-mcp

# Fetch MCP 서버 (Python 기반)
uvx install mcp-server-fetch

# 설치 확인
npx @modelcontextprotocol/server-filesystem --help
npx g-search-mcp --help
```

#### **1.3 MCP 서버 연동 테스트 (10분)**
```bash
# 기본 연동 테스트
cd /home/user/workspace/mcp_agent
python3 -c "
import yaml
with open('configs/mcp_agent.config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('MCP 서버 설정:', list(config['mcp']['servers'].keys()))
"
```

### **Phase 2: Product Planner Agent 실제 테스트 (30분)**

#### **2.1 Streamlit 앱 실행 테스트**
```bash
# Product Planner 페이지 실행
streamlit run pages/product_planner.py --server.headless=true --server.port=8501

# 별도 터미널에서 기본 접근성 확인
curl -I http://localhost:8501
```

#### **2.2 Agent 기본 기능 테스트**
```python
# 간단한 기능 테스트 스크립트
import sys
sys.path.append('.')

from srcs.product_planner_agent.agents.coordinator_agent import CoordinatorAgent
from srcs.common.config import get_orchestrator

# 기본 Agent 생성 테스트
try:
    orchestrator = get_orchestrator()
    agent = CoordinatorAgent(orchestrator)
    print("✅ CoordinatorAgent 생성 성공")
except Exception as e:
    print(f"❌ Agent 생성 실패: {e}")
```

### **Phase 3: 실제 문제 발견 및 수정 (30분)**

#### **3.1 Import 에러 수정**
```bash
# 모든 페이지 import 테스트
for page in product_planner urban_hive seo_doctor decision_agent finance_health; do
    echo "Testing $page..."
    python3 -c "import sys; sys.path.append('.'); import pages.$page; print('✅ $page OK')" 2>&1 || echo "❌ $page FAILED"
done
```

#### **3.2 실제 사용자 시나리오 테스트**
- Product Planner에서 간단한 작업 입력
- ReAct 패턴 실행 확인
- 결과 파일 생성 확인

---

## 📋 **내일까지 목표 (2024년 12월 19일)**

### **🎯 Product Planner 완전 작동**
1. ✅ MCP 서버 인프라 구축 완료
2. ✅ Streamlit UI 정상 작동 확인
3. ✅ ReAct 패턴 실제 실행 성공
4. ✅ 실제 Figma URL 분석 1회 성공
5. ✅ PRD 파일 생성 및 저장 성공

### **🎯 사용자 테스트 성공**
- **시나리오**: "모바일 앱 MVP 기획서 작성"
- **입력**: Figma URL 또는 간단한 앱 아이디어
- **결과**: 실제 사용 가능한 PRD 문서 파일 생성

---

## ⚠️ **실패 시 대안 계획**

### **Plan B: Node.js 설치 실패 시**
```bash
# Node.js 없이 Python MCP 서버만 사용
uvx install mcp-server-fetch
python3 -m uvicorn srcs.urban_hive.providers.urban_hive_mcp_server:app --port 8002
```

### **Plan C: MCP 서버 연동 실패 시**
- Product Planner Agent를 일반 AI Agent로 동작
- 외부 API 직접 호출 방식으로 임시 구현
- 최소한 Streamlit UI는 작동하게 만들기

---

## 🔄 **실시간 진행 상황 추적**

### **오늘 (2024년 12월 18일) 체크리스트**
- [ ] MockDecisionAgent 삭제 ✅ **완료**
- [ ] Node.js 설치
- [ ] MCP 서버 설치
- [ ] Product Planner Streamlit 테스트
- [ ] 실제 ReAct 패턴 실행 테스트

### **진행 중 발견된 문제들**
1. **Node.js 미설치 확인**: `sudo apt install nodejs` 필요
2. **WSL 환경**: 브라우저 접근 제한으로 headless 모드 테스트 필요
3. **MCP 서버 의존성**: 대부분 Agent가 MCP 서버 없이는 동작 불가

---

## 💡 **성공 기준**

### **최소 성공 기준 (오늘)**
- ✅ Product Planner 페이지가 에러 없이 로드됨
- ✅ Agent 1개 기능이 실제로 실행됨
- ✅ 결과 파일이 생성됨

### **이상적 성공 기준 (내일까지)**
- ✅ MCP 서버 연동으로 실제 외부 데이터 수집
- ✅ 사용자가 입력한 내용을 바탕으로 의미있는 결과 생성
- ✅ End-to-End 사용자 시나리오 1개 완전 성공

---

**🚀 다음 작업: Node.js 설치부터 시작**
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
``` 