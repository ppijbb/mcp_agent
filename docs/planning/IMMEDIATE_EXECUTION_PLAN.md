# 🚀 MCP Agent 프로젝트 즉시 실행 계획

**📅 작성 일자**: 2025년 06월 19일  
**🎯 목표**: 2-3시간 내 실제 사용 가능한 MCP Agent 시스템 구축  
**⚡ 현재 상태**: 코드 준비 완료, 인프라만 구축하면 즉시 사용 가능

---

## 🔥 **Phase 1: MCP 서버 인프라 구축 (30분)**

### **Step 1.1: Node.js 환경 설치 (10분)**
```bash
# Ubuntu/WSL 환경에서 Node.js 최신 LTS 설치
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 설치 확인
node --version
npm --version
```

### **Step 1.2: 핵심 MCP 서버 설치 (15분)**
```bash
# 파일시스템 서버 (필수 - 결과 저장용)
npm install -g @modelcontextprotocol/server-filesystem

# 구글 검색 서버 (연구 및 데이터 수집용)
npm install -g g-search-mcp

# Puppeteer 서버 (웹 스크래핑 및 SEO 분석용)
npm install -g @modelcontextprotocol/server-puppeteer

# 설치 확인
npx @modelcontextprotocol/server-filesystem --help
npx g-search-mcp --help
npx @modelcontextprotocol/server-puppeteer --help
```

### **Step 1.3: Python MCP 도구 설치 (5분)**
```bash
# 웹 페치 서버 (API 호출용)
pip install uvicorn
uvx install mcp-server-fetch

# 설치 확인
uvx mcp-server-fetch --help
```

---

## 🎯 **Phase 2: Product Planner Agent 완전 구동 (1시간)**

### **Step 2.1: Streamlit 서버 실행 (5분)**
```bash
cd /home/user/workspace/mcp_agent
streamlit run pages/product_planner.py --server.port 8501 --server.headless true
```

### **Step 2.2: MCP 서버 연동 테스트 (30분)**

#### **테스트 시나리오 1: 기본 파일 저장**
1. **입력**: "Create a simple mobile app PRD"
2. **예상 동작**: 
   - CoordinatorAgent가 ReAct 패턴으로 작업 분석
   - PRDWriterAgent가 문서 생성
   - Filesystem MCP 서버를 통해 파일 저장
3. **성공 기준**: `product_planner_reports/` 폴더에 PRD 파일 생성

#### **테스트 시나리오 2: Figma URL 분석 (Mock)**
1. **입력**: "https://www.figma.com/design/example"
2. **예상 동작**: 
   - FigmaAnalyzerAgent가 URL 분석 (Mock 데이터 사용)
   - BusinessPlannerAgent가 비즈니스 계획 생성
   - 최종 결과물 통합 리포트 생성
3. **성공 기준**: 완전한 워크플로우 실행 및 결과 파일 저장

### **Step 2.3: 결과 검증 (15분)**
```bash
# 생성된 파일 확인
ls -la product_planner_reports/
cat product_planner_reports/latest_prd.md

# 로그 확인
tail -f logs/mcp-agent-*.jsonl
```

### **Step 2.4: 문제 해결 (10분)**
**예상 문제들과 해결책:**
- **MCP 서버 연결 실패**: config 파일의 서버 경로 수정
- **파일 저장 권한 오류**: 출력 디렉토리 권한 확인
- **Agent 메소드 호출 실패**: 메소드명 오타 수정

---

## 🚀 **Phase 3: Urban Hive Agent 안정화 (1시간)**

### **Step 3.1: Streamlit 경고 해결 (20분)**
```bash
# Urban Hive 페이지 import 테스트
python -c "
try:
    import pages.urban_hive
    print('✅ Urban Hive import success')
except Exception as e:
    print(f'❌ Import error: {e}')
"

# 경고 발생 시 pages/urban_hive.py 수정
# - ScriptRunContext 경고 무시 설정 추가
# - Session state 초기화 코드 추가
```

### **Step 3.2: Urban Hive MCP 서버 실행 (15분)**
```bash
# Urban Hive 자체 MCP 서버 실행 (별도 터미널)
python -m uvicorn srcs.urban_hive.providers.urban_hive_mcp_server:app --port 8002 --reload

# 서버 상태 확인
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/docs
```

### **Step 3.3: Urban Hive 실제 분석 테스트 (20분)**
```bash
# Streamlit 서버 실행
streamlit run pages/urban_hive.py --server.port 8502
```

#### **테스트 시나리오: 서울시 교통 분석**
1. **입력**: 
   - 도시: "Seoul"
   - 분석 카테고리: "Traffic Flow Analysis"
   - 시간 범위: "24h"
2. **예상 동작**:
   - UrbanHiveMCPAgent가 ReAct 패턴으로 분석 수행
   - 자체 MCP 서버에서 도시 데이터 조회
   - 분석 결과 리포트 생성
3. **성공 기준**: `urban_hive_reports/` 폴더에 분석 리포트 생성

### **Step 3.4: 결과 검증 (5분)**
```bash
ls -la urban_hive_reports/
cat urban_hive_reports/latest_analysis.json
```

---

## ⚡ **Phase 4: SEO Doctor Agent 완성 (1시간)**

### **Step 4.1: SEO Doctor 페이지 실행 (10분)**
```bash
streamlit run pages/seo_doctor.py --server.port 8503
```

### **Step 4.2: Lighthouse 연동 테스트 (30분)**

#### **테스트 시나리오: 실제 웹사이트 SEO 분석**
1. **입력**: "https://example.com"
2. **예상 동작**:
   - SEODoctorMCPAgent가 ReAct 패턴으로 분석 계획 수립
   - Puppeteer MCP 서버를 통해 웹사이트 접근
   - Lighthouse 분석 실행 (또는 Mock 분석)
   - SEO 개선 제안 생성
3. **성공 기준**: 완전한 SEO 분석 리포트 생성

### **Step 4.3: Mock 데이터 제거 (15분)**
**예상 수정 사항:**
```python
# srcs/seo_doctor/seo_doctor_agent.py에서
# Mock 데이터 반환 코드를 실제 MCP 서버 호출로 변경

async def analyze_seo(self, url: str):
    # 기존: return {"score": 85, "status": "mock"}
    # 변경: 실제 Puppeteer MCP 서버 호출
    result = await self.orchestrator.run_agent_method(
        "puppeteer", "analyze_page", {"url": url}
    )
    return result
```

### **Step 4.4: 결과 검증 (5분)**
```bash
ls -la seo_reports/
cat seo_reports/latest_analysis.json
```

---

## 🎉 **Phase 5: 통합 테스트 및 검증 (30분)**

### **Step 5.1: 3개 Agent 동시 실행 (10분)**
```bash
# 3개 터미널에서 동시 실행
Terminal 1: streamlit run pages/product_planner.py --server.port 8501
Terminal 2: streamlit run pages/urban_hive.py --server.port 8502  
Terminal 3: streamlit run pages/seo_doctor.py --server.port 8503
```

### **Step 5.2: End-to-End 사용자 시나리오 테스트 (15분)**

#### **시나리오 1: 신규 앱 개발 프로젝트**
1. **Product Planner**: "모바일 음식 배달 앱" PRD 생성
2. **Urban Hive**: "서울 강남구" 배달 시장 분석
3. **SEO Doctor**: 경쟁사 웹사이트 "https://baemin.com" SEO 분석

#### **시나리오 2: 부동산 서비스 기획**
1. **Product Planner**: "부동산 중개 플랫폼" 사업 계획
2. **Urban Hive**: "부산 해운대구" 부동산 시장 분석
3. **SEO Doctor**: "https://zigbang.com" SEO 벤치마킹

### **Step 5.3: 최종 검증 체크리스트 (5분)**

#### **✅ 성공 기준**
- [ ] **Node.js/NPM**: 정상 설치 및 버전 확인
- [ ] **MCP 서버**: 3개 이상 서버 정상 설치
- [ ] **Product Planner**: 완전한 PRD 생성 및 파일 저장
- [ ] **Urban Hive**: 도시 분석 리포트 생성
- [ ] **SEO Doctor**: 웹사이트 분석 리포트 생성
- [ ] **파일 저장**: 각 Agent별 결과물 파일 생성 확인
- [ ] **에러 없음**: 3개 Agent 모두 에러 없이 완전 실행
- [ ] **MCP 서버 연동**: filesystem 서버 통한 실제 파일 저장 확인

#### **🚀 성공 시 달성 상태**
- **MCP Agent 정체성**: ✅ **완전히 입증됨**
- **사용자 가치**: ✅ **즉시 제공 가능**
- **Demo 가능성**: ✅ **완전한 End-to-End Demo 가능**

---

## 🛠️ **문제 해결 가이드**

### **예상 문제 1: Node.js 설치 실패**
```bash
# 권한 문제 시
sudo chown -R $(whoami) ~/.npm

# 방화벽 문제 시  
sudo ufw allow 8501:8503/tcp
```

### **예상 문제 2: MCP 서버 연결 실패**
```bash
# 설정 파일 확인
cat configs/mcp_agent.config.yaml

# 포트 사용 확인
netstat -tlnp | grep :850
```

### **예상 문제 3: Streamlit 페이지 로드 실패**
```bash
# Python path 확인
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 가상환경 확인
which python
pip list | grep streamlit
```

### **예상 문제 4: 권한 오류**
```bash
# 출력 디렉토리 권한 설정
mkdir -p product_planner_reports urban_hive_reports seo_reports
chmod 755 product_planner_reports urban_hive_reports seo_reports
```

---

## 📊 **성공 지표**

### **단계별 성공 기준**
1. **Phase 1**: `node --version && npm --version` 성공
2. **Phase 2**: `ls product_planner_reports/` 에 파일 존재
3. **Phase 3**: `curl http://127.0.0.1:8002/health` 응답 성공
4. **Phase 4**: `ls seo_reports/` 에 분석 파일 존재
5. **Phase 5**: 3개 Agent 모두 에러 없이 완전 실행

### **최종 목표**
**"2-3시간 후: 실제 사용 가능한 MCP Agent 시스템 완성"**

---

**🚀 지금 바로 Phase 1부터 시작하세요!** 