# Table Game Mate

LangGraph 기반 멀티 에이전트 보드게임 플랫폼

## 프로젝트 구조

```
table_game_mate/
├── agents.py              # 통합 에이전트 시스템 (3개 에이전트)
├── core.py                # 통합 코어 시스템 (상태관리, 에러처리, 게임엔진)
├── tests.py               # 통합 테스트 시스템
├── dashboard.py           # Streamlit 대시보드
├── data/                  # 데이터 저장소
│   ├── game_data.json     # 게임 데이터
│   └── logs/              # 로그 파일
├── main.py               # 메인 진입점
├── requirements.txt      # 의존성
└── README.md            # 프로젝트 문서
```

## 설치 및 실행

```bash
pip install -r requirements.txt
python main.py
```

## 웹 대시보드 실행

```bash
streamlit run dashboard.py
```

## 테스트 실행

```bash
pytest tests.py
```

## 특징

- **통합 파일 구조**: 최소한의 파일로 모든 기능 구현
- **에이전트 중심 설계**: 각 기능이 독립적인 에이전트로 구현
- **LangGraph 패턴**: LangGraph의 모범 사례를 따르는 구조
- **No Fallback**: 명확한 에러 처리, fallback 없음
- **확장 가능**: 새로운 에이전트와 게임 쉽게 추가
