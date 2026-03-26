"""
Financial Agent LangGraph Workflow

This module defines the LangGraph workflow for the financial agent system.
The workflow implements a comprehensive financial management pipeline that includes:
- Financial analysis and budgeting
- Tax optimization
- Debt management
- Goal tracking
- Investment workflow (market data, news, chart analysis)
- Trading execution
- Commission calculation
- Audit reporting

The workflow uses a state graph architecture where each node represents
a specialized agent or processing step.
"""

from langgraph.graph import StateGraph, END, START
from .state import AgentState
from .config import get_workflow_config, initialize_config
from .llm_client import initialize_llm_client
# agents 패키지에서 모든 노드를 한 번에 임포트
from .agents import (
    market_data_collector_node,
    news_collector_node,
    news_analyzer_node,
    sync_node,
    chief_strategist_node,
    portfolio_manager_node,
    trader_node,
    auditor_node,
    financial_analyzer_node,
    tax_optimizer_node,
    debt_manager_node,
    goal_tracker_node,
    commission_calculator_node,
    chart_analyzer_node,
    technical_synthesizer_node,
    exit_point_predictor_node,
)

class FinancialAgentWorkflow:
    """
    LangGraph-based workflow for the financial agent system.
    
    This class manages the complete financial management pipeline including
    financial analysis, tax optimization, debt management, investment workflows,
    trading execution, and audit reporting.
    
    The workflow is built using LangGraph's StateGraph for flexible state
    management and node-based processing.
    
    Attributes:
        graph: Compiled LangGraph workflow
        
    Workflow Pipeline:
        1. Financial Analysis → Tax Optimization → Debt Management → Goal Tracking
        2. Investment Workflow: Market Data + News + Chart Analysis → Sync → Analysis → Strategy → Trading
        3. Commission Calculation → Audit Reporting
        
    Example:
        workflow = FinancialAgentWorkflow()
        result = workflow.run(initial_state)
    """
    
    def __init__(self):
        # LLM 클라이언트 초기화 (설정은 이미 초기화됨)
        try:
            initialize_llm_client()
            print("✅ Financial Agent 워크플로우 초기화 완료")
        except Exception as e:
            error_msg = f"워크플로우 초기화 실패: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph workflow.
        
        Defines the complete financial management pipeline as a state graph.
        The workflow connects the following stages:
        - Financial Management: Analyzer → Tax Optimizer → Debt Manager → Goal Tracker
        - Investment Workflow: Data Collectors → Sync → News Analyzer → Chief Strategist 
          → Technical Synthesizer → Exit Predictor → Portfolio Manager → Trader
        - Finalization: Commission Calculator → Auditor
        
        Returns:
            Compiled LangGraph workflow ready for execution
        """
        workflow = StateGraph(AgentState)

        # 1. 노드 추가 (재무 관리 단계)
        workflow.add_node("financial_analyzer", financial_analyzer_node)
        workflow.add_node("tax_optimizer", tax_optimizer_node)
        workflow.add_node("debt_manager", debt_manager_node)
        workflow.add_node("goal_tracker", goal_tracker_node)
        
        # 기존 투자 워크플로우 노드
        workflow.add_node("market_data_collector", market_data_collector_node)
        workflow.add_node("news_collector", news_collector_node)
        workflow.add_node("chart_analyzer", chart_analyzer_node)
        workflow.add_node("sync_data", sync_node)
        workflow.add_node("news_analyzer", news_analyzer_node)
        workflow.add_node("chief_strategist", chief_strategist_node)
        workflow.add_node("technical_synthesizer", technical_synthesizer_node)
        workflow.add_node("exit_point_predictor", exit_point_predictor_node)
        workflow.add_node("portfolio_manager", portfolio_manager_node)
        workflow.add_node("trader", trader_node)
        
        # 수수료 계산 및 감사 노드
        workflow.add_node("commission_calculator", commission_calculator_node)
        workflow.add_node("auditor", auditor_node)

        # 2. 엣지 연결
        # 재무 관리 단계
        workflow.add_edge(START, "financial_analyzer")
        workflow.add_edge("financial_analyzer", "tax_optimizer")
        workflow.add_edge("tax_optimizer", "debt_manager")
        workflow.add_edge("debt_manager", "goal_tracker")
        
        # 투자 워크플로우 단계
        workflow.add_edge("goal_tracker", "market_data_collector")
        workflow.add_edge("goal_tracker", "news_collector")
        workflow.add_edge("goal_tracker", "chart_analyzer")
        
        # 데이터 수집 노드 -> 동기화 노드
        workflow.add_edge("market_data_collector", "sync_data")
        workflow.add_edge("news_collector", "sync_data")
        workflow.add_edge("chart_analyzer", "sync_data")
        
        # 동기화 노드 -> 뉴스 분석가 -> 전략가 노드
        workflow.add_edge("sync_data", "news_analyzer")
        workflow.add_edge("news_analyzer", "chief_strategist")
        
        # 전략가 -> 기술적 지표 통합 -> 매도시점 추측 -> 포트폴리오 관리자
        workflow.add_edge("chief_strategist", "technical_synthesizer")
        workflow.add_edge("technical_synthesizer", "exit_point_predictor")
        workflow.add_edge("exit_point_predictor", "portfolio_manager")
        workflow.add_edge("portfolio_manager", "trader")
        
        # 수수료 계산 및 감사
        workflow.add_edge("trader", "commission_calculator")
        workflow.add_edge("commission_calculator", "auditor")
        workflow.add_edge("auditor", END)

        # 3. 그래프 컴파일
        return workflow.compile()

    def run(self, initial_state: AgentState):
        """
        워크플로우를 실행하고 최종 상태를 반환합니다.
        
        Args:
            initial_state: 초기 에이전트 상태
            
        Returns:
            최종 상태 또는 에러 메시지가 포함된 상태
        """
        try:
            return self.graph.invoke(initial_state)
        except Exception as e:
            error_msg = f"워크플로우 실행 중 에러 발생: {str(e)}"
            initial_state['error_message'] = error_msg
            initial_state['log'] = initial_state.get('log', []) + [error_msg]
            return initial_state

# 이 파일이 직접 실행될 때 테스트를 위한 코드
if __name__ == "__main__":
    import sys
    from datetime import datetime

    # 명령행 인자에서 설정값 읽기
    if len(sys.argv) < 3:
        print("사용법: python graph.py <tickers> <risk_profile>")
        print("예시: python graph.py 'NVDA,AMD,QCOM' aggressive")
        sys.exit(1)
    
    target_tickers = [ticker.strip() for ticker in sys.argv[1].split(',')]
    risk_profile = sys.argv[2]
    
    # 설정 초기화 및 유효한 리스크 프로필 검증
    try:
        initialize_config()
        workflow_config = get_workflow_config()
        valid_profiles = workflow_config.valid_risk_profiles
        
        if risk_profile not in valid_profiles:
            print(f"오류: 유효하지 않은 리스크 프로필입니다. 사용 가능한 값: {', '.join(valid_profiles)}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        sys.exit(1)

    # 워크플로우 인스턴스 생성 (설정 초기화 완료 후)
    workflow_runner = FinancialAgentWorkflow()
    
    # 초기 상태 정의
    initial_state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "risk_profile": risk_profile,
        "target_tickers": target_tickers,
        "user_id": "default_user",  # 재무 분석용 사용자 ID
        "log": [],
        # 재무 관리 단계 필드
        "financial_analysis": None,
        "budget_status": None,
        "savings_progress": None,
        "tax_optimization": None,
        "debt_management": None,
        "financial_goals": None,
        "goal_progress": None,
        # 투자 워크플로우 필드
        "technical_analysis": {},
        "news_data": {},
        # 차트 분석 필드 (신규)
        "ohlcv_data": None,
        "chart_analysis": None,
        "chart_images": None,
        "technical_indicators_advanced": None,
        # 최종 지표 및 매도시점 추측 필드 (신규)
        "synthesized_indicators": None,
        "exit_point_predictions": None,
        "sentiment_analysis": None,
        "market_outlook": None,
        "investment_plan": None,
        "trade_results": None,
        "daily_pnl": None,
        # 구조적 상업성 필드
        "commission_rate": None,
        "total_commission": None,
        "affiliate_commission": None,
        # 에러 필드
        "error_message": None,
    }

    print("🚀 지능형 금융 에이전트 워크플로우 시작 (LLM-Powered)")
    print(f"분석 대상: {initial_state['target_tickers']}")
    print(f"투자 성향: {initial_state['risk_profile']}")
    print("-" * 30)

    # 워크플로우 실행
    final_state = workflow_runner.run(initial_state)

    print("-" * 30)
    print("🏁 금융 에이전트 워크플로우 종료")
    print("\n최종 결과 요약:")
    print(f"  - 최종 손익 (PNL): ${final_state.get('daily_pnl', 0):.2f}")
    print("  - 실행된 거래 내역:")
    for trade in final_state.get("trade_results", []):
        print(f"    - {trade['action'].upper()}: {trade['ticker']} @ ${trade['price']} (수량: {trade['shares']})")
    
    print("\n상세 로그:")
    for log_entry in final_state.get("log", []):
        print(f"  - {log_entry}") 