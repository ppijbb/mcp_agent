from langgraph.graph import StateGraph, END, START
from .state import AgentState
# agents 패키지에서 모든 노드를 한 번에 임포트
from .agents import (
    market_data_collector_node,
    news_sentiment_analyzer_node,
    aggregator_node,
    chief_strategist_node,
    portfolio_manager_node,
    trader_node,
    auditor_node,
)

class FinancialAgentWorkflow:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        에이전트 워크플로우를 정의하고 그래프를 빌드합니다.
        데이터 집계 노드를 추가하여 병렬 실행 동기화를 보장합니다.
        """
        workflow = StateGraph(AgentState)

        # 1. 노드 추가
        workflow.add_node("market_data_collector", market_data_collector_node)
        workflow.add_node("news_sentiment_analyzer", news_sentiment_analyzer_node)
        workflow.add_node("aggregator", aggregator_node)
        workflow.add_node("chief_strategist", chief_strategist_node)
        workflow.add_node("portfolio_manager", portfolio_manager_node)
        workflow.add_node("trader", trader_node)
        workflow.add_node("auditor", auditor_node)

        # 2. 엣지 연결
        workflow.add_edge(START, "market_data_collector")
        workflow.add_edge(START, "news_sentiment_analyzer")

        # 데이터 수집 노드 -> 집계 노드
        workflow.add_edge("market_data_collector", "aggregator")
        workflow.add_edge("news_sentiment_analyzer", "aggregator")
        
        # 집계 노드 -> 전략가 노드
        workflow.add_edge("aggregator", "chief_strategist")
        
        workflow.add_edge("chief_strategist", "portfolio_manager")
        workflow.add_edge("portfolio_manager", "trader")
        
        # Trader -> Auditor -> END
        workflow.add_edge("trader", "auditor")
        workflow.add_edge("auditor", END)

        # 3. 그래프 컴파일
        return workflow.compile()

    def run(self, initial_state: AgentState):
        """
        워크플로우를 실행하고 최종 상태를 반환합니다.
        """
        return self.graph.invoke(initial_state)

# 이 파일이 직접 실행될 때 테스트를 위한 코드
if __name__ == "__main__":
    from datetime import datetime

    # 워크플로우 인스턴스 생성
    workflow_runner = FinancialAgentWorkflow()

    # 분석할 대상을 동적으로 지정
    target_tickers = ["NVDA", "AMD", "QCOM"] 
    
    # 초기 상태 정의
    initial_state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "risk_profile": "aggressive", # "conservative", "moderate", "aggressive"
        "target_tickers": target_tickers, # 동적으로 티커 리스트 전달
        "log": [],
        "technical_analysis": {}, # None 대신 빈 dict로 초기화
        "sentiment_analysis": {}, # None 대신 빈 dict로 초기화
        "market_outlook": None,
        "investment_plan": None,
        "trade_results": None,
        "daily_pnl": None,
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