import json
import os
from typing import Dict
from datetime import datetime

from ..state import AgentState

def auditor_node(state: AgentState) -> Dict:
    """
    감사 노드: 하루의 모든 활동을 요약하여 JSON 파일로 저장합니다.
    """
    print("--- AGENT: Auditor ---")
    log_message = "하루 활동 감사를 시작하고 보고서를 생성합니다."
    state["log"].append(log_message)
    print(log_message)

    # 보고서 데이터 구성
    report = {
        "date": state.get("date"),
        "risk_profile": state.get("risk_profile"),
        "target_tickers": state.get("target_tickers"),
        "technical_analysis": state.get("technical_analysis"),
        "news_data": state.get("news_data"),
        "sentiment_analysis": state.get("sentiment_analysis"),
        "market_outlook": state.get("market_outlook"),
        "investment_plan": state.get("investment_plan"),
        "trade_results": state.get("trade_results"),
        "daily_pnl": state.get("daily_pnl"),
        "full_log": state.get("log")
    }

    # 데이터 폴더 경로 설정
    # 이 파일의 위치(agents/)를 기준으로 상위 폴더(financial_agent/) 밑의 data/ 폴더를 찾음
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # data 폴더가 없으면 생성
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # 파일 저장
    report_filename = f"daily_report_{state.get('date')}.json"
    report_filepath = os.path.join(data_folder, report_filename)

    try:
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        log_message = f"성공적으로 보고서를 저장했습니다: {report_filepath}"
        print(log_message)
        state["log"].append(log_message)

    except Exception as e:
        error_message = f"보고서 저장 중 에러 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

    # 감사 노드는 상태를 변경하지 않고 종료되므로 빈 딕셔너리 반환
    return {} 