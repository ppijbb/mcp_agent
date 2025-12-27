import yfinance as yf
import pandas as pd
import json
import re
from typing import Dict, List, Any
from datetime import datetime
from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 절대 import 사용
from lang_graph.financial_agent.config import get_mcp_config, initialize_config, get_workflow_config
from lang_graph.financial_agent.graph import FinancialAgentWorkflow
from lang_graph.financial_agent.state import AgentState
from lang_graph.financial_agent.llm_client import initialize_llm_client, call_llm

# MCP 서버 초기화
mcp = FastMCP("FinancialTools")

# 설정 및 워크플로우 초기화 (한 번만)
_initialized = False
_workflow_runner = None

def _ensure_initialized():
    """설정 및 워크플로우 초기화 (한 번만 실행)"""
    global _initialized, _workflow_runner
    if not _initialized:
        try:
            initialize_config()
            initialize_llm_client()
            _workflow_runner = FinancialAgentWorkflow()
            _initialized = True
        except Exception as e:
            raise RuntimeError(f"Financial Agent 초기화 실패: {e}")

def _extract_tickers_and_risk_profile(user_query: str) -> Dict[str, Any]:
    """LLM을 사용하여 사용자 요청에서 티커와 리스크 프로필 추출"""
    try:
        workflow_config = get_workflow_config()
        valid_profiles = workflow_config.valid_risk_profiles
        default_tickers = workflow_config.default_tickers
        
        prompt = f"""
사용자 요청: {user_query}

위 요청에서 다음 정보를 추출하세요:
1. 주식 티커 심볼 (예: AAPL, MSFT, NVDA, TSLA 등) - 없으면 빈 리스트
2. 리스크 프로필 (conservative, moderate, aggressive 중 하나) - 없으면 "moderate"

출력 형식 (JSON only, 추가 텍스트 금지):
{{
    "tickers": ["TICKER1", "TICKER2"],
    "risk_profile": "conservative|moderate|aggressive"
}}

주의:
- 티커가 없으면 빈 리스트 반환
- 리스크 프로필이 없거나 불명확하면 "moderate" 사용
- 티커는 대문자로 정규화
- 유효한 리스크 프로필만 사용: {', '.join(valid_profiles)}
"""
        
        response = call_llm(prompt)
        
        # JSON 추출
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # JSON이 없으면 기본값 사용
            result = {
                "tickers": [],
                "risk_profile": "moderate"
            }
        
        # 검증 및 기본값 설정
        tickers = result.get("tickers", [])
        if not tickers:
            tickers = default_tickers
        
        risk_profile = result.get("risk_profile", "moderate")
        if risk_profile not in valid_profiles:
            risk_profile = "moderate"
        
        return {
            "tickers": [t.upper().strip() for t in tickers if t],
            "risk_profile": risk_profile
        }
    except Exception as e:
        # 추출 실패 시 기본값 사용
        workflow_config = get_workflow_config()
        return {
            "tickers": workflow_config.default_tickers,
            "risk_profile": "moderate"
        }

@mcp.tool()
def get_technical_indicators(ticker: str, period: str = None) -> Dict:
    """
    yfinance를 사용하여 특정 종목의 기술적 지표를 계산합니다.
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - 50일 이동평균
    """
    # 설정에서 기본 기간 가져오기
    if period is None:
        config = get_mcp_config()
        period = config.data_period
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        return {"error": f"No data found for ticker {ticker}"}

    close = hist['Close']
    
    # RSI 계산
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD 계산
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    
    # 50일 이동평균
    ma50 = close.rolling(window=50).mean()

    # NaN 값을 JSON 호환 가능한 None으로 변환
    return {
        "price": float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None,
        "rsi": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
        "macd": float(macd.iloc[-1]) if pd.notna(macd.iloc[-1]) else None,
        "moving_average_50": float(ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else None,
        "volume": int(hist['Volume'].iloc[-1]) if pd.notna(hist['Volume'].iloc[-1]) else None,
    }

@mcp.tool()
def get_market_news(ticker: str) -> List[Dict]:
    """
    yfinance를 사용하여 특정 종목에 대한 최신 뉴스를 가져옵니다.
    """
    stock = yf.Ticker(ticker)
    try:
        # yfinance의 news 속성은 딕셔너리 리스트를 반환합니다.
        news = stock.news
        if not news:
            return [{"title": "No recent news found.", "publisher": "System", "link": ""}]
        
        # 필요한 정보만 추출하여 반환
        return [
            {"title": item.get('title'), "publisher": item.get('publisher'), "link": item.get('link')}
            for item in news[:5] # 최신 5개 뉴스
        ]
    except Exception as e:
        return [{"title": f"An error occurred while fetching news: {e}", "publisher": "Error", "link": ""}]

@mcp.tool()
def get_ohlcv_data(ticker: str, period: str = None) -> Dict:
    """
    yfinance를 사용하여 특정 종목의 OHLCV(Open, High, Low, Close, Volume) 데이터를 수집합니다.
    
    Args:
        ticker: 주식 티커 심볼
        period: 데이터 기간 (예: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
                None인 경우 설정에서 기본 기간 사용
    
    Returns:
        Dict: OHLCV 데이터 리스트 및 메타데이터
    """
    # 설정에서 기본 기간 가져오기
    if period is None:
        config = get_mcp_config()
        period = config.data_period
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        return {"error": f"No data found for ticker {ticker}"}

    # OHLCV 데이터를 리스트로 변환 (시간순)
    ohlcv_list = []
    for idx, row in hist.iterrows():
        # timestamp를 초 단위로 변환
        timestamp = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(pd.Timestamp(idx).timestamp())
        
        ohlcv_list.append({
            "time": timestamp,
            "open": float(row['Open']) if pd.notna(row['Open']) else None,
            "high": float(row['High']) if pd.notna(row['High']) else None,
            "low": float(row['Low']) if pd.notna(row['Low']) else None,
            "close": float(row['Close']) if pd.notna(row['Close']) else None,
            "volume": int(row['Volume']) if pd.notna(row['Volume']) else None,
        })
    
    return {
        "ticker": ticker,
        "period": period,
        "data_count": len(ohlcv_list),
        "data": ohlcv_list,
        "latest_price": float(hist['Close'].iloc[-1]) if pd.notna(hist['Close'].iloc[-1]) else None,
        "latest_volume": int(hist['Volume'].iloc[-1]) if pd.notna(hist['Volume'].iloc[-1]) else None,
    }

@mcp.tool()
def run_financial_analysis(user_query: str) -> Dict[str, Any]:
    """
    사용자 요청을 받아 Financial Agent의 LangGraph 워크플로우를 실행하여 경제 지표 분석을 수행합니다.
    
    Args:
        user_query: 사용자의 경제/금융 관련 요청 (전체 쿼리)
    
    Returns:
        Dict: 경제 지표 분석 결과 (JSON)
    """
    try:
        # 초기화 확인
        _ensure_initialized()
        
        # 사용자 요청에서 티커와 리스크 프로필 추출
        extracted = _extract_tickers_and_risk_profile(user_query)
        target_tickers = extracted["tickers"]
        risk_profile = extracted["risk_profile"]
        
        if not target_tickers:
            return {
                "success": False,
                "error": "티커를 추출할 수 없습니다. 사용자 요청에 주식 티커 심볼을 포함해주세요.",
                "user_query": user_query
            }
        
        # 초기 상태 정의
        initial_state: AgentState = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "risk_profile": risk_profile,
            "target_tickers": target_tickers,
            "user_id": "mcp_user",
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
            # 차트 분석 필드
            "ohlcv_data": None,
            "chart_analysis": None,
            "chart_images": None,
            "technical_indicators_advanced": None,
            # 최종 지표 및 매도시점 추측 필드
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
        
        # 워크플로우 실행
        final_state = _workflow_runner.run(initial_state)
        
        # 결과 정리 (JSON 직렬화 가능한 형태로 변환)
        result = {
            "success": True,
            "user_query": user_query,
            "extracted_info": {
                "tickers": target_tickers,
                "risk_profile": risk_profile
            },
            "financial_analysis": final_state.get("financial_analysis"),
            "tax_optimization": final_state.get("tax_optimization"),
            "debt_management": final_state.get("debt_management"),
            "financial_goals": final_state.get("financial_goals"),
            "goal_progress": final_state.get("goal_progress"),
            "technical_analysis": final_state.get("technical_analysis", {}),
            "news_data": final_state.get("news_data", {}),
            "sentiment_analysis": final_state.get("sentiment_analysis"),
            "market_outlook": final_state.get("market_outlook"),
            "investment_plan": final_state.get("investment_plan"),
            "daily_pnl": final_state.get("daily_pnl"),
            "log": final_state.get("log", []),
            "error_message": final_state.get("error_message")
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Financial Agent 실행 중 오류 발생: {str(e)}",
            "user_query": user_query
        }

if __name__ == "__main__":
    # Stdio를 통해 서버 실행
    mcp.run() 