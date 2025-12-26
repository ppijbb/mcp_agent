"""
차트 분석 에이전트 노드
OHLCV 데이터 수집, 고급 기술적 지표 계산, 차트 패턴 인식, LLM 기반 분석, 차트 이미지 생성
"""

from typing import Dict, Any
import pandas as pd
from ..state import AgentState
from ..mcp_client import call_ohlcv_data_tool
from ..tools.chart_tools import (
    calculate_technical_indicators,
    recognize_chart_patterns,
    analyze_chart_with_llm,
    generate_chart_image
)
from ..config import get_mcp_config


def chart_analyzer_node(state: AgentState) -> Dict:
    """
    차트 분석 노드: OHLCV 데이터를 수집하고 고급 기술적 분석을 수행합니다.
    """
    print("--- AGENT: Chart Analyzer (CandleView Style) ---")
    log_message = "고급 차트 분석을 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    tickers = state.get("target_tickers", [])
    if not tickers:
        error_message = "분석할 티커가 지정되지 않았습니다."
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}
    
    try:
        # MCP 설정에서 기간 가져오기
        mcp_config = get_mcp_config()
        period = mcp_config.data_period
        
        # OHLCV 데이터 수집
        print(f"OHLCV 데이터 수집 중: {tickers} (period: {period})")
        ohlcv_data_all = call_ohlcv_data_tool(tickers, period=period)
        
        # 결과 저장용 딕셔너리 초기화
        ohlcv_data_dict = {}
        chart_analysis_dict = {}
        chart_images_dict = {}
        technical_indicators_advanced_dict = {}
        
        # 각 티커별로 분석 수행
        for ticker in tickers:
            if ticker not in ohlcv_data_all:
                error_message = f"{ticker}의 OHLCV 데이터를 가져올 수 없습니다."
                print(f"⚠️ {error_message}")
                state["log"].append(error_message)
                continue
            
            ticker_data = ohlcv_data_all[ticker]
            if "error" in ticker_data:
                error_message = f"{ticker} 데이터 수집 오류: {ticker_data['error']}"
                print(f"⚠️ {error_message}")
                state["log"].append(error_message)
                continue
            
            # OHLCV 데이터 저장
            ohlcv_data_dict[ticker] = ticker_data.get("data", [])
            
            if not ticker_data.get("data"):
                error_message = f"{ticker}의 OHLCV 데이터가 비어있습니다."
                print(f"⚠️ {error_message}")
                state["log"].append(error_message)
                continue
            
            # DataFrame 생성
            df_data = []
            for d in ticker_data["data"]:
                df_data.append({
                    'Open': d['open'],
                    'High': d['high'],
                    'Low': d['low'],
                    'Close': d['close'],
                    'Volume': d.get('volume', 0)
                })
            
            df = pd.DataFrame(df_data)
            
            # 고급 기술적 지표 계산
            print(f"{ticker}: 기술적 지표 계산 중...")
            indicators = calculate_technical_indicators(df)
            technical_indicators_advanced_dict[ticker] = indicators
            
            # 차트 패턴 인식
            print(f"{ticker}: 차트 패턴 인식 중...")
            patterns = recognize_chart_patterns(df)
            
            # LLM 기반 차트 분석
            print(f"{ticker}: LLM 차트 분석 중...")
            llm_analysis = analyze_chart_with_llm(ticker_data["data"], analysis_type="comprehensive")
            
            # 차트 분석 결과 통합
            chart_analysis_dict[ticker] = {
                "technical_indicators": indicators,
                "patterns": patterns,
                "llm_analysis": llm_analysis,
                "data_summary": {
                    "period": period,
                    "data_count": len(ticker_data["data"]),
                    "latest_price": ticker_data.get("latest_price"),
                    "latest_volume": ticker_data.get("latest_volume")
                }
            }
            
            # 차트 이미지 생성 (base64 인코딩)
            print(f"{ticker}: 차트 이미지 생성 중...")
            chart_image = generate_chart_image(
                ticker_data["data"],
                indicators=indicators,
                return_base64=True
            )
            if chart_image:
                chart_images_dict[ticker] = chart_image
                print(f"{ticker}: 차트 이미지 생성 완료")
            else:
                print(f"⚠️ {ticker}: 차트 이미지 생성 실패 (mplfinance 미설치 또는 오류)")
            
            log_message = f"{ticker} 차트 분석 완료"
            state["log"].append(log_message)
            print(f"✅ {log_message}")
        
        log_message = f"{len(tickers)}개 티커에 대한 차트 분석 완료"
        state["log"].append(log_message)
        print(f"✅ {log_message}")
        
        return {
            "ohlcv_data": ohlcv_data_dict,
            "chart_analysis": chart_analysis_dict,
            "chart_images": chart_images_dict,
            "technical_indicators_advanced": technical_indicators_advanced_dict
        }
        
    except Exception as e:
        error_message = f"차트 분석 중 오류 발생: {e}"
        print(f"❌ {error_message}")
        state["log"].append(error_message)
        raise ValueError(error_message)

