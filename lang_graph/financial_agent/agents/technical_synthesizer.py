"""
기술적 지표 통합 에이전트 노드
재무 분석 + 뉴스 감성 분석 + 기술 분석(차트 분석)을 통합하여 최종 지표 산출
"""

import json
import numpy as np
from typing import Dict, Any
from ..state import AgentState
from ..llm_client import call_llm


class NumpyEncoder(json.JSONEncoder):
    """Numpy 데이터 타입을 JSON으로 변환하기 위한 커스텀 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def technical_synthesizer_node(state: AgentState) -> Dict:
    """
    기술적 지표 통합 노드: 재무 분석, 뉴스 감성, 기술 분석을 통합하여 최종 지표를 산출합니다.
    """
    print("--- AGENT: Technical Indicator Synthesizer ---")
    log_message = "최종 지표 산출을 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    # 입력 데이터 수집
    financial_analysis = state.get("financial_analysis")
    sentiment_analysis = state.get("sentiment_analysis")
    chart_analysis = state.get("chart_analysis")
    technical_indicators_advanced = state.get("technical_indicators_advanced")
    market_outlook = state.get("market_outlook")
    tickers = state.get("target_tickers", [])
    
    if not tickers:
        error_message = "분석할 티커가 지정되지 않았습니다."
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}
    
    try:
        synthesized_indicators = {}
        
        # 각 티커별로 최종 지표 산출
        for ticker in tickers:
            ticker_chart_analysis = chart_analysis.get(ticker, {}) if chart_analysis else {}
            ticker_indicators = technical_indicators_advanced.get(ticker, {}) if technical_indicators_advanced else {}
            
            # LLM을 사용하여 통합 분석
            prompt = f"""
역할: 최종 지표 통합 분석가. 재무 분석, 뉴스 감성, 기술 분석을 종합하여 최종 투자 지표를 산출하세요.

티커: {ticker}

입력 데이터:
1. 재무 분석:
{json.dumps(financial_analysis, indent=2, ensure_ascii=False, cls=NumpyEncoder) if financial_analysis else "없음"}

2. 뉴스 감성 분석:
{json.dumps(sentiment_analysis, indent=2, ensure_ascii=False) if sentiment_analysis else "없음"}

3. 시장 전망:
{market_outlook if market_outlook else "없음"}

4. 기술 분석 (차트 분석):
{json.dumps(ticker_chart_analysis, indent=2, ensure_ascii=False, cls=NumpyEncoder) if ticker_chart_analysis else "없음"}

5. 고급 기술적 지표:
{json.dumps(ticker_indicators, indent=2, ensure_ascii=False, cls=NumpyEncoder) if ticker_indicators else "없음"}

요구사항:
- 모든 입력 데이터를 종합적으로 고려하여 최종 지표를 산출하세요.
- 다음 JSON 형식으로만 응답하세요. 추가 텍스트 금지.

출력(JSON only):
{{
    "composite_score": 0.0,  // 0-100 점수
    "buy_signal_strength": 0.0,  // 0-1 (매수 신호 강도)
    "sell_signal_strength": 0.0,  // 0-1 (매도 신호 강도)
    "hold_signal_strength": 0.0,  // 0-1 (보유 신호 강도)
    "risk_level": "low|medium|high",  // 리스크 레벨
    "confidence": 0.0,  // 0-1 (신뢰도)
    "rationale": "종합 분석 근거를 한국어로 3-5문장",
    "key_factors": ["주요 요인 1", "주요 요인 2", "주요 요인 3"],
    "recommendation": "strong_buy|buy|hold|sell|strong_sell"
}}
"""
            
            try:
                response = call_llm(prompt)
                # JSON 파싱
                cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
                ticker_synthesized = json.loads(cleaned_response)
                
                # 필수 키 검증
                required_keys = ["composite_score", "buy_signal_strength", "sell_signal_strength", 
                               "hold_signal_strength", "risk_level", "confidence", "rationale", 
                               "key_factors", "recommendation"]
                missing_keys = [key for key in required_keys if key not in ticker_synthesized]
                if missing_keys:
                    raise KeyError(f"필수 키가 누락되었습니다: {missing_keys}")
                
                synthesized_indicators[ticker] = ticker_synthesized
                print(f"✅ {ticker}: 최종 지표 산출 완료 (종합 점수: {ticker_synthesized.get('composite_score', 0):.2f})")
                
            except (json.JSONDecodeError, KeyError) as e:
                error_message = f"{ticker} 최종 지표 산출 중 파싱 오류: {e}"
                print(f"⚠️ {error_message}")
                state["log"].append(error_message)
                # 기본값 설정
                synthesized_indicators[ticker] = {
                    "composite_score": 50.0,
                    "buy_signal_strength": 0.0,
                    "sell_signal_strength": 0.0,
                    "hold_signal_strength": 1.0,
                    "risk_level": "medium",
                    "confidence": 0.5,
                    "rationale": "데이터 부족으로 기본값 사용",
                    "key_factors": [],
                    "recommendation": "hold"
                }
        
        log_message = f"{len(tickers)}개 티커에 대한 최종 지표 산출 완료"
        state["log"].append(log_message)
        print(f"✅ {log_message}")
        
        return {"synthesized_indicators": synthesized_indicators}
        
    except Exception as e:
        error_message = f"최종 지표 산출 중 오류 발생: {e}"
        print(f"❌ {error_message}")
        state["log"].append(error_message)
        raise ValueError(error_message)

