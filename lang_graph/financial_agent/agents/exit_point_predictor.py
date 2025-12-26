"""
매도시점 추측 에이전트 노드
기술적 지표 기반 매도시점 추측 (예상 가격대, 시점, 신호 강도, 손절/익절 기준선)
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


def exit_point_predictor_node(state: AgentState) -> Dict:
    """
    매도시점 추측 노드: 기술적 지표를 기반으로 매도시점을 추측합니다.
    """
    print("--- AGENT: Exit Point Predictor ---")
    log_message = "매도시점 추측을 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    # 입력 데이터 수집
    chart_analysis = state.get("chart_analysis")
    technical_indicators_advanced = state.get("technical_indicators_advanced")
    synthesized_indicators = state.get("synthesized_indicators")
    ohlcv_data = state.get("ohlcv_data")
    tickers = state.get("target_tickers", [])
    
    if not tickers:
        error_message = "분석할 티커가 지정되지 않았습니다."
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}
    
    try:
        exit_point_predictions = {}
        
        # 각 티커별로 매도시점 추측
        for ticker in tickers:
            ticker_chart_analysis = chart_analysis.get(ticker, {}) if chart_analysis else {}
            ticker_indicators = technical_indicators_advanced.get(ticker, {}) if technical_indicators_advanced else {}
            ticker_synthesized = synthesized_indicators.get(ticker, {}) if synthesized_indicators else {}
            ticker_ohlcv = ohlcv_data.get(ticker, []) if ohlcv_data else []
            
            # 현재 가격 추출
            current_price = None
            if ticker_ohlcv and len(ticker_ohlcv) > 0:
                current_price = ticker_ohlcv[-1].get('close')
            elif ticker_indicators:
                # 기술적 지표에서 가격 정보 추출 시도
                pass
            
            # LLM을 사용하여 매도시점 추측
            prompt = f"""
역할: 전문 매도시점 분석가. 기술적 지표를 기반으로 매도시점을 추측하세요.

티커: {ticker}
현재 가격: {current_price if current_price else "알 수 없음"}

입력 데이터:
1. 기술 분석 (차트 분석):
{json.dumps(ticker_chart_analysis, indent=2, ensure_ascii=False, cls=NumpyEncoder) if ticker_chart_analysis else "없음"}

2. 고급 기술적 지표:
{json.dumps(ticker_indicators, indent=2, ensure_ascii=False, cls=NumpyEncoder) if ticker_indicators else "없음"}

3. 최종 통합 지표:
{json.dumps(ticker_synthesized, indent=2, ensure_ascii=False, cls=NumpyEncoder) if ticker_synthesized else "없음"}

4. 최근 OHLCV 데이터 (최근 10개):
{json.dumps(ticker_ohlcv[-10:], indent=2, ensure_ascii=False) if ticker_ohlcv and len(ticker_ohlcv) > 0 else "없음"}

요구사항:
- 기술적 지표를 종합적으로 분석하여 매도시점을 추측하세요.
- 다음 JSON 형식으로만 응답하세요. 추가 텍스트 금지.

출력(JSON only):
{{
    "exit_price_range": {{
        "target_min": 0.0,  // 목표 매도 가격 (최소)
        "target_max": 0.0,  // 목표 매도 가격 (최대)
        "target_optimal": 0.0  // 최적 매도 가격
    }},
    "exit_timing": {{
        "estimated_days": 0,  // 예상 매도 시점 (일 단위, 0은 즉시)
        "estimated_hours": 0,  // 예상 매도 시점 (시간 단위)
        "urgency": "low|medium|high"  // 긴급도
    }},
    "exit_signal_strength": 0.0,  // 매도 신호 강도 (0-1)
    "stop_loss": 0.0,  // 손절 기준선 (가격)
    "take_profit": 0.0,  // 익절 기준선 (가격)
    "rationale": "매도시점 추측 근거를 한국어로 3-5문장",
    "risk_factors": ["리스크 요인 1", "리스크 요인 2"],
    "confidence": 0.0  // 신뢰도 (0-1)
}}
"""
            
            try:
                response = call_llm(prompt)
                # JSON 파싱
                cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
                ticker_exit_prediction = json.loads(cleaned_response)
                
                # 필수 키 검증
                required_keys = ["exit_price_range", "exit_timing", "exit_signal_strength", 
                               "stop_loss", "take_profit", "rationale", "risk_factors", "confidence"]
                missing_keys = [key for key in required_keys if key not in ticker_exit_prediction]
                if missing_keys:
                    raise KeyError(f"필수 키가 누락되었습니다: {missing_keys}")
                
                exit_point_predictions[ticker] = ticker_exit_prediction
                
                exit_price = ticker_exit_prediction.get("exit_price_range", {}).get("target_optimal", 0)
                exit_days = ticker_exit_prediction.get("exit_timing", {}).get("estimated_days", 0)
                print(f"✅ {ticker}: 매도시점 추측 완료 (목표 가격: ${exit_price:.2f}, 예상 시점: {exit_days}일)")
                
            except (json.JSONDecodeError, KeyError) as e:
                error_message = f"{ticker} 매도시점 추측 중 파싱 오류: {e}"
                print(f"⚠️ {error_message}")
                state["log"].append(error_message)
                # 기본값 설정
                exit_point_predictions[ticker] = {
                    "exit_price_range": {
                        "target_min": current_price * 0.95 if current_price else 0,
                        "target_max": current_price * 1.05 if current_price else 0,
                        "target_optimal": current_price if current_price else 0
                    },
                    "exit_timing": {
                        "estimated_days": 7,
                        "estimated_hours": 0,
                        "urgency": "medium"
                    },
                    "exit_signal_strength": 0.5,
                    "stop_loss": current_price * 0.90 if current_price else 0,
                    "take_profit": current_price * 1.10 if current_price else 0,
                    "rationale": "데이터 부족으로 기본값 사용",
                    "risk_factors": [],
                    "confidence": 0.5
                }
        
        log_message = f"{len(tickers)}개 티커에 대한 매도시점 추측 완료"
        state["log"].append(log_message)
        print(f"✅ {log_message}")
        
        return {"exit_point_predictions": exit_point_predictions}
        
    except Exception as e:
        error_message = f"매도시점 추측 중 오류 발생: {e}"
        print(f"❌ {error_message}")
        state["log"].append(error_message)
        raise ValueError(error_message)

