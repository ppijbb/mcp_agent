"""
차트 분석 도구 모음
candleview 스타일의 고급 기술적 지표 계산, 차트 이미지 생성, LLM 기반 분석, 패턴 인식 기능 제공
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import base64
import io
from datetime import datetime
import os

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("⚠️ pandas-ta가 설치되지 않았습니다. 기술적 지표 계산 기능이 제한됩니다.")

try:
    import mplfinance as mpf
    import matplotlib
    matplotlib.use('Agg')  # 백엔드 설정
    import matplotlib.pyplot as plt
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    print("⚠️ mplfinance가 설치되지 않았습니다. 차트 이미지 생성 기능이 제한됩니다.")

from ..llm_client import call_llm


def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    OHLCV 데이터프레임에서 고급 기술적 지표를 계산합니다.
    
    Args:
        df: OHLCV 데이터프레임 (Open, High, Low, Close, Volume 컬럼 필요)
    
    Returns:
        Dict: 계산된 기술적 지표들
    """
    if not PANDAS_TA_AVAILABLE:
        return {"error": "pandas-ta가 설치되지 않았습니다."}
    
    if df.empty or len(df) < 2:
        return {"error": "데이터가 부족합니다."}
    
    indicators = {}
    
    try:
        # Main Chart Indicators
        # Moving Average (MA)
        indicators['ma_5'] = ta.sma(df['Close'], length=5).iloc[-1] if len(df) >= 5 else None
        indicators['ma_10'] = ta.sma(df['Close'], length=10).iloc[-1] if len(df) >= 10 else None
        indicators['ma_20'] = ta.sma(df['Close'], length=20).iloc[-1] if len(df) >= 20 else None
        indicators['ma_50'] = ta.sma(df['Close'], length=50).iloc[-1] if len(df) >= 50 else None
        indicators['ma_200'] = ta.sma(df['Close'], length=200).iloc[-1] if len(df) >= 200 else None
        
        # Exponential Moving Average (EMA)
        indicators['ema_12'] = ta.ema(df['Close'], length=12).iloc[-1] if len(df) >= 12 else None
        indicators['ema_26'] = ta.ema(df['Close'], length=26).iloc[-1] if len(df) >= 26 else None
        
        # Bollinger Bands
        try:
            bb = ta.bbands(df['Close'], length=20)
            if bb is not None:
                if isinstance(bb, pd.DataFrame) and not bb.empty:
                    indicators['bb_upper'] = bb.iloc[-1, 0] if pd.notna(bb.iloc[-1, 0]) else None
                    indicators['bb_middle'] = bb.iloc[-1, 1] if pd.notna(bb.iloc[-1, 1]) else None
                    indicators['bb_lower'] = bb.iloc[-1, 2] if pd.notna(bb.iloc[-1, 2]) else None
                    indicators['bb_width'] = (bb.iloc[-1, 0] - bb.iloc[-1, 2]) if pd.notna(bb.iloc[-1, 0]) and pd.notna(bb.iloc[-1, 2]) else None
        except Exception:
            pass
        
        # Ichimoku Cloud
        try:
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
            if ichimoku is not None:
                if isinstance(ichimoku, pd.DataFrame) and not ichimoku.empty:
                    indicators['ichimoku_tenkan'] = ichimoku.iloc[-1, 0] if pd.notna(ichimoku.iloc[-1, 0]) else None
                    indicators['ichimoku_kijun'] = ichimoku.iloc[-1, 1] if pd.notna(ichimoku.iloc[-1, 1]) else None
                    indicators['ichimoku_senkou_a'] = ichimoku.iloc[-1, 2] if pd.notna(ichimoku.iloc[-1, 2]) else None
                    indicators['ichimoku_senkou_b'] = ichimoku.iloc[-1, 3] if pd.notna(ichimoku.iloc[-1, 3]) else None
                    indicators['ichimoku_chikou'] = ichimoku.iloc[-1, 4] if pd.notna(ichimoku.iloc[-1, 4]) else None
        except Exception:
            pass
        
        # Donchian Channel
        try:
            donchian = ta.donchian(df['High'], df['Low'], length=20)
            if donchian is not None:
                if isinstance(donchian, pd.DataFrame) and not donchian.empty:
                    indicators['donchian_upper'] = donchian.iloc[-1, 0] if pd.notna(donchian.iloc[-1, 0]) else None
                    indicators['donchian_lower'] = donchian.iloc[-1, 1] if pd.notna(donchian.iloc[-1, 1]) else None
        except Exception:
            pass
        
        # VWAP (Volume Weighted Average Price)
        if 'Volume' in df.columns:
            vwap = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            indicators['vwap'] = vwap.iloc[-1] if vwap is not None and not vwap.empty and pd.notna(vwap.iloc[-1]) else None
        
        # Sub Chart Indicators
        # RSI (Relative Strength Index)
        rsi = ta.rsi(df['Close'], length=14)
        indicators['rsi'] = rsi.iloc[-1] if rsi is not None and not rsi.empty and pd.notna(rsi.iloc[-1]) else None
        
        # MACD (Moving Average Convergence Divergence)
        try:
            macd = ta.macd(df['Close'])
            if macd is not None:
                if isinstance(macd, pd.DataFrame) and not macd.empty:
                    indicators['macd'] = macd.iloc[-1, 0] if pd.notna(macd.iloc[-1, 0]) else None
                    indicators['macd_signal'] = macd.iloc[-1, 1] if pd.notna(macd.iloc[-1, 1]) else None
                    indicators['macd_hist'] = macd.iloc[-1, 2] if pd.notna(macd.iloc[-1, 2]) else None
        except Exception:
            pass
        
        # KDJ (Stochastic Oscillator)
        try:
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            if stoch is not None:
                if isinstance(stoch, pd.DataFrame) and not stoch.empty:
                    indicators['kdj_k'] = stoch.iloc[-1, 0] if pd.notna(stoch.iloc[-1, 0]) else None
                    indicators['kdj_d'] = stoch.iloc[-1, 1] if pd.notna(stoch.iloc[-1, 1]) else None
                    indicators['kdj_j'] = stoch.iloc[-1, 2] if pd.notna(stoch.iloc[-1, 2]) else None
        except Exception:
            pass
        
        # ATR (Average True Range)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        indicators['atr'] = atr.iloc[-1] if atr is not None and not atr.empty and pd.notna(atr.iloc[-1]) else None
        
        # ADX (Average Directional Index)
        try:
            adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            if adx is not None:
                if isinstance(adx, pd.DataFrame) and not adx.empty:
                    indicators['adx'] = adx.iloc[-1, 0] if pd.notna(adx.iloc[-1, 0]) else None
                    indicators['adx_pos'] = adx.iloc[-1, 1] if pd.notna(adx.iloc[-1, 1]) else None
                    indicators['adx_neg'] = adx.iloc[-1, 2] if pd.notna(adx.iloc[-1, 2]) else None
        except Exception:
            pass
        
        # OBV (On Balance Volume)
        if 'Volume' in df.columns:
            obv = ta.obv(df['Close'], df['Volume'])
            indicators['obv'] = obv.iloc[-1] if obv is not None and not obv.empty and pd.notna(obv.iloc[-1]) else None
        
        # CCI (Commodity Channel Index)
        cci = ta.cci(df['High'], df['Low'], df['Close'], length=20)
        indicators['cci'] = cci.iloc[-1] if cci is not None and not cci.empty and pd.notna(cci.iloc[-1]) else None
        
        # Stochastic Oscillator (별도)
        stoch_rsi = ta.stochrsi(df['Close'], length=14)
        if stoch_rsi is not None and not stoch_rsi.empty:
            indicators['stochastic'] = stoch_rsi.iloc[-1] if pd.notna(stoch_rsi.iloc[-1]) else None
        
        # NaN 값을 None으로 변환
        for key, value in indicators.items():
            if pd.isna(value):
                indicators[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                indicators[key] = float(value)
        
    except Exception as e:
        return {"error": f"기술적 지표 계산 중 오류 발생: {e}"}
    
    return indicators


def recognize_chart_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    차트 패턴을 인식합니다.
    
    Args:
        df: OHLCV 데이터프레임
    
    Returns:
        Dict: 인식된 패턴들
    """
    if df.empty or len(df) < 10:
        return {"patterns": [], "message": "패턴 인식을 위한 데이터가 부족합니다."}
    
    patterns = []
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    try:
        # 간단한 패턴 인식 로직
        # Head and Shoulders 패턴 (간단한 버전)
        if len(close) >= 20:
            recent_highs = []
            recent_lows = []
            for i in range(len(close) - 20, len(close) - 5):
                if i > 0 and i < len(close) - 1:
                    if high[i] > high[i-1] and high[i] > high[i+1]:
                        recent_highs.append((i, high[i]))
                    if low[i] < low[i-1] and low[i] < low[i+1]:
                        recent_lows.append((i, low[i]))
            
            # Double Top/Bottom 패턴
            if len(recent_highs) >= 2:
                peaks = sorted(recent_highs, key=lambda x: x[1], reverse=True)[:2]
                if abs(peaks[0][1] - peaks[1][1]) / peaks[0][1] < 0.02:  # 2% 이내 차이
                    patterns.append({
                        "type": "double_top",
                        "confidence": 0.6,
                        "description": "이중 천장 패턴이 감지되었습니다."
                    })
            
            if len(recent_lows) >= 2:
                troughs = sorted(recent_lows, key=lambda x: x[1])[:2]
                if abs(troughs[0][1] - troughs[1][1]) / troughs[0][1] < 0.02:  # 2% 이내 차이
                    patterns.append({
                        "type": "double_bottom",
                        "confidence": 0.6,
                        "description": "이중 바닥 패턴이 감지되었습니다."
                    })
        
        # Triangle 패턴 (간단한 버전)
        if len(close) >= 15:
            recent_range = close[-15:]
            if np.std(recent_range) / np.mean(recent_range) < 0.05:  # 변동성이 낮음
                patterns.append({
                    "type": "triangle",
                    "confidence": 0.5,
                    "description": "삼각형 패턴이 감지되었습니다."
                })
        
    except Exception as e:
        return {"patterns": [], "error": f"패턴 인식 중 오류 발생: {e}"}
    
    return {"patterns": patterns}


def analyze_chart_with_llm(ohlcv_data: List[Dict], analysis_type: str = "comprehensive") -> str:
    """
    LLM을 사용하여 OHLCV 데이터를 분석합니다 (ohlcv-ai 스타일).
    
    Args:
        ohlcv_data: OHLCV 데이터 리스트
        analysis_type: 분석 타입 ("trend", "volume", "technical", "comprehensive")
    
    Returns:
        str: LLM 분석 결과
    """
    if not ohlcv_data or len(ohlcv_data) < 5:
        return "분석을 위한 데이터가 부족합니다."
    
    # 최근 데이터만 사용 (너무 많은 데이터는 LLM 토큰 제한에 걸릴 수 있음)
    max_data_points = 50
    data_to_analyze = ohlcv_data[-max_data_points:] if len(ohlcv_data) > max_data_points else ohlcv_data
    
    # 데이터를 구조화된 형식으로 변환
    data_summary = {
        "total_periods": len(data_to_analyze),
        "price_range": {
            "highest": max(d['high'] for d in data_to_analyze),
            "lowest": min(d['low'] for d in data_to_analyze),
            "current": data_to_analyze[-1]['close'],
            "first": data_to_analyze[0]['open']
        },
        "volume_summary": {
            "average": sum(d.get('volume', 0) for d in data_to_analyze) / len(data_to_analyze),
            "latest": data_to_analyze[-1].get('volume', 0)
        },
        "recent_trend": "상승" if data_to_analyze[-1]['close'] > data_to_analyze[0]['open'] else "하락",
        "recent_data": data_to_analyze[-10:]  # 최근 10개 데이터 포인트
    }
    
    analysis_prompts = {
        "trend": "최근 가격 추세를 분석하고 향후 추세 전망을 제시하세요.",
        "volume": "거래량 패턴을 분석하고 거래량과 가격의 관계를 평가하세요.",
        "technical": "기술적 지표 관점에서 차트를 분석하고 매수/매도 신호를 평가하세요.",
        "comprehensive": "가격, 거래량, 추세를 종합적으로 분석하고 투자 의견을 제시하세요."
    }
    
    prompt = f"""
역할: 전문 차트 분석가. OHLCV 데이터를 분석하여 투자 인사이트를 제공하세요.

분석 타입: {analysis_type}
{analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])}

데이터 요약:
{json.dumps(data_summary, indent=2, ensure_ascii=False)}

다음 항목을 포함하여 한국어로 분석하세요:
1. 현재 시장 상황 요약
2. 주요 기술적 신호
3. 추세 분석
4. 리스크 요인
5. 투자 권고사항

분석 결과만 반환하세요 (JSON 형식 불필요).
"""
    
    try:
        analysis = call_llm(prompt)
        return analysis
    except Exception as e:
        return f"LLM 분석 중 오류 발생: {e}"


def generate_chart_image(
    ohlcv_data: List[Dict],
    indicators: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    return_base64: bool = False
) -> Optional[str]:
    """
    OHLCV 데이터로부터 캔들스틱 차트 이미지를 생성합니다.
    
    Args:
        ohlcv_data: OHLCV 데이터 리스트
        indicators: 기술적 지표 딕셔너리 (선택)
        save_path: 이미지 저장 경로 (선택)
        return_base64: True인 경우 base64 인코딩된 문자열 반환
    
    Returns:
        str: base64 인코딩된 이미지 또는 파일 경로, None (오류 시)
    """
    if not MPLFINANCE_AVAILABLE:
        return None
    
    if not ohlcv_data or len(ohlcv_data) < 2:
        return None
    
    try:
        # OHLCV 데이터를 DataFrame으로 변환
        df_data = []
        for d in ohlcv_data:
            timestamp = pd.Timestamp.fromtimestamp(d['time'])
            df_data.append({
                'Open': d['open'],
                'High': d['high'],
                'Low': d['low'],
                'Close': d['close'],
                'Volume': d.get('volume', 0)
            })
        
        df = pd.DataFrame(df_data, index=[pd.Timestamp.fromtimestamp(d['time']) for d in ohlcv_data])
        
        # mplfinance 스타일 설정 (기본 스타일 사용)
        # mplfinance 버전에 따라 make_marketcolors API가 다를 수 있으므로 기본 스타일 사용
        style = 'default'
        
        # 추가 플롯 (기술적 지표)
        addplot = []
        if indicators and PANDAS_TA_AVAILABLE:
            # MA 추가
            if indicators.get('ma_20') is not None and len(df) >= 20:
                try:
                    ma20 = ta.sma(df['Close'], length=20)
                    if ma20 is not None and not ma20.empty:
                        addplot.append(mpf.make_addplot(ma20, color='orange', width=1, alpha=0.7))
                except Exception:
                    pass
            
            if indicators.get('ma_50') is not None and len(df) >= 50:
                try:
                    ma50 = ta.sma(df['Close'], length=50)
                    if ma50 is not None and not ma50.empty:
                        addplot.append(mpf.make_addplot(ma50, color='purple', width=1, alpha=0.7))
                except Exception:
                    pass
        
        # 차트 생성
        plot_kwargs = {
            'type': 'candle',
            'style': style,
            'volume': True,
            'savefig': {}
        }
        
        if addplot:
            plot_kwargs['addplot'] = addplot
        
        if save_path:
            plot_kwargs['savefig'] = dict(fname=save_path, dpi=100, bbox_inches='tight')
            mpf.plot(df, **plot_kwargs)
            return save_path
        elif return_base64:
            # 메모리 버퍼에 저장
            buf = io.BytesIO()
            plot_kwargs['savefig'] = dict(fname=buf, format='png', dpi=100, bbox_inches='tight')
            mpf.plot(df, **plot_kwargs)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return img_base64
        else:
            return None
            
    except Exception as e:
        import traceback
        error_msg = f"차트 이미지 생성 중 오류 발생: {e}"
        print(error_msg)
        print(traceback.format_exc())
        return None

