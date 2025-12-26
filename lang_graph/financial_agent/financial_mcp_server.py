import yfinance as yf
import pandas as pd
from typing import Dict, List
from mcp.server.fastmcp import FastMCP
from .config import get_mcp_config

# MCP 서버 초기화
mcp = FastMCP("FinancialTools")

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

if __name__ == "__main__":
    # Stdio를 통해 서버 실행
    mcp.run() 