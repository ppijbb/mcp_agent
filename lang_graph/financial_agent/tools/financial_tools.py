import yfinance as yf
import pandas as pd
from typing import Dict, List

def get_technical_indicators(ticker: str, period: str = "3mo") -> Dict:
    """
    yfinance를 사용하여 특정 종목의 기술적 지표를 계산합니다.
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - 50일 이동평균
    """
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

    return {
        "price": close.iloc[-1],
        "rsi": rsi.iloc[-1],
        "macd": macd.iloc[-1],
        "moving_average_50": ma50.iloc[-1],
        "volume": hist['Volume'].iloc[-1]
    }

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