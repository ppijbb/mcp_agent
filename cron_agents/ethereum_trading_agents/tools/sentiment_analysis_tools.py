"""
Advanced Sentiment Analysis Tools
Comprehensive sentiment analysis for Ethereum trading
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    NEWS = "news"
    BLOGS = "blogs"
    FORUMS = "forums"
    YOUTUBE = "youtube"

@dataclass
class SentimentAnalysisConfig:
    """Configuration for sentiment analysis"""
    twitter_api_key: Optional[str] = None
    reddit_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1
    sentiment_threshold: float = 0.1

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with multiple sources and methods"""
    
    def __init__(self, config: SentimentAnalysisConfig):
        self.config = config
        self.session = None
        self.rate_limiter = asyncio.Semaphore(5)
        
        # Initialize NLTK sentiment analyzer
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK: {e}")
            self.sia = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_comprehensive_sentiment(self, symbol: str = "ethereum") -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis"""
        try:
            async with self.rate_limiter:
                # Collect sentiment from multiple sources
                tasks = [
                    self._analyze_social_media_sentiment(symbol),
                    self._analyze_news_sentiment(symbol),
                    self._analyze_forum_sentiment(symbol),
                    self._analyze_youtube_sentiment(symbol),
                    self._analyze_blog_sentiment(symbol),
                    self._analyze_reddit_sentiment(symbol),
                    self._analyze_twitter_sentiment(symbol),
                    self._analyze_telegram_sentiment(symbol)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                sentiment_analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "status": "success",
                    "sources": {},
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "trend": "stable"
                }
                
                source_names = [
                    "social_media", "news", "forums", "youtube",
                    "blogs", "reddit", "twitter", "telegram"
                ]
                
                valid_scores = []
                total_weight = 0
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to analyze {source_names[i]}: {result}")
                        sentiment_analysis["sources"][source_names[i]] = {"status": "error", "error": str(result)}
                    else:
                        sentiment_analysis["sources"][source_names[i]] = result
                        
                        # Collect valid scores for overall calculation
                        if result.get("status") == "success" and result.get("sentiment_score") is not None:
                            weight = self._get_source_weight(source_names[i])
                            valid_scores.append(result["sentiment_score"] * weight)
                            total_weight += weight
                
                # Calculate overall sentiment
                if valid_scores and total_weight > 0:
                    overall_score = sum(valid_scores) / total_weight
                    sentiment_analysis["sentiment_score"] = overall_score
                    sentiment_analysis["overall_sentiment"] = self._classify_sentiment(overall_score)
                    sentiment_analysis["confidence"] = min(len(valid_scores) / 8, 1.0)  # Based on data sources
                    
                    # Calculate trend
                    sentiment_analysis["trend"] = self._calculate_sentiment_trend(sentiment_analysis["sources"])
                
                return sentiment_analysis
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _analyze_social_media_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze social media sentiment"""
        try:
            # Aggregate from multiple social platforms
            tasks = [
                self._get_twitter_sentiment(symbol),
                self._get_reddit_sentiment(symbol),
                self._get_telegram_sentiment(symbol),
                self._get_discord_sentiment(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            valid_scores = []
            total_mentions = 0
            
            for result in results:
                if not isinstance(result, Exception) and result.get("status") == "success":
                    if result.get("sentiment_score") is not None:
                        valid_scores.append(result["sentiment_score"])
                    total_mentions += result.get("mentions", 0)
            
            if valid_scores:
                avg_score = np.mean(valid_scores)
                return {
                    "status": "success",
                    "sentiment_score": avg_score,
                    "sentiment": self._classify_sentiment(avg_score),
                    "mentions": total_mentions,
                    "sources_analyzed": len([r for r in results if not isinstance(r, Exception)])
                }
            
            return {"status": "error", "message": "No valid social media data"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze news sentiment"""
        try:
            # Get news articles
            news_data = await self._get_news_articles(symbol)
            
            if news_data.get("status") != "success":
                return {"status": "error", "message": "Failed to get news data"}
            
            articles = news_data.get("articles", [])
            if not articles:
                return {"status": "error", "message": "No news articles found"}
            
            # Analyze sentiment of each article
            sentiments = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self._analyze_text_sentiment(text)
                if sentiment is not None:
                    sentiments.append(sentiment)
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                return {
                    "status": "success",
                    "sentiment_score": avg_sentiment,
                    "sentiment": self._classify_sentiment(avg_sentiment),
                    "articles_analyzed": len(articles),
                    "sentiment_distribution": self._get_sentiment_distribution(sentiments)
                }
            
            return {"status": "error", "message": "No valid sentiment data"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _analyze_forum_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze forum sentiment"""
        try:
            # Get forum posts
            forum_data = await self._get_forum_posts(symbol)
            
            if forum_data.get("status") != "success":
                return {"status": "error", "message": "Failed to get forum data"}
            
            posts = forum_data.get("posts", [])
            if not posts:
                return {"status": "error", "message": "No forum posts found"}
            
            # Analyze sentiment of posts
            sentiments = []
            for post in posts:
                text = post.get("content", "")
                sentiment = self._analyze_text_sentiment(text)
                if sentiment is not None:
                    sentiments.append(sentiment)
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                return {
                    "status": "success",
                    "sentiment_score": avg_sentiment,
                    "sentiment": self._classify_sentiment(avg_sentiment),
                    "posts_analyzed": len(posts),
                    "engagement_score": self._calculate_engagement_score(posts)
                }
            
            return {"status": "error", "message": "No valid sentiment data"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _analyze_youtube_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze YouTube sentiment"""
        try:
            # Get YouTube data
            youtube_data = await self._get_youtube_data(symbol)
            
            if youtube_data.get("status") != "success":
                return {"status": "error", "message": "Failed to get YouTube data"}
            
            videos = youtube_data.get("videos", [])
            if not videos:
                return {"status": "error", "message": "No YouTube videos found"}
            
            # Analyze sentiment of video titles and descriptions
            sentiments = []
            total_views = 0
            
            for video in videos:
                text = f"{video.get('title', '')} {video.get('description', '')}"
                sentiment = self._analyze_text_sentiment(text)
                if sentiment is not None:
                    # Weight by view count
                    views = video.get("view_count", 0)
                    sentiments.append(sentiment * views)
                    total_views += views
            
            if sentiments and total_views > 0:
                weighted_sentiment = sum(sentiments) / total_views
                return {
                    "status": "success",
                    "sentiment_score": weighted_sentiment,
                    "sentiment": self._classify_sentiment(weighted_sentiment),
                    "videos_analyzed": len(videos),
                    "total_views": total_views
                }
            
            return {"status": "error", "message": "No valid sentiment data"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _analyze_blog_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze blog sentiment"""
        try:
            # Get blog posts
            blog_data = await self._get_blog_posts(symbol)
            
            if blog_data.get("status") != "success":
                return {"status": "error", "message": "Failed to get blog data"}
            
            posts = blog_data.get("posts", [])
            if not posts:
                return {"status": "error", "message": "No blog posts found"}
            
            # Analyze sentiment of blog posts
            sentiments = []
            for post in posts:
                text = f"{post.get('title', '')} {post.get('content', '')}"
                sentiment = self._analyze_text_sentiment(text)
                if sentiment is not None:
                    sentiments.append(sentiment)
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                return {
                    "status": "success",
                    "sentiment_score": avg_sentiment,
                    "sentiment": self._classify_sentiment(avg_sentiment),
                    "posts_analyzed": len(posts),
                    "authority_score": self._calculate_authority_score(posts)
                }
            
            return {"status": "error", "message": "No valid sentiment data"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _analyze_text_sentiment(self, text: str) -> Optional[float]:
        """Analyze sentiment of text using multiple methods"""
        try:
            if not text or len(text.strip()) < 10:
                return None
            
            # Clean text
            text = self._clean_text(text)
            
            # Use multiple sentiment analysis methods
            scores = []
            
            # TextBlob sentiment
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                scores.append(polarity)
            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")
            
            # NLTK VADER sentiment
            if self.sia:
                try:
                    vader_scores = self.sia.polarity_scores(text)
                    compound_score = vader_scores['compound']
                    scores.append(compound_score)
                except Exception as e:
                    logger.debug(f"VADER analysis failed: {e}")
            
            # Custom keyword-based sentiment
            try:
                keyword_score = self._analyze_keyword_sentiment(text)
                if keyword_score is not None:
                    scores.append(keyword_score)
            except Exception as e:
                logger.debug(f"Keyword analysis failed: {e}")
            
            if scores:
                return np.mean(scores)
            
            return None
            
        except Exception as e:
            logger.error(f"Text sentiment analysis failed: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        try:
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            # Remove special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text.lower()
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text
    
    def _analyze_keyword_sentiment(self, text: str) -> Optional[float]:
        """Analyze sentiment using keyword matching"""
        try:
            # Positive keywords
            positive_keywords = [
                'bullish', 'moon', 'pump', 'surge', 'rally', 'breakout',
                'adoption', 'growth', 'innovation', 'breakthrough', 'success',
                'profit', 'gain', 'rise', 'increase', 'up', 'positive'
            ]
            
            # Negative keywords
            negative_keywords = [
                'bearish', 'dump', 'crash', 'fall', 'drop', 'decline',
                'fear', 'panic', 'sell', 'loss', 'down', 'negative',
                'problem', 'issue', 'concern', 'risk', 'danger', 'warning'
            ]
            
            text_lower = text.lower()
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
            
            total_keywords = positive_count + negative_count
            if total_keywords == 0:
                return None
            
            # Calculate sentiment score
            sentiment_score = (positive_count - negative_count) / total_keywords
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Keyword sentiment analysis failed: {e}")
            return None
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score into category"""
        if score > self.config.sentiment_threshold:
            return "bullish"
        elif score < -self.config.sentiment_threshold:
            return "bearish"
        else:
            return "neutral"
    
    def _get_source_weight(self, source: str) -> float:
        """Get weight for sentiment source"""
        weights = {
            "twitter": 0.2,
            "reddit": 0.15,
            "news": 0.2,
            "forums": 0.1,
            "youtube": 0.15,
            "blogs": 0.1,
            "telegram": 0.05,
            "discord": 0.05
        }
        return weights.get(source, 0.1)
    
    def _get_sentiment_distribution(self, sentiments: List[float]) -> Dict[str, int]:
        """Get distribution of sentiment scores"""
        bullish = sum(1 for s in sentiments if s > self.config.sentiment_threshold)
        bearish = sum(1 for s in sentiments if s < -self.config.sentiment_threshold)
        neutral = len(sentiments) - bullish - bearish
        
        return {
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral
        }
    
    def _calculate_engagement_score(self, posts: List[Dict]) -> float:
        """Calculate engagement score for posts"""
        try:
            if not posts:
                return 0.0
            
            total_engagement = 0
            for post in posts:
                likes = post.get("likes", 0)
                comments = post.get("comments", 0)
                shares = post.get("shares", 0)
                total_engagement += likes + comments + shares
            
            return min(total_engagement / len(posts), 1000) / 1000  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Failed to calculate engagement score: {e}")
            return 0.0
    
    def _calculate_authority_score(self, posts: List[Dict]) -> float:
        """Calculate authority score for blog posts"""
        try:
            if not posts:
                return 0.0
            
            total_authority = 0
            for post in posts:
                author_followers = post.get("author_followers", 0)
                post_views = post.get("views", 0)
                total_authority += author_followers + post_views
            
            return min(total_authority / len(posts), 100000) / 100000  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Failed to calculate authority score: {e}")
            return 0.0
    
    def _calculate_sentiment_trend(self, sources: Dict[str, Any]) -> str:
        """Calculate sentiment trend across sources"""
        try:
            recent_scores = []
            for source_data in sources.values():
                if source_data.get("status") == "success" and source_data.get("sentiment_score") is not None:
                    recent_scores.append(source_data["sentiment_score"])
            
            if len(recent_scores) < 2:
                return "stable"
            
            # Simple trend calculation
            if recent_scores[-1] > recent_scores[0] + 0.1:
                return "improving"
            elif recent_scores[-1] < recent_scores[0] - 0.1:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Failed to calculate sentiment trend: {e}")
            return "unknown"
    
    # Placeholder methods for data collection
    async def _get_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Twitter sentiment data"""
        return {"status": "success", "sentiment_score": 0.0, "mentions": 0}
    
    async def _get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Reddit sentiment data"""
        return {"status": "success", "sentiment_score": 0.0, "mentions": 0}
    
    async def _get_telegram_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Telegram sentiment data"""
        return {"status": "success", "sentiment_score": 0.0, "mentions": 0}
    
    async def _get_discord_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Discord sentiment data"""
        return {"status": "success", "sentiment_score": 0.0, "mentions": 0}
    
    async def _get_news_articles(self, symbol: str) -> Dict[str, Any]:
        """Get news articles"""
        return {"status": "success", "articles": []}
    
    async def _get_forum_posts(self, symbol: str) -> Dict[str, Any]:
        """Get forum posts"""
        return {"status": "success", "posts": []}
    
    async def _get_youtube_data(self, symbol: str) -> Dict[str, Any]:
        """Get YouTube data"""
        return {"status": "success", "videos": []}
    
    async def _get_blog_posts(self, symbol: str) -> Dict[str, Any]:
        """Get blog posts"""
        return {"status": "success", "posts": []}
