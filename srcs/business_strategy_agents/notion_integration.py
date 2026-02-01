"""
Notion Integration for Most Hooking Business Strategy Agent

This module handles automated documentation of business insights and strategies
to Notion databases for easy team collaboration and tracking.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
import aiohttp

from .config import get_config
from .architecture import ProcessedInsight, BusinessStrategy, BusinessOpportunityLevel

logger = logging.getLogger(__name__)


class NotionClient:
    """Notion API 클라이언트"""

    def __init__(self):
        self.config = get_config()
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.config.notion.api_key if self.config.notion else ''}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 반환"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
        return self.session

    async def close(self):
        """세션 종료"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def create_page(self, database_id: str, properties: Dict[str, Any],
                         content_blocks: List[Dict] = None) -> Optional[str]:
        """Notion 페이지 생성"""
        try:
            session = await self._get_session()

            payload = {
                "parent": {"database_id": database_id},
                "properties": properties
            }

            if content_blocks:
                payload["children"] = content_blocks

            async with session.post(f"{self.base_url}/pages", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    page_id = result.get('id')
                    logger.info(f"Created Notion page: {page_id}")
                    return page_id
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create Notion page: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"Notion page creation failed: {e}")
            return None

    async def query_database(self, database_id: str, filter_conditions: Dict = None) -> List[Dict]:
        """데이터베이스 쿼리"""
        try:
            session = await self._get_session()

            payload = {}
            if filter_conditions:
                payload["filter"] = filter_conditions

            async with session.post(f"{self.base_url}/databases/{database_id}/query",
                                  json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('results', [])
                else:
                    logger.warning(f"Database query failed: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []


class NotionFormatter:
    """Notion 형식 변환기"""

    @staticmethod
    def format_insight_properties(insight: ProcessedInsight) -> Dict[str, Any]:
        """인사이트를 Notion 속성으로 변환"""
        return {
            "Title": {
                "title": [{"text": {"content": f"Insight: {', '.join(insight.key_topics[:3])}"}}]
            },
            "Date": {
                "date": {"start": insight.timestamp.isoformat()}
            },
            "Region": {
                "select": {"name": insight.region.value.replace('_', ' ').title()}
            },
            "Category": {
                "select": {"name": insight.category.replace('_', ' ').title()}
            },
            "Hooking Score": {
                "number": insight.hooking_score
            },
            "Business Opportunity": {
                "select": {"name": insight.business_opportunity.value.title()}
            },
            "Key Insights": {
                "rich_text": [{"text": {"content": "; ".join(insight.actionable_insights)}}]
            },
            "Sentiment Score": {
                "number": insight.sentiment_score
            },
            "Trend Direction": {
                "select": {"name": insight.trend_direction.title()}
            }
        }

    @staticmethod
    def format_strategy_properties(strategy: BusinessStrategy) -> Dict[str, Any]:
        """전략을 Notion 속성으로 변환"""
        return {
            "Title": {
                "title": [{"text": {"content": strategy.title}}]
            },
            "Date": {
                "date": {"start": strategy.created_at.isoformat()}
            },
            "Region": {
                "select": {"name": strategy.region.value.replace('_', ' ').title()}
            },
            "Category": {
                "select": {"name": strategy.category.replace('_', ' ').title()}
            },
            "Opportunity Level": {
                "select": {"name": strategy.opportunity_level.value.title()}
            },
            "Timeline": {
                "rich_text": [{"text": {"content": strategy.timeline}}]
            },
            "ROI Prediction": {
                "rich_text": [{"text": {"content": json.dumps(strategy.roi_prediction, indent=2)}}]
            },
            "Status": {
                "select": {"name": "New"}
            }
        }

    @staticmethod
    def create_insight_content_blocks(insight: ProcessedInsight) -> List[Dict]:
        """인사이트 내용 블록 생성"""
        blocks = []

        # 개요 섹션
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"text": {"content": "📊 Insight Overview"}}]
            }
        })

        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"text": {"content": f"Content ID: {insight.content_id}"}}]
            }
        })

        # 핵심 주제
        if insight.key_topics:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"text": {"content": "🏷️ Key Topics"}}]
                }
            })

            topics_text = "• " + "\n• ".join(insight.key_topics)
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"text": {"content": topics_text}}]
                }
            })

        # 실행 가능한 인사이트
        if insight.actionable_insights:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"text": {"content": "💡 Actionable Insights"}}]
                }
            })

            for i, action_insight in enumerate(insight.actionable_insights, 1):
                blocks.append({
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"text": {"content": action_insight}}]
                    }
                })

        # 시장 규모 추정
        if insight.market_size_estimate:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"text": {"content": "📈 Market Size Estimate"}}]
                }
            })

            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"text": {"content": insight.market_size_estimate}}]
                }
            })

        return blocks

    @staticmethod
    def create_strategy_content_blocks(strategy: BusinessStrategy) -> List[Dict]:
        """전략 내용 블록 생성"""
        blocks = []

        # 전략 설명
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"text": {"content": "🎯 Strategy Description"}}]
            }
        })

        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"text": {"content": strategy.description}}]
            }
        })

        # 실행 항목
        if strategy.action_items:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"text": {"content": "✅ Action Items"}}]
                }
            })

            for item in strategy.action_items:
                task = item.get('task', 'No task specified')
                timeline = item.get('timeline', 'TBD')
                resources = item.get('resources', 'TBD')

                blocks.append({
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": f"{task} | Timeline: {timeline} | Resources: {resources}"}}],
                        "checked": False
                    }
                })

        # ROI 예측
        if strategy.roi_prediction:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"text": {"content": "💰 ROI Prediction"}}]
                }
            })

            roi_text = []
            for key, value in strategy.roi_prediction.items():
                roi_text.append(f"**{key.replace('_', ' ').title()}**: {value}")

            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"text": {"content": "\n".join(roi_text)}}]
                }
            })

        # 리스크 요인
        if strategy.risk_factors:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"text": {"content": "⚠️ Risk Factors"}}]
                }
            })

            for risk in strategy.risk_factors:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": risk}}]
                    }
                })

        # 성공 지표
        if strategy.success_metrics:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"text": {"content": "📊 Success Metrics"}}]
                }
            })

            for metric in strategy.success_metrics:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": metric}}]
                    }
                })

        return blocks


class NotionIntegration:
    """Notion 통합 서비스"""

    def __init__(self):
        self.client = NotionClient()
        self.formatter = NotionFormatter()
        self.config = get_config()

    async def initialize(self) -> bool:
        """통합 서비스 초기화"""
        if not self.config.notion:
            logger.warning("Notion configuration not found")
            return False

        logger.info("Notion integration initialized")
        return True

    async def create_daily_insights_page(self, insights: List[ProcessedInsight]) -> Optional[str]:
        """일일 인사이트 페이지 생성"""
        if not insights:
            logger.info("No insights to document")
            return None

        try:
            # 오늘 날짜로 제목 생성
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            top_insight = max(insights, key=lambda x: x.hooking_score)

            # 페이지 속성
            properties = {
                "Title": {
                    "title": [{"text": {"content": f"Daily Business Insights - {today}"}}]
                },
                "Date": {
                    "date": {"start": datetime.now(timezone.utc).isoformat()}
                },
                "Top Hooking Score": {
                    "number": top_insight.hooking_score
                },
                "Insights Count": {
                    "number": len(insights)
                },
                "Status": {
                    "select": {"name": "New"}
                }
            }

            # 내용 블록 생성
            content_blocks = []

            # 요약 섹션
            content_blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"text": {"content": f"📈 Daily Insights Summary - {today}"}}]
                }
            })

            # 통계
            high_value_count = len([i for i in insights if i.hooking_score >= 0.7])
            avg_score = sum(i.hooking_score for i in insights) / len(insights)

            stats_text = f"""
📊 **Statistics**:
• Total Insights: {len(insights)}
• High-Value Insights (≥0.7): {high_value_count}
• Average Hooking Score: {avg_score:.2f}
• Top Score: {top_insight.hooking_score:.2f}
"""

            content_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"text": {"content": stats_text}}]
                }
            })

            # 상위 인사이트들
            content_blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "🎯 Top Insights"}}]
                }
            })

            top_insights = sorted(insights, key=lambda x: x.hooking_score, reverse=True)[:5]

            for i, insight in enumerate(top_insights, 1):
                content_blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"text": {"content": f"{i}. Score: {insight.hooking_score:.2f} | {', '.join(insight.key_topics[:2])}"}}]
                    }
                })

                # 각 인사이트의 세부 내용
                content_blocks.extend(self.formatter.create_insight_content_blocks(insight))

            # 데이터베이스에 페이지 생성
            database_id = self.config.notion.database_id if self.config.notion else ""
            page_id = await self.client.create_page(database_id, properties, content_blocks)

            if page_id:
                logger.info(f"Created daily insights page: {page_id}")

            return page_id

        except Exception as e:
            logger.error(f"Failed to create daily insights page: {e}")
            return None

    async def create_strategy_page(self, strategy: BusinessStrategy) -> Optional[str]:
        """전략 페이지 생성"""
        try:
            properties = self.formatter.format_strategy_properties(strategy)
            content_blocks = self.formatter.create_strategy_content_blocks(strategy)

            database_id = self.config.notion.database_id if self.config.notion else ""
            page_id = await self.client.create_page(database_id, properties, content_blocks)

            if page_id:
                logger.info(f"Created strategy page: {strategy.title}")

            return page_id

        except Exception as e:
            logger.error(f"Failed to create strategy page: {e}")
            return None

    async def create_weekly_summary(self, insights: List[ProcessedInsight],
                                  strategies: List[BusinessStrategy]) -> Optional[str]:
        """주간 요약 페이지 생성"""
        try:
            week_start = datetime.now(timezone.utc).strftime("%Y-W%U")

            properties = {
                "Title": {
                    "title": [{"text": {"content": f"Weekly Business Intelligence Summary - {week_start}"}}]
                },
                "Date": {
                    "date": {"start": datetime.now(timezone.utc).isoformat()}
                },
                "Total Insights": {
                    "number": len(insights)
                },
                "Total Strategies": {
                    "number": len(strategies)
                },
                "Type": {
                    "select": {"name": "Weekly Summary"}
                }
            }

            # 주간 요약 내용 생성
            content_blocks = []

            content_blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"text": {"content": f"📊 Weekly Summary - {week_start}"}}]
                }
            })

            # 핵심 지표
            if insights:
                avg_score = sum(i.hooking_score for i in insights) / len(insights)
                top_regions = {}
                for insight in insights:
                    region = insight.region.value
                    top_regions[region] = top_regions.get(region, 0) + 1

                metrics_text = f"""
**📈 Key Metrics**:
• Total Insights Analyzed: {len(insights)}
• Average Hooking Score: {avg_score:.2f}
• Strategies Generated: {len(strategies)}
• Top Region: {max(top_regions.keys(), key=top_regions.get) if top_regions else 'N/A'}
"""

                content_blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": metrics_text}}]
                    }
                })

            # 최고 전략들
            if strategies:
                content_blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "🏆 Top Strategies This Week"}}]
                    }
                })

                critical_strategies = [s for s in strategies if s.opportunity_level == BusinessOpportunityLevel.CRITICAL]
                high_strategies = [s for s in strategies if s.opportunity_level == BusinessOpportunityLevel.HIGH]

                for strategy in critical_strategies[:3]:
                    content_blocks.append({
                        "object": "block",
                        "type": "callout",
                        "callout": {
                            "rich_text": [{"text": {"content": f"🔥 CRITICAL: {strategy.title}\n{strategy.description[:200]}..."}}],
                            "icon": {"emoji": "🚨"}
                        }
                    })

                for strategy in high_strategies[:2]:
                    content_blocks.append({
                        "object": "block",
                        "type": "callout",
                        "callout": {
                            "rich_text": [{"text": {"content": f"⭐ HIGH: {strategy.title}\n{strategy.description[:200]}..."}}],
                            "icon": {"emoji": "⚡"}
                        }
                    })

            database_id = self.config.notion.database_id if self.config.notion else ""
            page_id = await self.client.create_page(database_id, properties, content_blocks)

            if page_id:
                logger.info(f"Created weekly summary page: {page_id}")

            return page_id

        except Exception as e:
            logger.error(f"Failed to create weekly summary: {e}")
            return None

    async def shutdown(self):
        """서비스 종료"""
        await self.client.close()
        logger.info("Notion integration shut down")


# 글로벌 인스턴스
notion_integration = NotionIntegration()


async def get_notion_integration() -> NotionIntegration:
    """Notion 통합 인스턴스 반환"""
    await notion_integration.initialize()
    return notion_integration
