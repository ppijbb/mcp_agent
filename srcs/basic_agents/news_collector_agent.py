"""
간단한 뉴스 수집 Agent

MCP를 사용하여 국내뉴스와 국제뉴스를 수집하고 날짜별로 정리합니다.
"""

import asyncio
import argparse
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional

from mcp_agent.agents.agent import Agent
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.logging.logger import LoggingConfig
from rich import print as rprint

from srcs.common.utils import setup_agent_app, ensure_output_directory


class NewsCollectorAgent:
    """MCP를 사용하여 뉴스를 수집하는 Agent"""

    def __init__(self, target_date: Optional[str] = None):
        """
        Args:
            target_date: 수집할 날짜 (YYYY-MM-DD 형식). None이면 오늘 날짜 사용
        """
        if target_date:
            try:
                self.target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid date format: {target_date}. Use YYYY-MM-DD format.")
        else:
            self.target_date = date.today()

        self.date_str = self.target_date.strftime("%Y-%m-%d")
        self.output_dir = Path("outputs/news") / self.date_str
        ensure_output_directory(str(self.output_dir))

    async def collect_news(self) -> Dict[str, Any]:
        """뉴스를 수집하고 저장합니다."""
        app = setup_agent_app("news_collector_app")

        async with app.run() as app_context:
            context = app_context.context
            logger = app_context.logger

            # MCP 서버 확인
            available_servers = list(context.config.mcp.servers.keys())
            logger.info(f"Available MCP servers: {available_servers}")

            # 웹 검색 가능한 MCP 서버 찾기
            search_servers = []
            for server_name in ["brave", "g-search", "tavily", "exa"]:
                if server_name in available_servers:
                    search_servers.append(server_name)

            if not search_servers:
                raise RuntimeError(
                    "No web search MCP servers available. "
                    "Please configure at least one of: brave, g-search, tavily, exa"
                )

            logger.info(f"Using MCP search servers: {search_servers}")

            async with MCPConnectionManager(context.server_registry):
                # 뉴스 수집 Agent 생성
                news_agent = Agent(
                    name="news_collector",
                    instruction=f"""You are a news collection assistant with access to web search via MCP.
                    Your task is to collect news articles for the date {self.date_str}.
                    You have access to web search tools through MCP servers: {', '.join(search_servers)}.
                    Use these tools to search for and collect news articles.
                    """,
                    server_names=search_servers,
                )

                try:
                    llm = await news_agent.attach_llm(OpenAIAugmentedLLM)

                    # 국내뉴스 수집
                    rprint(f"[bold blue]Collecting domestic news for {self.date_str}...[/bold blue]")
                    domestic_news = await self._collect_domestic_news(llm, news_agent)

                    # 국제뉴스 수집
                    rprint(f"[bold blue]Collecting international news for {self.date_str}...[/bold blue]")
                    international_news = await self._collect_international_news(llm, news_agent)

                    # 결과 정리
                    news_data = {
                        "date": self.date_str,
                        "domestic_news": domestic_news,
                        "international_news": international_news,
                        "collected_at": datetime.now().isoformat(),
                    }

                    # 마크다운 파일로 저장
                    output_path = self._save_to_markdown(news_data)
                    rprint(f"[bold green]News saved to: {output_path}[/bold green]")

                    return news_data

                finally:
                    await news_agent.close()

        await LoggingConfig.shutdown()

    async def _collect_domestic_news(self, llm, agent: Agent) -> List[Dict[str, Any]]:
        """국내뉴스를 수집합니다."""
        date_formatted = self.target_date.strftime("%Y년 %m월 %d일")

        prompt = f"""You are a news collection assistant. Your task is to collect Korean domestic news for the date {date_formatted} ({self.date_str}).

INSTRUCTIONS:
1. Use the available MCP web search tools (brave, g-search, tavily, or exa) to search for: "{date_formatted} 한국 뉴스" or "{self.date_str} 한국 주요뉴스"
2. Collect at least 10-20 news articles from actual web search results
3. For each article, extract:
   - title: News headline
   - source: News publisher/outlet name
   - url: Full URL to the article
   - summary: Brief summary or snippet of the article content
4. Return ONLY valid JSON format with no additional text:
{{
  "news": [
    {{
      "title": "뉴스 제목",
      "source": "언론사명",
      "url": "https://example.com/article",
      "summary": "뉴스 요약 내용"
    }}
  ]
}}

CRITICAL:
- You MUST use MCP search tools to get real news articles
- Do NOT generate fake or dummy data
- Return ONLY the JSON object, no markdown, no explanations
- If search fails, return {{"news": []}}
"""

        try:
            result = await llm.generate_str(prompt)

            # JSON 파싱
            import json
            import re

            # JSON 부분 추출 (더 정확한 패턴)
            json_match = re.search(r'\{[\s\S]*"news"[\s\S]*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)
                    news_list = parsed.get("news", [])
                    if isinstance(news_list, list):
                        rprint(f"[green]Collected {len(news_list)} domestic news articles[/green]")
                        return news_list
                except json.JSONDecodeError as e:
                    rprint(f"[yellow]JSON parse error: {e}[/yellow]")

            # JSON 파싱 실패 시 경고
            rprint(f"[yellow]Warning: Could not parse JSON. Showing first 500 chars:[/yellow]")
            rprint(result[:500])
            return []

        except Exception as e:
            rprint(f"[red]Error collecting domestic news: {e}[/red]")
            raise RuntimeError(f"Failed to collect domestic news: {e}")

    async def _collect_international_news(self, llm, agent: Agent) -> List[Dict[str, Any]]:
        """국제뉴스를 수집합니다."""
        date_formatted = self.target_date.strftime("%B %d, %Y")

        prompt = f"""You are a news collection assistant. Your task is to collect international/world news for the date {date_formatted} ({self.date_str}).

INSTRUCTIONS:
1. Use the available MCP web search tools (brave, g-search, tavily, or exa) to search for: "{date_formatted} world news" or "{self.date_str} international news"
2. Collect at least 10-20 news articles from actual web search results
3. For each article, extract:
   - title: News headline
   - source: News publisher/outlet name
   - url: Full URL to the article
   - summary: Brief summary or snippet of the article content
4. Return ONLY valid JSON format with no additional text:
{{
  "news": [
    {{
      "title": "News Title",
      "source": "Publisher Name",
      "url": "https://example.com/article",
      "summary": "News summary content"
    }}
  ]
}}

CRITICAL:
- You MUST use MCP search tools to get real news articles
- Do NOT generate fake or dummy data
- Return ONLY the JSON object, no markdown, no explanations
- If search fails, return {{"news": []}}
"""

        try:
            result = await llm.generate_str(prompt)

            # JSON 파싱
            import json
            import re

            # JSON 부분 추출 (더 정확한 패턴)
            json_match = re.search(r'\{[\s\S]*"news"[\s\S]*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)
                    news_list = parsed.get("news", [])
                    if isinstance(news_list, list):
                        rprint(f"[green]Collected {len(news_list)} international news articles[/green]")
                        return news_list
                except json.JSONDecodeError as e:
                    rprint(f"[yellow]JSON parse error: {e}[/yellow]")

            # JSON 파싱 실패 시 경고
            rprint(f"[yellow]Warning: Could not parse JSON. Showing first 500 chars:[/yellow]")
            rprint(result[:500])
            return []

        except Exception as e:
            rprint(f"[red]Error collecting international news: {e}[/red]")
            raise RuntimeError(f"Failed to collect international news: {e}")

    def _save_to_markdown(self, news_data: Dict[str, Any]) -> str:
        """뉴스 데이터를 마크다운 파일로 저장합니다."""
        output_file = self.output_dir / f"news_{self.date_str}.md"

        with open(output_file, "w", encoding="utf-8") as f:
            # 헤더
            f.write(f"# 뉴스 수집 보고서\n\n")
            f.write(f"**수집 날짜**: {self.date_str}\n\n")
            f.write(f"**생성 시간**: {news_data['collected_at']}\n\n")
            f.write("---\n\n")

            # 국내뉴스
            f.write("## 국내뉴스\n\n")
            domestic_news = news_data.get("domestic_news", [])
            if domestic_news:
                for i, article in enumerate(domestic_news, 1):
                    f.write(f"### {i}. {article.get('title', '제목 없음')}\n\n")
                    f.write(f"**출처**: {article.get('source', '알 수 없음')}\n\n")
                    if article.get('url'):
                        f.write(f"**링크**: [{article['url']}]({article['url']})\n\n")
                    if article.get('summary'):
                        f.write(f"**요약**: {article['summary']}\n\n")
                    f.write("---\n\n")
            else:
                f.write("수집된 국내뉴스가 없습니다.\n\n")

            # 국제뉴스
            f.write("## 국제뉴스\n\n")
            international_news = news_data.get("international_news", [])
            if international_news:
                for i, article in enumerate(international_news, 1):
                    f.write(f"### {i}. {article.get('title', 'No Title')}\n\n")
                    f.write(f"**Source**: {article.get('source', 'Unknown')}\n\n")
                    if article.get('url'):
                        f.write(f"**Link**: [{article['url']}]({article['url']})\n\n")
                    if article.get('summary'):
                        f.write(f"**Summary**: {article['summary']}\n\n")
                    f.write("---\n\n")
            else:
                f.write("No international news collected.\n\n")

            # 통계
            f.write("## 수집 통계\n\n")
            f.write(f"- 국내뉴스: {len(domestic_news)}건\n")
            f.write(f"- 국제뉴스: {len(international_news)}건\n")
            f.write(f"- 총계: {len(domestic_news) + len(international_news)}건\n")

        return str(output_file.absolute())


async def main(target_date: Optional[str] = None):
    """메인 실행 함수"""
    try:
        agent = NewsCollectorAgent(target_date=target_date)
        news_data = await agent.collect_news()

        rprint(f"\n[bold green]✓ News collection completed successfully![/bold green]")
        rprint(f"[green]Domestic news: {len(news_data['domestic_news'])} articles[/green]")
        rprint(f"[green]International news: {len(news_data['international_news'])} articles[/green]")

        return news_data

    except Exception as e:
        rprint(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="뉴스 수집 Agent")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="수집할 날짜 (YYYY-MM-DD 형식). 지정하지 않으면 오늘 날짜 사용"
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(target_date=args.date))
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        rprint(f"[bold red]Fatal error: {e}[/bold red]")
        raise
