#!/usr/bin/env python3
"""
진짜 LangGraph 에이전트 연동 UI 시스템
실제 AI 에이전트가 게임을 분석하고 UI를 생성하는 시스템
"""

import streamlit as st
import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# A2A 실행을 위한 import
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a
from configs.settings import get_reports_path

# LLM import (게임 이름 추출용)
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

# BGG 접근용 - 웹 스크래핑
import aiohttp
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import re

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# 페이지 설정
st.set_page_config(page_title="🤖 Agent-driven UI", page_icon="🤖", layout="wide")

class RealLangGraphUI:
    """실제 LangGraph 에이전트 기반 UI 시스템 (A2A 통합)"""
    
    def __init__(self):
        # A2A를 사용하므로 직접 에이전트 초기화 불필요
        # if "ui_analyzer" not in st.session_state:
        #     with st.spinner("LangGraph 에이전트 초기화 중..."):
        #         try:
        #             st.session_state.ui_analyzer = get_game_ui_analyzer()
        #         except Exception as e:
        #             st.error(f"❌ 에이전트 초기화 실패: {str(e)}")
        #             st.session_state.ui_analyzer = None

        # MCP 클라이언트 제거됨 - BGG API 직접 호출 사용
        
        # 세션 상태 초기화
        for key, default in {
            "generated_games": {},
            "current_game_id": None,
            "analysis_log": [],
            "analysis_steps": [],
            "analysis_in_progress": False,
            "bgg_search_results": None,
            "game_selection_needed": False,
            "bgg_game_details": None,
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default

    async def _search_bgg_direct(self, name: str) -> Dict[str, Any]:
        """BGG 검색 - 웹 페이지 스크래핑 (API가 401 반환하므로)"""
        encoded_name = quote_plus(name)
        
        # BGG 웹 검색 페이지 사용
        search_url = f"https://boardgamegeek.com/geeksearch.php?action=search&objecttype=boardgame&q={encoded_name}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://boardgamegeek.com/",
            "Upgrade-Insecure-Requests": "1",
        }
        
        logger.info(f"BGG 웹 검색 시도: {name} -> {search_url}")
        
        try:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=20, connect=10)
            
            async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
                async with session.get(search_url, allow_redirects=True) as response:
                    logger.info(f"BGG 웹 검색 응답 상태: {response.status}, URL: {response.url}")
                    
                    if response.status == 200:
                        html_content = await response.text()
                        
                        if not html_content or len(html_content.strip()) == 0:
                            logger.warning("빈 HTML 응답")
                            return {"success": False, "error": "빈 응답", "games": []}
                        
                        # HTML에서 게임 정보 추출 (정규식 사용)
                        games = []
                        seen_ids = set()
                        
                        # 패턴 1: /boardgame/{id}/ 형태의 링크 찾기
                        game_link_pattern = r'/boardgame/(\d+)/([^"\'<>/]+)'
                        matches = re.finditer(game_link_pattern, html_content)
                        
                        for match in matches:
                            try:
                                game_id = int(match.group(1))
                                if game_id in seen_ids:
                                    continue
                                seen_ids.add(game_id)
                                
                                # 게임 이름 찾기 - 링크 텍스트에서
                                # <a href="/boardgame/12345/...">Game Name</a> 형식
                                link_start = match.start()
                                # 링크 태그 찾기
                                link_tag_pattern = rf'<a[^>]*href="/boardgame/{game_id}/[^"]*"[^>]*>([^<]+)</a>'
                                name_match = re.search(link_tag_pattern, html_content[max(0, link_start-500):link_start+500], re.IGNORECASE)
                                
                                if name_match:
                                    game_name = name_match.group(1).strip()
                                    # HTML 엔티티 디코딩
                                    game_name = game_name.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
                                else:
                                    # URL에서 이름 추출
                                    game_name_url = match.group(2).replace('-', ' ').strip()
                                    game_name = game_name_url
                                
                                # 년도 찾기 (게임 이름 근처에서)
                                context_start = max(0, match.start() - 200)
                                context_end = min(len(html_content), match.end() + 200)
                                context = html_content[context_start:context_end]
                                
                                year = None
                                year_patterns = [
                                    rf'\((\d{{4}})\)',
                                    rf'<span[^>]*>(\d{{4}})</span>',
                                    rf'year[^>]*>(\d{{4}})<',
                                ]
                                
                                for pattern in year_patterns:
                                    year_match = re.search(pattern, context, re.IGNORECASE)
                                    if year_match:
                                        try:
                                            year_val = int(year_match.group(1))
                                            if 1900 <= year_val <= 2100:  # 합리적인 범위
                                                year = year_val
                                                break
                                        except ValueError:
                                            continue
                                
                                games.append({
                                    "id": game_id,
                                    "name": game_name,
                                    "year": year
                                })
                                
                                # 최대 20개까지만
                                if len(games) >= 20:
                                    break
                                    
                            except (ValueError, AttributeError) as e:
                                logger.warning(f"게임 정보 파싱 실패: {e}")
                                continue
                        
                        if games:
                            logger.info(f"✅ BGG 웹 검색 성공: {len(games)}개 게임 발견")
                            return {
                                "success": True,
                                "games": games,
                                "total": len(games)
                            }
                        else:
                            # 방법 2: 다른 패턴 시도 - 검색 결과 테이블에서
                            logger.warning("방법 1 실패, 방법 2 시도...")
                            
                            # 모든 게임 ID 찾기
                            all_game_ids = re.findall(r'/boardgame/(\d+)/', html_content)
                            unique_ids = list(set([int(gid) for gid in all_game_ids[:20]]))
                            
                            for game_id in unique_ids:
                                # 게임 이름 찾기 (다양한 패턴 시도)
                                name_patterns = [
                                    rf'<a[^>]*href="/boardgame/{game_id}/[^"]*"[^>]*>([^<]+)</a>',
                                    rf'/boardgame/{game_id}/([^"\'<>/]+)',
                                    rf'boardgame/{game_id}[^>]*>([^<]+)</a>',
                                ]
                                
                                game_name = None
                                for pattern in name_patterns:
                                    name_match = re.search(pattern, html_content, re.IGNORECASE)
                                    if name_match:
                                        game_name = name_match.group(1).strip().replace('-', ' ').replace('&amp;', '&')
                                        break
                                
                                if not game_name:
                                    game_name = f"Game {game_id}"
                                
                                games.append({
                                    "id": game_id,
                                    "name": game_name,
                                    "year": None
                                })
                            
                            if games:
                                logger.info(f"✅ BGG 웹 검색 성공 (방법 2): {len(games)}개 게임 발견")
                                return {
                                    "success": True,
                                    "games": games,
                                    "total": len(games)
                                }
                            
                            logger.warning("BGG 웹 검색: 게임을 찾을 수 없음")
                            return {
                                "success": False,
                                "error": "검색 결과를 찾을 수 없습니다",
                                "games": []
                            }
                    else:
                        error_text = await response.text()
                        logger.error(f"BGG 웹 검색 오류 {response.status}: {error_text[:500]}")
                        return {
                            "success": False,
                            "error": f"BGG 웹 검색 오류: {response.status}",
                            "games": []
                        }
                        
        except Exception as e:
            logger.error(f"BGG 웹 검색 실패: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "games": []
            }
        finally:
            try:
                await connector.close()
            except:
                pass
    
    async def _get_bgg_game_details_direct(self, bgg_id: int) -> Dict[str, Any]:
        """BGG 게임 상세 정보 - 웹 페이지 스크래핑"""
        game_url = f"https://boardgamegeek.com/boardgame/{bgg_id}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://boardgamegeek.com/",
        }
        
        logger.info(f"BGG 게임 상세 정보 시도: game_id={bgg_id}")
        
        try:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=20, connect=10)
            
            async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
                async with session.get(game_url, allow_redirects=True) as response:
                    logger.info(f"BGG 게임 페이지 응답 상태: {response.status}")
                    
                    if response.status == 200:
                        html_content = await response.text()
                        
                        if not html_content or len(html_content.strip()) == 0:
                            logger.warning("빈 HTML 응답")
                            return {"success": False, "error": "빈 응답", "game": None}
                        
                        # HTML에서 게임 정보 추출
                        game_info = {
                            "id": bgg_id,
                            "name": "Unknown",
                            "description": "",
                            "year_published": None,
                            "min_players": None,
                            "max_players": None,
                            "playing_time": None,
                            "min_age": None,
                            "categories": [],
                            "mechanics": [],
                            "rating": {}
                        }
                        
                        # 게임 이름 추출
                        name_patterns = [
                            r'<h1[^>]*class="game-header-title"[^>]*>([^<]+)</h1>',
                            r'<h1[^>]*>([^<]+)</h1>',
                            r'<title>([^<]+)</title>',
                        ]
                        
                        for pattern in name_patterns:
                            name_match = re.search(pattern, html_content, re.IGNORECASE)
                            if name_match:
                                game_info["name"] = name_match.group(1).strip().replace('&amp;', '&')
                                break
                        
                        # 설명 추출
                        desc_patterns = [
                            r'<div[^>]*class="game-description"[^>]*>([^<]+)</div>',
                            r'<meta[^>]*name="description"[^>]*content="([^"]+)"',
                        ]
                        
                        for pattern in desc_patterns:
                            desc_match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
                            if desc_match:
                                game_info["description"] = desc_match.group(1).strip()[:1000]  # 최대 1000자
                                break
                        
                        # 게임 통계 추출
                        stats_patterns = {
                            "year_published": [
                                r'Year Published[^>]*>(\d{4})',
                                r'Published[^>]*>(\d{4})',
                            ],
                            "min_players": [
                                r'Min Players[^>]*>(\d+)',
                                r'Players[^>]*>(\d+)[^<]*-\s*(\d+)',
                            ],
                            "max_players": [
                                r'Max Players[^>]*>(\d+)',
                                r'Players[^>]*>(\d+)[^<]*-\s*(\d+)',
                            ],
                            "playing_time": [
                                r'Playing Time[^>]*>(\d+)',
                                r'Play Time[^>]*>(\d+)',
                            ],
                            "min_age": [
                                r'Min Age[^>]*>(\d+)',
                                r'Age[^>]*>(\d+)',
                            ],
                        }
                        
                        for key, patterns in stats_patterns.items():
                            for pattern in patterns:
                                match = re.search(pattern, html_content, re.IGNORECASE)
                                if match:
                                    try:
                                        if key == "max_players" and len(match.groups()) > 1:
                                            game_info[key] = int(match.group(2))
                                        else:
                                            game_info[key] = int(match.group(1))
                                        break
                                    except (ValueError, IndexError):
                                        continue
                        
                        # 평점 추출
                        rating_patterns = [
                            r'Geek Rating[^>]*>([\d.]+)',
                            r'Average[^>]*>([\d.]+)',
                        ]
                        
                        for pattern in rating_patterns:
                            rating_match = re.search(pattern, html_content, re.IGNORECASE)
                            if rating_match:
                                try:
                                    game_info["rating"]["average"] = float(rating_match.group(1))
                                    break
                                except ValueError:
                                    continue
                        
                        logger.info(f"✅ BGG 게임 상세 정보 성공: {game_info['name']}")
                        return {
                            "success": True,
                            "game": game_info
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"BGG 게임 페이지 오류 {response.status}: {error_text[:200]}")
                        return {
                            "success": False,
                            "error": f"BGG 게임 페이지 오류: {response.status}",
                            "game": None
                        }
                        
        except Exception as e:
            logger.error(f"BGG 게임 상세 정보 실패: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "game": None
            }
        finally:
            try:
                await connector.close()
            except:
                pass
    
    async def handle_game_search(self, game_description: str):
        st.session_state.analysis_in_progress = True
        st.session_state.game_selection_needed = False
        st.session_state.bgg_search_results = None
        st.session_state.current_game_id = None
        
        try:
            # 1단계: LLM으로 사용자 설명에서 게임 이름 추출
            with st.spinner("게임 설명을 분석하여 검색어를 생성 중..."):
                try:
                    from srcs.common.llm.fallback_llm import create_fallback_llm_factory
                    llm_factory = create_fallback_llm_factory("gemini-2.5-flash-lite", logger)
                    llm = llm_factory()
                    
                    extraction_prompt = f"""사용자가 원하는 보드게임을 찾기 위해, 다음 설명에서 실제 보드게임 이름이나 검색에 적합한 키워드를 추출해주세요.

사용자 설명: {game_description}

요구사항:
1. 설명에 실제 게임 이름이 언급되어 있으면 그 이름을 그대로 사용
2. 게임 이름이 없으면, 설명의 핵심 키워드 2-3개를 추출 (예: "마피아", "심리게임", "협상")
3. 검색에 적합한 간단한 키워드로 변환 (최대 5단어)
4. 영어 게임 이름이 있으면 영어로, 한국어 게임이면 한국어로

응답 형식: 추출된 게임 이름이나 키워드만 출력 (설명 없이)"""
                    
                    extracted_name = await llm.generate_str(extraction_prompt)
                    # LLM 응답 정리 (불필요한 설명 제거)
                    extracted_name = extracted_name.strip().split('\n')[0].strip()
                    if not extracted_name or len(extracted_name) > 100:
                        # 추출 실패 시 원본 설명 사용
                        extracted_name = game_description
                        logger.warning(f"LLM 추출 실패, 원본 설명 사용: {game_description}")
                except Exception as e:
                    logger.error(f"LLM 호출 실패: {e}", exc_info=True)
                    st.warning(f"게임 이름 추출 실패, 원본 설명으로 검색합니다: {e}")
                    extracted_name = game_description
            
            # 2단계: 추출된 이름으로 BGG 웹 검색
            with st.spinner(f"'{extracted_name}' 게임을 BoardGameGeek에서 검색 중..."):
                logger.info(f"BGG 검색 시작: {extracted_name}")
                search_result = await self._search_bgg_direct(extracted_name)
                logger.info(f"BGG 검색 결과: success={search_result.get('success')}, total={search_result.get('total', 0)}, error={search_result.get('error', 'None')}")

            if search_result.get("success") and search_result.get("total", 0) > 0:
                games = search_result.get("games", [])
                if len(games) == 1:
                    # 결과가 하나면 바로 분석 진행
                    st.session_state.bgg_search_results = games
                    await self.handle_game_selection(games[0])
                else:
                    # 결과가 여러 개면 사용자에게 선택 요청
                    st.session_state.bgg_search_results = games
                    st.session_state.game_selection_needed = True
            else:
                error_msg = search_result.get("error", "알 수 없는 오류")
                st.error(f"'{game_description}'에 대한 게임을 BGG에서 찾을 수 없습니다. ({error_msg}) 더 일반적인 이름으로 시도해보세요.")
                st.session_state.analysis_in_progress = False

        except Exception as e:
            st.error(f"게임 검색 중 오류 발생: {e}")
            logger.error(f"게임 검색 오류: {e}", exc_info=True)
            st.session_state.analysis_in_progress = False
        
        st.rerun()

    async def handle_game_selection(self, selected_game: Dict[str, Any]):
        st.session_state.game_selection_needed = False
        st.session_state.analysis_in_progress = True
        
        game_id = f"bgg_{selected_game['id']}"
        st.session_state.current_game_id = game_id

        # 상세 정보 및 웹 규칙 가져오기
        try:
            game_name_for_search = selected_game.get('name', 'board game')

            with st.spinner(f"'{selected_game['name']}' 상세 정보 조회 중..."):
                # BGG 웹 페이지에서 상세 정보 가져오기
                details_result = await self._get_bgg_game_details_direct(selected_game['id'])
            
            if not details_result.get("success"):
                raise Exception(f"BGG 상세 정보 조회 실패: {details_result.get('error')}")
            
            st.session_state.bgg_game_details = details_result["game"]
            
            # 웹 검색 제거 - BGG 정보만으로 충분
            web_rules_content = ""

            # 이제 LangGraph 분석 시작
            game_name = st.session_state.bgg_game_details.get('name', '분석 중...')
            st.session_state.generated_games[game_id] = {
                "name": game_name,
                "description": st.session_state.bgg_game_details.get('description', ''),
                "rules": web_rules_content # 웹에서 가져온 규칙
            }

        except Exception as e:
            st.error(f"게임 상세 정보 및 규칙 조회 중 오류 발생: {e}")
            logger.error(f"게임 상세 정보 오류: {e}", exc_info=True)
            st.session_state.analysis_in_progress = False
        
        st.rerun()

    def render_game_creator(self):
        st.subheader("1. AI에게 분석을 요청할 게임 설명하기")
        game_description = st.text_area(
            "어떤 보드게임을 플레이하고 싶으신가요? 자유롭게 설명해주세요.", 
            placeholder="예시: 친구들과 할 수 있는 마피아 같은 심리 게임인데, 너무 무겁지 않고 간단하게 한 판 할 수 있는 거 없을까? 서로 속이고 정체를 밝혀내는 요소가 있었으면 좋겠어.", 
            height=150
        )
        
        if st.button("🧠 이 설명으로 UI 생성 분석 요청", type="primary", width='stretch', disabled=st.session_state.analysis_in_progress):
            if game_description.strip():
                # Streamlit에서 비동기 함수 실행
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                try:
                    loop.run_until_complete(self.handle_game_search(game_description))
                except Exception as e:
                    logger.error(f"게임 검색 실행 오류: {e}", exc_info=True)
                    st.error(f"게임 검색 중 오류 발생: {e}")
                    st.session_state.analysis_in_progress = False
            else:
                st.error("게임 설명을 입력해주세요!")

    def render_game_selection(self):
        st.subheader("BGG 검색 결과")
        st.write("분석하려는 게임을 선택하세요. 너무 많은 결과가 나온 경우 설명을 더 구체적으로 입력해주세요.")
        
        results = st.session_state.bgg_search_results
        
        for game in results:
            col1, col2 = st.columns([4, 1])
            with col1:
                year = f"({game.get('year')})" if game.get('year') else ""
                st.info(f"**{game.get('name')}** {year}")
            with col2:
                if st.button("이 게임으로 분석", key=f"select_{game.get('id')}", width='stretch'):
                    # Streamlit에서 비동기 함수 실행
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    try:
                        loop.run_until_complete(self.handle_game_selection(game))
                    except Exception as e:
                        logger.error(f"게임 선택 실행 오류: {e}", exc_info=True)
                        st.error(f"게임 선택 중 오류 발생: {e}")
                        st.session_state.analysis_in_progress = False

    def render_generated_games_list(self):
        st.subheader("2. 분석된 게임 목록")
        if not st.session_state.generated_games:
            st.info("아직 분석된 게임이 없습니다.")
            return

        for game_id, game_info in st.session_state.generated_games.items():
            name = game_info.get('name', '이름 없음')
            col_name, col_button = st.columns([4, 1])
            col_name.write(f"🎮 **{name}**")
            if col_button.button("결과 보기", key=f"load_{game_id}", width='stretch'):
                st.session_state.current_game_id = game_id
                st.session_state.analysis_in_progress = False
                st.rerun()

    def run_analysis_via_a2a(self):
        """A2A를 통해 게임 UI 분석 실행"""
        game_id = st.session_state.current_game_id
        game_info = st.session_state.generated_games[game_id]
        
        # 리포트 경로 설정
        reports_path = Path(get_reports_path('boardgame_ui_generator'))
        reports_path.mkdir(parents=True, exist_ok=True)
        result_json_path = reports_path / f"game_ui_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 입력 데이터 구성 시 BGG 상세 정보 사용
        if st.session_state.bgg_game_details:
            input_description = (f"게임명: {st.session_state.bgg_game_details.get('name')}\n\n"
                                 f"설명: {st.session_state.bgg_game_details.get('description')}")
        else:
            input_description = game_info["description"]
        
        # GameUIAnalysisState 데이터 구성
        from lang_graph.table_game_mate.agents.game_ui_analyzer import GameUIAnalysisState
        input_state_data = {
            "game_description": input_description,
            "detailed_rules": game_info.get("rules", ""),
            "messages": []
        }
        
        # A2A를 통한 agent 실행
        agent_metadata = {
            "agent_id": "game_ui_analyzer",
            "agent_name": "Game UI Analyzer",
            "entry_point": "lang_graph.table_game_mate.agents.game_ui_analyzer",
            "agent_type": "langgraph_agent",
            "capabilities": ["game_analysis", "ui_spec_generation", "board_game_analysis"],
            "description": "LangGraph 기반 보드게임 UI 분석 및 명세서 생성 시스템"
        }
        
        input_data = {
            "game_description": input_state_data["game_description"],
            "detailed_rules": input_state_data["detailed_rules"],
            "messages": input_state_data["messages"],
            "result_json_path": str(result_json_path)
        }
        
        result_placeholder = st.empty()
        
        result = run_agent_via_a2a(
            placeholder=result_placeholder,
            agent_metadata=agent_metadata,
            input_data=input_data,
            result_json_path=result_json_path,
            use_a2a=True
        )
        
        if result and result.get("success") and result.get("data"):
            # 결과 처리
            final_result = result["data"]
            ui_spec = final_result.get("ui_spec", {})
            analysis_result = {
                "id": game_id,
                "success": "error_message" not in final_result or not final_result.get("error_message"),
                "name": ui_spec.get("game_name", "분석 완료"),
                "board_type": ui_spec.get("board_type", "unknown"),
                "confidence": final_result.get("confidence_score", 0.0),
                "full_spec": ui_spec,
                "analysis_summary": final_result.get("analysis_result", {}),
                "error_message": final_result.get("error_message", ""),
            }
            st.session_state.generated_games[game_id].update(analysis_result)
            st.session_state.analysis_log.append(analysis_result)
        else:
            error_msg = result.get("error", "알 수 없는 오류") if result else "결과를 받지 못했습니다"
            st.session_state.generated_games[game_id].update({
                "success": False, 
                "error_message": error_msg
            })
        
        st.session_state.analysis_in_progress = False
        st.rerun()

    async def _format_result_as_markdown(self, game_info: dict) -> str:
        """LLM을 사용해서 게임 분석 결과를 마크다운 형식으로 변환"""
        try:
            from srcs.common.llm.fallback_llm import _try_fallback_llm
            import json
            
            # Fallback LLM 가져오기
            llm = _try_fallback_llm("gemini-2.5-flash-lite", logger)
            if not llm:
                # Fallback LLM이 없으면 기본 포맷팅
                return self._format_result_basic(game_info)
            
            # JSON 결과를 마크다운으로 변환하는 프롬프트
            prompt = f"""다음 보드게임 UI 분석 결과를 읽기 쉬운 마크다운 형식의 보고서로 변환해주세요.

게임 정보:
- 게임 이름: {game_info.get('name', 'N/A')}
- 보드 타입: {game_info.get('board_type', 'N/A')}
- AI 신뢰도: {game_info.get('confidence', 0.0):.1%}

UI 명세서:
{json.dumps(game_info.get('full_spec', {}), ensure_ascii=False, indent=2)}

분석 결과:
{json.dumps(game_info.get('analysis_summary', {}), ensure_ascii=False, indent=2)}

다음 형식으로 마크다운 보고서를 작성해주세요:
1. 게임 개요 (게임 이름, 타입, 신뢰도)
2. UI 컴포넌트 설명 (각 컴포넌트의 역할과 기능)
3. 레이아웃 구조 (화면 배치 설명)
4. 상호작용 방식 (플레이어가 어떻게 게임을 조작하는지)
5. 플레이어 인터페이스 (손패, 액션 버튼, 상태 표시 등)

기술적인 JSON 구조보다는 실제 게임을 플레이할 때 어떻게 보이고 작동하는지 설명하는 방식으로 작성해주세요."""

            # LLM 호출
            if hasattr(llm, 'generate_str'):
                result = await llm.generate_str(message=prompt, request_params=None)
                return result
            else:
                return self._format_result_basic(game_info)
        except Exception as e:
            logger.error(f"마크다운 변환 오류: {e}", exc_info=True)
            return self._format_result_basic(game_info)
    
    def _format_result_basic(self, game_info: dict) -> str:
        """기본 마크다운 포맷팅 (LLM 없이)"""
        md = f"""# 🎲 {game_info.get('name', '게임')} - UI 분석 결과

## 📊 분석 개요

- **게임 이름**: {game_info.get('name', 'N/A')}
- **보드 타입**: {game_info.get('board_type', 'N/A')}
- **AI 신뢰도**: {game_info.get('confidence', 0.0):.1%}

## 🎮 UI 컴포넌트

"""
        full_spec = game_info.get('full_spec', {})
        components = full_spec.get('components', [])
        for comp in components:
            md += f"### {comp.get('name', '컴포넌트')}\n"
            md += f"- **타입**: {comp.get('type', 'N/A')}\n"
            md += f"- **설명**: {comp.get('description', 'N/A')}\n"
            md += f"- **UI 컴포넌트**: {comp.get('ui_component', 'N/A')}\n\n"
        
        md += "## 📐 레이아웃\n\n"
        layout = full_spec.get('layout', {})
        md += f"- **타입**: {layout.get('type', 'N/A')}\n"
        md += f"- **설명**: {layout.get('description', 'N/A')}\n\n"
        
        md += "## 🎯 상호작용\n\n"
        interactions = full_spec.get('interactions', [])
        for inter in interactions:
            md += f"- **{inter.get('type', 'N/A')}**: {inter.get('description', 'N/A')}\n"
        
        return md

    def render_text_based_interface(self):
        game_id = st.session_state.current_game_id
        game_info = st.session_state.generated_games.get(game_id)
        
        if not game_info:
            st.warning("게임 정보를 찾을 수 없습니다.")
            return

        st.header(f"🎲 {game_info.get('name', '게임')}: AI 분석 결과")

        if not game_info.get("success", True):
             st.error(f"분석 실패: {game_info.get('error_message', '알 수 없는 오류')}")
             return

        col1, col2, col3 = st.columns(3)
        col1.metric("AI 신뢰도", f"{game_info.get('confidence', 0.0):.1%}")
        col2.metric("보드 타입", game_info.get('board_type', "N/A"))
        col3.metric("복잡도", game_info.get('analysis_summary', {}).get('게임_복잡도', "N/A"))

        # 마크다운 형식으로 결과 표시
        with st.spinner("📝 분석 결과를 마크다운 형식으로 변환 중..."):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            try:
                markdown_result = loop.run_until_complete(self._format_result_as_markdown(game_info))
                st.markdown(markdown_result)
            except Exception as e:
                logger.error(f"마크다운 변환 실패: {e}", exc_info=True)
                # 기본 포맷팅으로 대체
                markdown_result = self._format_result_basic(game_info)
                st.markdown(markdown_result)
        
        # 원본 JSON은 접을 수 있는 섹션에 숨김
        with st.expander("🔧 원본 JSON 데이터 (개발자용)", expanded=False):
            st.json(game_info.get('full_spec', {}))

    def render_main_content(self):
        st.title("🤖 LangGraph AI Game Mate")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            with st.container(border=True):
                self.render_game_creator()
            with st.container(border=True):
                self.render_generated_games_list()
        
        with col2:
            if st.session_state.analysis_in_progress:
                # 만약 BGG 검색 결과가 있고 선택이 필요하다면, 선택 UI를 렌더링
                if st.session_state.game_selection_needed:
                    self.render_game_selection()
                else:
                    # A2A를 통해 분석 실행
                    self.run_analysis_via_a2a()
            elif st.session_state.current_game_id:
                self.render_text_based_interface()
            else:
                st.info("👈 왼쪽에서 게임을 검색하고 분석을 시작하세요!")

def main():
    ui = RealLangGraphUI()
    ui.render_main_content()

if __name__ == "__main__":
    main()
