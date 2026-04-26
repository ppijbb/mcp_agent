#!/usr/bin/env python3
"""
MCP Browser Client - Configuration Based

✅ 하드코딩 제거
✅ 설정 파일 기반
✅ 에러 핸들링 강화
✅ 메서드 분리
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    import streamlit as st
except ImportError:
    st = None

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config_loader import config

logger = logging.getLogger(__name__)


class MCPBrowserClient:
    """MCP Browser Client - 설정 기반"""

    def __init__(self,
                 headless: bool = None,
                 disable_gpu: bool = None,
                 streamlit_container = None,
                 screenshot_dir: Optional[str] = None):

        # 설정 파일에서 브라우저 설정 로드
        browser_config = config.get_browser_config()
        self.browser_settings = {
            'headless': headless if headless is not None else browser_config.get('headless', True),
            'disable_gpu': disable_gpu if disable_gpu is not None else browser_config.get('disable_gpu', True),
            'timeout': browser_config.get('timeout', 30000),
            'window_size': browser_config.get('window_size', [1280, 720]),
            'debug_screenshots': browser_config.get('debug_screenshots', True)
        }

        # 브라우저 설정
        self.headless = self.browser_settings['headless']
        self.disable_gpu = self.browser_settings['disable_gpu']

        # Streamlit 통합
        self.streamlit_container = streamlit_container

        # 상태 관리
        self.session: Optional[ClientSession] = None
        self.session_context = None  # 세션 컨텍스트 관리
        self.process = None  # MCP 서버 프로세스 관리
        self.current_url: Optional[str] = None
        self.last_screenshot: Optional[str] = None
        self.screenshots: List[str] = []  # 파일 경로만 저장하도록 변경

        # 성능 설정
        self.max_screenshots = browser_config.get('max_screenshots', 20)
        self.timeout = self.browser_settings['timeout']

        # 디버그 설정
        self.debug_screenshots = self.browser_settings['debug_screenshots']
        self.screenshots_dir = screenshot_dir or self._setup_screenshots_dir()

    def _setup_screenshots_dir(self) -> str:
        """
        Setup screenshots directory for debug storage.
        
        Returns:
            str: Path to the screenshots directory
            
        Creates a temporary directory for storing browser screenshots
        during debugging sessions. Directory is created if it doesn't exist.
        """
        base_dir = "tmp/debug_screenshots"
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _get_server_command(self) -> List[str]:
        """MCP 서버 명령어 생성 - 설정 기반"""
        mcp_config = config.get_mcp_server_config()
        possible_paths = mcp_config.get('possible_paths', [])

        for path in possible_paths:
            # ~ 경로 확장
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return ["node", expanded_path]

        # npm으로 설치된 패키지 찾기
        try:
            result = subprocess.run(
                ["npm", "list", "@modelcontextprotocol/server-puppeteer", "--depth=0", "--json"],
                capture_output=True, text=True, check=True
            )
            npm_data = json.loads(result.stdout)
            if "dependencies" in npm_data and "@modelcontextprotocol/server-puppeteer" in npm_data["dependencies"]:
                return ["npx", "@modelcontextprotocol/server-puppeteer"]
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"npm 패키지 검색 실패: {e}")

        # 명시적 에러 발생 - fallback 제거
        raise FileNotFoundError(
            "MCP Puppeteer 서버를 찾을 수 없습니다. "
            "다음 명령어로 설치하세요: npm install @modelcontextprotocol/server-puppeteer"
        )

    def _get_browser_args(self) -> List[str]:
        """브라우저 인수 생성 - 설정 기반"""
        args = []

        if self.headless:
            args.append("--headless=new")

        if self.disable_gpu:
            args.extend([
                "--disable-gpu",
                "--disable-gpu-sandbox",
                "--disable-software-rasterizer",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding"
            ])

        # 창 크기 설정
        window_size = self.browser_settings.get('window_size', [1280, 720])
        args.append(f"--window-size={window_size[0]},{window_size[1]}")

        # 추가 보안 설정
        args.extend([
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor"
        ])

        return args

    async def connect_to_mcp_server(self) -> bool:
        """MCP 서버 연결"""
        try:
            command = self._get_server_command()
            browser_args = self._get_browser_args()

            # 환경 변수 설정
            env = os.environ.copy()
            if self.disable_gpu:
                env.update({
                    'DISPLAY': ':99',  # Xvfb 디스플레이
                    'LIBGL_ALWAYS_INDIRECT': '1',
                    'GPU_FORCE_64BIT_PTR': '0',
                    'GPU_MAX_HEAP_SIZE': '100',
                    'GPU_USE_SYNC_OBJECTS': '1'
                })

            # 서버 파라미터 설정
            server_params = StdioServerParameters(
                command=command[0],
                args=command[1:] + [f"--args={','.join(browser_args)}"],
                env=env
            )

            # 연결 시도 및 세션 초기화
            self.session_context = stdio_client(server_params)
            receive_stream, write_stream = await self.session_context.__aenter__()

            # ClientSession을 '받는 통로'와 '보내는 통로'로 초기화
            self.session = ClientSession(receive_stream, write_stream)
            self.process = None  # self.process는 스트림이 아니므로 혼동을 막기 위해 초기화

            logger.info("✅ MCP 서버 연결 성공 및 세션 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"❌ MCP 서버 연결 실패: {e}", exc_info=True)
            return False

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self.session is not None

    async def navigate_to_url(self, url: str) -> Dict[str, Any]:
        """URL 탐색"""
        if not self.is_connected():
            return {"success": False, "error": "브라우저가 연결되지 않음"}

        try:
            result = await self.session.call_tool("puppeteer_navigate", {"url": url})

            if result and not result.isError:
                self.current_url = url
                logger.info(f"✅ 탐색 성공: {url}")
                return {"success": True, "url": url}
            else:
                error_msg = result.content[0].text if result and result.content and len(result.content) > 0 else "알 수 없는 오류"
                logger.error(f"❌ 탐색 실패: {error_msg}")
                return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"❌ 탐색 중 오류: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def take_screenshot_async(self) -> Dict[str, Any]:
        """스크린샷 캡처 (비동기)"""
        if not self.is_connected():
            return {"success": False, "error": "브라우저가 연결되지 않음"}

        try:
            result = await self.session.call_tool("puppeteer_screenshot", {})

            if result and not result.isError and result.content:
                # Base64 데이터 처리
                screenshot_data = result.content[0].text

                # 파일 저장은 항상 시도
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{timestamp}_screenshot.png"
                filepath = os.path.join(self.screenshots_dir, filename)

                try:
                    # Base64 데이터에서 이미지 저장
                    if screenshot_data.startswith('data:image'):
                        base64_data = screenshot_data.split(',')[1]
                    else:
                        base64_data = screenshot_data

                    with open(filepath, 'wb') as f:
                        f.write(base64.b64decode(base64_data))

                    logger.info(f"📸 스크린샷 저장: {filepath}")

                    # 성공적으로 저장된 경우에만 경로를 리스트에 추가
                    self.screenshots.append(filepath)

                    # 마지막 스크린샷은 base64 데이터로 streamlit UI에 표시하기 위해 유지
                    self.last_screenshot = screenshot_data

                except Exception as save_error:
                    logger.warning(f"스크린샷 저장 실패: {save_error}")
                    filepath = None

                # 최대 개수 제한 및 비동기 파일 삭제
                if len(self.screenshots) > self.max_screenshots:
                    # 오래된 파일 삭제 (비동기 방식)
                    old_files = self.screenshots[:-self.max_screenshots]
                    self.screenshots = self.screenshots[-self.max_screenshots:]
                    
                    # 백그라운드에서 파일 삭제
                    for old_file in old_files:
                        try:
                            if os.path.exists(old_file):
                                os.remove(old_file)
                        except OSError as e:
                            logger.warning(f"오래된 스크린샷 파일 삭제 실패: {e}")

                # Streamlit UI 업데이트 (컨테이너가 있는 경우)
                if self.streamlit_container:
                    try:
                        # 비동기 방식으로 이미지 표시를 피하기 위해 동기 처리
                        self.streamlit_container.image(
                            base64.b64decode(base64_data),
                            caption=f"[{datetime.now().strftime('%H:%M:%S')}] {self.current_url}",
                            use_column_width=True
                        )
                    except Exception as e:
                        logger.warning(f"Streamlit에 스크린샷 표시 실패: {e}")

                return {"success": True, "filepath": filepath, "data": screenshot_data}

            else:
                error_msg = result.content[0].text if result and result.content and len(result.content) > 0 else "스크린샷 캡처 실패"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"❌ 스크린샷 캡처 중 오류: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def clear_screenshots(self):
        """
        Clear all screenshot records and update UI.
        
        Clears the internal screenshots list and empties the Streamlit
        container if one is available. Logs the operation for debugging.
        """
        self.screenshots = []
        if self.streamlit_container:
            self.streamlit_container.empty()
        logger.info("스크린샷 기록이 삭제되었습니다.")

    async def navigate_and_capture(self, url: str) -> Dict[str, Any]:
        """URL 탐색 및 스크린샷 캡처"""
        try:
            if not self.is_connected():
                connected = await self.connect_to_mcp_server()
                if not connected:
                    return {"success": False, "error": "브라우저 연결 실패"}

            nav_result = await self.navigate_to_url(url)
            if not nav_result.get("success"):
                return nav_result

            await asyncio.sleep(1)
            screenshot_result = await self.take_screenshot_async()

            return {
                "success": True,
                "url": url,
                "screenshot": screenshot_result
            }
        except Exception as e:
            logger.error(f"탐색 및 캡처 오류: {e}")
            return {"success": False, "error": str(e)}

    async def cleanup(self):
        """
        Clean up MCP session and browser process.
        
        Properly closes the MCP session context and terminates the browser
        process. This should be called when the client is no longer needed
        to prevent resource leaks.
        """
        if self.session_context:
            try:
                await self.session_context.__aexit__(None, None, None)
                logger.info("✅ MCP 세션이 성공적으로 정리되었습니다.")
            except Exception as e:
                logger.error(f"❌ MCP 세션 정리 중 오류 발생: {e}")
            finally:
                self.session = None
                self.session_context = None
                if self.process and self.process.returncode is None:
                    self.process.terminate()
                self.process = None

    def __del__(self):
        """소멸자 - 비동기 정리 방지"""
        # 소멸자에서 비동기 작업을 피하기 위해 동기 정리 수행
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
            except Exception:
                pass
        self.session = None
        self.session_context = None
