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
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    import streamlit as st
except ImportError:
    st = None

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPBrowserClient:
    """MCP Browser Client - 설정 기반"""
    
    def __init__(self, 
                 headless: bool = True, 
                 disable_gpu: bool = True,
                 streamlit_container = None,
                 screenshot_dir: Optional[str] = None):
        
        # 기본 설정값 직접 지정
        self.browser_settings = {
            'headless': headless,
            'disable_gpu': disable_gpu,
            'timeout': 30000,
            'window_size': [1280, 720],
            'debug_screenshots': True
        }
        
        # 브라우저 설정
        self.headless = headless
        self.disable_gpu = disable_gpu
        
        # Streamlit 통합
        self.streamlit_container = streamlit_container
        
        # 상태 관리
        self.session: Optional[ClientSession] = None
        self.session_context = None  # 세션 컨텍스트 관리
        self.process = None  # MCP 서버 프로세스 관리
        self.current_url: Optional[str] = None
        self.last_screenshot: Optional[str] = None
        self.screenshots: List[str] = [] # 파일 경로만 저장하도록 변경
        
        # 성능 설정
        self.max_screenshots = 20 # 스크린샷 최대 개수 증가
        self.timeout = self.browser_settings.get('timeout', 30000)
        
        # 디버그 설정
        self.debug_screenshots = self.browser_settings.get('debug_screenshots', True)
        self.screenshots_dir = screenshot_dir or self._setup_screenshots_dir()
    
    def _setup_screenshots_dir(self) -> str:
        """스크린샷 디렉토리 설정"""
        base_dir = "tmp/debug_screenshots"
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    
    def _get_server_command(self) -> List[str]:
        """MCP 서버 명령어 생성 - 하드코딩 제거"""
        possible_paths = [
            "node_modules/@modelcontextprotocol/server-puppeteer/dist/index.js",
            "../node_modules/@modelcontextprotocol/server-puppeteer/dist/index.js",
            "/usr/local/lib/node_modules/@modelcontextprotocol/server-puppeteer/dist/index.js",
            os.path.expanduser("~/.npm-global/lib/node_modules/@modelcontextprotocol/server-puppeteer/dist/index.js")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return ["node", path]
        
        # 마지막 시도: npm으로 설치된 패키지 찾기
        try:
            result = subprocess.run(
                ["npm", "list", "@modelcontextprotocol/server-puppeteer", "--depth=0", "--json"],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                npm_data = json.loads(result.stdout)
                if "dependencies" in npm_data and "@modelcontextprotocol/server-puppeteer" in npm_data["dependencies"]:
                    return ["npx", "@modelcontextprotocol/server-puppeteer"]
        except Exception as e:
            logger.warning(f"npm 패키지 검색 실패: {e}")
        
        # 기본 fallback
        return ["npx", "@modelcontextprotocol/server-puppeteer"]
    
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
            self.process = None # self.process는 스트림이 아니므로 혼동을 막기 위해 초기화

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
                error_msg = result.content[0].text if result and result.content else "알 수 없는 오류"
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

                # 최대 개수 제한
                if len(self.screenshots) > self.max_screenshots:
                    # 오래된 파일 삭제
                    try:
                        os.remove(self.screenshots[0])
                    except OSError as e:
                        logger.warning(f"오래된 스크린샷 파일 삭제 실패: {e}")
                    self.screenshots = self.screenshots[-self.max_screenshots:]
                
                # Streamlit UI 업데이트 (컨테이너가 있는 경우)
                if self.streamlit_container:
                    try:
                        self.streamlit_container.image(
                            base64.b64decode(base64_data),
                            caption=f"[{datetime.now().strftime('%H:%M:%S')}] {self.current_url}",
                            use_column_width=True
                        )
                    except Exception as e:
                        logger.warning(f"Streamlit에 스크린샷 표시 실패: {e}")

                return {"success": True, "filepath": filepath, "data": screenshot_data}
            
            else:
                error_msg = result.content[0].text if result and result.content else "스크린샷 캡처 실패"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"❌ 스크린샷 캡처 중 오류: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def take_screenshot(self) -> Dict[str, Any]:
        """스크린샷 캡처 (동기 wrapper)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프가 있는 경우
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.take_screenshot_async())
                    return future.result()
            else:
                return loop.run_until_complete(self.take_screenshot_async())
        except Exception as e:
            logger.error(f"스크린샷 동기 실행 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def clear_screenshots(self):
        """스크린샷 기록 삭제"""
        self.screenshots = []
        if self.streamlit_container:
            self.streamlit_container.empty()
        logger.info("스크린샷 기록이 삭제되었습니다.")

    async def navigate_and_capture(self, url: str) -> Dict[str, Any]:
        """URL 탐색 및 스크린샷 캡처"""
        async def _navigate_and_capture():
            # 연결 확인
            if not self.is_connected():
                connected = await self.connect_to_mcp_server()
                if not connected:
                    return {"success": False, "error": "브라우저 연결 실패"}
            
            # 탐색
            nav_result = await self.navigate_to_url(url)
            if not nav_result["success"]:
                return nav_result
            
            # 잠시 대기 (페이지 로딩)
            await asyncio.sleep(2)
            
            # 스크린샷
            screenshot_result = await self.take_screenshot_async()
            
            return {
                "success": True,
                "url": url,
                "screenshot": screenshot_result
            }
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _navigate_and_capture())
                    return future.result()
            else:
                return loop.run_until_complete(_navigate_and_capture())
        except Exception as e:
            logger.error(f"탐색 및 캡처 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def search_hotels(self, destination: str, guests: int = 2, url: Optional[str] = None) -> Dict[str, Any]:
        """호텔 검색"""
        search_url = url or self.browser_settings.get('default_url', 'booking')
        
        # 실제 검색 로직 구현 필요
        # 현재는 기본 탐색만 수행
        return self.navigate_and_capture(search_url)
    
    def search_flights(self, origin: str, destination: str, url: Optional[str] = None) -> Dict[str, Any]:
        """항공편 검색"""
        search_url = url or self.browser_settings.get('default_url', 'google_flights')
        
        # 실제 검색 로직 구현 필요
        # 현재는 기본 탐색만 수행
        return self.navigate_and_capture(search_url)
    
    async def cleanup(self):
        """MCP 세션 정리"""
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
        """소멸자"""
        if self.session:
            try:
                asyncio.create_task(self.cleanup())
            except Exception:
                pass