"""
ERA Agent 서버 자동 관리 모듈

ERA Agent 서버의 자동 시작, 종료, 상태 모니터링을 담당합니다.
"""

import asyncio
import logging
import os
import signal
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any
import atexit
import httpx

logger = logging.getLogger(__name__)


class ERAServerManager:
    """ERA Agent 서버 자동 관리 클래스"""
    
    def __init__(
        self,
        agent_binary_path: Optional[str] = None,
        server_url: str = "http://localhost:8080",
        server_addr: str = ":8080",
        auto_start: bool = True
    ):
        """
        ERA 서버 관리자 초기화
        
        Args:
            agent_binary_path: ERA Agent 바이너리 경로 (None이면 자동 탐지)
            server_url: 서버 URL
            server_addr: 서버 주소 (예: ":8080")
            auto_start: 자동 시작 여부
        """
        self.agent_binary_path = agent_binary_path or self._find_agent_binary()
        self.server_url = server_url
        self.server_addr = server_addr
        self.auto_start = auto_start
        self._process: Optional[subprocess.Popen] = None
        self._started_by_us = False
        
        # 종료 시 정리
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        logger.info(f"Received signal {signum}, cleaning up ERA server...")
        self.cleanup()
    
    def _find_agent_binary(self) -> Optional[str]:
        """
        ERA Agent 바이너리 경로 자동 탐지 (자동 빌드 포함)
        
        Returns:
            바이너리 경로 또는 None
        """
        # 1. 프로젝트 내부 경로 우선 확인 (가장 가능성 높음)
        project_root = Path(__file__).parent.parent.parent.parent
        project_paths = [
            project_root / "open_researcher" / "ERA" / "era-agent" / "agent",
            project_root.parent / "open_researcher" / "ERA" / "era-agent" / "agent",
            project_root / "era-agent" / "agent",
            project_root / "agent",
        ]
        
        # 2. 일반적인 시스템 경로
        system_paths = [
            "agent",  # PATH에 있는 경우
            "era-agent",
            "/usr/local/bin/agent",
            "/usr/bin/agent",
            os.path.expanduser("~/.local/bin/agent"),
            os.path.expanduser("~/go/bin/agent"),
        ]
        
        # 3. 환경변수 확인 (선택사항)
        env_path = os.getenv("ERA_AGENT_BINARY")
        if env_path:
            system_paths.insert(0, env_path)
        
        # 4. PATH에서 찾기
        agent_path = shutil.which("agent")
        if agent_path:
            system_paths.insert(0, agent_path)
        
        # 모든 경로 확인
        all_paths = project_paths + system_paths
        
        for path in all_paths:
            path_str = str(path) if isinstance(path, Path) else path
            if os.path.isfile(path_str) and os.access(path_str, os.X_OK):
                logger.info(f"Found ERA Agent binary at: {path_str}")
                return path_str
        
        # 바이너리를 찾지 못했으면 자동 빌드 시도
        logger.info("ERA Agent binary not found, attempting to build...")
        built_path = self._try_build_agent(project_root)
        if built_path:
            return built_path
        
        logger.error("ERA Agent binary not found and could not be built automatically.")
        logger.error("Please install Go and build ERA Agent manually:")
        logger.error("  1. sudo apt install golang-go")
        logger.error("  2. cd open_researcher/ERA/era-agent")
        logger.error("  3. make agent")
        return None
    
    def _try_build_agent(self, project_root: Path) -> Optional[str]:
        """
        ERA Agent 자동 빌드 시도
        
        Args:
            project_root: 프로젝트 루트 경로
            
        Returns:
            빌드된 바이너리 경로 또는 None
        """
        # Go가 설치되어 있는지 확인
        go_path = shutil.which("go")
        if not go_path:
            logger.warning("Go is not installed. Cannot build ERA Agent automatically.")
            return None
        
        # ERA Agent 소스 경로 확인
        era_agent_dir = project_root / "open_researcher" / "ERA" / "era-agent"
        if not era_agent_dir.exists():
            # 상위 디렉토리 확인
            era_agent_dir = project_root.parent / "open_researcher" / "ERA" / "era-agent"
            if not era_agent_dir.exists():
                logger.warning(f"ERA Agent source not found at {era_agent_dir}")
                return None
        
        agent_binary = era_agent_dir / "agent"
        
        # 이미 빌드되어 있으면 반환
        if agent_binary.exists() and agent_binary.is_file() and os.access(str(agent_binary), os.X_OK):
            logger.info(f"Found existing ERA Agent binary at: {agent_binary}")
            return str(agent_binary)
        
        # Makefile 확인
        makefile = era_agent_dir / "Makefile"
        if not makefile.exists():
            logger.warning(f"Makefile not found at {makefile}")
            return None
        
        # 빌드 시도
        logger.info(f"Building ERA Agent from {era_agent_dir}...")
        try:
            import subprocess
            result = subprocess.run(
                ["make", "agent"],
                cwd=str(era_agent_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            if result.returncode == 0 and agent_binary.exists():
                # 실행 권한 부여
                os.chmod(str(agent_binary), 0o755)
                logger.info(f"Successfully built ERA Agent at: {agent_binary}")
                return str(agent_binary)
            else:
                logger.error(f"Failed to build ERA Agent: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("ERA Agent build timed out")
            return None
        except Exception as e:
            logger.error(f"Error building ERA Agent: {e}")
            return None
    
    def is_server_running(self) -> bool:
        """
        ERA 서버가 실행 중인지 확인
        
        Returns:
            서버가 실행 중이면 True
        """
        try:
            # HTTP 요청으로 확인
            response = httpx.get(f"{self.server_url}/api/vm/list", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_port_available(self, port: int) -> bool:
        """
        포트가 사용 가능한지 확인
        
        Args:
            port: 포트 번호
            
        Returns:
            포트가 사용 가능하면 True
        """
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result != 0  # 연결 실패 = 포트 사용 가능
        except Exception:
            return True  # 확인 실패 시 사용 가능한 것으로 간주
    
    async def start_server(self) -> bool:
        """
        ERA 서버 시작
        
        Returns:
            시작 성공 여부
        """
        # 이미 실행 중이면 스킵
        if self.is_server_running():
            logger.info("ERA server is already running")
            return True
        
        if not self.agent_binary_path:
            logger.warning("ERA Agent binary not found, cannot start server")
            return False
        
        # 포트 확인
        try:
            port = int(self.server_addr.split(':')[-1])
            if not self._check_port_available(port):
                logger.warning(f"Port {port} is already in use. Assuming ERA server is running externally.")
                return True
        except (ValueError, IndexError):
            pass
        
        try:
            # 환경변수 설정 (ERA Agent가 필요로 하는 환경변수)
            env = os.environ.copy()
            
            # ERA Agent가 사용하는 환경변수들
            era_env_vars = [
                "AGENT_STATE_DIR",
                "KRUNVM_DATA_DIR",
                "CONTAINERS_STORAGE_CONF",
                "CONTAINERS_STORAGE_CONFIG",
                "CONTAINERS_POLICY",
                "CONTAINERS_REGISTRIES_CONF",
                "DYLD_LIBRARY_PATH",  # macOS에서 krunvm 라이브러리 경로
            ]
            
            # 기존 환경변수 유지
            for var in era_env_vars:
                if var in env:
                    logger.debug(f"Using existing {var}={env[var]}")
            
            # 서버 시작
            cmd = [self.agent_binary_path, "server", "--addr", self.server_addr]
            logger.info(f"Starting ERA server: {' '.join(cmd)}")
            
            self._process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # 새 세션으로 시작하여 부모 프로세스 종료 시 영향 없도록
            )
            
            self._started_by_us = True
            
            # 서버 시작 대기 (최대 10초)
            max_wait = 10
            for i in range(max_wait):
                await asyncio.sleep(1)
                if self.is_server_running():
                    logger.info(f"ERA server started successfully on {self.server_url}")
                    return True
                
                # 프로세스가 종료되었는지 확인
                if self._process.poll() is not None:
                    stdout, stderr = self._process.communicate()
                    logger.error(f"ERA server process exited with code {self._process.returncode}")
                    logger.error(f"stdout: {stdout.decode('utf-8', errors='ignore')}")
                    logger.error(f"stderr: {stderr.decode('utf-8', errors='ignore')}")
                    self._process = None
                    self._started_by_us = False
                    return False
            
            logger.warning(f"ERA server did not respond within {max_wait} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start ERA server: {e}")
            if self._process:
                try:
                    self._process.terminate()
                except:
                    pass
                self._process = None
            self._started_by_us = False
            return False
    
    def stop_server(self):
        """ERA 서버 중지 (우리가 시작한 경우만)"""
        if not self._started_by_us or not self._process:
            return
        
        try:
            logger.info("Stopping ERA server...")
            self._process.terminate()
            
            # 최대 5초 대기
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("ERA server did not terminate gracefully, forcing kill...")
                self._process.kill()
                self._process.wait()
            
            logger.info("ERA server stopped")
        except Exception as e:
            logger.error(f"Error stopping ERA server: {e}")
        finally:
            self._process = None
            self._started_by_us = False
    
    def cleanup(self):
        """정리 작업"""
        self.stop_server()
    
    async def ensure_server_running(self) -> bool:
        """
        서버가 실행 중인지 확인하고, 필요시 시작
        
        Returns:
            서버가 실행 중이면 True
        """
        if self.is_server_running():
            return True
        
        if not self.auto_start:
            logger.warning("ERA server is not running and auto_start is disabled")
            return False
        
        return await self.start_server()
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        서버 정보 반환
        
        Returns:
            서버 정보 딕셔너리
        """
        return {
            "binary_path": self.agent_binary_path,
            "server_url": self.server_url,
            "server_addr": self.server_addr,
            "auto_start": self.auto_start,
            "is_running": self.is_server_running(),
            "started_by_us": self._started_by_us,
            "process_pid": self._process.pid if self._process else None
        }

