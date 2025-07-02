#!/usr/bin/env python3
"""
Filesystem MCP Server - 파일 시스템 관리

게임 데이터 저장, 설정 파일 관리, 로그 기록 등
안전한 파일 시스템 접근을 위한 MCP 서버
"""

import asyncio
import json
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiofiles
from datetime import datetime

# MCP 관련 임포트
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️ MCP 패키지가 설치되지 않음. 시뮬레이션 모드로 실행")
    MCP_AVAILABLE = False


class FilesystemMCPServer:
    """파일 시스템 관리를 위한 MCP 서버"""
    
    def __init__(self, base_path: str = "./game_data"):
        self.base_path = Path(base_path).resolve()
        self.server = Server("filesystem-server") if MCP_AVAILABLE else None
        
        # 허용된 디렉토리 (보안)
        self.allowed_paths = [
            self.base_path / "games",           # 게임 데이터
            self.base_path / "sessions",        # 게임 세션
            self.base_path / "configs",         # 설정 파일
            self.base_path / "logs",            # 로그 파일
            self.base_path / "personas",        # 페르소나 데이터
            self.base_path / "cache",           # 캐시 데이터
            self.base_path / "exports"          # 내보내기 데이터
        ]
        
        # 허용된 파일 확장자
        self.allowed_extensions = {
            '.json', '.txt', '.log', '.csv', '.yaml', '.yml', '.md'
        }
        
        # 최대 파일 크기 (MB)
        self.max_file_size = 50
        
        self._ensure_directories()
        
        if MCP_AVAILABLE and self.server:
            self._register_tools()
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        for path in self.allowed_paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def _register_tools(self):
        """MCP 도구들 등록"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="read_file",
                    description="파일 읽기",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "읽을 파일 경로"},
                            "encoding": {"type": "string", "description": "파일 인코딩 (기본값: utf-8)"}
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="write_file",
                    description="파일 쓰기",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "쓸 파일 경로"},
                            "content": {"type": "string", "description": "파일 내용"},
                            "encoding": {"type": "string", "description": "파일 인코딩 (기본값: utf-8)"},
                            "create_dirs": {"type": "boolean", "description": "디렉토리 자동 생성 (기본값: true)"}
                        },
                        "required": ["file_path", "content"]
                    }
                ),
                Tool(
                    name="list_directory",
                    description="디렉토리 목록 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory_path": {"type": "string", "description": "조회할 디렉토리 경로"},
                            "recursive": {"type": "boolean", "description": "하위 디렉토리 포함 (기본값: false)"},
                            "include_hidden": {"type": "boolean", "description": "숨김 파일 포함 (기본값: false)"}
                        },
                        "required": ["directory_path"]
                    }
                ),
                Tool(
                    name="create_directory",
                    description="디렉토리 생성",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory_path": {"type": "string", "description": "생성할 디렉토리 경로"},
                            "parents": {"type": "boolean", "description": "상위 디렉토리도 생성 (기본값: true)"}
                        },
                        "required": ["directory_path"]
                    }
                ),
                Tool(
                    name="delete_file",
                    description="파일 삭제",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "삭제할 파일 경로"},
                            "force": {"type": "boolean", "description": "강제 삭제 (기본값: false)"}
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="copy_file",
                    description="파일 복사",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_path": {"type": "string", "description": "원본 파일 경로"},
                            "destination_path": {"type": "string", "description": "대상 파일 경로"},
                            "overwrite": {"type": "boolean", "description": "덮어쓰기 허용 (기본값: false)"}
                        },
                        "required": ["source_path", "destination_path"]
                    }
                ),
                Tool(
                    name="get_file_info",
                    description="파일 정보 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "조회할 파일 경로"}
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="save_game_session",
                    description="게임 세션 저장",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"},
                            "session_data": {"type": "object", "description": "세션 데이터"},
                            "metadata": {"type": "object", "description": "메타데이터 (선택사항)"}
                        },
                        "required": ["session_id", "session_data"]
                    }
                ),
                Tool(
                    name="load_game_session",
                    description="게임 세션 로드",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "세션 ID"}
                        },
                        "required": ["session_id"]
                    }
                ),
                Tool(
                    name="list_game_sessions",
                    description="게임 세션 목록 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "number", "description": "조회할 세션 수 (기본값: 20)"},
                            "game_name": {"type": "string", "description": "게임 이름 필터 (선택사항)"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """도구 호출 처리"""
            
            try:
                if name == "read_file":
                    result = await self.read_file(**arguments)
                elif name == "write_file":
                    result = await self.write_file(**arguments)
                elif name == "list_directory":
                    result = await self.list_directory(**arguments)
                elif name == "create_directory":
                    result = await self.create_directory(**arguments)
                elif name == "delete_file":
                    result = await self.delete_file(**arguments)
                elif name == "copy_file":
                    result = await self.copy_file(**arguments)
                elif name == "get_file_info":
                    result = await self.get_file_info(**arguments)
                elif name == "save_game_session":
                    result = await self.save_game_session(**arguments)
                elif name == "load_game_session":
                    result = await self.load_game_session(**arguments)
                elif name == "list_game_sessions":
                    result = await self.list_game_sessions(**arguments)
                else:
                    result = {"error": f"알 수 없는 도구: {name}"}
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
                
            except Exception as e:
                error_result = {
                    "error": f"도구 실행 실패: {str(e)}",
                    "tool": name,
                    "arguments": arguments
                }
                return [TextContent(
                    type="text",
                    text=json.dumps(error_result, ensure_ascii=False, indent=2)
                )]
    
    def _is_path_allowed(self, file_path: str) -> bool:
        """경로 허용 여부 확인"""
        try:
            path = Path(file_path).resolve()
            
            # 허용된 디렉토리 내부인지 확인
            for allowed_path in self.allowed_paths:
                if path.is_relative_to(allowed_path):
                    return True
            
            return False
        except:
            return False
    
    def _is_extension_allowed(self, file_path: str) -> bool:
        """파일 확장자 허용 여부 확인"""
        ext = Path(file_path).suffix.lower()
        return ext in self.allowed_extensions
    
    def _get_full_path(self, file_path: str) -> Path:
        """전체 경로 반환"""
        if Path(file_path).is_absolute():
            return Path(file_path)
        else:
            return self.base_path / file_path
    
    async def read_file(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """파일 읽기"""
        
        full_path = self._get_full_path(file_path)
        
        if not self._is_path_allowed(str(full_path)):
            return {"error": f"허용되지 않은 경로: {file_path}"}
        
        if not self._is_extension_allowed(str(full_path)):
            return {"error": f"허용되지 않은 파일 형식: {full_path.suffix}"}
        
        try:
            if not full_path.exists():
                return {"error": f"파일이 존재하지 않음: {file_path}"}
            
            if not full_path.is_file():
                return {"error": f"파일이 아님: {file_path}"}
            
            # 파일 크기 확인
            file_size = full_path.stat().st_size / (1024 * 1024)  # MB
            if file_size > self.max_file_size:
                return {"error": f"파일 크기 초과: {file_size:.2f}MB > {self.max_file_size}MB"}
            
            async with aiofiles.open(full_path, 'r', encoding=encoding) as f:
                content = await f.read()
            
            return {
                "success": True,
                "content": content,
                "file_path": str(full_path),
                "size_mb": file_size,
                "encoding": encoding
            }
            
        except UnicodeDecodeError:
            return {"error": f"파일 인코딩 오류: {encoding}"}
        except Exception as e:
            return {"error": f"파일 읽기 실패: {str(e)}"}
    
    async def write_file(
        self, 
        file_path: str, 
        content: str, 
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """파일 쓰기"""
        
        full_path = self._get_full_path(file_path)
        
        if not self._is_path_allowed(str(full_path)):
            return {"error": f"허용되지 않은 경로: {file_path}"}
        
        if not self._is_extension_allowed(str(full_path)):
            return {"error": f"허용되지 않은 파일 형식: {full_path.suffix}"}
        
        try:
            # 내용 크기 확인
            content_size = len(content.encode(encoding)) / (1024 * 1024)  # MB
            if content_size > self.max_file_size:
                return {"error": f"내용 크기 초과: {content_size:.2f}MB > {self.max_file_size}MB"}
            
            # 디렉토리 생성
            if create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(full_path, 'w', encoding=encoding) as f:
                await f.write(content)
            
            return {
                "success": True,
                "file_path": str(full_path),
                "size_mb": content_size,
                "encoding": encoding
            }
            
        except Exception as e:
            return {"error": f"파일 쓰기 실패: {str(e)}"}
    
    async def list_directory(
        self, 
        directory_path: str, 
        recursive: bool = False,
        include_hidden: bool = False
    ) -> Dict[str, Any]:
        """디렉토리 목록 조회"""
        
        full_path = self._get_full_path(directory_path)
        
        if not self._is_path_allowed(str(full_path)):
            return {"error": f"허용되지 않은 경로: {directory_path}"}
        
        try:
            if not full_path.exists():
                return {"error": f"디렉토리가 존재하지 않음: {directory_path}"}
            
            if not full_path.is_dir():
                return {"error": f"디렉토리가 아님: {directory_path}"}
            
            items = []
            
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for item in full_path.glob(pattern):
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                item_info = {
                    "name": item.name,
                    "path": str(item.relative_to(self.base_path)),
                    "is_file": item.is_file(),
                    "is_directory": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
                
                items.append(item_info)
            
            # 정렬 (디렉토리 먼저, 그 다음 파일명 순)
            items.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
            
            return {
                "success": True,
                "directory_path": str(full_path),
                "items": items,
                "total_count": len(items)
            }
            
        except Exception as e:
            return {"error": f"디렉토리 조회 실패: {str(e)}"}
    
    async def create_directory(self, directory_path: str, parents: bool = True) -> Dict[str, Any]:
        """디렉토리 생성"""
        
        full_path = self._get_full_path(directory_path)
        
        if not self._is_path_allowed(str(full_path)):
            return {"error": f"허용되지 않은 경로: {directory_path}"}
        
        try:
            full_path.mkdir(parents=parents, exist_ok=True)
            
            return {
                "success": True,
                "directory_path": str(full_path),
                "created": not full_path.exists()
            }
            
        except Exception as e:
            return {"error": f"디렉토리 생성 실패: {str(e)}"}
    
    async def delete_file(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        """파일 삭제"""
        
        full_path = self._get_full_path(file_path)
        
        if not self._is_path_allowed(str(full_path)):
            return {"error": f"허용되지 않은 경로: {file_path}"}
        
        try:
            if not full_path.exists():
                return {"error": f"파일이 존재하지 않음: {file_path}"}
            
            # 중요 파일 보호 (force가 False인 경우)
            if not force and full_path.name in ['config.json', 'settings.json']:
                return {"error": f"중요 파일은 force=true로만 삭제 가능: {file_path}"}
            
            if full_path.is_file():
                full_path.unlink()
            elif full_path.is_dir():
                shutil.rmtree(full_path)
            
            return {
                "success": True,
                "deleted_path": str(full_path)
            }
            
        except Exception as e:
            return {"error": f"파일 삭제 실패: {str(e)}"}
    
    async def copy_file(
        self, 
        source_path: str, 
        destination_path: str, 
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """파일 복사"""
        
        source_full = self._get_full_path(source_path)
        dest_full = self._get_full_path(destination_path)
        
        if not self._is_path_allowed(str(source_full)):
            return {"error": f"허용되지 않은 원본 경로: {source_path}"}
        
        if not self._is_path_allowed(str(dest_full)):
            return {"error": f"허용되지 않은 대상 경로: {destination_path}"}
        
        try:
            if not source_full.exists():
                return {"error": f"원본 파일이 존재하지 않음: {source_path}"}
            
            if dest_full.exists() and not overwrite:
                return {"error": f"대상 파일이 이미 존재함: {destination_path}"}
            
            # 디렉토리 생성
            dest_full.parent.mkdir(parents=True, exist_ok=True)
            
            if source_full.is_file():
                shutil.copy2(source_full, dest_full)
            else:
                shutil.copytree(source_full, dest_full, dirs_exist_ok=overwrite)
            
            return {
                "success": True,
                "source_path": str(source_full),
                "destination_path": str(dest_full)
            }
            
        except Exception as e:
            return {"error": f"파일 복사 실패: {str(e)}"}
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """파일 정보 조회"""
        
        full_path = self._get_full_path(file_path)
        
        if not self._is_path_allowed(str(full_path)):
            return {"error": f"허용되지 않은 경로: {file_path}"}
        
        try:
            if not full_path.exists():
                return {"error": f"파일이 존재하지 않음: {file_path}"}
            
            stat = full_path.stat()
            
            # 파일 해시 (작은 파일만)
            file_hash = None
            if full_path.is_file() and stat.st_size < 1024 * 1024:  # 1MB 미만
                with open(full_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
            
            return {
                "success": True,
                "path": str(full_path),
                "name": full_path.name,
                "extension": full_path.suffix,
                "is_file": full_path.is_file(),
                "is_directory": full_path.is_dir(),
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "hash_md5": file_hash
            }
            
        except Exception as e:
            return {"error": f"파일 정보 조회 실패: {str(e)}"}
    
    async def save_game_session(
        self, 
        session_id: str, 
        session_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """게임 세션 저장"""
        
        try:
            session_file = self.base_path / "sessions" / f"{session_id}.json"
            
            # 세션 데이터 구성
            save_data = {
                "session_id": session_id,
                "session_data": session_data,
                "metadata": metadata or {},
                "saved_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            async with aiofiles.open(session_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(save_data, ensure_ascii=False, indent=2))
            
            return {
                "success": True,
                "session_id": session_id,
                "file_path": str(session_file),
                "saved_at": save_data["saved_at"]
            }
            
        except Exception as e:
            return {"error": f"세션 저장 실패: {str(e)}"}
    
    async def load_game_session(self, session_id: str) -> Dict[str, Any]:
        """게임 세션 로드"""
        
        try:
            session_file = self.base_path / "sessions" / f"{session_id}.json"
            
            if not session_file.exists():
                return {"error": f"세션 파일이 존재하지 않음: {session_id}"}
            
            async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                save_data = json.loads(content)
            
            return {
                "success": True,
                "session_id": session_id,
                "session_data": save_data.get("session_data", {}),
                "metadata": save_data.get("metadata", {}),
                "saved_at": save_data.get("saved_at"),
                "version": save_data.get("version", "unknown")
            }
            
        except json.JSONDecodeError:
            return {"error": f"세션 파일 형식 오류: {session_id}"}
        except Exception as e:
            return {"error": f"세션 로드 실패: {str(e)}"}
    
    async def list_game_sessions(
        self, 
        limit: int = 20, 
        game_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """게임 세션 목록 조회"""
        
        try:
            sessions_dir = self.base_path / "sessions"
            
            if not sessions_dir.exists():
                return {"success": True, "sessions": [], "total_count": 0}
            
            sessions = []
            
            for session_file in sessions_dir.glob("*.json"):
                try:
                    async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        save_data = json.loads(content)
                    
                    # 게임 이름 필터링
                    if game_name:
                        session_game = save_data.get("session_data", {}).get("game_config", {}).get("target_game_name", "")
                        if game_name.lower() not in session_game.lower():
                            continue
                    
                    session_info = {
                        "session_id": save_data.get("session_id"),
                        "game_name": save_data.get("session_data", {}).get("game_config", {}).get("target_game_name", "Unknown"),
                        "player_count": save_data.get("session_data", {}).get("game_config", {}).get("desired_player_count", 0),
                        "saved_at": save_data.get("saved_at"),
                        "version": save_data.get("version", "unknown"),
                        "file_size": session_file.stat().st_size
                    }
                    
                    sessions.append(session_info)
                    
                except:
                    # 손상된 세션 파일 무시
                    continue
            
            # 저장 시간 순으로 정렬 (최신 먼저)
            sessions.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
            
            # 제한 적용
            sessions = sessions[:limit]
            
            return {
                "success": True,
                "sessions": sessions,
                "total_count": len(sessions),
                "limit": limit
            }
            
        except Exception as e:
            return {"error": f"세션 목록 조회 실패: {str(e)}"}


# MCP 서버 실행
async def main():
    """Filesystem MCP 서버 실행"""
    
    if not MCP_AVAILABLE:
        print("❌ MCP 패키지가 필요합니다: pip install mcp")
        return
    
    filesystem_server = FilesystemMCPServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await filesystem_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="filesystem-server",
                server_version="1.0.0",
                capabilities={}
            )
        )


if __name__ == "__main__":
    asyncio.run(main())