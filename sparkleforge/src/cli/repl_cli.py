"""
SparkleForge REPL CLI (완전 CLI 환경)

prompt_toolkit 기반의 완전한 REPL 환경 제공.
- 히스토리 관리 (파일 기반)
- 자동완성 (명령어, 파일 경로, 세션 ID 등)
- 역검색 (Ctrl+R)
- 컬러 프롬프트 및 출력
"""

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.cli.completion import SparkleForgeCompleter
from src.cli.history import SparkleForgeHistory

logger = logging.getLogger(__name__)


class REPLCLI:
    """SparkleForge REPL CLI."""
    
    def __init__(self):
        """초기화."""
        self.console = Console()
        
        # 히스토리 초기화
        self.history_manager = SparkleForgeHistory()
        self.history = self.history_manager.get_file_history()
        
        # PromptSession 초기화
        self.session = PromptSession(
            history=self.history,
            completer=SparkleForgeCompleter(self),
            enable_history_search=True,
            complete_while_typing=True,
        )
        
        # 명령어 핸들러
        self.command_handlers = {}
        self._register_handlers()
        
        # 컨텍스트 및 체크포인트 매니저
        self.context_loader = None
        self.checkpoint_manager = None
        self.session_control = None
        
        try:
            from src.core.context_loader import ContextLoader
            from src.core.checkpoint_manager import CheckpointManager
            from src.core.session_control import get_session_control
            
            self.context_loader = ContextLoader()
            self.checkpoint_manager = CheckpointManager()
            self.session_control = get_session_control()
        except Exception as e:
            logger.warning(f"Failed to initialize context/checkpoint/session: {e}")
    
    def _register_handlers(self):
        """명령어 핸들러 등록."""
        from src.cli.commands.research import research_command
        from src.cli.commands.session import (
            session_list_command, session_show_command,
            session_pause_command, session_resume_command,
            session_cancel_command, session_delete_command,
            session_search_command, session_stats_command,
            session_tasks_command
        )
        from src.cli.commands.context import context_show_command, context_reload_command
        from src.cli.commands.checkpoint import (
            checkpoint_save_command, checkpoint_list_command,
            checkpoint_restore_command, checkpoint_delete_command
        )
        from src.cli.commands.schedule import (
            schedule_list_command, schedule_add_command,
            schedule_remove_command, schedule_enable_command,
            schedule_disable_command
        )
        from src.cli.commands.config import config_show_command, config_set_command, config_get_command
        from src.cli.commands.help import help_command
        
        self.command_handlers = {
            'research': research_command,
            'session': {
                'list': session_list_command,
                'show': session_show_command,
                'pause': session_pause_command,
                'resume': session_resume_command,
                'cancel': session_cancel_command,
                'delete': session_delete_command,
                'search': session_search_command,
                'stats': session_stats_command,
                'tasks': session_tasks_command,
            },
            'context': {
                'show': context_show_command,
                'reload': context_reload_command,
            },
            'checkpoint': {
                'save': checkpoint_save_command,
                'list': checkpoint_list_command,
                'restore': checkpoint_restore_command,
                'delete': checkpoint_delete_command,
            },
            'schedule': {
                'list': schedule_list_command,
                'add': schedule_add_command,
                'remove': schedule_remove_command,
                'enable': schedule_enable_command,
                'disable': schedule_disable_command,
            },
            'config': {
                'show': config_show_command,
                'set': config_set_command,
                'get': config_get_command,
            },
            'help': help_command,
            'exit': self._handle_exit,
            'quit': self._handle_exit,
            'clear': self._handle_clear,
        }
    
    
    async def run(self):
        """REPL 루프 실행."""
        # 시작 배너
        self._show_banner()
        
        # 컨텍스트 로드
        if self.context_loader:
            try:
                context = await self.context_loader.load_context()
                if context:
                    self.console.print("[dim]📄 Project context loaded from SPARKLEFORGE.md[/dim]\n")
            except Exception as e:
                logger.debug(f"Failed to load context: {e}")
        
        # REPL 루프
        while True:
            try:
                text = await self.session.prompt_async(
                    "[bold cyan]sparkleforge[/bold cyan]> ",
                )
                
                if not text.strip():
                    continue
                
                await self.handle_command(text)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except EOFError:
                self.console.print("\n[bold]Goodbye! 👋[/bold]")
                break
            except Exception as e:
                logger.error(f"Error in REPL CLI: {e}", exc_info=True)
                self.console.print(f"[red]❌ Error: {e}[/red]")
    
    def _show_banner(self):
        """시작 배너 표시."""
        banner = Panel(
            Text("⚒️  SparkleForge - REPL Mode", style="bold cyan"),
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(banner)
        self.console.print("[dim]Type 'help' for commands, 'exit' to quit[/dim]\n")
    
    async def handle_command(self, text: str):
        """명령어 처리."""
        try:
            # shlex로 파싱 (따옴표 처리)
            parts = shlex.split(text)
            if not parts:
                return
            
            command = parts[0].lower()
            
            if command in ['exit', 'quit']:
                await self._handle_exit()
                return
            
            if command == 'clear':
                await self._handle_clear()
                return
            
            if command == 'help':
                await self.command_handlers['help'](self.console)
                return
            
            # 명령어 라우팅
            if command in self.command_handlers:
                handler = self.command_handlers[command]
                
                if isinstance(handler, dict):
                    # 서브 명령어
                    if len(parts) < 2:
                        self.console.print(f"[red]Usage: {command} <subcommand>[/red]")
                        self.console.print(f"[dim]Type '{command} help' for subcommands[/dim]")
                        return
                    
                    subcommand = parts[1].lower()
                    if subcommand in handler:
                        await handler[subcommand](self, parts[2:])
                    else:
                        self.console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
                        self.console.print(f"[dim]Available: {', '.join(handler.keys())}[/dim]")
                else:
                    # 직접 명령어
                    await handler(self, parts[1:])
            else:
                # 연구 요청으로 처리 (명령어가 없으면)
                await self.command_handlers['research'](self, [text])
                
        except Exception as e:
            logger.error(f"Error handling command: {e}", exc_info=True)
            self.console.print(f"[red]❌ Error: {e}[/red]")
    
    async def _handle_exit(self):
        """종료 처리."""
        self.console.print("[bold]Goodbye! 👋[/bold]")
        raise EOFError()
    
    async def _handle_clear(self):
        """화면 지우기."""
        self.console.clear()
