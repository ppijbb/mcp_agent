"""
연구 요청 명령어
"""

import asyncio
import logging
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

logger = logging.getLogger(__name__)


async def research_command(cli, args: List[str]):
    """연구 요청 실행."""
    if not args:
        cli.console.print("[red]Usage: research <query>[/red]")
        cli.console.print("[dim]Or just type your query directly[/dim]")
        return
    
    query = " ".join(args)
    
    cli.console.print(f"\n[bold cyan]🔬 Research Request:[/bold cyan] {query}\n")
    
    try:
        from src.core.autonomous_orchestrator import AutonomousOrchestrator
        
        orchestrator = AutonomousOrchestrator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=cli.console,
        ) as progress:
            task = progress.add_task("Executing research...", total=None)
            
            result = await orchestrator.execute_full_research_workflow(query)
            
            progress.update(task, completed=True)
        
        # 결과 출력
        if isinstance(result, dict):
            if "final_synthesis" in result:
                content = result["final_synthesis"].get("content", "")
                if content:
                    cli.console.print(Panel(content, title="Research Result", border_style="green"))
            elif "content" in result:
                cli.console.print(Panel(result["content"], title="Research Result", border_style="green"))
            else:
                cli.console.print("[green]✅ Research completed[/green]")
                cli.console.print(f"[dim]Result keys: {list(result.keys())[:5]}...[/dim]")
        else:
            cli.console.print(f"[green]✅ Research completed[/green]")
            cli.console.print(str(result))
            
    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        cli.console.print(f"[red]❌ Research failed: {e}[/red]")
