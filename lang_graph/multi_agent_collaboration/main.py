import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from .graph import app
from .security import security_manager, privacy_manager, audit_logger
from .utils import performance_manager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich 콘솔 설정
console = Console()

async def initialize_enterprise_system():
    """엔터프라이즈 시스템 초기화"""
    try:
        console.print(Panel.fit(
            "[bold blue]🔒 엔터프라이즈급 보안 시스템 초기화 중...[/bold blue]",
            border_style="blue"
        ))
        
        # 보안 시스템 초기화
        security_metrics = await security_manager.get_security_metrics()
        console.print(f"[green]✓ 보안 시스템 초기화 완료[/green]")
        
        # 프라이버시 시스템 초기화
        privacy_metrics = await privacy_manager.get_privacy_metrics()
        console.print(f"[green]✓ 프라이버시 시스템 초기화 완료[/green]")
        
        # 감사 시스템 초기화
        audit_metrics = await audit_logger.get_audit_metrics()
        console.print(f"[green]✓ 감사 시스템 초기화 완료[/green]")
        
        # 성능 시스템 초기화
        performance_metrics = await performance_manager.get_performance_metrics()
        console.print(f"[green]✓ 성능 시스템 초기화 완료[/green]")
        
        console.print(Panel.fit(
            "[bold green]🚀 모든 시스템이 성공적으로 초기화되었습니다![/bold green]",
            border_style="green"
        ))
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ 시스템 초기화 실패: {str(e)}[/red]")
        return False

async def display_system_status():
    """시스템 상태 표시"""
    try:
        # 상태 테이블 생성
        table = Table(title="🔍 시스템 상태 대시보드")
        table.add_column("시스템", style="cyan", no_wrap=True)
        table.add_column("상태", style="green")
        table.add_column("메트릭", style="yellow")
        
        # 보안 상태
        security_metrics = await security_manager.get_security_metrics()
        security_status = "🟢 정상" if security_metrics.get("active_sessions", 0) > 0 else "🔴 오류"
        table.add_row("보안", security_status, f"활성 세션: {security_metrics.get('active_sessions', 0)}")
        
        # 프라이버시 상태
        privacy_metrics = await privacy_manager.get_privacy_metrics()
        privacy_status = "🟢 정상" if privacy_metrics.get("encryption_enabled", False) else "🔴 비활성"
        table.add_row("프라이버시", privacy_status, f"암호화: {'활성' if privacy_metrics.get('encryption_enabled') else '비활성'}")
        
        # 성능 상태
        performance_metrics = await performance_manager.get_performance_metrics()
        perf_status = "🟢 정상" if performance_metrics.get("success_rate", 0) > 0.8 else "🟡 주의"
        table.add_row("성능", perf_status, f"성공률: {performance_metrics.get('success_rate', 0):.1%}")
        
        # 감사 상태
        audit_metrics = await audit_logger.get_audit_metrics()
        audit_status = "🟢 정상" if audit_metrics.get("redis_enabled", False) else "🟡 제한적"
        table.add_row("감사", audit_status, f"Redis: {'활성' if audit_metrics.get('redis_enabled') else '비활성'}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]시스템 상태 확인 실패: {str(e)}[/red]")

async def main_async():
    """비동기 메인 함수"""
    try:
        # .env 파일에서 API 키 로드
        load_dotenv()
        
        # API 키 존재 여부 확인
        if 'OPENAI_API_KEY' not in os.environ or 'TAVILY_API_KEY' not in os.environ:
            console.print("[red]❌ '.env' 파일에 OPENAI_API_KEY와 TAVILY_API_KEY를 설정해주세요.[/red]")
            return
        
        # 엔터프라이즈 시스템 초기화
        if not await initialize_enterprise_system():
            return
        
        # 시스템 상태 표시
        await display_system_status()
        
        console.print("\n" + "="*60)
        console.print("[bold blue]🔬 향상된 멀티 에이전트 협업 시스템[/bold blue]")
        console.print("="*60)
        
        # 사용자로부터 리서치 주제 입력받기
        query = console.input("[bold cyan]안녕하세요! 어떤 주제에 대한 보고서를 작성해드릴까요?[/bold cyan]\n> ")
        
        # 프라이버시 검사
        privacy_result = await privacy_manager.process_data_with_privacy(
            query, 
            context={"operation": "user_input", "source": "main"},
            operation="user_query_processing"
        )
        
        console.print(f"[green]✓ 사용자 입력 처리 완료 (프라이버시 레벨: {privacy_result['data_category']})[/green]")
        
        # 그래프 실행 (진행 상황 표시)
        inputs = {"query": privacy_result["processed_data"]}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("리서치 진행 중...", total=None)
            
            # 스트림 실행
            for event in app.stream(inputs):
                for key, value in event.items():
                    progress.update(task, description=f"단계 완료: {key}")
                    console.print(f"\n[bold green]--- 이벤트: {key} ---[/bold green]")
                    console.print(value)
                    console.print("\n" + "="*50 + "\n")
        
        # 최종 결과 출력
        final_state = app.invoke(inputs)
        final_report = final_state.get("final_report", "보고서 생성에 실패했습니다.")
        
        console.print("\n" + "="*60)
        console.print("[bold green]📋 최종 보고서[/bold green]")
        console.print("="*60)
        console.print(final_report)
        
        # 성능 메트릭 표시
        final_performance = await performance_manager.get_performance_metrics()
        console.print(f"\n[dim]성능 통계: 총 작업 {final_performance.get('total_operations', 0)}개, "
                     f"성공률 {final_performance.get('success_rate', 0):.1%}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ 오류 발생: {str(e)}[/red]")
        logger.error(f"Main execution failed: {str(e)}")

def main():
    """메인 함수"""
    try:
        # 비동기 실행
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ 사용자에 의해 중단되었습니다.[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ 치명적 오류: {str(e)}[/red]")
        logger.error(f"Critical error in main: {str(e)}")
    finally:
        # 정리 작업
        console.print("\n[dim]시스템 정리 중...[/dim]")
        try:
            # 백그라운드 작업 정리
            asyncio.run(performance_manager.cleanup_old_metrics())
            console.print("[green]✓ 정리 작업 완료[/green]")
        except Exception as e:
            console.print(f"[red]❌ 정리 작업 실패: {str(e)}[/red]")

if __name__ == "__main__":
    main() 