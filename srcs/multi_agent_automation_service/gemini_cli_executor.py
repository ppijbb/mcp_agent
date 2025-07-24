"""
Gemini CLI Executor
==================

Multi-Agent에서 생성된 Gemini CLI 명령어를 실제로 실행하는 Executor
"""

import asyncio
import subprocess
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class GeminiExecutionResult:
    """Gemini CLI 실행 결과"""
    command: str
    output: str
    error: str
    exit_code: int
    execution_time: float
    timestamp: str

class GeminiCLIExecutor:
    """Gemini CLI 명령어 실행기"""
    
    def __init__(self, gemini_path: str = "gemini"):
        self.gemini_path = gemini_path
        self.execution_history: List[GeminiExecutionResult] = []
        
        # Gemini CLI 설치 확인
        self._check_gemini_installation()
    
    def _check_gemini_installation(self):
        """Gemini CLI 설치 확인"""
        try:
            result = subprocess.run(
                [self.gemini_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✅ Gemini CLI 설치 확인됨: {result.stdout.strip()}")
            else:
                print("⚠️ Gemini CLI 설치가 필요합니다.")
                print("설치 명령어: npx https://github.com/google-gemini/gemini-cli")
                
        except FileNotFoundError:
            print("❌ Gemini CLI가 설치되지 않았습니다.")
            print("설치 명령어: npx https://github.com/google-gemini/gemini-cli")
        except subprocess.TimeoutExpired:
            print("⚠️ Gemini CLI 응답 시간 초과")
    
    async def execute_command(self, command: str) -> GeminiExecutionResult:
        """Gemini CLI 명령어 실행"""
        start_time = datetime.now()
        
        try:
            # 명령어에서 "gemini" 부분 추출
            if command.startswith("gemini "):
                gemini_command = command[7:]  # "gemini " 제거
            else:
                gemini_command = command
            
            # Gemini CLI 실행
            process = await asyncio.create_subprocess_exec(
                self.gemini_path,
                *gemini_command.split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)  # 5분 타임아웃
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = GeminiExecutionResult(
                command=command,
                output=stdout.decode('utf-8', errors='ignore'),
                error=stderr.decode('utf-8', errors='ignore'),
                exit_code=process.returncode,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
            # 히스토리 저장
            self.execution_history.append(result)
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = GeminiExecutionResult(
                command=command,
                output="",
                error="Command execution timed out (5 minutes)",
                exit_code=-1,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = GeminiExecutionResult(
                command=command,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.execution_history.append(result)
            return result
    
    async def execute_batch_commands(self, commands: List[str]) -> List[GeminiExecutionResult]:
        """배치 명령어 실행"""
        results = []
        
        for command in commands:
            result = await self.execute_command(command)
            results.append(result)
            
            # 명령어 간 간격 (API 제한 고려)
            await asyncio.sleep(1)
        
        return results
    
    async def execute_with_context(self, command: str, context_files: List[str] = None) -> GeminiExecutionResult:
        """컨텍스트 파일과 함께 명령어 실행"""
        if not context_files:
            return await self.execute_command(command)
        
        # 컨텍스트 파일들을 명령어에 포함
        context_command = command
        for file_path in context_files:
            if os.path.exists(file_path):
                context_command += f" @{file_path}"
        
        return await self.execute_command(context_command)
    
    def get_execution_summary(self) -> str:
        """실행 히스토리 요약"""
        if not self.execution_history:
            return "실행된 명령어가 없습니다."
        
        total_commands = len(self.execution_history)
        successful_commands = sum(1 for r in self.execution_history if r.exit_code == 0)
        failed_commands = total_commands - successful_commands
        
        total_execution_time = sum(r.execution_time for r in self.execution_history)
        avg_execution_time = total_execution_time / total_commands if total_commands > 0 else 0
        
        summary = f"""
Gemini CLI 실행 요약
===================

📊 전체 통계:
- 총 실행 명령어: {total_commands}개
- 성공: {successful_commands}개
- 실패: {failed_commands}개
- 성공률: {(successful_commands/total_commands)*100:.1f}%

⏱️ 실행 시간:
- 총 실행 시간: {total_execution_time:.2f}초
- 평균 실행 시간: {avg_execution_time:.2f}초

최근 실행된 명령어:
"""
        
        # 최근 5개 명령어 표시
        for result in self.execution_history[-5:]:
            status = "✅" if result.exit_code == 0 else "❌"
            summary += f"{status} {result.command[:50]}... ({result.execution_time:.2f}초)\n"
        
        return summary
    
    def get_failed_commands(self) -> List[GeminiExecutionResult]:
        """실패한 명령어 목록"""
        return [result for result in self.execution_history if result.exit_code != 0]
    
    def get_slow_commands(self, threshold: float = 30.0) -> List[GeminiExecutionResult]:
        """느린 명령어 목록 (기본 30초 이상)"""
        return [result for result in self.execution_history if result.execution_time > threshold]
    
    def clear_history(self):
        """실행 히스토리 초기화"""
        self.execution_history.clear()
    
    def export_history(self, file_path: str):
        """실행 히스토리를 JSON 파일로 내보내기"""
        history_data = [asdict(result) for result in self.execution_history]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"실행 히스토리가 {file_path}에 저장되었습니다.")
    
    def import_history(self, file_path: str):
        """JSON 파일에서 실행 히스토리 가져오기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            self.execution_history = [
                GeminiExecutionResult(**data) for data in history_data
            ]
            
            print(f"{file_path}에서 실행 히스토리를 가져왔습니다.")
            
        except Exception as e:
            print(f"히스토리 가져오기 실패: {e}")

# 사용 예시
async def main():
    """사용 예시"""
    executor = GeminiCLIExecutor()
    
    # 단일 명령어 실행
    result = await executor.execute_command(
        "gemini '현재 디렉토리의 Python 파일들을 분석해줘'"
    )
    print(f"실행 결과: {result.exit_code}")
    print(f"출력: {result.output[:200]}...")
    
    # 배치 명령어 실행
    commands = [
        "gemini 'README.md를 업데이트해줘'",
        "gemini '코드 품질을 개선해줘'",
        "gemini '테스트 케이스를 생성해줘'"
    ]
    
    results = await executor.execute_batch_commands(commands)
    print(f"배치 실행 완료: {len(results)}개 명령어")
    
    # 실행 요약
    print(executor.get_execution_summary())
    
    # 실패한 명령어 확인
    failed_commands = executor.get_failed_commands()
    if failed_commands:
        print(f"실패한 명령어: {len(failed_commands)}개")

if __name__ == "__main__":
    asyncio.run(main()) 