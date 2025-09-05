"""
Gemini Service - gemini-cli를 통한 무료 로컬 AI 코드 리뷰

이 모듈은 gemini-cli를 사용하여 로컬에서 무료로 코드 리뷰를 수행합니다.
API 호출 없이 터미널 명령어를 통해 Google의 Gemini AI를 활용합니다.
"""

import subprocess
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

from .config import config
from .cache import cache_manager

logger = logging.getLogger(__name__)

@dataclass
class GeminiUsage:
    """Gemini 사용량 추적"""
    date: str
    request_count: int
    last_request_time: str

class GeminiService:
    """Gemini CLI 서비스"""
    
    def __init__(self):
        """Gemini 서비스 초기화"""
        self.gemini_path = config.gemini.gemini_cli_path
        self.model = config.gemini.gemini_model
        self.max_requests_per_day = config.gemini.max_requests_per_day
        self.timeout = config.gemini.timeout
        self.prompt_template = config.gemini.review_prompt_template
        
        # 사용량 추적
        self.usage_file = Path("gemini_usage.json")
        self.usage_data = self._load_usage_data()
        
        logger.info(f"Gemini Service 초기화 완료 - CLI: {self.gemini_path}, 모델: {self.model}")
    
    def _load_usage_data(self) -> Dict[str, GeminiUsage]:
        """사용량 데이터 로드"""
        try:
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    return {
                        date: GeminiUsage(**usage_data) 
                        for date, usage_data in data.items()
                    }
        except Exception as e:
            logger.error(f"사용량 데이터 로드 실패: {e}")
        
        return {}
    
    def _save_usage_data(self):
        """사용량 데이터 저장"""
        try:
            data = {
                date: {
                    "date": usage.date,
                    "request_count": usage.request_count,
                    "last_request_time": usage.last_request_time
                }
                for date, usage in self.usage_data.items()
            }
            
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"사용량 데이터 저장 실패: {e}")
    
    def _check_daily_limit(self) -> bool:
        """일일 사용량 한도 체크"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in self.usage_data:
            self.usage_data[today] = GeminiUsage(
                date=today,
                request_count=0,
                last_request_time=""
            )
        
        current_usage = self.usage_data[today]
        return current_usage.request_count < self.max_requests_per_day
    
    def _record_usage(self):
        """사용량 기록"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().isoformat()
        
        if today not in self.usage_data:
            self.usage_data[today] = GeminiUsage(
                date=today,
                request_count=0,
                last_request_time=""
            )
        
        self.usage_data[today].request_count += 1
        self.usage_data[today].last_request_time = current_time
        self._save_usage_data()
    
    def _generate_prompt(self, code: str, language: str, file_path: str, context: Dict[str, Any] = None) -> str:
        """리뷰 프롬프트 생성"""
        context = context or {}
        
        # 기본 프롬프트 템플릿 사용
        prompt = self.prompt_template.format(
            code=code,
            language=language,
            file_path=file_path
        )
        
        # 추가 컨텍스트가 있으면 추가
        if context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
            prompt += f"\n\n추가 컨텍스트:\n{context_str}"
        
        return prompt
    
    def review_code(self, code: str, language: str, file_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        코드 리뷰 수행
        
        Args:
            code (str): 리뷰할 코드
            language (str): 프로그래밍 언어
            file_path (str): 파일 경로
            context (Dict[str, Any], optional): 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 리뷰 결과
        """
        if not code or not file_path:
            raise ValueError("code와 file_path는 필수입니다.")
        
        # 캐시 확인
        cache_key = f"gemini_review:{hash(code)}:{file_path}"
        if config.optimization.cache_review_results:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"캐시된 리뷰 결과 사용: {file_path}")
                return cached_result
        
        # 일일 한도 체크
        if not self._check_daily_limit():
            raise ValueError(f"일일 Gemini 사용량 한도 초과 ({self.max_requests_per_day}회)")
        
        # 프롬프트 생성
        prompt = self._generate_prompt(code, language, file_path, context)
        
        try:
            # Gemini CLI 호출
            result = self._call_gemini_cli(prompt)
            
            # 사용량 기록
            self._record_usage()
            
            # 결과 포맷팅
            review_result = {
                "review": result,
                "language": language,
                "file_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "source": "gemini-cli"
            }
            
            # 캐시에 저장
            if config.optimization.cache_review_results:
                cache_manager.set(cache_key, review_result, ttl=config.optimization.cache_ttl_hours * 3600)
            
            logger.info(f"Gemini 리뷰 완료: {file_path}")
            return review_result
            
        except Exception as e:
            logger.error(f"Gemini 리뷰 실패: {e}")
            if config.gemini.fail_on_gemini_error:
                raise
            return {"error": str(e), "file_path": file_path}
    
    def _call_gemini_cli(self, prompt: str) -> str:
        """Gemini CLI 호출"""
        try:
            # Gemini CLI 명령어 구성
            cmd = [
                self.gemini_path,
                "-m", self.model,
                "-p", prompt
            ]
            
            logger.debug(f"Gemini CLI 명령어 실행: {' '.join(cmd[:3])}...")
            
            # 서브프로세스 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True
            )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            # 응답 검증
            if not result.stdout or not result.stdout.strip():
                raise ValueError("Gemini CLI에서 빈 응답을 받았습니다.")
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            raise ValueError(f"Gemini CLI 호출 시간 초과 ({self.timeout}초)")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Gemini CLI 실행 실패: {e.stderr}")
        except FileNotFoundError:
            raise ValueError(f"Gemini CLI를 찾을 수 없습니다: {self.gemini_path}")
        except Exception as e:
            raise ValueError(f"Gemini CLI 호출 중 오류: {e}")
    
    def review_diff(self, diff_content: str, file_path: str, language: str = None) -> Dict[str, Any]:
        """
        diff 내용 리뷰
        
        Args:
            diff_content (str): diff 내용
            file_path (str): 파일 경로
            language (str, optional): 프로그래밍 언어
            
        Returns:
            Dict[str, Any]: 리뷰 결과
        """
        if not diff_content or not file_path:
            raise ValueError("diff_content와 file_path는 필수입니다.")
        
        # 언어 감지
        if not language:
            language = self._detect_language_from_path(file_path)
        
        # diff 전용 프롬프트
        diff_prompt = f"""다음 diff 내용을 GitHub PR 리뷰 관점에서 분석해주세요:

파일: {file_path}
언어: {language}

{diff_content}

변경사항을 분석하고 다음 관점에서 리뷰해주세요:
1. 코드 품질 개선사항
2. 잠재적 버그나 문제점
3. 보안 취약점
4. 성능 최적화 제안
5. 코딩 스타일 및 가독성

구체적이고 실행 가능한 개선사항을 제안해주세요."""

        return self.review_code(diff_content, language, file_path, {"type": "diff"})
    
    def _detect_language_from_path(self, file_path: str) -> str:
        """파일 경로에서 언어 감지"""
        extension = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'matlab',
            '.sh': 'bash',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown'
        }
        return language_map.get(extension, 'unknown')
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """사용량 통계 조회"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 오늘 사용량
        today_usage = self.usage_data.get(today, GeminiUsage(today, 0, ""))
        
        # 최근 7일 사용량
        recent_usage = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            usage = self.usage_data.get(date, GeminiUsage(date, 0, ""))
            recent_usage.append({
                "date": date,
                "request_count": usage.request_count
            })
        
        return {
            "today": {
                "date": today,
                "request_count": today_usage.request_count,
                "remaining": max(0, self.max_requests_per_day - today_usage.request_count),
                "last_request": today_usage.last_request_time
            },
            "recent_7_days": recent_usage,
            "daily_limit": self.max_requests_per_day,
            "model": self.model
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Gemini CLI 상태 확인"""
        try:
            # Gemini CLI 버전 확인
            result = subprocess.run(
                [self.gemini_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    "status": "healthy",
                    "gemini_path": self.gemini_path,
                    "model": self.model,
                    "version": result.stdout.strip(),
                    "daily_usage": self.get_usage_stats()["today"]
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Gemini CLI 버전 확인 실패: {result.stderr}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Gemini CLI 상태 확인 실패: {e}"
            }
    
    def clear_usage_data(self):
        """사용량 데이터 정리"""
        self.usage_data.clear()
        if self.usage_file.exists():
            self.usage_file.unlink()
        logger.info("Gemini 사용량 데이터 정리 완료")

# 전역 인스턴스
gemini_service = GeminiService()
