#!/usr/bin/env python3
"""
Services Layer Tests

서비스 레이어의 각 서비스에 대한 단위 테스트입니다.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.github_service import GitHubService
from services.mcp_service import MCPService
from services.review_service import ReviewService
from services.webhook_service import WebhookService

class TestGitHubService:
    """GitHub 서비스 테스트"""
    
    @patch('services.github_service.Github')
    def test_github_service_initialization(self, mock_github):
        """GitHub 서비스 초기화 테스트"""
        # Mock 설정
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github.return_value.get_user.return_value = mock_user
        
        # 서비스 초기화
        service = GitHubService()
        
        # 검증
        assert service.github is not None
        mock_github.assert_called_once()
    
    @patch('services.github_service.Github')
    def test_get_repository_success(self, mock_github):
        """저장소 조회 성공 테스트"""
        # Mock 설정
        mock_repo = Mock()
        mock_github.return_value.get_repo.return_value = mock_repo
        
        service = GitHubService()
        result = service.get_repository("owner/repo")
        
        # 검증
        assert result == mock_repo
        mock_github.return_value.get_repo.assert_called_once_with("owner/repo")
    
    @patch('services.github_service.Github')
    def test_get_repository_invalid_format(self, mock_github):
        """저장소 이름 형식 오류 테스트"""
        service = GitHubService()
        
        with pytest.raises(ValueError, match="저장소 이름 형식이 올바르지 않습니다"):
            service.get_repository("invalid-repo-name")

class TestMCPService:
    """MCP 서비스 테스트"""
    
    @patch('services.mcp_service.MultiServerMCPClient')
    @patch('services.mcp_service.ChatOpenAI')
    def test_mcp_service_initialization(self, mock_chat, mock_client):
        """MCP 서비스 초기화 테스트"""
        # Mock 설정
        mock_client.return_value.get_tools.return_value = [Mock(), Mock()]
        
        service = MCPService()
        
        # 검증
        assert service.mcp_client is not None
        assert service.agent is not None
        assert service.langgraph_app is not None
        assert len(service.tools) == 2
    
    @patch('services.mcp_service.MultiServerMCPClient')
    @patch('services.mcp_service.ChatOpenAI')
    def test_analyze_code(self, mock_chat, mock_client):
        """코드 분석 테스트"""
        # Mock 설정
        mock_tool = Mock()
        mock_tool.name = "github_tool"
        mock_tool.invoke.return_value = {"result": "analysis result"}
        mock_client.return_value.get_tools.return_value = [mock_tool]
        
        service = MCPService()
        result = service.analyze_code("def test(): pass", "python")
        
        # 검증
        assert "timestamp" in result
        assert result["language"] == "python"
        assert "analysis_results" in result

class TestReviewService:
    """리뷰 서비스 테스트"""
    
    def test_should_review_pr_with_keywords(self):
        """리뷰 키워드가 있는 PR 테스트"""
        mock_github_service = Mock()
        mock_mcp_service = Mock()
        
        service = ReviewService(mock_github_service, mock_mcp_service)
        
        # @review-bot 키워드가 있는 PR
        pr_body = "This PR fixes a bug @review-bot"
        result = service.should_review_pr(pr_body)
        
        assert result is True
    
    def test_should_review_pr_without_keywords(self):
        """리뷰 키워드가 없는 PR 테스트"""
        mock_github_service = Mock()
        mock_mcp_service = Mock()
        
        service = ReviewService(mock_github_service, mock_mcp_service)
        
        # 키워드가 없는 PR
        pr_body = "This PR fixes a bug"
        result = service.should_review_pr(pr_body)
        
        assert result is False
    
    def test_detect_language_python(self):
        """Python 언어 감지 테스트"""
        mock_github_service = Mock()
        mock_mcp_service = Mock()
        
        service = ReviewService(mock_github_service, mock_mcp_service)
        
        # Python 파일들
        mock_files = [Mock(filename="test.py"), Mock(filename="main.py")]
        language = service._detect_language(mock_files)
        
        assert language == "python"
    
    def test_detect_language_javascript(self):
        """JavaScript 언어 감지 테스트"""
        mock_github_service = Mock()
        mock_mcp_service = Mock()
        
        service = ReviewService(mock_github_service, mock_mcp_service)
        
        # JavaScript 파일들
        mock_files = [Mock(filename="app.js"), Mock(filename="utils.js")]
        language = service._detect_language(mock_files)
        
        assert language == "javascript"

class TestWebhookService:
    """웹훅 서비스 테스트"""
    
    def test_verify_signature_valid(self):
        """유효한 서명 검증 테스트"""
        mock_review_service = Mock()
        service = WebhookService(mock_review_service)
        
        # Mock 설정
        with patch('services.webhook_service.config') as mock_config:
            mock_config.github.webhook_secret = "test_secret"
            
            payload = b'{"test": "data"}'
            signature = "sha256=valid_signature"
            
            with patch('services.webhook_service.hmac.compare_digest', return_value=True):
                result = service.verify_signature(payload, signature)
                assert result is True
    
    def test_parse_webhook_payload(self):
        """웹훅 페이로드 파싱 테스트"""
        mock_review_service = Mock()
        service = WebhookService(mock_review_service)
        
        payload = b'{"action": "opened", "pull_request": {"number": 123}}'
        result = service.parse_webhook_payload(payload)
        
        assert result["action"] == "opened"
        assert result["pull_request"]["number"] == 123
    
    def test_should_process_pr_event_opened(self):
        """opened 액션 PR 이벤트 처리 테스트"""
        mock_review_service = Mock()
        service = WebhookService(mock_review_service)
        
        event_data = {
            "action": "opened",
            "pull_request": {
                "state": "open",
                "draft": False,
                "body": "@review-bot please review"
            }
        }
        
        with patch('services.webhook_service.config') as mock_config:
            mock_config.github.auto_review_enabled = True
            mock_config.github.require_explicit_review_request = True
            
            result = service.should_process_pr_event(event_data)
            assert result is True
    
    def test_extract_pr_info(self):
        """PR 정보 추출 테스트"""
        mock_review_service = Mock()
        service = WebhookService(mock_review_service)
        
        event_data = {
            "action": "opened",
            "pull_request": {
                "number": 123,
                "title": "Test PR",
                "body": "Test body",
                "state": "open",
                "draft": False,
                "head": {"ref": "feature-branch"},
                "base": {"ref": "main"}
            },
            "repository": {
                "full_name": "owner/repo",
                "name": "repo",
                "owner": {"login": "owner"}
            }
        }
        
        result = service.extract_pr_info(event_data)
        
        assert result["pr_number"] == 123
        assert result["repo_full_name"] == "owner/repo"
        assert result["pr_title"] == "Test PR"
        assert result["action"] == "opened"

if __name__ == "__main__":
    pytest.main([__file__])
