#!/usr/bin/env python3
"""GitHub MCP 서버 실제 연결 및 작업 처리 테스트"""

import asyncio
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 설정 로드
from src.core.researcher_config import load_config_from_env
config = load_config_from_env()

from src.core.mcp_integration import get_mcp_hub, execute_tool

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_github_mcp():
    """GitHub MCP 서버 실제 작업 테스트"""
    print("=" * 80)
    print("🐙 GitHub MCP 서버 실제 작업 처리 테스트")
    print("=" * 80)
    
    hub = get_mcp_hub()
    
    try:
        # MCP 서버 초기화
        print("\n1️⃣ MCP 서버 초기화 중...")
        await hub.initialize_mcp()
        
        # GitHub 서버 확인
        if 'github' not in hub.mcp_sessions:
            print("❌ GitHub 서버가 연결되지 않았습니다")
            print(f"연결된 서버: {list(hub.mcp_sessions.keys())}")
            return False
        
        print("✅ GitHub 서버 연결 확인됨")
        
        # 사용 가능한 도구 확인
        github_tools = hub.mcp_tools_map.get('github', {})
        print(f"✅ 사용 가능한 GitHub 도구: {len(github_tools)}개")
        print(f"   예시: {list(github_tools.keys())[:5]}")
        
        # 테스트 1: search_repositories (인증 불필요)
        print("\n2️⃣ 테스트: 리포지토리 검색 (search_repositories)")
        print("   파라미터: query='modelcontextprotocol', limit=3")
        
        result = await execute_tool('github::search_repositories', {
            'query': 'modelcontextprotocol',
            'limit': 3
        })
        
        print(f"\n   결과:")
        print(f"   - success: {result.get('success')}")
        print(f"   - 실행 시간: {result.get('execution_time', 0):.2f}초")
        
        if result.get('success'):
            data = result.get('data', {})
            print(f"   - 데이터 타입: {type(data).__name__}")
            
            # 결과가 CallToolResult 형태로 반환된 경우 JSON 파싱 시도
            if isinstance(data, dict) and 'result' in data:
                result_str = str(data['result'])
                # JSON 추출 시도
                import json
                import re
                
                # JSON 객체 찾기 (더 넓은 범위로)
                json_match = re.search(r'\{[^{}]*"total_count"[^{}]*"items"[^{}]*\[.*?\].*?\}', result_str, re.DOTALL)
                if not json_match:
                    # 더 간단한 패턴 시도
                    json_match = re.search(r'\{.*?"total_count".*?"items".*?\}', result_str, re.DOTALL)
                
                if json_match:
                    try:
                        json_str = json_match.group()
                        # 이스케이프된 따옴표 처리
                        json_str = json_str.replace('\\n', '\n').replace('\\"', '"')
                        json_data = json.loads(json_str)
                        if 'items' in json_data:
                            items = json_data['items']
                            total = json_data.get('total_count', len(items))
                            print(f"   ✅ 검색 성공! 총 {total}개 리포지토리 발견 (표시: {min(len(items), 3)}개)")
                            print(f"\n   📋 검색 결과:")
                            for i, repo in enumerate(items[:3], 1):
                                name = repo.get('name') or repo.get('full_name') or 'N/A'
                                desc = repo.get('description') or ''
                                url = repo.get('html_url') or repo.get('url') or ''
                                print(f"   {i}. {name}")
                                if desc:
                                    print(f"      설명: {desc[:80]}...")
                                if url:
                                    print(f"      URL: {url}")
                            return True
                    except (json.JSONDecodeError, KeyError) as e:
                        # JSON 파싱 실패 시 원본 데이터에서 직접 추출 시도
                        if '"total_count"' in result_str and '"items"' in result_str:
                            # 간단한 추출
                            total_match = re.search(r'"total_count":\s*(\d+)', result_str)
                            if total_match:
                                total = int(total_match.group(1))
                                print(f"   ✅ 검색 성공! 총 {total}개 리포지토리 발견")
                                print(f"   (상세 결과는 JSON 파싱 필요)")
                                return True
            
            if isinstance(data, dict):
                keys = list(data.keys())
                print(f"   - 데이터 키: {keys[:10]}")
                
                # items 또는 repositories 키 확인
                items = None
                if 'items' in data:
                    items = data['items']
                elif 'repositories' in data:
                    items = data['repositories']
                elif 'results' in data:
                    items = data['results']
                
                if items and isinstance(items, list) and len(items) > 0:
                    print(f"   ✅ 검색 성공! {len(items)}개 리포지토리 발견")
                    print(f"\n   📋 검색 결과:")
                    for i, repo in enumerate(items[:3], 1):
                        name = repo.get('name') or repo.get('full_name') or repo.get('repo') or 'N/A'
                        desc = repo.get('description') or repo.get('desc') or ''
                        url = repo.get('html_url') or repo.get('url') or ''
                        print(f"   {i}. {name}")
                        if desc:
                            print(f"      설명: {desc[:80]}...")
                        if url:
                            print(f"      URL: {url}")
                    return True
                else:
                    print(f"   ⚠️ 리포지토리 목록을 찾을 수 없습니다")
                    print(f"   데이터 구조: {str(data)[:500]}")
            elif isinstance(data, list):
                if len(data) > 0:
                    print(f"   ✅ 검색 성공! {len(data)}개 리포지토리 발견")
                    print(f"\n   📋 검색 결과:")
                    for i, repo in enumerate(data[:3], 1):
                        if isinstance(repo, dict):
                            name = repo.get('name') or repo.get('full_name') or 'N/A'
                            print(f"   {i}. {name}")
                        else:
                            print(f"   {i}. {str(repo)[:100]}")
                    return True
                else:
                    print(f"   ⚠️ 빈 결과 반환")
            else:
                print(f"   ⚠️ 예상치 못한 데이터 타입")
                print(f"   데이터: {str(data)[:500]}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"   ❌ 실패: {error}")
            
            # 인증 오류인지 확인
            if '401' in str(error) or 'unauthorized' in str(error).lower() or 'token' in str(error).lower():
                print(f"\n   💡 인증이 필요합니다. GITHUB_TOKEN 환경 변수를 설정하세요.")
        
        # 테스트 2: search_code (인증 불필요) - 파라미터 이름이 'q'임
        print("\n3️⃣ 테스트: 코드 검색 (search_code)")
        print("   파라미터: q='MCP server'")
        
        result2 = await execute_tool('github::search_code', {
            'q': 'MCP server'
        })
        
        print(f"\n   결과:")
        print(f"   - success: {result2.get('success')}")
        
        if result2.get('success'):
            data2 = result2.get('data', {})
            if isinstance(data2, dict) and ('items' in data2 or 'results' in data2):
                items2 = data2.get('items', data2.get('results', []))
                if items2 and len(items2) > 0:
                    print(f"   ✅ 코드 검색 성공! {len(items2)}개 결과 발견")
                    return True
        
        return False
        
    except Exception as e:
        print(f"\n❌ 테스트 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await hub.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_github_mcp())
    print("\n" + "=" * 80)
    if success:
        print("✅ GitHub MCP 서버를 통한 실제 작업 처리 성공!")
    else:
        print("❌ GitHub MCP 서버 작업 처리 실패 또는 부분 성공")
    print("=" * 80)
    sys.exit(0 if success else 1)

