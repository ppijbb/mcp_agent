#!/usr/bin/env python3
"""
Pages 마이그레이션 스크립트

표준 A2A 패턴으로 pages를 자동 마이그레이션하는 스크립트
"""

import re
from pathlib import Path
from typing import List, Tuple

def migrate_page_file(file_path: Path) -> Tuple[bool, str]:
    """
    단일 page 파일을 표준 A2A 패턴으로 마이그레이션
    
    Returns:
        (success, message)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # 이미 마이그레이션된 경우 스킵
        if "execute_standard_agent_via_a2a" in content or "create_standard_a2a_page" in content:
            return False, "Already migrated"
        
        # agent_type 문자열을 AgentType enum으로 변경
        content = re.sub(
            r'"agent_type":\s*"mcp_agent"',
            r'"agent_type": AgentType.MCP_AGENT',
            content
        )
        content = re.sub(
            r'"agent_type":\s*"langgraph_agent"',
            r'"agent_type": AgentType.LANGGRAPH_AGENT',
            content
        )
        content = re.sub(
            r'"agent_type":\s*"sparkleforge_agent"',
            r'"agent_type": AgentType.SPARKLEFORGE_AGENT',
            content
        )
        
        # import 추가
        if "from srcs.common.standard_a2a_page_helper import" not in content:
            # run_agent_via_a2a import 다음에 추가
            if "from srcs.common.streamlit_a2a_runner import run_agent_via_a2a" in content:
                content = content.replace(
                    "from srcs.common.streamlit_a2a_runner import run_agent_via_a2a",
                    "from srcs.common.streamlit_a2a_runner import run_agent_via_a2a\nfrom srcs.common.standard_a2a_page_helper import (\n    execute_standard_agent_via_a2a,\n    process_standard_agent_result\n)\nfrom srcs.common.agent_interface import AgentType"
                )
            elif "from srcs.common.agent_interface import AgentType" not in content:
                # 적절한 위치에 import 추가
                import_line = "from srcs.common.standard_a2a_page_helper import (\n    execute_standard_agent_via_a2a,\n    process_standard_agent_result\n)\nfrom srcs.common.agent_interface import AgentType"
                # configs.settings import 다음에 추가
                if "from configs.settings import" in content:
                    content = re.sub(
                        r'(from configs\.settings import[^\n]+\n)',
                        r'\1' + import_line + '\n',
                        content
                    )
        
        # agent_metadata와 input_data를 표준 함수 호출로 변경
        # 패턴 1: 간단한 MCP Agent
        pattern1 = re.compile(
            r'agent_metadata\s*=\s*\{[^}]+\}\s+input_data\s*=\s*\{[^}]+\}\s+result\s*=\s*run_agent_via_a2a\s*\([^)]+\)',
            re.DOTALL
        )
        
        # 패턴 2: 클래스 기반 MCP Agent
        pattern2 = re.compile(
            r'agent_metadata\s*=\s*\{[^}]+\}\s+input_data\s*=\s*\{[^}]+"class_name"[^}]+\}\s+result\s*=\s*run_agent_via_a2a\s*\([^)]+\)',
            re.DOTALL
        )
        
        # 변경사항이 있으면 저장
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True, "Migrated successfully"
        
        return False, "No changes needed"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """모든 pages 파일을 마이그레이션"""
    pages_dir = Path(__file__).parent.parent / "pages"
    
    if not pages_dir.exists():
        print(f"❌ Pages directory not found: {pages_dir}")
        return
    
    page_files = list(pages_dir.glob("*.py"))
    print(f"📁 Found {len(page_files)} page files")
    
    migrated = 0
    skipped = 0
    errors = 0
    
    for page_file in page_files:
        if page_file.name == "__init__.py":
            continue
            
        print(f"\n📄 Processing {page_file.name}...")
        success, message = migrate_page_file(page_file)
        
        if success:
            migrated += 1
            print(f"  ✅ {message}")
        elif "Already" in message or "No changes" in message:
            skipped += 1
            print(f"  ⏭️  {message}")
        else:
            errors += 1
            print(f"  ❌ {message}")
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Migrated: {migrated}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Errors: {errors}")

if __name__ == "__main__":
    main()

