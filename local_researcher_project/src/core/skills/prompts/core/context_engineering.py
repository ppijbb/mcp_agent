"""
Context Engineering 프롬프트 모듈

Context Engineer에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 시스템 프롬프트
system_prompt = {
    'template': '''{SYSTEM_PROMPT}''',
    'variables': ['SYSTEM_PROMPT'],
    'description': '시스템 프롬프트 템플릿'
}

# 프롬프트들을 딕셔너리로 묶어서 export
context_engineering_prompts = {
    'system_prompt': system_prompt
}
