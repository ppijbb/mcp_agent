"""
Task Analyzer 프롬프트 모듈

Task Analyzer에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 작업 분석
task_analysis = {
    'system_message': 'You are an expert task analyzer with comprehensive decomposition capabilities.',
    'template': '''다음 작업을 분석하여 세부 구성 요소로 분해하세요:

작업: {task}
맥락: {context}

작업 분석:
1. 작업의 주요 구성 요소를 식별하세요
2. 각 요소의 복잡성을 평가하세요
3. 의존 관계를 분석하세요
4. 실행 순서를 제안하세요

작업 분석 결과를 반환하세요.''',
    'variables': ['task', 'context'],
    'description': '작업 분석 및 분해 프롬프트'
}

# 프롬프트들을 딕셔너리로 묶어서 export
task_analyzer_prompts = {
    'task_analysis': task_analysis
}
