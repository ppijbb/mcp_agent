"""
Task Analyzer 프롬프트 모듈

Task Analyzer에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 작업 분석
task_analysis = {
    'system_message': 'You are an expert task analyzer with comprehensive decomposition capabilities. You validate variables and determine collaboration needs with other agents. IMPORTANT: All internal communication between agents must be in English. Only the final report to the user should be in the user\'s requested language.',
    'template': '''다음 작업을 분석하여 세부 구성 요소로 분해하세요:

작업: {task}
맥락: {context}

**변수 검증 (Variable Validation):**
- `task`가 존재하고 None이 아닌지 확인하세요. 없거나 None인 경우 명시적으로 보고하세요.
- `context`가 존재하는지 확인하세요 (None일 수 있음).
- 빈 값이나 불완전한 데이터가 있는 경우 이를 명시하세요.

작업 분석:
1. 작업의 주요 구성 요소를 식별하세요
2. 각 요소의 복잡성을 평가하세요
3. 의존 관계를 분석하세요
4. 실행 순서를 제안하세요
5. **협동 필요성 판단**: 다른 agent와의 협동이 필요한지 판단하세요
   - Validation Agent가 필요한지 판단
   - Research Agent가 필요한지 판단
   - Synthesis Agent가 필요한지 판단
   - 다른 agent와의 협동 필요성 평가

**중요: 모든 내부 agent 간 소통은 영어로 해야 합니다.**

작업 분석 결과를 반환하세요.''',
    'variables': ['task', 'context'],
    'description': '작업 분석 및 분해 프롬프트 (변수 검증 및 협동 필요성 판단 포함)'
}

# 프롬프트들을 딕셔너리로 묶어서 export
task_analyzer_prompts = {
    'task_analysis': task_analysis
}

