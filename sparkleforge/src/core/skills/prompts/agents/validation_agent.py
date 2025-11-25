"""
Validation Agent 프롬프트 모듈

Validation Agent에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 목표 일치성
objective_alignment = {
    'system_message': 'You are an expert research validator evaluating objective alignment.',
    'template': '''연구 결과가 원래 목표와 일치하는지 검증하세요:

원래 목표: {original_objective}
연구 결과: {research_results}

검증 작업:
1. 목표와 결과의 일치성을 평가하세요
2. 누락된 측면을 식별하세요
3. 추가 조사가 필요한 부분을 제안하세요

검증 결과를 반환하세요.''',
    'variables': ['original_objective', 'research_results'],
    'description': '목표 일치성 검증 프롬프트'
}

# 품질 평가
quality_assessment = {
    'system_message': 'You are an expert quality assessor.',
    'template': '''연구 결과의 전반적인 품질을 평가하세요:

연구 결과: {research_results}
품질 기준: {quality_criteria}

품질 평가:
1. 각 품질 측면을 평가하세요
2. 강점과 약점을 식별하세요
3. 개선 방안을 제시하세요

품질 평가 결과를 반환하세요.''',
    'variables': ['research_results', 'quality_criteria'],
    'description': '품질 평가 프롬프트'
}

# 완전성 평가
completeness_evaluation = {
    'system_message': 'You are an expert completeness evaluator.',
    'template': '''연구 결과의 완전성을 평가하세요:

연구 결과: {research_results}
완전성 기준: {completeness_criteria}

완전성 평가:
1. 커버된 범위를 분석하세요
2. 누락된 부분을 식별하세요
3. 추가 조사가 필요한 영역을 제안하세요

완전성 평가 결과를 반환하세요.''',
    'variables': ['research_results', 'completeness_criteria'],
    'description': '완전성 평가 프롬프트'
}

# 정확성 평가
accuracy_assessment = {
    'system_message': 'You are an expert accuracy assessor.',
    'template': '''연구 결과의 정확성을 평가하세요:

연구 결과: {research_results}
정확성 기준: {accuracy_criteria}

정확성 평가:
1. 사실 관계를 검증하세요
2. 오류를 식별하세요
3. 정확성 수준을 평가하세요

정확성 평가 결과를 반환하세요.''',
    'variables': ['research_results', 'accuracy_criteria'],
    'description': '정확성 평가 프롬프트'
}

# 관련성 평가
relevance_evaluation = {
    'system_message': 'You are an expert relevance evaluator.',
    'template': '''연구 결과의 관련성을 평가하세요:

연구 쿼리: {query}
연구 결과: {research_results}

관련성 평가:
1. 쿼리와 결과의 관련성을 분석하세요
2. 관련성 수준을 평가하세요
3. 개선 방안을 제시하세요

관련성 평가 결과를 반환하세요.''',
    'variables': ['query', 'research_results'],
    'description': '관련성 평가 프롬프트'
}

# 프롬프트들을 딕셔너리로 묶어서 export
validation_agent_prompts = {
    'objective_alignment': objective_alignment,
    'quality_assessment': quality_assessment,
    'completeness_evaluation': completeness_evaluation,
    'accuracy_assessment': accuracy_assessment,
    'relevance_evaluation': relevance_evaluation
}

