"""
Autonomous Orchestrator 프롬프트 모듈

Autonomous Orchestrator에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 분석
analysis = {
    'system_message': 'You are an expert research analyst with comprehensive domain knowledge.',
    'template': '''다음 데이터를 분석하고 평가하세요:

데이터: {data}
분석 목표: {analysis_goal}

분석 작업:
1. 데이터의 품질과 완전성을 평가하세요
2. 주요 패턴과 트렌드를 식별하세요
3. 잠재적 문제를 발견하세요
4. 개선 방안을 제시하세요

분석 결과를 구조화하여 반환하세요.''',
    'variables': ['data', 'analysis_goal'],
    'description': '데이터 분석 프롬프트'
}

# 검증
verification = {
    'system_message': 'You are an expert research planner and quality auditor with deep knowledge of research methodologies and resource optimization.',
    'template': '''연구 결과를 검증하고 품질을 평가하세요:

연구 결과: {research_results}
검증 기준: {verification_criteria}

검증 작업:
1. 결과의 정확성과 신뢰성을 확인하세요
2. 방법론의 적절성을 평가하세요
3. 잠재적 편향이나 오류를 식별하세요
4. 품질 점수와 개선 제안을 제시하세요

검증 결과를 반환하세요.''',
    'variables': ['research_results', 'verification_criteria'],
    'description': '연구 결과 검증 프롬프트'
}

# 평가
evaluation = {
    'system_message': 'You are an expert research evaluator with comprehensive quality assessment capabilities.',
    'template': '''연구 프로세스와 결과를 종합적으로 평가하세요:

평가 대상: {evaluation_target}
평가 기준: {evaluation_criteria}

평가 작업:
1. 프로세스의 효과성을 평가하세요
2. 결과의 품질을 분석하세요
3. 강점과 약점을 식별하세요
4. 개선 방안을 제시하세요

평가 결과를 반환하세요.''',
    'variables': ['evaluation_target', 'evaluation_criteria'],
    'description': '프로세스 및 결과 평가 프롬프트'
}

# 종합
synthesis = {
    'system_message': 'You are an expert research synthesizer with adaptive context window capabilities.',
    'template': '''다양한 출처의 데이터를 종합하여 통합된 결과를 생성하세요:

데이터 출처: {data_sources}
종합 목표: {synthesis_goal}

종합 작업:
1. 서로 다른 출처의 정보를 통합하세요
2. 일관성과 모순을 분석하세요
3. 종합된 narrative를 구성하세요
4. 실행 가능한 결론을 도출하세요

종합 결과를 반환하세요.''',
    'variables': ['data_sources', 'synthesis_goal'],
    'description': '데이터 종합 프롬프트'
}

# 작업 분해
decomposition = {
    'system_message': 'You are an expert research project manager with deep knowledge of task decomposition and resource allocation.',
    'template': '''복잡한 연구 작업을 독립적으로 실행 가능한 하위 작업으로 분해하세요:

연구 작업: {research_task}
분해 기준: {decomposition_criteria}

작업 분해:
1. 작업의 주요 구성 요소를 식별하세요
2. 각 하위 작업의 범위를 정의하세요
3. 의존 관계를 분석하세요
4. 우선순위와 예상 시간을 설정하세요
5. 리소스 요구사항을 지정하세요

분해된 작업 목록을 반환하세요.''',
    'variables': ['research_task', 'decomposition_criteria'],
    'description': '작업 분해 프롬프트'
}

# 프롬프트들을 딕셔너리로 묶어서 export
autonomous_orchestrator_prompts = {
    'analysis': analysis,
    'verification': verification,
    'evaluation': evaluation,
    'synthesis': synthesis,
    'decomposition': decomposition
}

