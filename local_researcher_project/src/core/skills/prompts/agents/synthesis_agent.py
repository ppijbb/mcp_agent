"""
Synthesis Agent 프롬프트 모듈

Synthesis Agent에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 데이터 분석
data_analysis = {
    'system_message': 'You are an expert data analyst with pattern recognition capabilities.',
    'template': '''다음 데이터를 분석하세요:

데이터: {data}
분석 목표: {analysis_goal}

분석 작업:
1. 데이터의 패턴과 트렌드를 식별하세요
2. 주요 통찰력을 도출하세요
3. 데이터의 의미를 해석하세요

분석 결과를 구조화하여 반환하세요.''',
    'variables': ['data', 'analysis_goal'],
    'description': '데이터 분석 프롬프트'
}

# 출처 비교
source_comparison = {
    'system_message': 'You are an expert comparative analyst with cross-source evaluation capabilities.',
    'template': '''여러 출처의 데이터를 비교 분석하세요:

데이터 출처: {data_sources}
비교 기준: {comparison_criteria}

비교 분석:
1. 각 출처의 강점과 약점을 평가하세요
2. 일관성과 차이점을 식별하세요
3. 종합적인 평가를 제시하세요

비교 결과를 구조화하여 반환하세요.''',
    'variables': ['data_sources', 'comparison_criteria'],
    'description': '출처 비교 분석 프롬프트'
}

# 패턴 인식
pattern_recognition = {
    'system_message': 'You are an expert pattern analyst with advanced recognition capabilities.',
    'template': '''데이터에서 패턴을 식별하고 분석하세요:

데이터: {data}
패턴 유형: {pattern_type}

패턴 분석:
1. 반복되는 패턴을 식별하세요
2. 패턴의 의미를 해석하세요
3. 미래 트렌드를 예측하세요

패턴 분석 결과를 반환하세요.''',
    'variables': ['data', 'pattern_type'],
    'description': '패턴 인식 프롬프트'
}

# 예측 분석
predictive_analysis = {
    'system_message': 'You are an expert predictive analyst with forecasting capabilities.',
    'template': '''데이터를 바탕으로 예측 분석을 수행하세요:

데이터: {data}
예측 목표: {prediction_goal}

예측 분석:
1. 현재 트렌드를 분석하세요
2. 미래 시나리오를 예측하세요
3. 예측의 불확실성을 평가하세요

예측 결과를 반환하세요.''',
    'variables': ['data', 'prediction_goal'],
    'description': '예측 분석 프롬프트'
}

# 전략적 조언
strategic_advice = {
    'system_message': 'You are an expert strategic advisor with recommendation generation capabilities.',
    'template': '''데이터를 바탕으로 전략적 조언을 제시하세요:

데이터: {data}
전략적 맥락: {strategic_context}

전략적 조언:
1. 현재 상황을 분석하세요
2. 전략적 옵션을 제시하세요
3. 실행 가능한 권장사항을 제시하세요

전략적 조언을 구조화하여 반환하세요.''',
    'variables': ['data', 'strategic_context'],
    'description': '전략적 조언 프롬프트'
}

# 종합
synthesis = {
    'system_message': 'You are an expert research synthesizer with professional writing capabilities.',
    'template': '''여러 출처의 연구 데이터를 종합하여 보고서를 작성하세요:

연구 데이터: {research_data}
종합 목표: {synthesis_goal}

종합 작업:
1. 다양한 출처의 정보를 통합하세요
2. 일관된 narrative를 구성하세요
3. 실행 가능한 결론을 도출하세요

종합 보고서를 작성하세요.''',
    'variables': ['research_data', 'synthesis_goal'],
    'description': '연구 데이터 종합 프롬프트'
}

# 검증
validation = {
    'system_message': 'You are an expert quality validator with synthesis assessment capabilities.',
    'template': '''종합 결과의 품질을 검증하세요:

종합 결과: {synthesis_results}
검증 기준: {validation_criteria}

검증 작업:
1. 종합의 정확성을 검증하세요
2. 논리의 일관성을 확인하세요
3. 개선 방안을 제시하세요

검증 결과를 반환하세요.''',
    'variables': ['synthesis_results', 'validation_criteria'],
    'description': '종합 결과 검증 프롬프트'
}

# 프롬프트들을 딕셔너리로 묶어서 export
synthesis_agent_prompts = {
    'data_analysis': data_analysis,
    'source_comparison': source_comparison,
    'pattern_recognition': pattern_recognition,
    'predictive_analysis': predictive_analysis,
    'strategic_advice': strategic_advice,
    'synthesis': synthesis,
    'validation': validation
}
