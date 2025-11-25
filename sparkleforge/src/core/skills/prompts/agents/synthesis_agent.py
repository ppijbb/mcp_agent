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
    'system_message': 'You are an expert research synthesizer with professional writing capabilities and DEEP ANALYTICAL THINKING. You create insightful, comprehensive reports through deep reflection and critical analysis.',
    'template': '''여러 출처의 연구 데이터를 종합하여 보고서를 작성하세요:

연구 데이터: {research_data}
종합 목표: {synthesis_goal}

**DEEP SYNTHESIS PROCESS - 깊이 있는 사고를 통해 종합하세요:**

1. **현재 상태 분석 (Current State Analysis)**:
   - 현재 상황은 무엇인가? 우리가 알고 있는 것은 무엇인가?
   - 맥락과 배경은 무엇인가?
   - 주요 사실, 트렌드, 발전 상황은 무엇인가?

2. **패턴 인식 (Pattern Recognition)**:
   - 여러 출처에서 나타나는 패턴, 트렌드, 관계는 무엇인가?
   - 어떤 연결고리와 상관관계가 있는가?
   - 역사적 맥락이나 선례는 무엇인가?

3. **비판적 통합 (Critical Integration)**:
   - 서로 다른 정보들이 어떻게 연결되는가?
   - 전체적인 그림은 무엇인가?
   - 어떤 관점들이 있고, 어떤 것이 누락되었는가?

4. **통찰력 도출 (Insight Generation)**:
   - 더 깊은 통찰, 함의, 의미는 무엇인가?
   - 이 정보가 의미하는 바는 무엇인가?
   - 어떤 질문이 남아있는가?

**종합 보고서 작성** (깊이 있는 사고를 반영):
1. **현재 상태 섹션**: 현재 상태, 맥락, 알려진 정보에 대한 명확한 평가
2. **깊이 있는 분석**: 단순한 사실 나열이 아닌 패턴, 연결, 함의 분석
3. **비판적 통찰**: 깊은 사고를 통해 도출된 의미 있는 통찰
4. **종합적 이해**: 깊은 이해를 보여주는 완전한 그림
5. **실행 가능한 결론**: 표면적 사실이 아닌 깊은 분석에 기반한 결론

단순히 발견 사항을 보고하는 것이 아니라, 다음을 포함한 **깊은 이해**를 제공하세요:
- 현재 상태와 맥락
- 패턴 인식과 연결
- 비판적 통찰과 함의
- 종합적 분석
- 의미 있는 결론

종합 보고서를 작성하세요.''',
    'variables': ['research_data', 'synthesis_goal'],
    'description': '연구 데이터 종합 프롬프트 (깊이 있는 사고 포함)'
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

