"""
Evaluation Agent 프롬프트 모듈

Evaluation Agent에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 평가
evaluation = {
    'system_message': 'You are an expert research evaluator with comprehensive quality assessment capabilities.',
    'template': '''다음 연구 결과를 평가하세요:

연구 결과: {research_results}
평가 기준: {evaluation_criteria}

다음 측면에서 평가하세요:
1. 품질 (Quality): 정보의 정확성과 신뢰성
2. 완전성 (Completeness): 쿼리에 대한 답변의 완전성
3. 관련성 (Relevance): 쿼리와의 관련성
4. 구조 (Structure): 결과의 조직화와 명확성
5. 유용성 (Usefulness): 실질적인 활용 가능성

각 측면에 대해 점수(1-10)와 자세한 설명을 제공하세요.
종합 평가와 개선 제안을 포함하세요.''',
    'variables': ['research_results', 'evaluation_criteria'],
    'description': '연구 결과 평가 프롬프트'
}

# 프롬프트들을 딕셔너리로 묶어서 export
evaluation_agent_prompts = {
    'evaluation': evaluation
}
