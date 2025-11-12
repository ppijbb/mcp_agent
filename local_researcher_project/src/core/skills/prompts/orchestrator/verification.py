"""
검증 프롬프트
"""

verification = {
    'system_message': 'You are a verification agent. Verify if search results are relevant and reliable.',
    'template': '''다음 검색 결과를 검증하세요:

검색 쿼리: {query}
검색 결과: {results}

다음 기준으로 검증하세요:
1. 관련성: 쿼리와의 관련성 (1-10점)
2. 신뢰성: 출처의 신뢰성 (1-10점)
3. 최신성: 정보의 최신성 (1-10점)
4. 완전성: 쿼리에 대한 답변 완전성 (1-10점)

각 결과에 대해 점수와 이유를 제공하세요.
전체 검색 품질을 평가하세요.''',
    'variables': ['query', 'results'],
    'description': '검색 결과를 검증하는 프롬프트'
}
