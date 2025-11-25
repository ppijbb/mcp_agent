"""
검색 쿼리 생성 프롬프트
"""

query_generation = {
    'system_message': 'You are a research query generator. Generate specific search queries based on research plans.',
    'template': '''연구 계획:
{plan}

원래 질문: {query}

위 연구 계획을 바탕으로 검색에 사용할 구체적인 검색 쿼리 3-5개를 생성하세요.
각 쿼리는 서로 다른 관점이나 측면을 다루어야 합니다.
응답 형식: 각 줄에 하나의 검색 쿼리만 작성하세요. 번호나 기호 없이 쿼리만 작성하세요.''',
    'variables': ['plan', 'query'],
    'description': '연구 계획을 바탕으로 검색 쿼리를 생성하는 프롬프트'
}

