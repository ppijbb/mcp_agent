"""
작업 분해 프롬프트
"""

task_decomposition = {
    'system_message': 'You are a task decomposition agent. Split research plans into independent parallel tasks.',
    'template': '''연구 계획:
{plan}

원래 질문: {query}

위 연구 계획을 분석하여 여러 독립적으로 실행 가능한 연구 작업으로 분할하세요.
각 작업은 별도의 연구자(ExecutorAgent)가 동시에 처리할 수 있어야 합니다.

⚠️ 중요: 검색 쿼리는 반드시 원래 질문("{query}")과 직접 관련된 실제 연구 주제여야 합니다.
작업 분할 방법론, 병렬화 전략, 태스크 분할 사례 등 메타 정보는 검색 쿼리가 아닙니다.
각 검색 쿼리는 원래 질문의 특정 측면이나 하위 주제를 다루어야 합니다.

응답 형식 (JSON):
{{
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "작업 설명 (원래 질문의 특정 측면)",
      "search_queries": ["원래 질문과 직접 관련된 검색 쿼리 1", "원래 질문과 직접 관련된 검색 쿼리 2"],
      "priority": 1,
      "estimated_time": "medium",
      "dependencies": []
    }},
    ...
  ]
}}

각 작업은:
- 독립적으로 실행 가능해야 함
- 검색 쿼리는 반드시 원래 질문("{query}")의 실제 내용과 직접 관련되어야 함
- 작업 분할 방법론, 병렬화, 태스크 분할 등 메타 정보는 검색 쿼리가 아님
- 우선순위와 예상 시간을 포함해야 함
- 의존성이 없어야 함 (병렬 실행을 위해)

작업 수: 3-5개 권장''',
    'variables': ['plan', 'query'],
    'description': '연구 계획을 독립적인 작업으로 분해하는 프롬프트'
}

