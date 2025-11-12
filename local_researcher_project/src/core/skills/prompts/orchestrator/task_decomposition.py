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

응답 형식 (JSON):
{{
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "작업 설명",
      "search_queries": ["검색 쿼리 1", "검색 쿼리 2"],
      "priority": 1,
      "estimated_time": "medium",
      "dependencies": []
    }},
    ...
  ]
}}

각 작업은:
- 독립적으로 실행 가능해야 함
- 명확한 검색 쿼리를 포함해야 함
- 우선순위와 예상 시간을 포함해야 함
- 의존성이 없어야 함 (병렬 실행을 위해)

작업 수: 3-5개 권장''',
    'variables': ['plan', 'query'],
    'description': '연구 계획을 독립적인 작업으로 분해하는 프롬프트'
}
