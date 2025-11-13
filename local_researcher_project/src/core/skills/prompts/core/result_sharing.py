"""
Result Sharing 프롬프트 모듈

Agent Result Sharing에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 토론
discussion = {
    'system_message': 'You are a collaborative research agent that provides constructive feedback.',
    'template': '''제공된 연구 결과를 검토하고 건설적인 피드백을 제공하세요:

연구 결과: {research_results}
토론 맥락: {discussion_context}

피드백 작업:
1. 결과의 강점을 인정하세요
2. 개선 가능한 부분을 제안하세요
3. 추가 분석 아이디어를 제시하세요
4. 협업 기회를 모색하세요

건설적인 피드백을 반환하세요.''',
    'variables': ['research_results', 'discussion_context'],
    'description': '연구 결과 토론 및 피드백 프롬프트'
}

# 프롬프트들을 딕셔너리로 묶어서 export
result_sharing_prompts = {
    'discussion': discussion
}

