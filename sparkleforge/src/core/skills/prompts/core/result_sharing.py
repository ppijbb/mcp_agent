"""
Result Sharing 프롬프트 모듈

Agent Result Sharing에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 토론
discussion = {
    'system_message': 'You are a collaborative research agent that provides constructive feedback. You collaborate with other agents (Validation Agent) to ensure comprehensive review. IMPORTANT: All internal communication between agents must be in English. Only the final report to the user should be in the user\'s requested language.',
    'template': '''제공된 연구 결과를 검토하고 건설적인 피드백을 제공하세요:

연구 결과: {research_results}
토론 맥락: {discussion_context}

**변수 검증 (Variable Validation):**
- `research_results`가 존재하고 None이 아닌지 확인하세요. 없거나 None인 경우 명시적으로 보고하세요.
- `discussion_context`가 존재하는지 확인하세요 (None일 수 있음).

피드백 작업:
1. 결과의 강점을 인정하세요
2. 개선 가능한 부분을 제안하세요
3. 추가 분석 아이디어를 제시하세요
4. 협업 기회를 모색하세요
5. **협동 검증**: 다른 agent의 결과를 검토할 때 다음을 체크하세요:
   - URL 중복이 없는지 확인
   - Markdown 문법이 올바른지 확인
   - 소스가 완전한지 확인
   - 변수가 올바르게 검증되었는지 확인

**Agent 간 협동 (Agent Collaboration):**
- Validation Agent에게 결과 검증을 요청하세요.
- 다른 agent의 결과를 검토할 때 URL 중복, Markdown 검증 등을 체크하세요.
- **중요: 모든 내부 agent 간 소통은 영어로 해야 합니다.**

건설적인 피드백을 반환하세요.''',
    'variables': ['research_results', 'discussion_context'],
    'description': '연구 결과 토론 및 피드백 프롬프트 (협동 검증 지시사항 포함)'
}

# 프롬프트들을 딕셔너리로 묶어서 export
result_sharing_prompts = {
    'discussion': discussion
}

