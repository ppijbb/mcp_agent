"""
보고서 생성 프롬프트
"""

report_generation = {
    'system_message': 'You are an expert assistant. Generate results in the exact format requested by the user. If they ask for a report, create a report. If they ask for code, create executable code. Follow the user\'s request precisely without adding unnecessary templates or structures.',
    'template': '''사용자 요청: {user_query}

연구 데이터: {research_data}

요청에 따라 정확한 형식으로 결과를 생성하세요.
불필요한 템플릿이나 구조를 추가하지 말고 사용자의 요청을 정확히 따르세요.''',
    'variables': ['user_query', 'research_data'],
    'description': '연구 결과를 바탕으로 보고서를 생성하는 프롬프트'
}

