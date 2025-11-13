"""
연구 계획 수립 프롬프트
"""

planning = {
    'system_message': 'You are an expert research planner with comprehensive knowledge of research methodologies and task decomposition.',
    'template': '''{instruction}

Task: Create a detailed research plan for: {user_query}

Based on previous research:
{previous_plans}

Create a comprehensive research plan with:
1. Research objectives
2. Key areas to investigate
3. Expected sources and methods
4. Success criteria

Keep it concise and actionable (max 300 words).''',
    'variables': ['instruction', 'user_query', 'previous_plans'],
    'description': '연구 계획 수립을 위한 프롬프트'
}

