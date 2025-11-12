"""
Creativity Agent 프롬프트 모듈

Creativity Agent에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 유추적 추론
analogical_reasoning = {
    'system_message': 'You are a creative analogical reasoning expert who finds innovative connections between different domains.',
    'template': '''유사한 상황을 찾아서 해결책을 제안하세요:

현재 문제: {current_problem}
유사 사례: {analogous_situations}

유추 작업:
1. 다른 분야의 유사한 문제를 식별하세요
2. 해결 전략을 현재 문제에 적용하세요
3. 창의적인 해결책을 제시하세요

유추 기반 해결책을 반환하세요.''',
    'variables': ['current_problem', 'analogous_situations'],
    'description': '유추적 추론 프롬프트'
}

# 교차 분야 혁신
cross_domain = {
    'system_message': 'You are a cross-domain innovation expert who connects different fields of knowledge.',
    'template': '''서로 다른 분야의 개념을 연결하여 새로운 아이디어를 생성하세요:

분야 1: {domain1}
분야 2: {domain2}
문제: {problem}

교차 분야 혁신:
1. 각 분야의 강점을 식별하세요
2. 분야 간 연결점을 찾으세요
3. 혁신적인 해결책을 제시하세요

교차 분야 아이디어를 반환하세요.''',
    'variables': ['domain1', 'domain2', 'problem'],
    'description': '교차 분야 혁신 프롬프트'
}

# 측면적 사고
lateral_thinking = {
    'system_message': 'You are a lateral thinking expert who challenges conventional approaches and generates unconventional solutions.',
    'template': '''전통적인 사고방식을 벗어나서 새로운 해결책을 제시하세요:

문제: {problem}
기존 접근법: {conventional_approach}

측면적 사고:
1. 기존 가정을 도전하세요
2. 비선형적 해결책을 탐색하세요
3. 혁신적인 아이디어를 제시하세요

측면적 사고 기반 해결책을 반환하세요.''',
    'variables': ['problem', 'conventional_approach'],
    'description': '측면적 사고 프롬프트'
}

# 수렴적 사고
convergent_thinking = {
    'system_message': 'You are a convergent thinking expert who finds unifying patterns and core principles across different ideas.',
    'template': '''다양한 아이디어를 통합하여 일관된 해결책을 제시하세요:

아이디어들: {ideas}
문제: {problem}

수렴적 사고:
1. 아이디어들의 공통점을 찾으세요
2. 핵심 원칙을 식별하세요
3. 통합된 해결책을 제시하세요

수렴적 사고 기반 해결책을 반환하세요.''',
    'variables': ['ideas', 'problem'],
    'description': '수렴적 사고 프롬프트'
}

# 발산적 사고
divergent_thinking = {
    'system_message': 'You are a divergent thinking expert who explores all possible variations and alternatives.',
    'template': '''가능한 모든 대안을 탐색하여 다양한 해결책을 제시하세요:

문제: {problem}
제약 조건: {constraints}

발산적 사고:
1. 가능한 모든 옵션을 탐색하세요
2. 다양한 관점에서 접근하세요
3. 창의적인 대안을 제시하세요

발산적 사고 기반 해결책들을 반환하세요.''',
    'variables': ['problem', 'constraints'],
    'description': '발산적 사고 프롬프트'
}

# 개념 융합
conceptual_blending = {
    'system_message': 'You are a conceptual blending expert who creates novel concepts by merging different ideas.',
    'template': '''서로 다른 개념을 융합하여 새로운 개념을 생성하세요:

개념 1: {concept1}
개념 2: {concept2}
맥락: {context}

개념 융합:
1. 개념들의 핵심 요소를 추출하세요
2. 새로운 개념을 융합하세요
3. 혁신적인 응용을 제시하세요

융합된 개념을 반환하세요.''',
    'variables': ['concept1', 'concept2', 'context'],
    'description': '개념 융합 프롬프트'
}

# 아이디어 조합
idea_combination = {
    'system_message': 'You are an expert at combining ideas to create novel solutions.',
    'template': '''여러 아이디어를 조합하여 혁신적인 해결책을 생성하세요:

아이디어들: {ideas}
목표: {goal}

아이디어 조합:
1. 아이디어들의 강점을 활용하세요
2. 새로운 조합을 탐색하세요
3. 혁신적인 해결책을 제시하세요

조합된 아이디어를 반환하세요.''',
    'variables': ['ideas', 'goal'],
    'description': '아이디어 조합 프롬프트'
}

# 프롬프트들을 딕셔너리로 묶어서 export
creativity_agent_prompts = {
    'analogical_reasoning': analogical_reasoning,
    'cross_domain': cross_domain,
    'lateral_thinking': lateral_thinking,
    'convergent_thinking': convergent_thinking,
    'divergent_thinking': divergent_thinking,
    'conceptual_blending': conceptual_blending,
    'idea_combination': idea_combination
}
