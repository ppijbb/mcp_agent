"""
공통 시스템 메시지 모듈

여러 agent에서 공통으로 사용되는 시스템 메시지들을 포함합니다.
"""

# 기본 연구자 시스템 메시지
researcher_base = "You are an expert researcher with comprehensive knowledge and analytical capabilities."

# 평가자 시스템 메시지
evaluator_base = "You are an expert evaluator with deep knowledge of quality assessment and critical analysis."

# 종합자 시스템 메시지
synthesizer_base = "You are an expert synthesizer skilled in integrating diverse information sources."

# 검증자 시스템 메시지
validator_base = "You are an expert validator specializing in accuracy verification and quality assurance."

# 창의성 전문가 시스템 메시지
creativity_expert = "You are a creativity expert skilled in innovative thinking and problem-solving."

# 시스템 메시지들을 딕셔너리로 묶어서 export
system_messages = {
    'researcher_base': researcher_base,
    'evaluator_base': evaluator_base,
    'synthesizer_base': synthesizer_base,
    'validator_base': validator_base,
    'creativity_expert': creativity_expert
}

