from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from .config import get_llm_config

# LLM 클라이언트 인스턴스 - 초기화 시 생성됨
llm: ChatGoogleGenerativeAI = None


def initialize_llm_client() -> None:
    """LLM 클라이언트 초기화 - 설정 로드 후 호출"""
    global llm
    
    try:
        config = get_llm_config()
        
        # LangChain을 통해 Gemini 모델 초기화
        llm = ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.api_key,
            temperature=config.temperature,
            convert_system_message_to_human=True
        )
        print(f"✅ Gemini LLM 클라이언트가 성공적으로 초기화되었습니다. (모델: {config.model})")
        
    except Exception as e:
        error_msg = f"LLM 클라이언트 초기화 실패: {e}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

def call_llm(prompt: str) -> str:
    """
    지정된 프롬프트로 Gemini LLM을 호출하고 응답을 문자열로 반환합니다.
    초기화되지 않은 경우 RuntimeError 발생 (NO FALLBACK)
    """
    if llm is None:
        raise RuntimeError("LLM 클라이언트가 초기화되지 않았습니다. initialize_llm_client()를 먼저 호출하세요.")

    try:
        # LangChain의 invoke 메서드를 사용
        messages = [
            SystemMessage(content=(
                "당신은 지시를 엄격히 준수하는 수석 금융 에이전트다.\n"
                "- 목표: 입력 데이터로부터 일관되고 검증 가능한 결론을 산출한다.\n"
                "- 원칙: 명확성, 간결성, 근거 기반, 결정성(회피적 표현 금지).\n"
                "- 금지: 사족/서론/사과/불필요한 수사. 요구된 출력 이외의 텍스트.\n"
                "- 형식: 요청된 출력 스키마와 언어(한국어)를 반드시 준수한다."
            )),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        content = response.content
        return content if isinstance(content, str) else "Error: Empty or invalid response from LLM."
    except Exception as e:
        error_msg = f"LLM 호출 중 에러 발생: {e}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)