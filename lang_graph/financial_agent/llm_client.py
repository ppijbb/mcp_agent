import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# .env 파일에서 환경 변수 로드
# 이 파일이 있는 디렉토리를 기준으로 .env 파일을 찾습니다.
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# 성공한 코드와 동일하게 환경 변수 이름과 모델 로직을 수정
api_key = os.getenv("GEMINI_API_KEY") 
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite-preview-0607") # 기본값을 사용자가 지정한 모델로

llm = None

if not api_key:
    # 환경변수가 없을 경우 경고 메시지를 출력하고, None으로 설정
    print("⚠️ 경고: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. LLM 호출은 오류를 발생시킵니다.")
else:
    try:
        # LangChain을 통해 Gemini 모델 초기화
        llm = ChatGoogleGenerativeAI(
            model=model_name, # 환경 변수에서 읽어온 모델 이름 사용
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True # 시스템 메시지를 지원하지 않는 모델을 위한 설정
        )
        print(f"✅ Gemini LLM 클라이언트가 성공적으로 초기화되었습니다. (모델: {model_name})")
    except Exception as e:
        print(f"❌ Gemini LLM 클라이언트 초기화 중 에러 발생: {e}")

def call_llm(prompt: str) -> str:
    """
    지정된 프롬프트로 Gemini LLM을 호출하고 응답을 문자열로 반환합니다.
    API 키가 없거나 초기화에 실패하면 예외를 발생시킵니다. (NO FALLBACK)
    """
    if not llm:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았거나 LLM 클라이언트 초기화에 실패했습니다.")

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
        print(f"LLM 호출 중 에러 발생: {e}")
        raise