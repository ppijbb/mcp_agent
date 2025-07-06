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
    print("⚠️ 경고: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. LLM 기능이 제한됩니다.")
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
    API 키가 없거나 초기화에 실패하면, 규칙 기반의 기본 응답을 반환합니다.
    """
    if not llm:
        # LLM 사용 불가 시, 간단한 규칙 기반 응답 (폴백)
        print("LLM 클라이언트가 초기화되지 않아 폴백 로직을 실행합니다.")
        if "시장 전망" in prompt:
            return "시장 데이터에 기반한 분석이 필요하지만, 현재는 시장이 혼조세인 것으로 보입니다."
        elif "투자 계획" in prompt:
            return '{"buy": [], "sell": [], "hold": ["NVDA", "AMD", "QCOM"]}'
        return "LLM 호출에 필요한 API 키가 없거나 클라이언트 초기화에 실패했습니다."

    try:
        # LangChain의 invoke 메서드를 사용
        messages = [
            SystemMessage(content="You are an expert financial analyst. Provide clear, concise, and data-driven insights."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        content = response.content
        return content if isinstance(content, str) else "Error: Empty or invalid response from LLM."
    except Exception as e:
        print(f"LLM 호출 중 에러 발생: {e}")
        return f"Error: LLM으로부터 응답을 받을 수 없습니다. 상세 정보: {e}" 