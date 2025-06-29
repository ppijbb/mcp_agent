import os
from dotenv import load_dotenv
from .graph import app

def main():
    # .env 파일에서 API 키 로드
    load_dotenv()

    # API 키 존재 여부 확인
    if 'OPENAI_API_KEY' not in os.environ or 'TAVILY_API_KEY' not in os.environ:
        print("'.env' 파일에 OPENAI_API_KEY와 TAVILY_API_KEY를 설정해주세요.")
        return

    # 사용자로부터 리서치 주제 입력받기
    query = input("안녕하세요! 어떤 주제에 대한 보고서를 작성해드릴까요?\n> ")

    # 그래프 실행
    # stream을 사용하면 각 단계의 진행 상황을 실시간으로 확인할 수 있습니다.
    inputs = {"query": query}
    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"--- 이벤트: {key} ---")
            print(value)
            print("\n" + "="*50 + "\n")

    # 최종 결과 출력 (스트림의 마지막 이벤트에서 final_report를 가져올 수도 있음)
    final_state = app.invoke(inputs)
    print("\n\n--- 최종 보고서 ---")
    print(final_state.get("final_report", "보고서 생성에 실패했습니다."))

if __name__ == "__main__":
    main() 