"""
mcp_agent에 `attach_llm`을 사용하여 로컬 LLM을 연결하는 테스트 에이전트.

이 스크립트는 `basic.py`의 패턴을 따라, `AugmentedLLM`을 상속한
커스텀 LLM 클래스를 에이전트에 직접 연결하는 방법을 보여줍니다.

실행 방법:
- python tests/test_local_llm_agent.py
"""
import asyncio
import http.server
import socketserver
import threading
import requests
import json
import sys
from typing import Any

try:
    from mcp_agent.agents.agent import Agent
    from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
except ImportError as e:
    print(f"Error: {e}", file=sys.stderr)
    print("`mcp_agent` 라이브러리에서 Agent 또는 AugmentedLLM을 찾을 수 없습니다.", file=sys.stderr)
    exit(1)

USER_PREFIX_PATH = "34.47.83.72/llmservice/v1/generate"


class LocalAugmentedLLM(OpenAIAugmentedLLM):
    """로컬 LLM 서버에 연결하는 AugmentedLLM 구현체"""
    base_url: str = f"http://{USER_PREFIX_PATH}"
    model_id: str = "local-llm"

    async def generate_str(self, message: str, **kwargs: Any) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with the shortest key points only. Try to answer with Korean."},
                {"role": "user", "content": message}],
            **kwargs
        }
        
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=10)
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"로컬 LLM 서버 연결 실패: {e}"


async def main():
    """메인 실행 함수"""
    try:

        # MCP 서버가 필요 없으므로 간단하게 Agent를 생성합니다.
        agent = Agent(name="LocalLLMAttachTestAgent")
        
        print("에이전트에 LocalAugmentedLLM을 연결합니다...")
        # `attach_llm`에 LLM 클래스와 생성자 인자를 전달합니다.
        llm = await agent.attach_llm(LocalAugmentedLLM)
        print(f"LLM ({llm.__class__.__name__}) 연결 완료.")

        print("\n--- LLM과 직접 통신 테스트 ---")
        prompt1 = "Hello, what can you do?"
        print(f"You: {prompt1}")
        response1 = await llm.generate_str(prompt1)
        print(f"LLM: {response1}")

        prompt2 = "what is your model name?"
        print(f"You: {prompt2}")
        response2 = await llm.generate_str(prompt2)
        print(f"LLM: {response2}")

    except Exception as e:
        print(f"\n에이전트 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n에이전트 실행 중 오류 발생: {e}")
