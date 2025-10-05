#!/usr/bin/env python3
"""
Hobby Starter Pack Agent - 레거시 메인 실행 스크립트
새로운 app.py를 사용하는 것을 권장합니다.

이 파일은 하위 호환성을 위해 유지됩니다.
실제 실행은 app.py를 사용해주세요:

python app.py

또는 직접 모드 설정:
HSP_MODE=server python app.py  # API 서버만 실행  
"""

import asyncio
import uuid
import sys
import os

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("⚠️  main.py는 레거시 파일입니다.")
print("📝 새로운 app.py를 사용해주세요:")
print("   python app.py")
print("\n🔄 기존 테스트 코드 실행 중... (하위 호환성)")

from .autogen.agents import HSPAutoGenAgents
from .mcp.manager import MCPServerManager
from .langgraph_workflow.workflow import HSPLangGraphWorkflow

async def main():
    """Main function to run the Hobby Starter Pack Agent."""
    
    # 1. LLM Configuration (replace with your actual configuration)
    # It is recommended to use environment variables for API keys.
    llm_config = {
        "config_list": [
            {
                "model": "gemini-4",
                # "api_key": "sk-...", # Use environment variable OPENAI_API_KEY
            }
        ],
        "cache_seed": 42, # Use a number or None to disable caching.
    }

    # 2. Initialize the core components
    autogen_agents = HSPAutoGenAgents(llm_config=llm_config)
    mcp_manager = MCPServerManager()
    
    # 3. Initialize the LangGraph workflow
    hsp_workflow_app = HSPLangGraphWorkflow(
        autogen_agents=autogen_agents,
        mcp_manager=mcp_manager
    ).workflow
    
    # 4. Prepare a sample input for the workflow
    # In a real application, this would come from a user request.
    user_id = str(uuid.uuid4())
    initial_input = {"user_id": user_id}
    
    print("🚀 Starting Hobby Starter Pack Agent Workflow...")
    print(f"   User ID: {user_id}")
    
    # 5. Invoke the workflow
    # The `astream` method returns an async generator of states.
    final_state = None
    async for state in hsp_workflow_app.astream(initial_input):
        print("\n" + "="*50)
        # Print the current node that was just executed
        # The `__end__` node is the final state.
        last_node = list(state.keys())[-1]
        print(f"✅ Executed Node: '{last_node}'")
        print("─"*50)
        
        # You can inspect the state at each step
        # print(state[last_node]) 
        
        if "__end__" in state:
            final_state = state
            print("🏁 Workflow finished!")

    # 6. Print the final state
    print("\n" + "="*50)
    print("🔍 Final Workflow State:")
    if final_state:
        # Removing the compiled graph object for cleaner printing
        final_state.pop('__input__', None)
        print(final_state)
    else:
        print("Workflow did not reach the end state.")
    print("="*50)


if __name__ == "__main__":
    # To run this async function, we use asyncio.run()
    try:
        asyncio.run(main())
        print("\n💡 다음부터는 직접 app.py를 실행해주세요:")
        print("   python app.py")
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
        print("\n💡 다음부터는 직접 app.py를 실행해주세요:")
        print("   python app.py") 