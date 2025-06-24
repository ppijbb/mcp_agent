"""
🏗️ AI Architect Agent Page

진화형 AI 아키텍처 설계 및 최적화
"""

import streamlit as st
import asyncio
from mcp_agent.app import MCPApp
from mcp_agent.config import get_settings

from srcs.common.streamlit_utils import get_agent_state
from srcs.common.streamlit_log_handler import setup_streamlit_logging
from srcs.evolutionary_ai_architect.evolutionary_ai_architect_agent import EvolutionaryAIArchitectMCP
import json

app = MCPApp(
    name="ai_architect_app",
    settings=get_settings("configs/mcp_agent.config.yaml"),
)

def format_architect_output(result) -> str:
    """Format the rich result object from the agent into a markdown string."""
    if not result or not result.success:
        return "아키텍처 설계에 실패했습니다. 다시 시도해주세요."

    best_arch = result.best_architecture
    final_metrics = result.final_metrics

    output = [
        f"## 🏗️ AI 아키텍처 진화 완료 (처리 시간: {result.processing_time:.2f}초)",
        f"**최종 세대**: {result.generation_count}",
        "---",
        "### 🏆 최적 아키텍처",
        f"- **ID**: `{best_arch.unique_id}`",
        f"- **적합도 점수**: {best_arch.fitness_score:.4f}",
        f"- **레이어 수**: {len(best_arch.layers)}",
        "#### 레이어 구성:",
    ]
    for i, layer in enumerate(best_arch.layers):
        output.append(f"  - Layer {i+1}: `{json.dumps(layer)}`")

    output.extend([
        "\n### 📊 최종 성능 메트릭",
        f"- **정확도**: {final_metrics.accuracy:.4f}",
        f"- **훈련 시간**: {final_metrics.training_time:.2f}s",
        f"- **추론 시간**: {final_metrics.inference_time:.3f}s",
        f"- **메모리 사용량**: {final_metrics.memory_usage:.2f}MB",
        f"- **에너지 효율**: {final_metrics.energy_efficiency:.3f}",
        "\n### 💡 최적화 추천 사항",
    ])
    for recommendation in result.optimization_recommendations:
        output.append(f"- {recommendation}")

    return "\n".join(output)


async def main():
    await app.initialize()

    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🏗️ AI Architect Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            진화형 AI 아키텍처 설계 및 성능 최적화 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Real-time Log Display ---
    log_expander = st.expander("실시간 실행 로그", expanded=False)
    log_container = log_expander.empty()
    # Capture logs from the root mcp_agent logger and the agent's specific logger
    setup_streamlit_logging(["mcp_agent", "evolutionary_architect"], log_container)
    # --- End Log Display ---

    # Use the state management pattern
    # Note: EvolutionaryAIArchitectMCP is not a standard Agent subclass, so we handle it directly
    if 'architect_agent' not in st.session_state:
        st.session_state.architect_agent = EvolutionaryAIArchitectMCP(output_dir="ai_architect_reports")

    agent = st.session_state.architect_agent

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요! 어떤 AI 아키텍처 문제를 해결하고 싶으신가요? 문제 상황을 자세히 설명해주세요."}
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("예: '이미지 분류를 위한 경량 CNN 아키텍처를 설계하고 싶어'"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response_md = ""
            with st.spinner("🧬 AI 아키텍처를 진화시키는 중... (시간이 다소 소요될 수 있습니다)"):
                # Call the agent's main method
                # This agent is more complex and doesn't follow the simple run(prompt) pattern
                # We call its specific `evolve_architecture` method instead.
                result = await agent.evolve_architecture(
                    problem_description=prompt,
                    # We can add UI elements later to configure these if needed
                    max_generations=5,
                    population_size=10,
                )
                response_md = format_architect_output(result)

            st.markdown(response_md)

        st.session_state["messages"].append({"role": "assistant", "content": response_md})


if __name__ == "__main__":
    asyncio.run(main()) 