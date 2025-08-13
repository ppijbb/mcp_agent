import os
import json
import httpx
import argparse
import asyncio
from typing import Dict, Any, List
from string import Template

# This is a self-contained script with no dependencies on the mcp_agent framework.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def create_prompt(high_level_goal: str, available_agents: List[str]) -> str:
    """Create a strict, agentic prompt with JSON-only schema and constraints (Korean)."""
    agents_joined = ", ".join(available_agents)
    default_agent = available_agents[0] if available_agents else "GeneralAgent"
    tmpl = Template(
        """
역할: 수석 전략 기획 에이전트. 다음 상위 목표를 구체적이고 실행 가능한 계획으로 분해하라.
원칙: 명확성, 간결성, 결정성. 불필요한 문장/사족/사과 금지. 출력은 오직 JSON만.

상위 목표: "$high_level_goal"

사용 가능 에이전트(정확한 이름만 사용, 임의 생성 금지): $agents_joined

요구 사항:
1) 2~4개의 독립적이며 측정 가능한 하위 목표(sub_goal)를 도출하라(SMART).
2) 각 sub_goal에 KPI 1~2개를 정의하라. KPI는 name/metric/target/data_source를 포함한다.
3) 각 sub_goal에 대한 실행 계획(action_plan) 2~5개를 정의하라.
   - 각 action은 action_item, suggested_agent(반드시 위 목록 중 하나), due_days(1~30 정수),
     acceptance_criteria(검증 기준), dependencies(선택, action_item 참조 리스트)를 포함한다.
4) 각 sub_goal에 risks(선택, 최대 3개)를 기술하라.
5) 전체 성공 기준(overall_success_criteria)을 간결히 제시하라.
6) 모든 내용은 한국어로 작성하라.

출력은 아래 JSON 스키마를 오직 그대로 충족하는 단일 JSON 객체로만 반환하라. 마크다운/코드펜스/설명 금지.
{
  "original_goal": "$high_level_goal",
  "decomposed_plan": [
    {
      "sub_goal": "구체적이며 측정 가능한 하위 목표",
      "rationale": "왜 중요한지",
      "priority": "high|medium|low",
      "kpis": [
        {
          "name": "KPI 이름",
          "metric": "측정 방법",
          "target": "목표치",
          "data_source": "데이터 출처"
        }
      ],
      "action_plan": [
        {
          "action_item": "구체적 작업",
          "suggested_agent": "$default_agent",
          "due_days": 7,
          "acceptance_criteria": "완료 판정 기준",
          "dependencies": ["선행 작업 이름"]
        }
      ],
      "risks": ["위험 1", "위험 2"]
    }
  ],
  "overall_success_criteria": "전반적 성공 기준"
}
"""
    )
    return tmpl.substitute(
        high_level_goal=high_level_goal,
        agents_joined=agents_joined,
        default_agent=default_agent,
    )

async def generate_goal_plan(goal: str, agents: List[str]) -> Dict[str, Any]:
    """Generates the goal plan by calling the OpenAI API."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    prompt = create_prompt(goal, agents)
    payload = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"OpenAI API request failed: {response.text}")

    message_content = response.json()["choices"][0]["message"]["content"]
    try:
        plan = json.loads(message_content)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from LLM response:\n{message_content}")
        raise

    _validate_plan(plan, agents)
    return plan


def _validate_plan(plan: Dict[str, Any], allowed_agents: List[str]) -> None:
    """Validate structure and constraints of the generated plan. Raises ValueError on violation."""
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")

    if plan.get("original_goal") is None:
        raise ValueError("Missing 'original_goal'.")

    decomposed = plan.get("decomposed_plan")
    if not isinstance(decomposed, list) or not (2 <= len(decomposed) <= 4):
        raise ValueError("'decomposed_plan' must be a list with 2~4 items.")

    for idx, sub in enumerate(decomposed, start=1):
        if not isinstance(sub, dict):
            raise ValueError(f"sub_goal[{idx}] must be an object.")
        for key in ["sub_goal", "rationale", "priority", "kpis", "action_plan"]:
            if key not in sub:
                raise ValueError(f"sub_goal[{idx}] missing '{key}'.")

        if sub["priority"] not in ("high", "medium", "low"):
            raise ValueError(f"sub_goal[{idx}].priority must be one of high|medium|low.")

        kpis = sub.get("kpis", [])
        if not isinstance(kpis, list) or not (1 <= len(kpis) <= 2):
            raise ValueError(f"sub_goal[{idx}].kpis must contain 1~2 items.")
        for k_i, kpi in enumerate(kpis, start=1):
            if not all(k in kpi for k in ("name", "metric", "target", "data_source")):
                raise ValueError(f"sub_goal[{idx}].kpis[{k_i}] missing required fields.")

        actions = sub.get("action_plan", [])
        if not isinstance(actions, list) or not (2 <= len(actions) <= 5):
            raise ValueError(f"sub_goal[{idx}].action_plan must contain 2~5 items.")
        for a_i, act in enumerate(actions, start=1):
            for key in ("action_item", "suggested_agent", "due_days", "acceptance_criteria"):
                if key not in act:
                    raise ValueError(f"sub_goal[{idx}].action_plan[{a_i}] missing '{key}'.")
            if act["suggested_agent"] not in allowed_agents:
                raise ValueError(
                    f"sub_goal[{idx}].action_plan[{a_i}].suggested_agent must be one of {allowed_agents}."
                )
            if not isinstance(act["due_days"], int) or not (1 <= act["due_days"] <= 30):
                raise ValueError(f"sub_goal[{idx}].action_plan[{a_i}].due_days must be an integer 1~30.")

    if not plan.get("overall_success_criteria"):
        raise ValueError("Missing 'overall_success_criteria'.")

def pretty_print_plan(plan: Dict[str, Any]):
    """Prints the generated plan in a human-readable format."""
    print("=" * 80)
    print(f"🎯 High-Level Goal: {plan.get('original_goal')}")
    print("=" * 80)
    
    for i, sub_plan in enumerate(plan.get('decomposed_plan', []), 1):
        print(f"\nSub-Goal {i}: {sub_plan.get('sub_goal')}")
        print(f"  - Rationale: {sub_plan.get('rationale')}")
        
        print("  - KPIs:")
        for kpi in sub_plan.get('kpis', []):
            print(f"    - {kpi.get('name')}: {kpi.get('metric')}")
            
        print("  - Action Plan:")
        for action in sub_plan.get('action_plan', []):
            print(f"    - Task: {action.get('action_item')}")
            print(f"      -> Suggested Agent: [{action.get('suggested_agent')}]")
    print("\n" + "=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Autonomous Goal-Setting Agent.")
    parser.add_argument("--goal", required=True, help="The high-level goal to be decomposed.")
    args = parser.parse_args()

    # 사용 가능한 에이전트 목록(실제 시스템 에이전트와 일치)
    available_agents = [
        "CodeReviewAgent",
        "DocumentationAgent",
        "PerformanceAgent",
        "SecurityAgent",
        "KubernetesAgent",
    ]

    try:
        print(f"🧠 Decomposing goal: \"{args.goal}\"...")
        plan = await generate_goal_plan(args.goal, available_agents)
        print("\n✅ Successfully generated a strategic plan!")
        pretty_print_plan(plan)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 