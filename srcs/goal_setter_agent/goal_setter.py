import os
import json
import httpx
import argparse
import asyncio
from typing import Dict, Any, List

# This is a self-contained script with no dependencies on the mcp_agent framework.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def create_prompt(high_level_goal: str, available_agents: List[str]) -> str:
    """Creates a sophisticated prompt to guide the LLM in goal decomposition."""
    return f"""
    You are a world-class Chief of Staff and strategic planner.
    Your task is to take a high-level, potentially ambiguous business goal and break it down into a concrete, actionable plan.

    **High-Level Goal:** "{high_level_goal}"

    **Available Agents/Tools:**
    The following agents are available to execute tasks:
    - {', '.join(available_agents)}

    **Your Process:**
    1.  **Clarify & Decompose:** Analyze the high-level goal. Identify ambiguities. Break it down into 2-4 specific, measurable, and independent sub-goals.
    2.  **Define KPIs:** For each sub-goal, define 1-2 primary Key Performance Indicators (KPIs) that directly measure its success.
    3.  **Formulate Action Plan:** For each sub-goal, devise a sequence of concrete actions to achieve the KPIs.
    4.  **Assign Agent:** For each action, identify the most suitable agent from the provided list to execute it. If no specific agent fits, suggest "GeneralPurposeAgent".

    **Output Format:**
    Your final output MUST be a single, valid JSON object. Do not include any other text or explanations.
    The JSON object should follow this precise structure:

    {{
      "original_goal": "{high_level_goal}",
      "decomposed_plan": [
        {{
          "sub_goal": "A specific, measurable sub-goal.",
          "rationale": "A brief explanation of why this sub-goal is important.",
          "kpis": [
            {{
              "name": "Name of the KPI (e.g., 'Daily Active Users')",
              "metric": "How to measure it (e.g., 'Count of unique users logging in daily')"
            }}
          ],
          "action_plan": [
            {{
              "action_item": "A concrete task to be performed.",
              "suggested_agent": "The most appropriate agent from the list."
            }}
          ]
        }}
      ]
    }}
    
    Ensure the response is a raw JSON string.
    """

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
        return json.loads(message_content)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from LLM response:\n{message_content}")
        raise

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

    # List of agents available in our ecosystem
    available_agents = [
        "SEODoctorMCPAgent",
        "ProductPlannerCoordinator",
        "BusinessStrategyMCPAgent",
        "CodeGeneratorAgent",
        "GeneralPurposeAgent" # A fallback option
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