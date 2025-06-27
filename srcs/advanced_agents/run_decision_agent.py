import asyncio
import argparse
import json
import sys
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

from srcs.advanced_agents.decision_agent import DecisionAgentMCP, MobileInteraction, UserProfile, InteractionType

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, 'value'):  # Enum
            return o.value
        return super().default(o)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Decision Agent workflow via CLI.")
    parser.add_argument("--user-profile", required=True, help="JSON string of the user profile.")
    parser.add_argument("--interaction-type", required=True, help="InteractionType enum name.")
    parser.add_argument("--app-name", required=True, help="Application or service name.")
    parser.add_argument("--context", required=True, help="JSON string of interaction context.")
    parser.add_argument("--result-json-path", required=True, help="Path to save structured JSON result.")
    args = parser.parse_args()
    return args

async def main():
    args = parse_args()

    # Deserialize inputs
    user_profile_dict = json.loads(args.user_profile)
    profile = UserProfile(**user_profile_dict)
    try:
        interaction_type = InteractionType[args.interaction_type]
    except KeyError:
        print(f"❌ Invalid interaction type: {args.interaction_type}")
        sys.exit(1)
    app_name = args.app_name
    context_dict = json.loads(args.context)

    interaction = MobileInteraction(
        interaction_type=interaction_type,
        app_name=app_name,
        timestamp=datetime.now(),
        context=context_dict
    )

    print("🚀 Starting Decision Agent workflow...")
    agent = DecisionAgentMCP(output_dir="decision_reports")
    try:
        result = await agent.analyze_and_decide(
            interaction=interaction,
            user_profile=profile
        )
        # Build structured result
        structured = {
            'success': True,
            'recommendation': result.decision.recommendation,
            'confidence_score': result.decision.confidence_score,
            'risk_level': result.decision.risk_level.value,
            'reasoning': result.decision.reasoning,
            'alternatives': result.decision.alternatives,
            'evidence': result.decision.evidence,
            'timestamp': datetime.now()
        }
        # Save to JSON
        os.makedirs(os.path.dirname(args.result_json_path), exist_ok=True)
        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump(structured, f, cls=EnhancedJSONEncoder, ensure_ascii=False, indent=2)
        print(f"✅ Decision result saved to {args.result_json_path}")
    except Exception as e:
        print(f"❌ Decision Agent workflow failed: {e}")
        import traceback; traceback.print_exc()
        structured = {'success': False, 'error': str(e)}
        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 