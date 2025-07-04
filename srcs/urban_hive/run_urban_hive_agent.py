import argparse
import json
import sys
from pathlib import Path
import asyncio
from dataclasses import asdict, is_dataclass
from enum import Enum

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.urban_hive.urban_hive_agent import UrbanHiveMCPAgent, UrbanDataCategory
from mcp_agent.workflows.llm.augmented_llm import RequestParams

class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            # Enum members need to be converted to their values
            if hasattr(o, '__dict__'):
                d = asdict(o)
                for k, v in d.items():
                    if isinstance(v, Enum):
                        d[k] = v.value
                return d
            return asdict(o)
        if isinstance(o, Enum):
            return o.value
        # Handle datetime
        if hasattr(o, 'isoformat'):
            return o.isoformat()
        return super().default(o)

async def main():
    """Urban Hive Agent 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the Urban Hive Agent from the command line.")
    parser.add_argument("--prompt", required=True, help="The natural language prompt for urban analysis.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")

    args = parser.parse_args()

    print(f"🏙️ Starting Urban Hive Agent for: '{args.prompt}'")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    agent = UrbanHiveMCPAgent(output_dir=str(result_json_path.parent))

    parsing_prompt = f"""
You are a helpful assistant that parses user queries for urban data analysis.
Extract the location, a suitable data category, and a time range from the user's query.

User Query: "{args.prompt}"

Available Data Categories:
- TRAFFIC_FLOW
- PUBLIC_SAFETY
- ILLEGAL_DUMPING
- COMMUNITY_EVENTS
- URBAN_PLANNING
- ENVIRONMENTAL
- REAL_ESTATE_TRENDS

Time range format should be a string like "24h", "7d", "1m", "3m", "1y".
If the user mentions a duration like "3개월", convert it to "3m". "하루" to "24h".
If no specific duration is mentioned but implies recent trends (e.g. "최근 시세"), use "1m".

Respond ONLY with a JSON object with the following keys: "location", "category", "time_range".
Default to "URBAN_PLANNING" for category if no clear match. Default to "1m" for time_range.
"""
    try:
        print("1. Parsing prompt to extract parameters...")
        parsed_params_str = await agent.parser_llm.generate_str(
            message=parsing_prompt,
            request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07", temperature=0.0)
        )
        parsed_params_str = parsed_params_str.strip().removeprefix("```json").removesuffix("```").strip()
        params = json.loads(parsed_params_str)
        print(f"   - Parameters parsed: {params}")

        location = params.get("location")
        category_str = params.get("category", "URBAN_PLANNING")
        time_range = params.get("time_range", "1m")

        if not location or location.lower() == "none":
            raise ValueError("Location could not be determined from the prompt.")

        category = UrbanDataCategory[category_str]
        
        print("\n2. Running urban data analysis...")
        analysis_result = await agent.analyze_urban_data(
            category=category,
            location=location,
            time_range=time_range
        )
        print("   - Analysis complete.")

        print("\n3. Saving results...")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            # We need a custom JSON encoder for dataclasses and enums
            json.dump(analysis_result, f, indent=2, ensure_ascii=False, cls=DataclassJSONEncoder)
        
        print(f"🎉 Results saved to {result_json_path}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        error_result = {
            'success': False,
            'message': f'An error occurred: {str(e)}',
            'error': str(e),
        }
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=2, ensure_ascii=False)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 