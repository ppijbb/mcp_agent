import argparse
import json
import sys
from pathlib import Path
import asyncio
from dataclasses import asdict, is_dataclass

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.evolutionary_ai_architect.evolutionary_ai_architect_agent import EvolutionaryAIArchitectMCP, ArchitectureEvolutionResult

class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

async def main():
    """AI Architect Agent 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the Evolutionary AI Architect Agent from the command line.")
    parser.add_argument("--problem-description", required=True, help="A description of the AI architecture problem to solve.")
    parser.add_argument("--max-generations", type=int, default=5, help="Maximum number of generations for evolution.")
    parser.add_argument("--population-size", type=int, default=10, help="Size of the population for evolution.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")

    args = parser.parse_args()

    print(f"🧬 Starting AI Architect Agent for: '{args.problem_description}'")
    print(f"Generations: {args.max_generations}, Population: {args.population_size}")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
    # 결과 파일의 디렉토리가 없으면 생성
    result_json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        agent = EvolutionaryAIArchitectMCP(output_dir=str(result_json_path.parent))
        
        result = await agent.evolve_architecture(
            problem_description=args.problem_description,
            max_generations=args.max_generations,
            population_size=args.population_size,
        )

        print("-" * 30)
        print("✅ AI Architect Agent finished successfully.")
        print(f"💾 Saving results to {result_json_path}...")

        # Dataclass를 JSON으로 변환하여 저장
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, cls=DataclassJSONEncoder)
        
        print("🎉 Results saved.")

    except Exception as e:
        print(f"❌ An error occurred during architecture evolution: {e}")
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