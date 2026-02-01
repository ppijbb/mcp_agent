import argparse
import json
import sys
from pathlib import Path
import asyncio
from datetime import datetime
from dataclasses import asdict, is_dataclass
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.evolutionary_ai_architect.evolutionary_ai_architect_agent import EvolutionaryAIArchitectMCP


def convert_to_serializable(obj):
    """재귀적으로 객체를 JSON 직렬화 가능한 형태로 변환합니다."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif is_dataclass(obj):
        return {k: convert_to_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj


class DataclassJSONEncoder(json.JSONEncoder):
    """JSONEncoder를 확장하여 dataclass와 datetime 객체를 처리합니다."""
    def default(self, o):
        return convert_to_serializable(o)


async def run_ai_architect_agent(
    problem_description: str,
    max_generations: int = 5,
    population_size: int = 10,
    result_json_path: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    AI Architect Agent 실행 함수
    Streamlit A2A runner에서 호출되는 함수

    Args:
        problem_description: 문제 설명
        max_generations: 최대 세대 수
        population_size: 인구 크기
        result_json_path: 결과 JSON 파일 경로
        **kwargs: 추가 인자

    Returns:
        실행 결과 딕셔너리
    """
    if result_json_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_json_path = Path("evolutionary_architect_reports") / f"architecture_{timestamp}.json"

    result_json_path = Path(result_json_path)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        agent = EvolutionaryAIArchitectMCP(output_dir=str(result_json_path.parent))

        result = await agent.evolve_architecture(
            problem_description=problem_description,
            max_generations=max_generations,
            population_size=population_size,
        )

        # 결과를 파일로 저장 (JSON 저장)
        result_dict = convert_to_serializable(result)
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, cls=DataclassJSONEncoder)

        # 간단한 요약만 반환 (전체 JSON 데이터는 반환하지 않음)
        best_arch = result_dict.get('best_architecture', {})
        evolution_history = result_dict.get('evolution_history', [])

        return {
            'success': True,
            'data': {
                'problem_description': result_dict.get('task', {}).get('problem_description', ''),
                'best_fitness': best_arch.get('fitness_score', 0.0),
                'generation_count': result_dict.get('generation_count', 0),
                'processing_time': result_dict.get('processing_time', 0.0),
                'evolution_summary': {
                    'total_generations': len(evolution_history),
                    'final_best_fitness': evolution_history[-1].get('best_fitness', 0.0) if evolution_history else 0.0,
                    'final_avg_fitness': evolution_history[-1].get('avg_fitness', 0.0) if evolution_history else 0.0,
                },
                'result_file_path': str(result_json_path),
                'message': f'AI Architect Agent 실행 완료. 최적 fitness: {best_arch.get("fitness_score", 0.0):.4f}'
            },
            'message': 'AI Architect Agent 실행 완료'
        }

    except Exception as e:
        error_result = {
            'success': False,
            'message': f'An error occurred: {str(e)}',
            'error': str(e),
        }
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=2, ensure_ascii=False)
        raise


async def run_evolution(args):
    """AI Architect Agent의 진화 프로세스를 실행합니다."""
    print(f"🧬 Starting AI Architect Agent for: '{args.problem_description}'")
    print(f"Generations: {args.max_generations}, Population: {args.population_size}")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
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


def main():
    """명령줄 인자를 파싱하고 에이전트를 실행합니다."""
    parser = argparse.ArgumentParser(description="Run the Evolutionary AI Architect Agent from the command line.")
    parser.add_argument("--problem-description", required=True, help="A description of the AI architecture problem to solve.")
    parser.add_argument("--max-generations", type=int, default=5, help="Maximum number of generations for evolution.")
    parser.add_argument("--population-size", type=int, default=10, help="Size of the population for evolution.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")

    args = parser.parse_args()
    asyncio.run(run_evolution(args))


if __name__ == "__main__":
    main()
