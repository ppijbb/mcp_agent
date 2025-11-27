import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import uuid

# Real MCP Agent imports
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from srcs.common.utils import setup_agent_app


class SelfEvolvingSwarm:
    """
    A swarm of agents that can evolve and adapt over time.
    """
    def __init__(self, num_agents: int = 5, output_dir: str = "swarm_reports"):
        self.output_dir = output_dir
        self.app = setup_agent_app("self_evolving_swarm")
        self.population: List[EvolvingAgent] = [
            EvolvingAgent(agent_id=f"agent_{i}") for i in range(num_agents)
        ]
        self.generation = 0
        self.task_history: List[Task] = []
        self.performance_logs: List[Dict[str, Any]] = []


class SelfEvolvingSwarmOrchestrator:
    """
    자기 진화형 스웜 아키텍처 오케스트레이터
    - 현재 스웜 성능 벤치마킹
    - 토폴로지 변이 생성 및 평가
    - 최적 아키텍처 적용
    """
    def __init__(
        self,
        name: str,
        config_path: str,
        base_swarm_agent: Agent,
        test_task: str,
        request_params: RequestParams,
    ):
        # MCP 앱 초기화
        self.app = MCPApp(
            name=name,
            settings=get_settings(config_path),
            human_input_callback=None,
        )
        self.base_swarm_agent = base_swarm_agent
        self.test_task = test_task
        self.request_params = request_params
        # 현재 토폴로지(임시 저장)
        self.current_architecture: Optional[Dict[str, Any]] = None

    async def benchmark_current_performance(self, agent: Agent = None) -> Dict[str, float]:
        """
        현재 스웜 토폴로지 성능 벤치마킹
        """
        start = time.time()
        async with self.app.run() as swarm_app:
            context = swarm_app.context
            # 파일 시스템 서버 인자 추가
            fs_args = context.config.mcp.servers.get("filesystem").args
            if os.getcwd() not in fs_args:
                fs_args.append(os.getcwd())
            swarm = AnthropicSwarm(
                agent=agent or self.base_swarm_agent,
                context_variables={"task": self.test_task},
            )
            # 실행 단일 테스트 태스크
            await swarm.generate_str(
                message=self.test_task,
                request_params=self.request_params,
            )
        elapsed = time.time() - start
        return {"run_time": elapsed}

    def generate_architecture_mutations(self) -> List[Dict[str, Any]]:
        """
        현재 토폴로지 변이 생성 (토폴로지 설계변경 예시)
        """
        mutations: List[Dict[str, Any]] = []
        # Example mutations: toggle instruction append based on index
        for i in range(3):
            mutations.append({
                "mutation_id": f"mut_{i}",
                "append_instruction": True if i % 2 == 0 else False
            })
        return mutations

    def apply_mutation(self, mutation: Dict[str, Any]) -> Agent:
        """
        Apply a mutation to base_swarm_agent, returning a new mutated agent instance.
        """
        agent_cls = self.base_swarm_agent.__class__
        # Build mutated name
        new_name = f"{self.base_swarm_agent.name}_{mutation['mutation_id']}"
        # Determine instruction mutation
        base_inst = self.base_swarm_agent.instruction
        if mutation.get("append_instruction") and isinstance(base_inst, str):
            new_inst = base_inst + f" (mutation: {mutation['mutation_id']})"
        else:
            new_inst = base_inst
        # Copy other properties
        funcs = getattr(self.base_swarm_agent, "functions", None)
        servers = getattr(self.base_swarm_agent, "server_names", None)
        hicb = getattr(self.base_swarm_agent, "human_input_callback", None)
        # Instantiate mutated agent
        mutated_agent = agent_cls(
            name=new_name,
            instruction=new_inst,
            functions=funcs,
            server_names=servers,
            human_input_callback=hicb,
        )
        return mutated_agent

    async def evaluate_mutations(
        self, mutations: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        각 변이에 대한 성능 평가
        """
        results: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
        for mutation in mutations:
            # Create mutated agent and benchmark its performance
            mutated_agent = self.apply_mutation(mutation)
            perf = await self.benchmark_current_performance(agent=mutated_agent)
            results.append((mutation, perf))
        return results

    async def evolve_swarm_architecture(
        self,
        generations: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        스웜 아키텍처 진화 실행
        """
        best_arch: Optional[Dict[str, Any]] = None
        best_perf: Optional[Dict[str, float]] = None
        for gen in range(generations):
            mutations = self.generate_architecture_mutations()
            evaluated = await self.evaluate_mutations(mutations)
            for arch, perf in evaluated:
                # Performance: minimize run_time
                if best_perf is None or perf.get("run_time", float('inf')) < best_perf.get("run_time", float('inf')):
                    best_arch = arch
                    best_perf = perf
            # Apply the best mutation found in this generation
            if best_arch:
                self.current_architecture = best_arch
                self.base_swarm_agent = self.apply_mutation(best_arch)
        return best_arch


# Example usage
async def main():
    # 테스트용 SwarmAgent만 입력합니다.
    from mcp_agent.workflows.swarm.swarm import SwarmAgent

    # 기본 트라이에이저 에이전트를 불러오거나 정의하세요
    triage_agent = SwarmAgent(
        name="Triage",
        instruction=lambda ctx: "... triage instructions ...",
        functions=[],
    )

    orchestrator = SelfEvolvingSwarmOrchestrator(
        name="self_evolving_swarm",
        config_path="configs/mcp_agent.config.yaml",
        base_swarm_agent=triage_agent,
        test_task="Test task for swarm",
        request_params=RequestParams(model="gemini-2.5-flash-lite"),
    )
    best_arch = await orchestrator.evolve_swarm_architecture(generations=2)
    print(f"Best architecture: {best_arch}")


if __name__ == "__main__":
    asyncio.run(main()) 