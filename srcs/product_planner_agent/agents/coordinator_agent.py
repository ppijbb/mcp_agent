"""
Coordinator Agent
Multi-Agent 간 소통, 워크플로우 조율 및 작업 협조를 관리하는 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from typing import Dict, List, Any
import json
import asyncio
import logging

from .figma_analyzer_agent import FigmaAnalyzerAgent
from .prd_writer_agent import PRDWriterAgent
from .business_planner_agent import BusinessPlannerAgent
from ..utils.status_logger import StatusLogger
from mcp_agent.logging.logger import get_logger
from srcs.product_planner_agent.prompts.coordinator_prompts import (
    PLANNING_AND_EXECUTION_PROMPT,
    GENERATE_FINAL_REPORT_PROMPT,
    REACT_PROMPT
)

logger = get_logger("coordinator_agent")

class CoordinatorAgent:
    """Agent 간 조율 및 워크플로우 관리를 위한 ReAct 기반 실행 Agent"""

    def __init__(self, orchestrator: Orchestrator, agents: Dict[str, Any] = None):
        self.orchestrator = orchestrator
        self.llm = orchestrator.llm_factory()
        # StatusLogger 초기화 시 기본 단계 목록 제공
        default_steps = [
            "Figma Design Analysis",
            "Product Requirements Document (PRD)",
            "Business Planning",
            "Final Report Generation"
        ]
        self.status_logger = StatusLogger(steps=default_steps)
        self.max_iterations = 7  # 최대 반복 횟수 설정
        
        # 에이전트들을 실제 인스턴스로 초기화
        self.agents = self._initialize_agents(agents)
        self.available_agents = list(self.agents.keys())
        
        logger.info(f"🔍 Initialized CoordinatorAgent with agents: {self.available_agents}")
        
        self.agent_instance = self._create_agent_instance()
        self.result = {}

    def _initialize_agents(self, provided_agents: Dict[str, Any] = None) -> Dict[str, Any]:
        """실제 에이전트 인스턴스들을 초기화"""
        agents = {}
        
        try:
            # FigmaAnalyzerAgent 초기화 (orchestrator 파라미터 제거)
            figma_url = "https://www.figma.com/design/sample"  # 기본 URL
            agents["figma_analyzer_agent"] = FigmaAnalyzerAgent(
                figma_url=figma_url
            )
            logger.info("✅ FigmaAnalyzerAgent initialized")
            
            # PRDWriterAgent 초기화 (orchestrator 파라미터 제거)
            output_path = "outputs/product_planner/prd_output.md"
            agents["prd_writer_agent"] = PRDWriterAgent(
                output_path=output_path
            )
            logger.info("✅ PRDWriterAgent initialized")
            
            # BusinessPlannerAgent 초기화 (orchestrator는 선택사항)
            agents["business_planner_agent"] = BusinessPlannerAgent(
                llm=self.llm,
                orchestrator=self.orchestrator
            )
            logger.info("✅ BusinessPlannerAgent initialized")
            
            # 제공된 추가 에이전트들이 있다면 추가
            if provided_agents:
                for name, agent in provided_agents.items():
                    if name not in agents:  # 중복 방지
                        agents[name] = agent
                        logger.info(f"✅ Additional agent added: {name}")
            
        except Exception as e:
            logger.error(f"💥 Error initializing agents: {e}", exc_info=True)
            # fallback 제거 - 오류 발생 시 빈 딕셔너리 반환
            raise RuntimeError(f"Failed to initialize Product Planner agents: {e}")
        
        return agents



    def _create_agent_instance(self) -> Agent:
        """
        조율 Agent의 기본 인스턴스 생성
        """
        instruction = self._get_base_instruction()
        return Agent(
            name="coordinator_agent",
            instruction=instruction,
            server_names=["filesystem"] # 필요한 MCP 서버 추가
        )

    async def run_react(self, initial_task: str) -> str:
        """
        ReAct 패턴을 사용하여 초기 과업을 자율적으로 해결합니다.
        
        :param initial_task: 사용자가 요청한 초기 과업
        :return: 최종 결과물
        """
        logger.info(f"🚀 Starting ReAct-based Product Planner Workflow for task: {initial_task}")

        context = {"initial_task": initial_task, "history": [], "result_store": {}}
        
        for i in range(self.max_iterations):
            logger.info(f"--- Iteration {i + 1}/{self.max_iterations} ---")

            # 1. THOUGHT
            thought_prompt = REACT_PROMPT.format(
                available_agents=json.dumps(self.available_agents),
                context=json.dumps(context, indent=2, default=str)
            )
            
            try:
                raw_thought = await self.llm.generate_str(thought_prompt)
                thought = raw_thought.strip()
                logger.info(f"🤔 THOUGHT: {thought}")
                context["history"].append({"role": "assistant", "thought": thought})

                action_json_str_part = thought.split("ACTION:")[1]
                action_json_str = action_json_str_part.strip()
                action_data = json.loads(action_json_str)

            except (IndexError, json.JSONDecodeError) as e:
                logger.warning(f"⚠️ Could not parse THOUGHT or ACTION. Raw response: {thought}. Error: {e}. Retrying.")
                observation = f"Error in thought process: {e}. I must provide a valid thought and a JSON object for ACTION."
                context["history"].append({"role": "system", "observation": observation})
                continue
            
            # 2. ACTION
            try:
                action_name = action_data.get("agent")
                action_method = action_data.get("method")
                action_params = action_data.get("params", {})
                
                logger.info(f"⚡ ACTION: Agent={action_name}, Method={action_method}, Params={action_params}")

                if action_name == "finish":
                    logger.info("🎉 Finishing the task based on 'finish' action.")
                    final_result = action_params.get("result", "No result provided.")
                    context["history"].append({"role": "assistant", "action": "finish", "result": final_result})
                    return json.dumps(final_result, indent=2, ensure_ascii=False)

            except KeyError as e:
                logger.warning(f"⚠️ Could not parse ACTION fields. Data: {action_data}. Error: {e}. Retrying.")
                observation = f"Error parsing action fields: {e}. The 'agent' and 'method' keys are required."
                context["history"].append({"role": "system", "observation": observation})
                continue
            
            # 3. OBSERVATION
            try:
                observation = await self._execute_action(action_name, action_method, action_params, context)
                obs_str = json.dumps(observation, indent=2, ensure_ascii=False, default=str)
                logger.info(f"👀 OBSERVATION: {obs_str[:1000]}...")
            except Exception as e:
                logger.error(f"💥 Error executing action '{action_name}.{action_method}'. Error: {e}", exc_info=True)
                observation = f"Error executing action: {e}. Check the agent name, method, and parameters."
                obs_str = str(observation)

            action_id = f"{action_name}_{action_method}_{i}"
            context["result_store"][action_id] = observation
            
            context["history"].append({
                "role": "assistant", 
                "action": f"{action_name}.{action_method}({json.dumps(action_params)})",
                "observation": f"Action executed. Result stored under id '{action_id}'. Result snippet: {obs_str[:200]}..."
            })

        logger.warning("🔚 Reached max iterations. Returning current accumulated results.")
        final_report = await self.generate_final_report(context['result_store'])
        return final_report

    async def _execute_action(self, agent_name: str, method_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """선택된 에이전트의 메서드를 동적으로 실행하고, 컨텍스트에서 파라미터를 주입합니다."""
        if agent_name not in self.agents:
            available_agents = ", ".join(self.available_agents)
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")

        agent = self.agents[agent_name]
        method_to_call = getattr(agent, method_name, None)

        if not method_to_call or not callable(method_to_call):
            # 에이전트의 사용 가능한 메소드 목록 제공
            available_methods = [method for method in dir(agent) if not method.startswith('_') and callable(getattr(agent, method))]
            raise AttributeError(f"Method '{method_name}' not found in agent '{agent_name}'. Available methods: {available_methods}")
        
        # 파라미터 처리: '@'로 시작하는 값을 컨텍스트에서 찾아 대체
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('@'):
                result_key = value[1:]
                if result_key in context['result_store']:
                    processed_params[key] = context['result_store'][result_key]
                else:
                    raise ValueError(f"Could not find result for key '{result_key}' in result_store. Available keys: {list(context['result_store'].keys())}")
            else:
                processed_params[key] = value

        logger.info(f"Executing: {agent_name}.{method_name} with processed params: {list(processed_params.keys())}")
        
        return await method_to_call(**processed_params)

    async def run_static_workflow(self, figma_api_key: str, figma_file_id: str, figma_node_id: str):
        logger.info("🚀 Starting Product Planner Static Workflow...")

        workflow_steps = [
            "Figma Design Analysis",
            "Product Requirements Document (PRD)",
            "Business Planning",
            "Final Report Generation"
        ]
        status_logger = StatusLogger(steps=workflow_steps)

        try:
            # 1. Figma 디자인 분석
            status_logger.update_status("Figma Design Analysis", "in_progress")
            figma_analyzer = self.agents.get("figma_analyzer_agent")
            prd_writer = self.agents.get("prd_writer_agent")
            business_planner = self.agents.get("business_planner_agent")

            if not all([figma_analyzer, prd_writer, business_planner]):
                msg = "One or more required agents (figma_analyzer, prd_writer, business_planner) not found for static workflow."
                logger.error(msg)
                raise ValueError(msg)

            figma_analysis_result = await figma_analyzer.analyze_figma_for_prd(
                figma_api_key=figma_api_key, figma_file_id=figma_file_id, figma_node_id=figma_node_id
            )
            self.result["figma_analysis"] = figma_analysis_result
            status_logger.update_status("Figma Design Analysis", "completed")
            logger.info("✅ Figma Design Analysis Completed")

            # 2. PRD 작성
            status_logger.update_status("Product Requirements Document (PRD)", "in_progress")
            prd_result = await prd_writer.write_prd(figma_analysis_result=figma_analysis_result)
            self.result["prd"] = prd_result
            status_logger.update_status("Product Requirements Document (PRD)", "completed")
            logger.info("✅ Product Requirements Document (PRD) Completed")

            # 3. 비즈니스 기획
            status_logger.update_status("Business Planning", "in_progress")
            business_plan_result = await business_planner.create_business_plan(prd_content=prd_result)
            self.result["business_plan"] = business_plan_result
            status_logger.update_status("Business Planning", "completed")
            logger.info("✅ Business Planning Completed")

            # 4. 최종 보고서 생성
            status_logger.update_status("Final Report Generation", "in_progress")
            final_report = await self.generate_final_report(self.result)
            self.result["final_report"] = final_report
            status_logger.update_status("Final Report Generation", "completed")
            logger.info("✅ Final Report Generation Completed")

            logger.info("🎉 Product Planner Static Workflow Completed Successfully!")
            return self.result

        except Exception as e:
            logger.error(f"💥 Static workflow execution failed: {e}", exc_info=True)
            current_step = next((step for step, status in status_logger.get_status().items() if status == "in_progress"), None)
            if current_step:
                status_logger.update_status(current_step, "failed")
            
            raise e

    async def run(self, initial_task: str) -> str:
        """
        ReAct 패턴을 사용하여 초기 과업을 자율적으로 해결합니다.
        이 메소드가 이제 클래스의 기본 진입점입니다.
        
        :param initial_task: 사용자가 요청한 초기 과업
        :return: 최종 결과물
        """
        return await self.run_react(initial_task)

    @staticmethod
    def _get_base_instruction() -> str:
        """
        Agent의 기본 지시사항을 반환합니다.
        """
        return """
        You are the ReAct-based coordination maestro for a multi-agent product planning system. 
        Your primary role is to understand user requests and autonomously orchestrate a team of specialist agents to achieve the goal.
        You must think step-by-step, choose an agent and its method, execute the action, and observe the result to plan your next move.
        Always use the 'finish' agent when the task is complete.
        """

    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "🎯 ReAct 기반으로 Multi-Agent 워크플로우를 자율적으로 조율하고 실행하는 중앙 조율 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "Multi-Agent 워크플로우 및 작업 순서 조율",
            "Agent 간 소통 및 정보 공유 촉진",
            "진행 상황 모니터링 및 품질 표준 보장",
            "Agent 작업 간 충돌 및 종속성 해결",
            "프로젝트 일정 및 마일스톤 조율"
        ]
    
    @staticmethod
    def get_workflow_phases() -> dict[str, dict[str, Any]]:
        """워크플로우 단계별 정보 반환"""
        return {
            "phase_1_analysis": {
                "name": "Figma Design Analysis",
                "description": "Comprehensive analysis of Figma design",
                "duration": "30 minutes"
            },
            "phase_2_prd": {
                "name": "Product Requirements Document", 
                "description": "Detailed PRD creation based on design analysis",
                "duration": "45 minutes"
            },
            "phase_3_business": {
                "name": "Business Planning",
                "description": "Strategic business plan development",
                "duration": "30 minutes"
            },
            "phase_4_integration": {
                "name": "Final Integration",
                "description": "Executive summary and recommendations",
                "duration": "15 minutes"
            }
        }
    
    @staticmethod
    def get_coordination_principles() -> list[str]:
        """조율 원칙 목록 반환"""
        return [
            "순차적 실행: 각 단계의 결과가 다음 단계의 입력이 됨",
            "품질 우선: 각 단계에서 완전한 결과물 생성",
            "오류 처리: 단계별 오류 발생 시 적절한 대응",
            "결과 통합: 모든 단계의 결과를 최종 보고서로 통합",
            "파일 저장: 최종 결과물을 지정된 경로에 저장"
        ]
    
    @staticmethod
    def get_success_metrics() -> list[str]:
        """성공 지표 목록 반환"""
        return [
            "모든 워크플로우 단계 완료",
            "각 단계별 품질 있는 결과물 생성",
            "최종 보고서 파일 저장 성공",
            "오류 발생 시 적절한 처리 및 계속 진행",
            "사용자 요구사항 충족"
        ]

    async def generate_final_report(self, results: Dict[str, Any]):
        logger.info("Generating final report...")
        
        final_report_content = "## 📝 Product Plan Final Report\n\n"
        for key, value in results.items():
            final_report_content += f"### {key.replace('_', ' ').title()}\n\n"
            if isinstance(value, (dict, list)):
                final_report_content += f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n\n"
            else:
                final_report_content += f"{str(value)}\n\n"
        
        prompt = GENERATE_FINAL_REPORT_PROMPT.format(
            report_data=final_report_content
        )
        
        try:
            final_report = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.7))
        except Exception as e:
            logger.warning(f"Final report generation failed: {e}")
            final_report = final_report_content  # 기본 보고서 사용

        # 파일 저장 로직 추가
        try:
            import os
            from datetime import datetime
            
            output_dir = "outputs/product_planner"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{output_dir}/product_plan_report_{timestamp}.md"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            logger.info(f"✅ Final report successfully saved to {file_path}")
            return {"report": final_report, "file_path": file_path}
            
        except Exception as e:
            logger.error(f"💥 Failed to save the final report. Error: {e}", exc_info=True)
            return {"report": final_report, "file_path": None}

# ... 기존의 analyze_figma, create_prd 등의 메소드는 ReAct 패턴 하에서는 직접 호출되지 않으므로
# 각 전문 에이전트로 옮겨지거나, 여기서 제거될 수 있습니다. 
# 이 예제에서는 일단 남겨두지만, 실제 리팩토링 과정에서는 정리하는 것이 좋습니다.
# ...