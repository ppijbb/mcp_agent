"""
Coordinator Agent
Multi-Agent 간 소통, 워크플로우 조율 및 작업 협조를 관리하는 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from typing import Dict, List, Any, Optional
import json
import asyncio
import logging
import aiohttp
from datetime import datetime

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
from mcp_agent.workflows.llm.openai_augmented_llm import OpenAIAugmentedLLM

logger = get_logger("coordinator_agent")

# Helper function to create the HTTP client session
def get_http_session():
    return aiohttp.ClientSession()

class CoordinatorAgent:
    """
    각 전문 Agent들을 조율하고 전체 워크플로우를 관리하는 핵심 Agent
    """

    def __init__(self, orchestrator: Orchestrator, 
                 google_drive_mcp_url: str = "http://localhost:3001",
                 figma_mcp_url: str = "http://localhost:3003",
                 notion_mcp_url: str = "http://localhost:3004"):
        self.orchestrator = orchestrator
        self.llm = orchestrator.llm_factory()
        self.google_drive_mcp_url = google_drive_mcp_url
        self.figma_mcp_url = figma_mcp_url
        self.notion_mcp_url = notion_mcp_url
        
        # 각 전문 Agent들을 초기화할 때 MCP URL을 전달합니다.
        self.figma_analyzer = FigmaAnalyzerAgent(orchestrator=orchestrator)
        self.prd_writer = PRDWriterAgent(
            google_drive_mcp_url=self.google_drive_mcp_url,
            figma_mcp_url=self.figma_mcp_url,
            notion_mcp_url=self.notion_mcp_url
        )
        self.business_planner = BusinessPlannerAgent(orchestrator=orchestrator)
        
        self.agent = self._create_agent_instance()

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
            # JSON 문자열의 중괄호가 .format()과 충돌하지 않도록 f-string 사용
            available_agents_json = json.dumps(self.available_agents)
            context_json = json.dumps(context, indent=2, default=str)
            thought_prompt = REACT_PROMPT.replace("{available_agents}", available_agents_json).replace("{context}", context_json)
            
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
        """
        모든 Agent의 결과물을 종합하여 최종 보고서를 생성하고 Google Drive에 업로드합니다.
        """
        logger.info("📄 Generating final comprehensive report...")
        
        prompt = GENERATE_FINAL_REPORT_PROMPT.format(
            figma_analysis=json.dumps(results.get("figma_analysis", {}), indent=2, ensure_ascii=False),
            prd=json.dumps(results.get("prd", {}), indent=2, ensure_ascii=False),
            business_plan=json.dumps(results.get("business_plan", {}), indent=2, ensure_ascii=False)
        )

        final_report = await self.llm.generate_str(
            message=prompt,
            request_params=RequestParams(
                model="gemini-2.5-pro-vision-preview-06-07",
                temperature=0.3,
                max_tokens=4096
            )
        )
        logger.info("Final report content generated.")

        # 파일 저장 로직을 Google Drive MCP 호출로 변경
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"product_plan_report_{timestamp}.md"
        
        upload_url = f"{self.google_drive_mcp_url}/upload"
        payload = {
            "fileName": file_name,
            "content": final_report
        }

        try:
            async with get_http_session() as session:
                async with session.post(upload_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()

                    if result.get("success"):
                        file_id = result.get("fileId")
                        logger.info(f"✅ Final report successfully uploaded to Google Drive. File ID: {file_id}")
                        return {
                            "report_content": final_report,
                            "drive_file_id": file_id,
                            "file_url": f"https://docs.google.com/document/d/{file_id}",
                            "status": "uploaded"
                        }
                    else:
                        raise Exception(f"MCP upload failed: {result.get('message')}")

        except Exception as e:
            logger.error(f"💥 Failed to upload the final report to Google Drive. Error: {e}", exc_info=True)
            return {"report_content": final_report, "drive_file_id": None, "status": "upload_failed", "error": str(e)}

    async def coordinate_prd_creation(self, 
                                      product_concept: str, 
                                      user_persona: str,
                                      figma_file_id: Optional[str] = None,
                                      notion_page_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Orchestrates the PRD creation workflow, now including Figma and Notion IDs.
        """
        print("Coordinating PRD Creation...")
        
        # 1. Generate Product Brief (existing logic)
        product_brief = await self._generate_product_brief(product_concept, user_persona)
        
        # 2. Draft PRD with Figma/Notion context
        print(f"Drafting PRD with context: Figma ID '{figma_file_id}', Notion ID '{notion_page_id}'")
        prd_draft = await self.prd_writer.draft_prd(
            product_brief=product_brief,
            figma_file_id=figma_file_id,
            notion_page_id=notion_page_id
        )
        
        # In a more complex scenario, we would have feedback loops here.
        # For now, we proceed directly to saving.
        
        # 3. Save the final PRD
        file_name = f"prd_{product_brief.get('product_name', 'untitled').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
        saved_info = await self.prd_writer.save_prd(prd_draft, file_name)
        
        print("PRD Coordination complete.")
        return {
            "final_prd": prd_draft,
            "saved_info": saved_info
        }

    async def _generate_product_brief(self, product_concept: str, user_persona: str) -> Dict[str, Any]:
        """
        Generates a structured product brief from the initial concept.
        (This is a simplified version of what a real agent would do)
        """
        prompt = f"""
        Based on the following product concept and user persona, create a structured product brief.

        **Product Concept**: {product_concept}
        **User Persona**: {user_persona}

        The brief must include:
        - A catchy, internal-facing 'product_name'.
        - A clear 'problem_statement'.
        - A list of 3-5 high-level 'key_features'.
        - The primary 'target_audience'.
        - Key 'success_metrics' (KPIs).

        Return the output as a JSON object.
        """
        llm = OpenAIAugmentedLLM()
        brief_str = await llm.generate_str(
            message=prompt,
            request_params=RequestParams(
                model="gemini-2.5-flash-lite-preview-06-07",
                temperature=0.3,
                response_format={"type": "json_object"},
            )
        )
        return json.loads(brief_str)

    async def generate_final_report(self, results: List[Dict[str, Any]]) -> str:
        """
        모든 Agent의 결과물을 종합하여 최종 보고서를 생성하고 Google Drive에 업로드합니다.
        """
        pass
        
    async def save_final_report(self, report_content: str, file_name: str) -> Dict[str, Any]:
        """
        최종 보고서를 파일로 저장합니다. (현재는 Google Drive MCP를 통해 업로드)
        """
        pass