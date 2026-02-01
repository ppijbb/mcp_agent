#!/usr/bin/env python3
"""
Product Planner Agent
"""
import re
from urllib.parse import unquote
from typing import Any, Dict, List
from datetime import datetime
import json

from srcs.product_planner_agent.agents.base_agent_simple import BaseAgentSimple as BaseAgent
from srcs.product_planner_agent.agents.figma_analyzer_agent import FigmaAnalyzerAgent
from srcs.product_planner_agent.agents.prd_writer_agent import PRDWriterAgent
from srcs.product_planner_agent.agents.figma_creator_agent import FigmaCreatorAgent
from srcs.product_planner_agent.coordinators.reporting_coordinator import ReportingCoordinator
from srcs.product_planner_agent.utils.logger import get_product_planner_logger
# from srcs.common.utils import get_gen_client  # mcp_agent 의존성 제거

logger = get_product_planner_logger("main_agent")


class ProductPlannerAgent(BaseAgent):
    """
    Coordinates the entire product planning process by orchestrating various sub-agents.
    This version is refactored to be simpler and delegate tasks to specialized agents
    and coordinators, following the new architecture.
    """

    def __init__(self):
        super().__init__("product_planner_agent")
        # Sub-agents are initialized here, but their LLM dependencies are handled by the app context.
        self.figma_analyzer = FigmaAnalyzerAgent()
        self.prd_writer = PRDWriterAgent()
        self.reporting_coordinator = ReportingCoordinator()
        self.figma_creator_agent = FigmaCreatorAgent()  # FigmaCreatorAgent 추가
        logger.info("ProductPlannerAgent and its sub-components initialized.")

        # Add state management for conversational mode
        self.state = {
            "step": "init",
            "data": {
                "product_concept": None,
                "user_persona": None,
                "figma_file_id": None,
                "figma_analysis": None,
                "prd_draft": None,
                "final_report": None
            },
            "history": []
        }

    async def _save_final_report(self, report_data: Dict[str, Any], product_concept: str) -> Dict[str, Any]:
        """Saves the final report to local file system."""
        logger.info("💾 Saving final report to local file system...")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize product_concept for use in a filename
            safe_concept_name = re.sub(r'[\\/*?:"<>|]', "", product_concept)[:50]
            file_name = f"Final_Report_{safe_concept_name}_{timestamp}.json"

            # 로컬 파일로 저장
            import os
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            file_path = os.path.join(reports_dir, file_name)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Final report saved successfully. File path: {file_path}")
            return {"status": "success", "file_path": file_path}
        except Exception as e:
            logger.error(f"❌ Failed to save final report: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _extract_figma_ids(self, figma_url: str) -> tuple[str, str]:
        """Extracts Figma file ID and node ID from a Figma URL."""
        try:
            # Remove query parameters and fragment
            url_path = unquote(figma_url).split('?', 1)[0].split('#', 1)[0]

            # Extract file ID and node ID
            file_id_match = re.search(r'/file/([a-zA-Z0-9_-]+)', url_path)
            node_id_match = re.search(r'/node/([a-zA-Z0-9_-]+)', url_path)

            file_id = file_id_match.group(1) if file_id_match else None
            node_id = node_id_match.group(1) if node_id_match else None

            if not file_id:
                raise ValueError("Could not extract Figma file ID from URL.")

            return file_id, node_id
        except Exception as e:
            logger.error(f"Error extracting Figma IDs from URL {figma_url}: {e}", exc_info=True)
            raise

    # --- PRD에서 다양한 컴포넌트 정보를 추출하는 고도화 함수 ---
    def _extract_figma_components_from_prd(self, prd_content: str) -> List[Dict[str, Any]]:
        """PRD 내용에서 Figma 컴포넌트 정보 추출 (고도화)"""
        components = []

        # LLM을 사용한 구조화된 컴포넌트 추출
        try:
            # 더 정교한 패턴 매칭과 LLM 기반 추출
            import re

            # 1. 기본 UI 컴포넌트 패턴 매칭
            button_patterns = [
                r'버튼[:\s]*([^\n]+)',
                r'button[:\s]*([^\n]+)',
                r'클릭[:\s]*([^\n]+)',
                r'submit[:\s]*([^\n]+)',
                r'확인[:\s]*([^\n]+)',
                r'취소[:\s]*([^\n]+)',
                r'로그인[:\s]*([^\n]+)',
                r'회원가입[:\s]*([^\n]+)',
                r'검색[:\s]*([^\n]+)',
                r'저장[:\s]*([^\n]+)',
                r'삭제[:\s]*([^\n]+)',
                r'편집[:\s]*([^\n]+)'
            ]

            for pattern in button_patterns:
                matches = re.findall(pattern, prd_content, re.IGNORECASE)
                for match in matches:
                    button_text = match.strip()
                    components.append({
                        "type": "button",
                        "content": button_text,
                        "x": len(components) * 150,  # 동적 위치 계산
                        "y": 50,
                        "width": max(120, len(button_text) * 10),
                        "height": 40,
                        "style": {
                            "bg_color": "#007AFF",
                            "text_color": "#FFFFFF",
                            "corner_radius": 8
                        },
                        "properties": {
                            "interactive": True,
                            "action": button_text.lower()
                        }
                    })

            # 2. 입력 필드 패턴
            input_patterns = [
                r'입력[:\s]*([^\n]+)',
                r'input[:\s]*([^\n]+)',
                r'텍스트[:\s]*([^\n]+)',
                r'검색[:\s]*([^\n]+)',
                r'이름[:\s]*([^\n]+)',
                r'이메일[:\s]*([^\n]+)',
                r'비밀번호[:\s]*([^\n]+)',
                r'전화번호[:\s]*([^\n]+)',
                r'주소[:\s]*([^\n]+)',
                r'설명[:\s]*([^\n]+)',
                r'코멘트[:\s]*([^\n]+)',
                r'메시지[:\s]*([^\n]+)'
            ]

            for pattern in input_patterns:
                matches = re.findall(pattern, prd_content, re.IGNORECASE)
                for match in matches:
                    placeholder = match.strip()
                    components.append({
                        "type": "input",
                        "content": placeholder,
                        "x": len(components) * 220,  # 동적 위치 계산
                        "y": 120,
                        "width": 200,
                        "height": 40,
                        "style": {
                            "border_color": "#CCCCCC",
                            "bg_color": "#FFFFFF",
                            "placeholder_color": "#999999"
                        },
                        "properties": {
                            "placeholder": placeholder,
                            "required": "필수" in placeholder or "required" in placeholder.lower()
                        }
                    })

            # 3. 텍스트/라벨 패턴
            text_patterns = [
                r'제목[:\s]*([^\n]+)',
                r'title[:\s]*([^\n]+)',
                r'설명[:\s]*([^\n]+)',
                r'description[:\s]*([^\n]+)',
                r'라벨[:\s]*([^\n]+)',
                r'label[:\s]*([^\n]+)',
                r'헤더[:\s]*([^\n]+)',
                r'header[:\s]*([^\n]+)',
                r'부제목[:\s]*([^\n]+)',
                r'subtitle[:\s]*([^\n]+)'
            ]

            for pattern in text_patterns:
                matches = re.findall(pattern, prd_content, re.IGNORECASE)
                for match in matches:
                    text_content = match.strip()
                    components.append({
                        "type": "text",
                        "content": text_content,
                        "x": len(components) * 250,  # 동적 위치 계산
                        "y": 200,
                        "width": len(text_content) * 12,
                        "height": 20,
                        "style": {
                            "font_size": 16,
                            "color": "#000000",
                            "font_family": "Inter",
                            "font_weight": 400
                        },
                        "properties": {
                            "text_type": "label" if "라벨" in pattern or "label" in pattern else "title"
                        }
                    })

            # 4. 카드/컨테이너 패턴
            card_patterns = [
                r'카드[:\s]*([^\n]+)',
                r'card[:\s]*([^\n]+)',
                r'아이템[:\s]*([^\n]+)',
                r'item[:\s]*([^\n]+)',
                r'컨테이너[:\s]*([^\n]+)',
                r'container[:\s]*([^\n]+)',
                r'섹션[:\s]*([^\n]+)',
                r'section[:\s]*([^\n]+)',
                r'패널[:\s]*([^\n]+)',
                r'panel[:\s]*([^\n]+)'
            ]

            for pattern in card_patterns:
                matches = re.findall(pattern, prd_content, re.IGNORECASE)
                for match in matches:
                    card_content = match.strip()
                    components.append({
                        "type": "card",
                        "content": card_content,
                        "x": len(components) * 320,  # 동적 위치 계산
                        "y": 250,
                        "width": 300,
                        "height": 200,
                        "style": {
                            "bg_color": "#FFFFFF",
                            "shadow": True,
                            "corner_radius": 8,
                            "border_color": "#E1E5E9"
                        },
                        "properties": {
                            "card_type": "content",
                            "interactive": True
                        }
                    })

            # 5. 이미지/아이콘 패턴
            image_patterns = [
                r'이미지[:\s]*([^\n]+)',
                r'image[:\s]*([^\n]+)',
                r'사진[:\s]*([^\n]+)',
                r'photo[:\s]*([^\n]+)',
                r'아이콘[:\s]*([^\n]+)',
                r'icon[:\s]*([^\n]+)',
                r'로고[:\s]*([^\n]+)',
                r'logo[:\s]*([^\n]+)'
            ]

            for pattern in image_patterns:
                matches = re.findall(pattern, prd_content, re.IGNORECASE)
                for match in matches:
                    image_content = match.strip()
                    components.append({
                        "type": "rectangle",  # 이미지는 사각형으로 표현
                        "content": image_content,
                        "x": len(components) * 350,  # 동적 위치 계산
                        "y": 480,
                        "width": 100,
                        "height": 100,
                        "style": {
                            "fill_color": "#F0F0F0",
                            "corner_radius": 8,
                            "border_color": "#CCCCCC"
                        },
                        "properties": {
                            "image_type": "placeholder",
                            "alt_text": image_content
                        }
                    })

            # 6. 네비게이션 패턴
            nav_patterns = [
                r'메뉴[:\s]*([^\n]+)',
                r'menu[:\s]*([^\n]+)',
                r'탭[:\s]*([^\n]+)',
                r'tab[:\s]*([^\n]+)',
                r'네비게이션[:\s]*([^\n]+)',
                r'navigation[:\s]*([^\n]+)',
                r'사이드바[:\s]*([^\n]+)',
                r'sidebar[:\s]*([^\n]+)'
            ]

            for pattern in nav_patterns:
                matches = re.findall(pattern, prd_content, re.IGNORECASE)
                for match in matches:
                    nav_content = match.strip()
                    components.append({
                        "type": "button",
                        "content": nav_content,
                        "x": len(components) * 120,  # 동적 위치 계산
                        "y": 600,
                        "width": 100,
                        "height": 35,
                        "style": {
                            "bg_color": "#6C757D",
                            "text_color": "#FFFFFF",
                            "corner_radius": 6
                        },
                        "properties": {
                            "nav_type": "menu",
                            "interactive": True
                        }
                    })

            # 7. 기본 컨테이너 (컴포넌트가 없을 경우)
            if not components:
                components.append({
                    "type": "rectangle",
                    "content": "기본 컨테이너",
                    "x": 50,
                    "y": 50,
                    "width": 400,
                    "height": 300,
                    "style": {
                        "fill_color": "#F5F5F5",
                        "corner_radius": 8,
                        "border_color": "#E1E5E9"
                    },
                    "properties": {
                        "container_type": "main",
                        "layout": "flex"
                    }
                })

            # 8. 레이아웃 최적화 - 겹치지 않도록 위치 조정
            self._optimize_component_layout(components)

            self.logger.info(f"PRD에서 {len(components)}개 컴포넌트 추출 완료")
            return components

        except Exception as e:
            self.logger.error(f"컴포넌트 추출 중 오류: {str(e)}")
            # 오류 시 기본 컴포넌트 반환
            return [{
                "type": "rectangle",
                "content": "기본 컨테이너",
                "x": 50,
                "y": 50,
                "width": 400,
                "height": 300,
                "style": {"fill_color": "#F5F5F5"},
                "properties": {"fallback": True}
            }]

    def _optimize_component_layout(self, components: List[Dict[str, Any]]) -> None:
        """컴포넌트 레이아웃 최적화 - 겹치지 않도록 위치 조정"""
        if not components:
            return

        # 컴포넌트 타입별로 그룹화
        buttons = [c for c in components if c["type"] == "button"]
        inputs = [c for c in components if c["type"] == "input"]
        texts = [c for c in components if c["type"] == "text"]
        cards = [c for c in components if c["type"] == "card"]
        rectangles = [c for c in components if c["type"] == "rectangle"]

        # 버튼들을 상단에 배치
        for i, button in enumerate(buttons):
            button["x"] = 50 + (i * 150)
            button["y"] = 50

        # 입력 필드들을 버튼 아래에 배치
        for i, input_field in enumerate(inputs):
            input_field["x"] = 50 + (i * 220)
            input_field["y"] = 120

        # 텍스트들을 입력 필드 아래에 배치
        for i, text in enumerate(texts):
            text["x"] = 50 + (i * 250)
            text["y"] = 200

        # 카드들을 텍스트 아래에 배치
        for i, card in enumerate(cards):
            card["x"] = 50 + (i * 320)
            card["y"] = 250

        # 사각형들을 카드 아래에 배치
        for i, rect in enumerate(rectangles):
            rect["x"] = 50 + (i * 350)
            rect["y"] = 480

    async def process_message(self, user_message: str) -> Dict[str, Any]:
        """Process a user message and advance the planning state."""
        self.state["history"].append({"role": "user", "content": user_message})
        response = {"message": "", "state": self.state["step"]}

        try:
            if self.state["step"] == "init":
                # Parse initial inputs from message or ask for them
                # For simplicity, assume message contains JSON with product_concept and user_persona
                try:
                    inputs = json.loads(user_message)
                    self.state["data"]["product_concept"] = inputs.get("product_concept")
                    self.state["data"]["user_persona"] = inputs.get("user_persona")
                    self.state["data"]["figma_url"] = inputs.get("figma_url")
                    if self.state["data"]["figma_url"]:
                        figma_file_id, node_id = self._extract_figma_ids(self.state["data"]["figma_url"])
                        self.state["data"]["figma_file_id"] = figma_file_id
                        self.state["data"]["figma_node_id"] = node_id
                except json.JSONDecodeError:
                    response["message"] = "Please provide product concept, user persona, and optional Figma URL in JSON format."
                    return response

                if not self.state["data"]["product_concept"] or not self.state["data"]["user_persona"]:
                    response["message"] = "Product concept and user persona are required."
                    return response

                self.state["step"] = "figma_analysis"
                response["message"] = "Starting product planning. Analyzing Figma if provided..."

            if self.state["step"] == "figma_analysis" and self.state["data"]["figma_file_id"]:
                logger.info(f"Analyzing Figma file with ID: {self.state['data']['figma_file_id']}")
                figma_context = self.state["data"]
                analysis_result = await self.figma_analyzer.run_workflow(figma_context)
                self.state["data"]["figma_analysis"] = analysis_result
                logger.info("Figma analysis completed.")
                response["message"] += "\nFigma analysis complete."
                self.state["step"] = "prd_drafting"

            if self.state["step"] == "figma_analysis" and not self.state["data"]["figma_file_id"]:
                self.state["data"]["figma_analysis"] = {"status": "skipped"}
                self.state["step"] = "prd_drafting"

            if self.state["step"] == "prd_drafting":
                logger.info("Drafting PRD...")
                prd_context = self.state["data"]
                prd_result = await self.prd_writer.run_workflow(prd_context)
                self.state["data"]["prd_draft"] = prd_result
                logger.info("PRD drafting completed.")
                response["message"] += "\nPRD draft complete. Generating Figma components..."
                # === Figma 컴포넌트 생성 단계 고도화 ===
                prd_content = str(prd_result)
                components = self._extract_figma_components_from_prd(prd_content)

                # 고도화된 FigmaCreatorAgent 호출
                try:
                    figma_result = await self.figma_creator_agent.run_workflow({
                        "prd_content": prd_content,
                        "components": components
                    })

                    # 추가로 특정 레이아웃 타입에 따른 생성도 시도
                    if "모바일" in prd_content or "앱" in prd_content:
                        mobile_result = await self.figma_creator_agent.create_mobile_app_layout(
                            app_name="제품 앱",
                            features=["로그인", "회원가입", "메인 기능", "설정", "프로필"]
                        )
                        figma_result["mobile_layout"] = mobile_result

                    elif "대시보드" in prd_content or "관리" in prd_content:
                        dashboard_result = await self.figma_creator_agent.create_web_dashboard_layout(
                            dashboard_title="관리 대시보드",
                            widgets=["사용자 통계", "매출 현황", "시스템 상태", "최근 활동", "알림", "설정"]
                        )
                        figma_result["dashboard_layout"] = dashboard_result

                    self.state["data"]["figma_creation_result"] = figma_result
                    response["message"] += f"\n🎨 Figma 레이아웃 스펙 생성 완료! {figma_result.get('components_spec_count', 0)}개 컴포넌트 스펙이 생성되었습니다. 레이아웃 최적화도 적용되었습니다."

                except Exception as e:
                    self.logger.error(f"Figma 생성 단계 오류: {str(e)}")
                    response["message"] += f"\n⚠️ Figma 컴포넌트 생성 중 오류가 발생했습니다: {str(e)}"
                    # 오류가 있어도 계속 진행
                self.state["step"] = "report_generation"

            if self.state["step"] == "report_generation":
                logger.info("Generating final report...")
                report_context = self.state["data"]
                final_report = await self.reporting_coordinator.generate_final_report(report_context)
                self.state["data"]["final_report"] = final_report
                logger.info("Final report generation completed.")
                response["message"] += "\nFinal report generated."
                self.state["step"] = "save_report"

            if self.state["step"] == "save_report":
                save_status = await self._save_final_report(self.state["data"]["final_report"], self.state["data"]["product_concept"])
                self.state["data"]["final_report"]["save_status"] = save_status
                saved_path = save_status.get("file_path") if isinstance(save_status, dict) else None
                if saved_path:
                    response["message"] += f"\nReport saved to local filesystem: {saved_path}"
                else:
                    response["message"] += "\nReport save status recorded (local filesystem)."
                self.state["step"] = "complete"

            if self.state["step"] == "complete":
                response["message"] += "\nPlanning complete!"
                response["final_report"] = self.state["data"]["final_report"]

            self.state["history"].append({"role": "assistant", "content": response["message"]})
            return response

        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            response["message"] = f"Error: {str(e)}"
            return response

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return self.state

    def set_state(self, state: Dict[str, Any]):
        """Set state from serialized data."""
        self.state = state

    async def run_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ProductPlannerAgent의 전체 워크플로우를 실행합니다.
        BaseAgentSimple의 추상 메서드 구현
        """
        logger.info("🚀 ProductPlannerAgent 전체 워크플로우 시작")

        try:
            # 입력 데이터에서 정보 추출
            product_concept = input_data.get("product_concept", "제품")
            user_persona = input_data.get("user_persona", "사용자")
            figma_url = input_data.get("figma_url")

            # 상태 초기화
            self.state["data"]["product_concept"] = product_concept
            self.state["data"]["user_persona"] = user_persona

            if figma_url:
                figma_file_id, figma_node_id = self._extract_figma_ids(figma_url)
                self.state["data"]["figma_file_id"] = figma_file_id
                self.state["data"]["figma_node_id"] = figma_node_id

            # 1. Figma 분석
            if self.state["data"]["figma_file_id"]:
                logger.info("🎨 Figma 분석 시작")
                figma_result = await self.figma_analyzer.run_workflow(self.state["data"])
                self.state["data"]["figma_analysis"] = figma_result

            # 2. PRD 작성
            logger.info("📝 PRD 작성 시작")
            prd_result = await self.prd_writer.run_workflow(self.state["data"])
            self.state["data"]["prd_draft"] = prd_result

            # 3. Figma 컴포넌트 생성
            logger.info("🔧 Figma 컴포넌트 생성 시작")
            prd_content = str(prd_result)
            components = self._extract_figma_components_from_prd(prd_content)

            figma_result = await self.figma_creator_agent.run_workflow({
                "prd_content": prd_content,
                "components": components
            })
            self.state["data"]["figma_creation_result"] = figma_result

            # 4. 최종 보고서 생성
            logger.info("📊 최종 보고서 생성 시작")
            final_report = await self.reporting_coordinator.generate_final_report(self.state["data"])
            self.state["data"]["final_report"] = final_report

            # 5. 보고서 저장
            logger.info("💾 보고서 저장 시작")
            save_result = await self._save_final_report(final_report, product_concept)
            final_report["save_status"] = save_result

            self.state["step"] = "complete"

            logger.info("✅ ProductPlannerAgent 워크플로우 완료")
            return {
                "status": "success",
                "final_report": final_report,
                "figma_creation_result": figma_result,
                "prd_draft": prd_result
            }

        except Exception as e:
            logger.error(f"❌ ProductPlannerAgent 워크플로우 실패: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
