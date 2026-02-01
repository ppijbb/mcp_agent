"""
Mental Care Chatbot Service with MCP Agent
------------------------------------------
A comprehensive mental health care system using MCP agents that:
1. Analyzes user's psychological state through conversation
2. Uses psychological schema therapy techniques for emotional analysis
3. Generates psychological state graphs
4. Provides detailed reports with emotions, causes, and solutions
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from srcs.common.llm.fallback_llm import create_fallback_orchestrator_llm_factory
from srcs.common.utils import setup_agent_app

# Import visualization module
from srcs.utils.mental_visualization import MentalStateVisualizer


class EmotionType(Enum):
    JOY = "기쁨"
    SADNESS = "슬픔"
    ANGER = "분노"
    FEAR = "두려움"
    ANXIETY = "불안"
    DEPRESSION = "우울"
    LONELINESS = "외로움"
    STRESS = "스트레스"
    CONFUSION = "혼란"
    HOPELESSNESS = "절망"


class SeverityLevel(Enum):
    MINIMAL = 1
    MILD = 2
    MODERATE = 3
    SEVERE = 4
    EXTREME = 5


@dataclass
class EmotionData:
    emotion: EmotionType
    severity: SeverityLevel
    triggers: List[str]
    duration: str  # "며칠", "몇 주", "몇 달" 등
    context: str


@dataclass
class PsychologicalSchema:
    """심리도식치료 기반 심리 도식"""
    schema_name: str
    description: str
    triggers: List[str]
    adaptive_responses: List[str]
    maladaptive_responses: List[str]


@dataclass
class ConversationSession:
    session_id: str
    start_time: datetime
    emotions: List[EmotionData]
    psychological_schemas: List[PsychologicalSchema]
    conversation_history: List[Dict[str, str]]
    analysis_results: Dict[str, Any]


# Configuration
OUTPUT_DIR = "mental_care_reports"
MAX_CONVERSATION_TURNS = 50

# Initialize app
app = setup_agent_app("mental_care_chatbot")


def get_schema_therapy_knowledge() -> str:
    """심리도식치료 관련 지식 베이스"""
    return """
    심리도식치료(Schema Therapy) 핵심 개념:

    1. 조기부적응도식 (Early Maladaptive Schemas):
       - 버림받음/불안정성: 중요한 사람들이 떠날 것이라는 믿음
       - 불신/남용: 다른 사람들이 자신을 이용하거나 해칠 것이라는 믿음
       - 정서적 결핍: 정서적 지지, 보살핌, 사랑을 받지 못할 것이라는 믿음
       - 결함/수치심: 자신이 결함이 있고, 나쁘고, 원하지 않는 존재라는 믿음
       - 사회적 고립/소외: 다른 사람들과 다르고 소속되지 못한다는 믿음

    2. 도식 대처 방식:
       - 항복: 도식을 받아들이고 포기
       - 회피: 도식을 활성화시키는 상황을 피함
       - 과잉보상: 도식과 반대되는 행동을 과도하게 함

    3. 도식 모드:
       - 아동 모드: 상처받은 아동, 분노한 아동, 충동적 아동
       - 부모 모드: 처벌하는 부모, 요구하는 부모
       - 대처 모드: 순응적 항복자, 분리된 보호자, 과잉보상자
       - 건강한 성인 모드: 균형잡힌 사고와 행동
    """


class MentalCareOrchestrator:
    def __init__(self):
        self.session = None
        self.conversation_agent = None
        self.emotion_analyzer = None
        self.schema_therapist = None
        self.graph_generator = None
        self.report_generator = None
        self.visualizer = MentalStateVisualizer(OUTPUT_DIR)

    async def initialize_agents(self, context, logger):
        """MCP agents 초기화"""

        # 대화 상담사 에이전트
        self.conversation_agent = Agent(
            name="conversation_counselor",
            instruction=f"""당신은 따뜻하고 공감적인 심리상담사입니다.

            대화 가이드라인:
            1. 내담자의 감정을 깊이 경청하고 공감적으로 반응하세요
            2. 판단하지 말고 수용적인 태도를 유지하세요
            3. 내담자가 자신의 감정과 생각을 자유롭게 표현할 수 있도록 도우세요
            4. 적절한 개방형 질문을 통해 더 깊은 탐색을 도우세요
            5. 내담자의 강점과 자원을 인정하고 격려하세요

            심리도식치료 관점에서:
            - 내담자의 핵심 감정과 그 뒤에 숨은 도식을 탐색하세요
            - 어린 시절 경험과 현재 문제의 연결점을 찾아보세요
            - 도식 대처 방식(항복/회피/과잉보상)을 파악하세요

            한국어로 자연스럽고 따뜻하게 대화하세요.
            """,
            server_names=["filesystem"],
        )

        # 감정 분석 에이전트 (강화된 분석 기능)
        self.emotion_analyzer = Agent(
            name="emotion_analyzer",
            instruction=f"""당신은 심리학 전문가로서 대화 내용을 분석하여 감정을 정확히 파악합니다.

            분석 영역:
            1. 주요 감정 (기쁨, 슬픔, 분노, 두려움, 불안, 우울, 외로움, 스트레스, 혼란, 절망)
            2. 감정의 강도 (1-5 척도)
            3. 감정의 트리거 (무엇이 이 감정을 유발했는가)
            4. 감정의 지속 기간
            5. 감정이 발생한 상황적 맥락
            6. 신체적 증상 (두통, 불면, 식욕 변화 등)
            7. 행동 변화 (회피, 과잉 활동, 고립 등)

            {get_schema_therapy_knowledge()}

            분석 결과를 다음 JSON 형식으로 제공하세요:
            {{
                "primary_emotions": [
                    {{
                        "emotion": "감정명",
                        "intensity": 1-5,
                        "triggers": ["트리거1", "트리거2"],
                        "duration": "지속기간",
                        "context": "상황적 맥락"
                    }}
                ],
                "secondary_emotions": [...],
                "physical_symptoms": ["증상1", "증상2"],
                "behavioral_changes": ["변화1", "변화2"],
                "risk_assessment": "low/medium/high"
            }}
            """,
            server_names=["filesystem"],
        )

        # 심리도식 치료사 에이전트
        self.schema_therapist = Agent(
            name="schema_therapist",
            instruction=f"""당신은 심리도식치료 전문가입니다.

            {get_schema_therapy_knowledge()}

            역할:
            1. 내담자의 대화에서 활성화된 심리도식을 식별
            2. 조기부적응도식과 현재 문제의 연관성 분석
            3. 도식 대처 방식(항복/회피/과잉보상) 파악
            4. 도식 모드 분석 (아동/부모/대처/건강한 성인)
            5. 적응적 대처 방안 제시
            6. 인지적 재구성 전략 제안
            7. 행동 실험 계획 수립

            분석 결과를 다음 JSON 형식으로 제공하세요:
            {{
                "activated_schemas": [
                    {{
                        "schema_name": "도식명",
                        "evidence": "근거",
                        "severity": "low/medium/high",
                        "triggers": ["트리거들"]
                    }}
                ],
                "coping_modes": {{
                    "surrender": "항복 방식 설명",
                    "avoidance": "회피 방식 설명",
                    "overcompensation": "과잉보상 방식 설명"
                }},
                "mode_analysis": {{
                    "current_mode": "현재 모드",
                    "target_mode": "목표 모드"
                }},
                "interventions": [
                    {{
                        "type": "개입 유형",
                        "description": "설명",
                        "priority": "high/medium/low"
                    }}
                ]
            }}
            """,
            server_names=["filesystem"],
        )

        # 개선된 그래프 생성 에이전트
        self.graph_generator = Agent(
            name="graph_generator",
            instruction="""당신은 심리 상태 시각화 전문가입니다.

            제공된 대화 데이터와 분석 결과를 바탕으로 다음과 같은 시각화를 생성하세요:

            1. 감정 변화 시계열 그래프
            2. 감정 분포 파이 차트
            3. 감정 강도 히트맵
            4. 주요 감정 막대 차트
            5. 감정 간 상관관계 매트릭스
            6. 종합 대시보드

            Python 코드를 작성하여 matplotlib와 seaborn을 사용해 시각화를 생성하고
            파일로 저장하는 코드를 제공하세요.

            한글 폰트 문제가 있을 수 있으니 적절한 대안을 제시하세요.
            """,
            server_names=["filesystem", "interpreter"],
        )

        # 강화된 보고서 생성 에이전트
        self.report_generator = Agent(
            name="report_generator",
            instruction="""당신은 심리 상담 보고서 작성 전문가입니다.

            보고서 구성:
            1. 세션 개요 (날짜, 시간, 주요 주제)
            2. 주요 감정 분석 결과
            3. 심리도식 분석 (활성화된 도식, 대처 방식)
            4. 위험도 평가 (자해/자살 위험, 기능 손상 정도)
            5. 감정의 원인 및 트리거 분석
            6. 단기 및 장기 치료 목표
            7. 구체적 개입 전략
            8. 셀프케어 및 자조 활동 제안
            9. 전문가 상담 권유사항
            10. 추적 관찰 계획
            11. 참고 자료 및 리소스

            보고서는 다음 형식을 따르세요:
            - 전문적이면서도 내담자가 이해하기 쉬운 한국어
            - 마크다운 형식으로 구조화
            - 구체적이고 실행 가능한 권장사항
            - 긍정적이고 희망적인 톤 유지
            - 비밀보장 및 윤리적 고려사항 포함

            보고서를 파일로 저장하는 코드도 포함하세요.
            """,
            server_names=["filesystem"],
        )

        logger.info("모든 MCP agents가 성공적으로 초기화되었습니다.")

    async def start_conversation_session(self):
        """새로운 상담 세션 시작"""
        session_id = f"mental_care_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session = ConversationSession(
            session_id=session_id,
            start_time=datetime.now(),
            emotions=[],
            psychological_schemas=[],
            conversation_history=[],
            analysis_results={}
        )

        print("🌟 안녕하세요! 마음 돌봄 상담실에 오신 것을 환영합니다.")
        print("저는 당신의 마음을 함께 돌보는 AI 상담사입니다.")
        print("편안하게 마음을 나누어 주세요. 언제든 '종료'라고 말씀하시면 상담을 마무리하겠습니다.")
        print("\n⚠️  주의사항: 이 서비스는 전문적인 심리 치료를 대체할 수 없습니다.")
        print("   심각한 심리적 위기 상황에서는 반드시 전문가의 도움을 받으시기 바랍니다.\n")

        return session_id

    async def process_user_input(self, user_input: str, orchestrator, logger):
        """사용자 입력 처리 및 분석"""

        # 위기 상황 키워드 체크
        crisis_keywords = ['자살', '죽고싶', '끝내고싶', '살기싫', '죽어버리고싶']
        if any(keyword in user_input for keyword in crisis_keywords):
            print("\n🚨 위급 상황 감지됨!")
            print("현재 위험한 생각을 하고 계신 것 같습니다.")
            print("즉시 전문가의 도움을 받으시기 바랍니다:")
            print("• 생명의 전화: 1588-9191")
            print("• 청소년 전화: 1388")
            print("• 응급실 방문 또는 119 신고")
            print("• 정신건강 위기상담전화: 1577-0199\n")

        # 대화 기록 저장
        self.session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "user",
            "message": user_input
        })

        # 상담사 응답 생성
        counselor_task = f"""
        사용자가 다음과 같이 말했습니다: "{user_input}"

        대화 맥락: {json.dumps(self.session.conversation_history[-3:], ensure_ascii=False, indent=2)}

        따뜻하고 공감적인 상담사로서 적절히 응답하세요.
        필요한 경우 더 깊은 탐색을 위한 질문을 포함하세요.
        심리도식치료 관점에서 도움이 될 수 있는 통찰을 제공하세요.
        """

        counselor_response = await orchestrator.generate_str(
            message=counselor_task,
            request_params=RequestParams(model="gemini-2.5-flash-lite")
        )

        # 상담사 응답 저장
        self.session.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "counselor",
            "message": counselor_response
        })

        # 감정 분석 (백그라운드에서 실행)
        emotion_task = f"""
        다음 대화 내용을 분석하여 사용자의 감정 상태를 파악하세요:

        사용자: "{user_input}"
        대화 맥락: {json.dumps(self.session.conversation_history[-5:], ensure_ascii=False, indent=2)}

        JSON 형식으로 분석 결과를 제공하세요.
        """

        # 심리도식 분석 (백그라운드에서 실행)
        schema_task = f"""
        사용자의 발화를 심리도식치료 관점에서 분석하세요:

        사용자 발화: "{user_input}"
        대화 맥락: {json.dumps(self.session.conversation_history[-5:], ensure_ascii=False, indent=2)}

        JSON 형식으로 도식 분석 결과를 제공하세요.
        """

        # 비동기적으로 분석 실행 (응답 속도 향상)
        emotion_analysis_task = orchestrator.generate_str(
            message=emotion_task,
            request_params=RequestParams(model="gemini-2.5-flash-lite")
        )

        schema_analysis_task = orchestrator.generate_str(
            message=schema_task,
            request_params=RequestParams(model="gemini-2.5-flash-lite")
        )

        print(f"\n🤖 상담사: {counselor_response}\n")

        # 분석 결과 저장 (백그라운드에서 완료되면)
        try:
            emotion_analysis = await emotion_analysis_task
            schema_analysis = await schema_analysis_task

            self.session.analysis_results[len(self.session.conversation_history)] = {
                "emotion_analysis": emotion_analysis,
                "schema_analysis": schema_analysis
            }
        except Exception as e:
            logger.warning(f"분석 중 오류 발생: {str(e)}")

        return counselor_response

    async def generate_final_report(self, orchestrator, logger):
        """최종 보고서 및 시각화 생성"""

        print("\n🔄 최종 분석 및 보고서를 생성하고 있습니다...")

        # 출력 디렉토리 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 시각화 생성
        try:
            session_info = {
                "start_time": self.session.start_time.isoformat(),
                "session_duration": str(datetime.now() - self.session.start_time),
                "total_turns": len([msg for msg in self.session.conversation_history if msg['speaker'] == 'user'])
            }

            visualization_files = self.visualizer.generate_all_visualizations(
                self.session.conversation_history,
                self.session.session_id,
                session_info
            )

            print(f"✅ {len(visualization_files)}개의 시각화 파일이 생성되었습니다.")

        except Exception as e:
            logger.warning(f"시각화 생성 중 오류 발생: {str(e)}")
            visualization_files = []

        # 최종 보고서 생성
        report_task = f"""
        다음 정보를 바탕으로 comprehensive한 심리 상담 보고서를 작성하세요:

        세션 정보:
        - 세션 ID: {self.session.session_id}
        - 시작 시간: {self.session.start_time}
        - 대화 횟수: {len([msg for msg in self.session.conversation_history if msg['speaker'] == 'user'])}
        - 총 소요 시간: {datetime.now() - self.session.start_time}

        대화 내용: {json.dumps(self.session.conversation_history, ensure_ascii=False, indent=2)}

        분석 결과: {json.dumps(self.session.analysis_results, ensure_ascii=False, indent=2)}

        생성된 시각화 파일들: {visualization_files}

        다음 구조로 보고서를 작성하세요:
        1. 세션 개요
        2. 주요 감정 분석
        3. 심리도식 분석 결과
        4. 위험도 평가
        5. 감정의 원인 및 트리거
        6. 치료 목표 및 개입 전략
        7. 셀프케어 제안
        8. 전문가 상담 권유사항
        9. 추적 관찰 계획
        10. 참고 자료

        보고서를 마크다운 형식으로 작성하여 {OUTPUT_DIR}/{self.session.session_id}_report.md 파일로 저장하세요.
        """

        try:
            report_result = await orchestrator.generate_str(
                message=report_task,
                request_params=RequestParams(model="gemini-2.5-flash-lite")
            )

            # 보고서 파일 직접 저장
            report_path = f"{OUTPUT_DIR}/{self.session.session_id}_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# 심리 상담 세션 보고서\n\n")
                f.write(f"**세션 ID:** {self.session.session_id}\n")
                f.write(f"**생성 일시:** {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}\n\n")
                f.write("---\n\n")
                f.write(report_result)

            print(f"✅ 상세 보고서가 생성되었습니다: {report_path}")

        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")

        print("\n" + "="*60)
        print("🌟 상담 세션이 완료되었습니다!")
        print(f"📊 상세한 분석 보고서: {OUTPUT_DIR}/{self.session.session_id}_report.md")
        print(f"📈 심리 상태 시각화 파일들:")
        for file in visualization_files:
            print(f"   • {os.path.basename(file)}")
        print("\n💙 오늘도 용기내어 상담에 참여해 주셔서 감사합니다.")
        print("   더 나은 내일을 위한 작은 걸음이었습니다.")
        print("="*60)

        return True


async def main():
    """메인 실행 함수"""

    async with app.run() as mental_app:
        context = mental_app.context
        logger = mental_app.logger

        # 파일시스템 서버 설정
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")

        # 인터프리터 서버 확인 (그래프 생성용)
        if "interpreter" not in context.config.mcp.servers:
            logger.warning("Python interpreter server not found - graph generation may be limited")

        # Mental Care Orchestrator 초기화
        mental_orchestrator = MentalCareOrchestrator()

        # MCP Orchestrator 생성
        orchestrator_llm_factory = create_fallback_orchestrator_llm_factory(

            primary_model="gemini-2.5-flash-lite",

            logger_instance=logger

        )

        orchestrator = Orchestrator(
            llm_factory=orchestrator_llm_factory,
            available_agents=[],  # agents will be added after initialization
            plan_type="full",
        )

        # Agents 초기화
        await mental_orchestrator.initialize_agents(context, logger)

        # Agents를 orchestrator에 추가
        orchestrator.available_agents = [
            mental_orchestrator.conversation_agent,
            mental_orchestrator.emotion_analyzer,
            mental_orchestrator.schema_therapist,
            mental_orchestrator.graph_generator,
            mental_orchestrator.report_generator,
        ]

        # 상담 세션 시작
        session_id = await mental_orchestrator.start_conversation_session()

        # 대화 루프
        conversation_count = 0
        while conversation_count < MAX_CONVERSATION_TURNS:
            try:
                user_input = input("당신: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['종료', '끝', 'quit', 'exit', 'bye']:
                    print("\n상담을 마무리하겠습니다. 잠시만 기다려주세요...")
                    break

                await mental_orchestrator.process_user_input(user_input, orchestrator, logger)
                conversation_count += 1

            except KeyboardInterrupt:
                print("\n\n상담을 중단합니다...")
                break
            except Exception as e:
                logger.error(f"오류가 발생했습니다: {str(e)}")
                print("죄송합니다. 일시적인 오류가 발생했습니다. 계속해주세요.")

        # 최종 보고서 생성
        await mental_orchestrator.generate_final_report(orchestrator, logger)


if __name__ == "__main__":
    asyncio.run(main())
