import asyncio
import os

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from rich import print

# The orchestrator is a high-level abstraction that allows you to generate dynamic plans
# and execute them using multiple agents and servers.
# Here is the example plan generate by a planner for the example below.
# {
#   "data": {
#     "steps": [
#       {
#         "description": "Load the short story from short_story.md.",
#         "tasks": [
#           {
#             "description": "Find and read the contents of short_story.md.",
#             "agent": "finder"
#           }
#         ]
#       },
#       {
#         "description": "Generate feedback on the short story.",
#         "tasks": [
#           {
#             "description": "Review the short story for grammar, spelling, and punctuation errors and provide detailed feedback.",
#             "agent": "proofreader"
#           },
#           {
#             "description": "Check the short story for factual consistency and logical coherence, and highlight any inconsistencies.",
#             "agent": "fact_checker"
#           },
#           {
#             "description": "Evaluate the short story for style adherence according to APA style guidelines and suggest improvements.",
#             "agent": "style_enforcer"
#           }
#         ]
#       },
#       {
#         "description": "Combine the feedback into a comprehensive report.",
#         "tasks": [
#           {
#             "description": "Compile the feedback on proofreading, factuality, and style adherence to create a comprehensive graded report.",
#             "agent": "writer"
#           }
#         ]
#       },
#       {
#         "description": "Write the graded report to graded_report.md.",
#         "tasks": [
#           {
#             "description": "Save the compiled feedback as graded_report.md in the same directory as short_story.md.",
#             "agent": "writer"
#           }
#         ]
#       }
#     ],
#     "is_complete": false
#   }
# }

# It produces a report like graded_report.md, which contains the feedback from the proofreader, fact checker, and style enforcer.
#  The objective to analyze "The Battle of Glimmerwood" and generate a comprehensive feedback report has been successfully accomplished. The process involved several sequential and
# detailed evaluation steps, each contributing to the final assessment:

# 1. **Content Retrieval**: The short story was successfully located and read from `short_story.md`. This enabled subsequent analyses on the complete narrative content.

# 2. **Proofreading**: The text was rigorously reviewed for grammar, spelling, and punctuation errors. Specific corrections were suggested, enhancing both clarity and readability. Suggestions for improving the narrative's clarity were also provided,
# advising more context for characters, stakes clarification, and detailed descriptions to immerse readers.

# 3. **Factual and Logical Consistency**: The story's overall consistency was verified, examining location, plot development, and character actions. Although largely logical within its mystical context, the narrative contained unresolved elements about
# the Glimmerstones' power. Addressing these potential inconsistencies would strengthen its coherence.

# 4. **Style Adherence**: Evaluated against APA guidelines, the story was reviewed for format compliance, grammatical correctness, clarity, and tone. Although the narrative inherently diverges due to its format, suggestions for more formal alignment in
# future academic contexts were provided.

# 5. **Report Compilation**: All findings, corrections, and enhancement suggestions were compiled into the graded report, `graded_report.md`, situated in the same directory as the original short story.

# The completed graded report encapsulates detailed feedback across all targeted areas, providing a comprehensive evaluation for the student's work. It highlights essential improvements and ensures adherence to APA style rules, where applicable,
# fulfilling the complete objective satisfactorily.
# Total run time: 89.78s

app = MCPApp(name="assignment_grader_orchestrator")


async def run_workflow(task: str, model_name: str, plan_type: str):
    """
    주어진 태스크를 실행하고 결과를 반환하는 워크플로우
    """
    async with app.run() as orchestrator_app:
        logger = orchestrator_app.logger

        context = orchestrator_app.context
        
        # 파일시스템 서버의 인자에 현재 디렉토리 추가
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        writer_agent = Agent(
            name="writer",
            instruction="""You are an agent that can write to the filesystem.
            You are tasked with taking the user's input, addressing it, and 
            writing the result to disk in the appropriate location.""",
            server_names=["filesystem"],
        )

        proofreader = Agent(
            name="proofreader",
            instruction=""""Review the provided text for grammar, spelling, and punctuation errors.
            Identify any awkward phrasing or structural issues that could improve clarity. 
            Provide detailed feedback on corrections.""",
            server_names=[], # No servers needed, works on input text
        )

        fact_checker = Agent(
            name="fact_checker",
            instruction="""Verify the factual consistency within the provided text. Identify any contradictions,
            logical inconsistencies, or inaccuracies. 
            Highlight potential issues with reasoning or coherence.""",
            server_names=[], # No servers needed
        )

        style_enforcer = Agent(
            name="style_enforcer",
            instruction="""Analyze the provided text for adherence to style guidelines.
            Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
            enhance storytelling, readability, and engagement.""",
            server_names=[], # No servers needed
        )

        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                finder_agent,
                writer_agent,
                proofreader,
                fact_checker,
                style_enforcer,
            ],
            plan_type=plan_type,
        )

        result = await orchestrator.generate_str(
            message=task, request_params=RequestParams(model=model_name)
        )
        logger.info(f"Workflow result: {result}")
        return result


if __name__ == "__main__":
    import time

    # 기본 작업을 사용하여 테스트
    default_task = """Load the student's short story from a file named 'short_story.md', 
    and generate a report with feedback across proofreading, 
    factuality/logical consistency and style adherence.
    Write the graded report to 'graded_report.md' in the same directory."""
    
    # 테스트용 임시 파일 생성
    with open("short_story.md", "w") as f:
        f.write("This is a tale of bravery and waffles. The knight sir Reginald ate 12 waffles before the dragon came.")

    start = time.time()
    # 수정된 함수 호출
    final_result = asyncio.run(run_workflow(default_task, "gemini-2.5-flash-lite-preview-06-07", "full"))
    print("--- FINAL ORCHESTRATOR RESULT ---")
    print(final_result)
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
    
    # 생성된 파일 내용 확인
    if os.path.exists("graded_report.md"):
        print("\n--- GRADED REPORT CONTENT ---")
        with open("graded_report.md", "r") as f:
            print(f.read())
