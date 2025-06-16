#!/usr/bin/env python3
"""
Simple Product Planner Agent - 실제 동작하는 최소 Agent
Basic Agent 패턴을 따라 구현한 진짜 Agent
"""

import asyncio
import os
import sys
from datetime import datetime

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Configuration
OUTPUT_DIR = "product_reports"
FIGMA_URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.figma.com/file/sample/test-design"

# Initialize MCP App (Basic Agent 패턴과 동일)
app = MCPApp(
    name="simple_product_planner",
    settings=get_settings("../../configs/mcp_agent.config.yaml"),
    human_input_callback=None
)

async def main():
    """Basic Agent와 동일한 패턴으로 실제 동작하는 Agent"""
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"product_analysis_{timestamp}.md"
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    # 🔥 실제 MCP App 실행 (Basic Agent와 동일)
    async with app.run() as planner_app:
        context = planner_app.context
        logger = planner_app.logger
        
        # 실제 MCP 서버 설정 확인 (Basic Agent 패턴)
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")
        else:
            logger.warning("Filesystem server not configured - report saving may fail")
        
        # --- 실제 Agent 정의 (Basic Agent 패턴) ---
        
        # Figma 분석 Agent
        figma_analyzer = Agent(
            name="figma_analyzer",
            instruction=f"""You are a Figma design analyst. Analyze the Figma design at: {FIGMA_URL}
            
            Provide analysis on:
            1. Design components and layout structure
            2. User interface patterns and interactions
            3. Design system consistency
            4. Technical implementation requirements
            5. User experience insights
            
            Be specific and actionable in your analysis.
            Focus on extracting product requirements from the design.""",
            server_names=["fetch", "filesystem"]
        )
        
        # PRD 생성 Agent
        prd_writer = Agent(
            name="prd_writer",
            instruction=f"""You are a product requirements document writer.
            
            Based on the Figma design analysis, create a comprehensive PRD with:
            1. Product Overview and Goals
            2. User Stories and Use Cases
            3. Functional Requirements
            4. Technical Specifications
            5. Success Metrics
            
            Format as clean markdown and save to: {output_path}
            
            Make it actionable for development teams.""",
            server_names=["filesystem"]
        )
        
        # --- 실제 Orchestrator 생성 (Basic Agent 패턴) ---
        logger.info(f"Initializing product planning workflow for: {FIGMA_URL}")
        
        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[figma_analyzer, prd_writer],
            plan_type="full"
        )
        
        # 실제 작업 정의
        task = f"""Create a product requirements document by analyzing the Figma design at: {FIGMA_URL}
        
        Steps:
        1. Use figma_analyzer to thoroughly analyze the Figma design
           - Extract all design components and patterns
           - Identify user interaction flows
           - Note technical requirements
        
        2. Use prd_writer to create a comprehensive PRD based on the analysis
           - Transform design insights into product requirements
           - Create actionable user stories
           - Define technical specifications
           - Save the final PRD to: {output_path}
        
        The final PRD should be professional, detailed, and ready for development teams."""
        
        # 🔥 실제 LLM 호출 (Basic Agent와 동일)
        logger.info("Starting the product planning workflow")
        try:
            await orchestrator.generate_str(
                message=task,
                request_params=RequestParams(model="gpt-4o-mini")  # 실제 LLM 호출
            )
            
            # 실제 결과 파일 생성 확인 (Basic Agent 패턴)
            if os.path.exists(output_path):
                logger.info(f"PRD successfully generated: {output_path}")
                print(f"✅ Product Requirements Document created: {output_path}")
                return True
            else:
                logger.error(f"Failed to create PRD at {output_path}")
                print(f"❌ Failed to create PRD")
                return False
                
        except Exception as e:
            logger.error(f"Error during workflow execution: {str(e)}")
            print(f"❌ Workflow failed: {str(e)}")
            return False

if __name__ == "__main__":
    print(f"🚀 Simple Product Planner Agent")
    print(f"📋 Analyzing Figma design: {FIGMA_URL}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 50)
    
    success = asyncio.run(main())
    
    if success:
        print("🎉 Product planning completed successfully!")
    else:
        print("💥 Product planning failed!")
    
    sys.exit(0 if success else 1) 