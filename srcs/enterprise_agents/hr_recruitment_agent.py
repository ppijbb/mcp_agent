import asyncio
import os
from datetime import datetime
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)

# Configuration
OUTPUT_DIR = "recruitment_reports"
POSITION_NAME = "Senior Software Engineer"
COMPANY_NAME = "TechCorp Inc."

app = MCPApp(
    name="hr_recruitment_system",
    settings=get_settings("configs/mcp_agent.config.yaml"),
    human_input_callback=None
)


async def main():
    """
    HR Recruitment Agent System
    
    Handles the complete recruitment lifecycle:
    1. Job description creation and posting
    2. Resume screening and candidate evaluation
    3. Interview question generation
    4. Reference checking automation
    5. Offer letter generation
    6. Onboarding checklist creation
    """
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async with app.run() as hr_app:
        context = hr_app.context
        logger = hr_app.logger
        
        # Configure servers
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")
        
        # --- HR RECRUITMENT AGENTS ---
        
        # Job Description Creator
        job_creator_agent = Agent(
            name="job_description_creator",
            instruction=f"""You are an expert HR professional specializing in job description creation.
            
            Create a comprehensive job description for {POSITION_NAME} at {COMPANY_NAME}:
            
            1. Company Overview section
            2. Position Summary (2-3 sentences)
            3. Key Responsibilities (8-10 bullet points)
            4. Required Qualifications:
               - Education requirements
               - Years of experience
               - Technical skills
               - Soft skills
            5. Preferred Qualifications
            6. Benefits & Compensation range
            7. Application process
            
            Make it engaging, inclusive, and compliant with employment laws.
            Use gender-neutral language and focus on essential vs. nice-to-have requirements.
            """,
            server_names=["filesystem", "fetch"],
        )
        
        # Resume Screener
        resume_screener_agent = Agent(
            name="resume_screener",
            instruction=f"""You are an experienced technical recruiter specializing in resume screening.
            
            Evaluate resumes for {POSITION_NAME} position based on:
            
            1. Technical Skills Match (40%)
               - Programming languages
               - Frameworks and tools
               - System design experience
               - Problem-solving abilities
            
            2. Experience Relevance (30%)
               - Years of relevant experience
               - Similar role responsibilities
               - Industry experience
               - Project complexity
            
            3. Education & Certifications (15%)
               - Relevant degree
               - Professional certifications
               - Continuous learning evidence
            
            4. Cultural Fit Indicators (15%)
               - Collaboration experience
               - Leadership examples
               - Growth mindset indicators
               - Communication skills
            
            Provide a score (1-100) and detailed reasoning for each candidate.
            Identify top 20% as "Strong Match", middle 60% as "Potential", bottom 20% as "Not a fit".
            """,
            server_names=["filesystem"],
        )
        
        # Interview Question Generator
        interview_agent = Agent(
            name="interview_question_generator",
            instruction=f"""You are a senior technical interviewer with expertise in behavioral and technical assessments.
            
            Generate comprehensive interview questions for {POSITION_NAME}:
            
            1. Technical Questions (60% of interview):
               - Coding problems (2-3 medium difficulty)
               - System design scenarios
               - Architecture discussions
               - Problem-solving approaches
            
            2. Behavioral Questions (25%):
               - Leadership examples
               - Conflict resolution
               - Team collaboration
               - Learning from failures
            
            3. Role-Specific Questions (15%):
               - Industry knowledge
               - Company culture fit
               - Career goals alignment
               - Motivation for role
            
            For each question, provide:
            - The question text
            - Expected answer framework
            - Evaluation criteria
            - Follow-up questions
            - Red flags to watch for
            """,
            server_names=["filesystem"],
        )
        
        # Reference Checker
        reference_agent = Agent(
            name="reference_checker",
            instruction="""You are a professional reference checker with experience in candidate verification.
            
            Create comprehensive reference check templates and processes:
            
            1. Reference Contact Script:
               - Professional introduction
               - Verification questions
               - Performance assessment
               - Rehire eligibility
               - Specific competency questions
            
            2. Documentation Template:
               - Reference contact information
               - Relationship to candidate
               - Performance ratings
               - Strengths and areas for improvement
               - Overall recommendation
            
            3. Red Flag Indicators:
               - Reluctance to provide reference
               - Inconsistent information
               - Performance concerns
               - Legal or ethical issues
            
            Ensure all processes comply with employment verification laws.
            """,
            server_names=["filesystem"],
        )
        
        # Offer Letter Generator
        offer_agent = Agent(
            name="offer_letter_generator",
            instruction=f"""You are an HR legal expert specializing in employment offers and contracts.
            
            Generate a professional offer letter for {POSITION_NAME} at {COMPANY_NAME}:
            
            1. Offer Letter Components:
               - Position title and reporting structure
               - Start date and location
               - Compensation package (salary, bonus, equity)
               - Benefits overview
               - Terms and conditions
               - At-will employment clause
               - Confidentiality and non-compete (if applicable)
            
            2. Legal Compliance:
               - Equal opportunity statement
               - Background check requirements
               - Drug testing policy (if applicable)
               - Document verification (I-9)
            
            3. Professional Tone:
               - Welcoming and enthusiastic
               - Clear and unambiguous
               - Professional formatting
               - Contact information for questions
            
            Ensure compliance with federal and state employment laws.
            """,
            server_names=["filesystem"],
        )
        
        # Onboarding Coordinator
        onboarding_agent = Agent(
            name="onboarding_coordinator",
            instruction=f"""You are an HR onboarding specialist focused on new hire success.
            
            Create a comprehensive onboarding program for {POSITION_NAME}:
            
            1. Pre-boarding (Week before start):
               - Welcome email sequence
               - Paperwork preparation
               - Equipment setup
               - First day logistics
               - Team introduction
            
            2. First Week Program:
               - Orientation schedule
               - Company culture immersion
               - Role-specific training
               - Manager meetings
               - Team introductions
               - Initial project assignments
            
            3. 30-60-90 Day Plan:
               - Learning objectives
               - Performance milestones
               - Check-in schedules
               - Feedback mechanisms
               - Integration activities
            
            4. Resources and Tools:
               - Employee handbook
               - System access guides
               - Mentor assignment
               - Training materials
               - Company directory
            
            Focus on engagement, productivity, and cultural integration.
            """,
            server_names=["filesystem"],
        )
        
        # Quality Evaluator for HR processes
        hr_evaluator = Agent(
            name="hr_process_evaluator",
            instruction="""You are an HR quality assurance expert evaluating recruitment processes.
            
            Evaluate HR deliverables based on:
            
            1. Legal Compliance (25%)
               - Employment law adherence
               - Non-discrimination practices
               - Privacy protection
               - Documentation requirements
            
            2. Process Efficiency (25%)
               - Time-to-hire optimization
               - Resource utilization
               - Candidate experience
               - Automation opportunities
            
            3. Quality Standards (25%)
               - Thoroughness of evaluation
               - Consistency in approach
               - Professional presentation
               - Clear communication
            
            4. Business Impact (25%)
               - Candidate quality
               - Cultural fit assessment
               - Long-term retention potential
               - Cost-effectiveness
            
            Provide EXCELLENT, GOOD, FAIR, or POOR ratings with specific improvement recommendations.
            """,
        )
        
        # Create quality controller
        hr_quality_controller = EvaluatorOptimizerLLM(
            optimizer=job_creator_agent,
            evaluator=hr_evaluator,
            llm_factory=OpenAIAugmentedLLM,
            min_rating=QualityRating.GOOD,
        )
        
        # --- CREATE ORCHESTRATOR ---
        logger.info(f"Initializing HR recruitment workflow for {POSITION_NAME}")
        
        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                hr_quality_controller,
                resume_screener_agent,
                interview_agent,
                reference_agent,
                offer_agent,
                onboarding_agent,
            ],
            plan_type="full",
        )
        
        # Define comprehensive recruitment task
        task = f"""Execute a complete HR recruitment process for {POSITION_NAME} at {COMPANY_NAME}:

        1. Use the hr_quality_controller to create a high-quality job description that is:
           - Legally compliant and inclusive
           - Technically accurate for the role
           - Attractive to top candidates
           - Clear about requirements and benefits
        
        2. Use the resume_screener to create evaluation criteria and scoring rubrics for:
           - Technical skill assessment
           - Experience evaluation
           - Cultural fit indicators
           - Overall candidate ranking system
        
        3. Use the interview_question_generator to develop:
           - Technical interview questions with solutions
           - Behavioral interview framework
           - Assessment criteria and scoring guides
           - Interview process timeline
        
        4. Use the reference_checker to create:
           - Reference check templates and scripts
           - Verification processes and documentation
           - Red flag identification guidelines
        
        5. Use the offer_letter_generator to create:
           - Professional offer letter template
           - Compensation and benefits summary
           - Legal compliance documentation
        
        6. Use the onboarding_coordinator to develop:
           - Pre-boarding communication plan
           - First week orientation program
           - 30-60-90 day integration plan
           - Resources and support materials
        
        Save all deliverables in the {OUTPUT_DIR} directory with appropriate naming:
        - job_description_{timestamp}.md
        - resume_screening_guide_{timestamp}.md
        - interview_questions_{timestamp}.md
        - reference_check_process_{timestamp}.md
        - offer_letter_template_{timestamp}.md
        - onboarding_program_{timestamp}.md
        """
        
        # Execute the workflow
        logger.info("Starting HR recruitment workflow")
        try:
            result = await orchestrator.generate_str(
                message=task,
                request_params=RequestParams(model="gpt-4o")
            )
            
            logger.info("HR recruitment workflow completed successfully")
            logger.info(f"All deliverables saved in {OUTPUT_DIR}/")
            return True
            
        except Exception as e:
            logger.error(f"Error during HR workflow execution: {str(e)}")
            return False


if __name__ == "__main__":
    asyncio.run(main()) 