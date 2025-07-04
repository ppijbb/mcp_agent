import asyncio
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

# Real MCP Agent imports
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)

# Keep existing enums and data classes (they're well designed)
class SEOEmergencyLevel(Enum):
    """SEO Emergency Level Classification"""
    CRITICAL = "🚨 응급실"
    HIGH = "⚠️ 위험"
    MEDIUM = "⚡ 주의"
    LOW = "✅ 안전"
    EXCELLENT = "🚀 완벽"

class CompetitorThreatLevel(Enum):
    """Competitor Threat Assessment"""
    DOMINATING = "👑 지배중"
    RISING = "📈 급상승"
    STABLE = "➡️ 안정"
    DECLINING = "📉 하락"
    WEAK = "😴 약함"

@dataclass
class SEOAnalysisResult:
    """Real SEO Analysis Result - No Mock Data"""
    url: str
    emergency_level: SEOEmergencyLevel
    overall_score: float  # From real Lighthouse analysis
    performance_score: float
    seo_score: float
    accessibility_score: float
    best_practices_score: float
    core_web_vitals: Dict[str, Any]
    critical_issues: List[str]
    quick_fixes: List[str]
    estimated_recovery_days: int
    competitor_analysis: List[Dict[str, Any]]
    recommendations: List[str]
    analysis_timestamp: datetime
    lighthouse_raw_data: Dict[str, Any]  # Full Lighthouse report

@dataclass
class SEOPrescription:
    """SEO Treatment Prescription"""
    prescription_id: str
    patient_url: str
    emergency_treatment: List[str]
    weekly_medicine: List[str]
    monthly_checkup: List[str]
    competitive_moves: List[str]
    expected_results: str
    follow_up_date: datetime
    implementation_priority: List[str]

class SEODoctorMCPAgent:
    """
    Real SEO Doctor MCP Agent Implementation
    
    Features:
    - Real Lighthouse performance analysis via Puppeteer MCP Server
    - Actual competitor research via Google Search MCP Server
    - File system integration for report generation
    - No mock data or simulations
    """
    
    def __init__(self, output_dir: str = "seo_doctor_reports"):
        self.output_dir = output_dir
        # NOTE: The app initialization is now deferred to the async context manager
        
    async def emergency_seo_diagnosis(
        self, 
        url: str, 
        include_competitors: bool = True,
        competitor_urls: Optional[List[str]] = None
    ) -> SEOAnalysisResult:
        """
        🚨 Real Emergency SEO Diagnosis
        
        Uses actual MCP servers for:
        - Lighthouse performance analysis
        - Google Search for competitor research
        - Real website crawling and analysis
        """
        
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize the app within the async context
        app = MCPApp(
            name="seo_doctor",
            settings=get_settings("configs/mcp_agent.config.yaml"),
            human_input_callback=None
        )
        
        async with app.run() as seo_app:
            context = seo_app.context
            logger = seo_app.logger
            
            await self._configure_seo_mcp_servers(context, logger)
            
            lighthouse_agent = Agent(
                name="lighthouse_analyzer",
                instruction=f"""You are a Lighthouse performance expert.
                Your SOLE purpose is to use the 'puppeteer' tool to run a Lighthouse audit for the URL: {url}.
                You MUST call the 'run' tool on the 'puppeteer' server with the parameter analysis_type='lighthouse'.
                The output of the tool is the only thing you should return.
                Do not add any additional text, explanation, or formatting. Just return the raw tool output as a JSON string.""",
                server_names=["puppeteer"]
            )
            
            seo_research_agent = Agent(
                name="seo_researcher",
                instruction=f"""You are an expert SEO analyst and researcher.
                Research and analyze: {url}. Use search and fetch tools to gather real data.
                Provide emergency-level recommendations for critical issues.""",
                server_names=["g-search", "fetch", "filesystem"]
            )
            
            logger.info(f"Starting emergency SEO diagnosis for: {url}")
            
            try:
                react_result = await self._simple_react_chain(
                    agents={
                        "lighthouse": lighthouse_agent,
                        "seo_research": seo_research_agent
                    },
                    url=url,
                    include_competitors=include_competitors,
                    competitor_urls=competitor_urls,
                    logger=logger
                )
                
                structured_result = await self._structure_seo_results(react_result, url, timestamp, logger)
                prescription = await self._generate_seo_prescription(structured_result, url, timestamp)
                await self._save_seo_analysis(structured_result, prescription, timestamp)
                
                logger.info(f"Emergency SEO diagnosis completed for: {url}")
                return structured_result
                
            except Exception as e:
                logger.error(f"Emergency SEO diagnosis failed for {url}: {e}", exc_info=True)
                return SEOAnalysisResult(
                    url=url,
                    emergency_level=SEOEmergencyLevel.CRITICAL,
                    overall_score=0, performance_score=0, seo_score=0, accessibility_score=0, best_practices_score=0,
                    core_web_vitals={},
                    critical_issues=[f"Analysis failed: {str(e)}"],
                    quick_fixes=["Fix MCP server configuration", "Check website accessibility"],
                    estimated_recovery_days=0, competitor_analysis=[],
                    recommendations=["Resolve analysis errors first"],
                    analysis_timestamp=datetime.now(timezone.utc),
                    lighthouse_raw_data={}
                )

    async def _simple_react_chain(
        self,
        agents: Dict[str, Agent],
        url: str,
        include_competitors: bool,
        competitor_urls: Optional[List[str]],
        logger
    ) -> str:
        llm = OpenAIAugmentedLLM()
        full_analysis = ""
        lighthouse_result_str = ""
        
        # Phase 1: Real Lighthouse Analysis via Agent Tool Call
        lighthouse_agent = agents.get("lighthouse")
        if lighthouse_agent:
            logger.info("REACT CHAIN: Running lighthouse_analyzer to trigger Lighthouse tool...")
            try:
                # Use the LLM to execute the agent's instruction, which triggers the tool.
                raw_lighthouse_output = await llm.generate_str(message=lighthouse_agent.instruction)
                # Validate that the output is valid JSON
                json.loads(raw_lighthouse_output) 
                # Embed the raw JSON in a clearly marked section for later extraction.
                lighthouse_result_str = f"---LIGHTHOUSE_RAW_JSON_START---\n{raw_lighthouse_output}\n---LIGHTHOUSE_RAW_JSON_END---\n\n"
                logger.info("REACT CHAIN: Real Lighthouse analysis tool call successful.")
            except Exception as e:
                logger.error(f"REACT CHAIN: Real Lighthouse analysis failed: {e}", exc_info=True)
                lighthouse_result_str = f"Lighthouse Analysis Failed: {e}\n\n"
        
        full_analysis += lighthouse_result_str

        # Phase 2: SEO Research
        seo_agent = agents.get("seo_research")
        if seo_agent:
            logger.info("REACT CHAIN: Running seo_researcher...")
            competitor_query = f"Research top 3 competitors for {url}."
            if competitor_urls:
                competitor_query = f"Analyze competitors: {', '.join(competitor_urls)}."
            
            # This prompt now explicitly uses the lighthouse result string, which contains the raw JSON
            prompt = f"""As the {seo_agent.name}, perform SEO research for {url}.
            Context from the Lighthouse audit is provided below (it may be raw JSON or an error message):
            ---
            {lighthouse_result_str if lighthouse_result_str else "Lighthouse audit was not run."}
            ---
            Your tasks:
            1. Analyze on-page SEO (titles, meta descriptions, headers).
            2. {'Include competitor analysis. ' + competitor_query if include_competitors else ''}
            3. Identify technical SEO issues not in Lighthouse (sitemap, robots.txt).
            4. Evaluate content quality and keyword strategy.
            Provide a concise summary of findings, focusing on critical issues.
            """
            try:
                seo_research_result = await llm.generate_str(message=prompt, request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07", temperature=0.1))
                full_analysis += f"SEO Research Results:\n{seo_research_result}\n\n"
                logger.info("REACT CHAIN: SEO research completed.")
            except Exception as e:
                logger.error(f"REACT CHAIN: SEO research failed: {e}", exc_info=True)
                full_analysis += f"SEO Research Failed: {e}\n\n"

        # Phase 3: Final Synthesis
        logger.info("REACT CHAIN: Synthesizing final report...")
        synthesis_prompt = f"""You are a master SEO Doctor. Synthesize the findings from the Lighthouse audit JSON and the SEO research text into a single, comprehensive SEO diagnosis for {url}.

        Combined Analysis Data:
        ---
        {full_analysis}
        ---

        Based on all available data (both the raw Lighthouse JSON and the text analysis), provide the final structured output. 
        It is critical that you parse the scores and Core Web Vitals from the Lighthouse JSON if it exists.
        Follow this format precisely:

        - Emergency Level: [CRITICAL/HIGH/MEDIUM/LOW/EXCELLENT]
        - Overall Score: [Calculated average score from Lighthouse data, if available]
        - Core Web Vitals: [LCP, FCP, CLS values from Lighthouse JSON]
        - Critical Issues: [List of 5-10 most urgent issues from all domains combined]
        - Quick Fixes: [List of 3-5 high-impact, low-effort fixes]
        - Estimated Recovery Days: [Number of days]
        - Competitor Analysis: [Summary of competitor landscape, if analyzed]
        - Recommendations: [List of 10-15 detailed, consolidated recommendations]
        """
        final_report = await llm.generate_str(message=synthesis_prompt, request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07", temperature=0.1))
        logger.info("REACT CHAIN: Synthesis complete.")
        # Embed the raw JSON again in the final report to ensure it's not lost
        return lighthouse_result_str + final_report

    async def _configure_seo_mcp_servers(self, context, logger):
        if "filesystem" in context.config.mcp.servers:
            # Ensure the output directory is part of the server args
            if self.output_dir not in context.config.mcp.servers["filesystem"].args:
                context.config.mcp.servers["filesystem"].args.extend([self.output_dir])
            logger.info(f"Filesystem server configured for output dir: {self.output_dir}")
            
        required_servers = ["g-search", "fetch", "filesystem", "puppeteer"]
        missing_servers = [s for s in required_servers if s not in context.config.mcp.servers]
        if missing_servers:
            logger.warning(f"Missing required MCP servers for SEO analysis: {missing_servers}")
            logger.info("You can install them by running: npm install -g @modelcontextprotocol/server-puppeteer g-search-mcp @modelcontextprotocol/server-fetch @modelcontextprotocol/server-filesystem")
    
    async def _structure_seo_results(
        self, 
        raw_analysis: str, 
        url: str, 
        timestamp: str,
        logger
    ) -> SEOAnalysisResult:
        """
        Structures the combined analysis string into the SEOAnalysisResult data class.
        It first extracts and parses the raw Lighthouse JSON, then parses the synthesized text report.
        """
        try:
            lighthouse_raw_data = {}
            metrics_data = {}
            
            # Step 1: Extract and parse Lighthouse JSON from the raw analysis string
            match = re.search(r"---LIGHTHOUSE_RAW_JSON_START---\n(.*?)\n---LIGHTHOUSE_RAW_JSON_END---", raw_analysis, re.DOTALL)
            if match:
                lighthouse_json_str = match.group(1)
                try:
                    lighthouse_raw_data = json.loads(lighthouse_json_str)
                    # Step 2: Extract structured metrics from the parsed JSON data
                    metrics_data = await self._extract_lighthouse_metrics(lighthouse_raw_data, logger)
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse Lighthouse JSON from analysis string: {e}")
            else:
                logger.warning("Could not find Lighthouse JSON in the analysis string.")

            # Step 3: Parse the synthesized text part of the report
            analysis_data = await self._parse_llm_analysis(raw_analysis)
            
            # Step 4: Combine data into the final result object
            return SEOAnalysisResult(
                url=url,
                emergency_level=await self._calculate_emergency_level(analysis_data, metrics_data),
                overall_score=metrics_data.get("overall_score", 0),
                performance_score=metrics_data.get("performance", 0),
                seo_score=metrics_data.get("seo", 0),
                accessibility_score=metrics_data.get("accessibility", 0),
                best_practices_score=metrics_data.get("best-practices", 0),
                core_web_vitals=metrics_data.get("core_web_vitals", {}),
                critical_issues=await self._extract_critical_issues(analysis_data),
                quick_fixes=await self._extract_quick_fixes(analysis_data),
                estimated_recovery_days=await self._estimate_recovery_time(analysis_data),
                competitor_analysis=await self._parse_competitor_analysis(analysis_data),
                recommendations=await self._extract_recommendations(analysis_data),
                analysis_timestamp=datetime.now(timezone.utc),
                lighthouse_raw_data=lighthouse_raw_data
            )
        except Exception as e:
            logger.error(f"Failed to structure SEO results for {url}: {e}", exc_info=True)
            raise

    async def _parse_llm_analysis(self, raw_analysis: str) -> Dict[str, Any]:
        """Parses the synthesized text report from the LLM to extract fields."""
        data = {}
        # Simple key-value parsing for fields that are on a single line
        for line in raw_analysis.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                # Normalize key for easier access
                data[key.strip().lower().replace(' ', '_')] = value.strip()
        
        # Regex-based parsing for multi-line list fields
        def _extract_list(field_name, text):
            # Matches list items that start with '- '
            pattern = re.compile(f"^- {field_name}:.*?\n((?:- .*?\n)+)", re.IGNORECASE | re.MULTILINE)
            match = re.search(pattern, text)
            if match:
                # Extract the matched group and split into a list
                items = [item.strip('- ').strip() for item in match.group(1).split('\n') if item.strip()]
                return items
            # Fallback for simple lists without a title on the same line
            pattern = re.compile(f"{field_name}:\n((?:- .*?\n)+)", re.IGNORECASE | re.MULTILINE)
            match = re.search(pattern, text)
            if match:
                items = [item.strip('- ').strip() for item in match.group(1).split('\n') if item.strip()]
                return items
            return []

        data['critical_issues'] = _extract_list("Critical Issues", raw_analysis)
        data['recommendations'] = _extract_list("Recommendations", raw_analysis)
        data['quick_fixes'] = _extract_list("Quick Fixes", raw_analysis)
        data['competitor_analysis'] = _extract_list("Competitor Analysis", raw_analysis)
        return data

    async def _extract_lighthouse_metrics(self, lighthouse_data: Dict[str, Any], logger) -> Dict[str, Any]:
        """Extracts key metrics directly from the parsed Lighthouse JSON report."""
        if not lighthouse_data or 'categories' not in lighthouse_data:
            logger.warning("Lighthouse data is empty or missing 'categories' key.")
            return {}
        
        try:
            categories = lighthouse_data.get('categories', {})
            # Convert scores from 0-1 to 0-100
            metrics = {cat: int(categories[cat]['score'] * 100) for cat in categories if cat in categories and isinstance(categories[cat], dict) and categories[cat].get('score') is not None}
            
            audits = lighthouse_data.get('audits', {})
            vitals = {}
            # Map audit keys to Core Web Vital names
            vital_map = {
                'largest-contentful-paint': 'LCP',
                'cumulative-layout-shift': 'CLS',
                'total-blocking-time': 'TBT', # TBT is often used as a proxy for FID
                'first-contentful-paint': 'FCP',
                'speed-index': 'Speed Index'
            }
            for key, name in vital_map.items():
                if key in audits:
                    vitals[name] = audits[key].get('displayValue', 'N/A')
            
            metrics['core_web_vitals'] = vitals
            
            # Calculate overall score as an average of the main category scores
            scores = [v for k, v in metrics.items() if k in ['performance', 'seo', 'accessibility', 'best-practices']]
            metrics['overall_score'] = sum(scores) / len(scores) if scores else 0
            
            return metrics
        except Exception as e:
            logger.error(f"Error extracting metrics from Lighthouse JSON: {e}", exc_info=True)
            return {}
    
    async def _parse_competitor_analysis(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # This is now based on the parsed LLM analysis
        items = analysis_data.get('competitor_analysis', [])
        # A more robust implementation could parse structured data here
        return [{"details": item} for item in items]

    async def _calculate_emergency_level(self, analysis_data: Dict[str, Any], metrics_data: Dict[str, Any]) -> SEOEmergencyLevel:
        """Calculates emergency level based on overall score and critical issues."""
        score = metrics_data.get('overall_score', 50) # Default to 50 if no score
        num_issues = len(analysis_data.get('critical_issues', []))
        
        if score < 40 or num_issues > 8: return SEOEmergencyLevel.CRITICAL
        if score < 60 or num_issues > 5: return SEOEmergencyLevel.HIGH
        if score < 85: return SEOEmergencyLevel.MEDIUM
        if score < 95: return SEOEmergencyLevel.LOW
        return SEOEmergencyLevel.EXCELLENT

    async def _extract_critical_issues(self, analysis_data: Dict[str, Any]) -> List[str]:
        return analysis_data.get('critical_issues', ["No specific issues extracted from report."])

    async def _extract_quick_fixes(self, analysis_data: Dict[str, Any]) -> List[str]:
        return analysis_data.get('quick_fixes', ["No specific quick fixes extracted from report."])

    async def _extract_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        return analysis_data.get('recommendations', ["No specific recommendations extracted from report."])

    async def _estimate_recovery_time(self, analysis_data: Dict[str, Any]) -> int:
        """Estimates recovery time based on the LLM's synthesized output."""
        try:
            # Try to find a number in the 'estimated_recovery_days' field
            recovery_str = analysis_data.get('estimated_recovery_days', '30')
            match = re.search(r'\d+', recovery_str)
            if match:
                return int(match.group())
        except:
            pass # Fallback to default
        return 30

    async def _generate_seo_prescription(self, analysis: SEOAnalysisResult, url: str, timestamp: str) -> SEOPrescription:
        return SEOPrescription(
            prescription_id=f"SEO_RX_{timestamp}",
            patient_url=url,
            emergency_treatment=analysis.quick_fixes,
            weekly_medicine=analysis.recommendations[:5],
            monthly_checkup=analysis.recommendations[5:],
            competitive_moves=[comp.get("details", "N/A") for comp in analysis.competitor_analysis],
            expected_results=f"Improve overall score from {analysis.overall_score:.0f} to {min(analysis.overall_score + 15, 100):.0f} in ~{analysis.estimated_recovery_days} days.",
            follow_up_date=datetime.now(timezone.utc) + timedelta(days=30),
            implementation_priority=analysis.critical_issues
        )

    async def _save_seo_analysis(self, analysis: SEOAnalysisResult, prescription: SEOPrescription, timestamp: str):
        report_path = os.path.join(self.output_dir, f"seo_emergency_diagnosis_{timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 🚨 SEO Emergency Diagnosis Report\n\n")
            f.write(f"**Patient URL**: {analysis.url}\n")
            f.write(f"**Diagnosis Date**: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"**Emergency Level**: {analysis.emergency_level.value}\n")
            f.write(f"**Overall Health Score**: {analysis.overall_score:.1f}/100\n\n")

            f.write("## 📊 Vital Signs (from Lighthouse)\n")
            f.write(f"- **Performance**: {analysis.performance_score}/100\n")
            f.write(f"- **SEO Health**: {analysis.seo_score}/100\n")
            f.write(f"- **Accessibility**: {analysis.accessibility_score}/100\n")
            f.write(f"- **Best Practices**: {analysis.best_practices_score}/100\n\n")

            f.write("## 🩺 Core Web Vitals (from Lighthouse)\n")
            if analysis.core_web_vitals:
                for vital, value in analysis.core_web_vitals.items():
                    f.write(f"- **{vital.replace('-', ' ').title()}**: {value}\n")
            else:
                f.write("- *No Core Web Vitals data available.*\n")
            f.write("\n")

            f.write("## 🚨 Critical Issues (Synthesized)\n")
            if analysis.critical_issues:
                for issue in analysis.critical_issues:
                    f.write(f"- {issue}\n")
            else:
                f.write("- *No critical issues were identified.*\n")
            f.write("\n")

            f.write("## ⚡ Emergency Treatment (Quick Fixes)\n")
            if analysis.quick_fixes:
                for fix in analysis.quick_fixes:
                    f.write(f"- {fix}\n")
            else:
                f.write("- *No quick fixes were recommended.*\n")
            f.write("\n")
            
            f.write(f"## 🗓️ Estimated Recovery\n")
            f.write(f"- **Estimated Days**: {analysis.estimated_recovery_days}\n\n")
            
            f.write(f"## 🏥 Prescription: {prescription.prescription_id}\n\n")
            f.write("### Priority Implementation\n")
            for item in prescription.implementation_priority:
                f.write(f"- {item}\n")
            f.write("\n### Weekly Medicine\n")
            for item in prescription.weekly_medicine:
                f.write(f"- {item}\n")
            f.write("\n### Monthly Checkup\n")
            for item in prescription.monthly_checkup:
                f.write(f"- {item}\n")
            f.write("\n")
            
            f.write(f"## 📈 Expected Recovery\n")
            f.write(f"- {prescription.expected_results}\n")
            f.write(f"- **Next Follow-up**: {prescription.follow_up_date.strftime('%Y-%m-%d')}\n")

# Export main functions
async def create_seo_doctor_agent(output_dir: str = "seo_doctor_reports") -> SEODoctorMCPAgent:
    return SEODoctorMCPAgent(output_dir=output_dir)

async def run_emergency_seo_diagnosis(
    url: str,
    include_competitors: bool = True,
    competitor_urls: Optional[List[str]] = None,
    output_dir: str = "seo_doctor_reports"
) -> SEOAnalysisResult:
    agent = await create_seo_doctor_agent(output_dir)
    return await agent.emergency_seo_diagnosis(
        url=url,
        include_competitors=include_competitors,
        competitor_urls=competitor_urls
    )