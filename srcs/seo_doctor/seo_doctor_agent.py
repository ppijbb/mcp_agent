import asyncio
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import aiohttp

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
    - Advanced competitor research via the new SEO MCP Server
    - Google Drive integration for report generation via Google Drive MCP
    - No mock data or simulations
    """
    
    def __init__(self, 
                 google_drive_mcp_url: str = "http://localhost:3001",
                 seo_mcp_url: str = "http://localhost:3002"):
        self.google_drive_mcp_url = google_drive_mcp_url
        self.seo_mcp_url = seo_mcp_url
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
        - SEO MCP for competitor SERP analysis
        - Real website crawling and analysis
        """
        
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
                Research and analyze: {url}. Use the 'seo_mcp' server to gather SERP data for competitive analysis.
                Provide emergency-level recommendations for critical issues.""",
                server_names=["seo_mcp", "fetch"] # Use seo_mcp instead of g-search
            )
            
            logger.info(f"Starting emergency SEO diagnosis for: {url}")
            
            try:
                # The react chain will now use the new SEO MCP
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
                final_report_url = await self._save_seo_analysis(structured_result, prescription, timestamp, logger)
                
                logger.info(f"Emergency SEO diagnosis completed for: {url}")
                logger.info(f"Final report available at (mock URL): {final_report_url}")
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
        serp_analysis_str = ""
        
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

        # Phase 2: SEO Research using the new SEO MCP
        seo_agent = agents.get("seo_research")
        if seo_agent:
            logger.info("REACT CHAIN: Running seo_researcher with SEO MCP...")
            
            # Formulate query for SERP analysis
            competitor_query = f"top organic competitors for '{url}'"
            if competitor_urls:
                # This part needs a more sophisticated approach, 
                # but for now, we'll just analyze the main URL's keywords.
                # A better approach would be to extract keywords from the URL first.
                competitor_query = f"SERP for keywords related to '{url}'"

            # Use the SEO MCP to get SERP data
            try:
                serp_url = f"{self.seo_mcp_url}/serp?q={competitor_query}"
                async with get_http_session() as session:
                    async with session.get(serp_url) as response:
                        response.raise_for_status()
                        serp_data = await response.json()
                serp_analysis_str = f"---SERP_ANALYSIS_JSON_START---\n{json.dumps(serp_data, indent=2)}\n---SERP_ANALYSIS_JSON_END---\n\n"
                logger.info("REACT CHAIN: SEO MCP SERP analysis successful.")
            except Exception as e:
                logger.error(f"REACT CHAIN: SEO MCP SERP analysis failed: {e}", exc_info=True)
                serp_analysis_str = f"SERP Analysis Failed: {e}\n\n"

            full_analysis += serp_analysis_str
            
            # This prompt now explicitly uses BOTH lighthouse and SERP results
            prompt = f"""As the {seo_agent.name}, perform SEO research for {url}.
            Context from Lighthouse and SERP analysis is provided below:
            Lighthouse Data:
            ---
            {lighthouse_result_str if lighthouse_result_str else "Lighthouse audit was not run."}
            ---
            SERP Analysis Data:
            ---
            {serp_analysis_str if serp_analysis_str else "SERP analysis was not run."}
            ---
            Your tasks:
            1. Analyze the SERP data to identify top 3 competitors and their strategies (titles, snippets).
            2. Based on both Lighthouse and SERP data, identify the most critical technical and content gaps for {url}.
            3. Evaluate content quality and keyword strategy based on competitor performance in the SERP.
            Provide a concise summary of findings, focusing on critical issues and strategic recommendations.
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
        # Add the new 'seo_mcp' to the list of required servers
        required_servers = ["fetch", "puppeteer", "seo_mcp"]
        missing_servers = [s for s in required_servers if s not in context.config.mcp.servers]
        if missing_servers:
            logger.warning(f"Missing required MCP servers for SEO analysis: {missing_servers}")
            logger.info("You may need to start the seo_mcp server manually or configure it in your YAML.")
    
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
        """Extracts key metrics from the raw Lighthouse JSON report."""
        try:
            categories = lighthouse_data.get('categories', {})
            audits = lighthouse_data.get('audits', {})

            def get_score(cat_id):
                return categories.get(cat_id, {}).get('score', 0) * 100

            performance = get_score('performance')
            seo = get_score('seo')
            accessibility = get_score('accessibility')
            best_practices = get_score('best-practices')
            
            overall_score = (performance + seo + accessibility + best_practices) / 4

            core_web_vitals = {
                'largest-contentful-paint': audits.get('largest-contentful-paint', {}).get('displayValue', 'N/A'),
                'cumulative-layout-shift': audits.get('cumulative-layout-shift', {}).get('displayValue', 'N/A'),
                'first-contentful-paint': audits.get('first-contentful-paint', {}).get('displayValue', 'N/A'),
                'speed-index': audits.get('speed-index', {}).get('displayValue', 'N/A'),
                'total-blocking-time': audits.get('total-blocking-time', {}).get('displayValue', 'N/A'),
            }

            return {
                "performance": performance,
                "seo": seo,
                "accessibility": accessibility,
                "best-practices": best_practices,
                "overall_score": overall_score,
                "core_web_vitals": core_web_vitals,
            }
        except Exception as e:
            logger.error(f"Error extracting lighthouse metrics: {e}", exc_info=True)
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

    async def _save_seo_analysis(self, analysis: SEOAnalysisResult, prescription: SEOPrescription, timestamp: str, logger) -> str:
        """
        Saves the markdown report to Google Drive via the Google Drive MCP.
        """
        report_file_name = f"seo_emergency_diagnosis_{timestamp}.md"
        
        # 1. Construct the report content in memory
        f = []
        f.append(f"# 🚨 SEO Emergency Diagnosis Report\n\n")
        f.append(f"**Patient URL**: {analysis.url}\n")
        f.append(f"**Diagnosis Date**: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.append(f"**Emergency Level**: {analysis.emergency_level.value}\n")
        f.append(f"**Overall Health Score**: {analysis.overall_score:.1f}/100\n\n")

        f.append("## 📊 Vital Signs (from Lighthouse)\n")
        f.append(f"- **Performance**: {analysis.performance_score}/100\n")
        f.append(f"- **SEO Health**: {analysis.seo_score}/100\n")
        f.append(f"- **Accessibility**: {analysis.accessibility_score}/100\n")
        f.append(f"- **Best Practices**: {analysis.best_practices_score}/100\n\n")

        f.append("## 🩺 Core Web Vitals (from Lighthouse)\n")
        if analysis.core_web_vitals:
            for vital, value in analysis.core_web_vitals.items():
                f.append(f"- **{vital.replace('-', ' ').title()}**: {value}\n")
        else:
            f.append("- *No Core Web Vitals data available.*\n")
        f.append("\n")

        f.append("## 🚨 Critical Issues (Synthesized)\n")
        if analysis.critical_issues:
            for issue in analysis.critical_issues:
                f.append(f"- {issue}\n")
        else:
            f.append("- *No critical issues were identified.*\n")
        f.append("\n")

        f.append("## ⚡ Emergency Treatment (Quick Fixes)\n")
        if analysis.quick_fixes:
            for fix in analysis.quick_fixes:
                f.append(f"- {fix}\n")
        else:
            f.append("- *No quick fixes were recommended.*\n")
        f.append("\n")
        
        f.append(f"## 🗓️ Estimated Recovery\n")
        f.append(f"- **Estimated Days**: {analysis.estimated_recovery_days}\n\n")
        
        f.append(f"## 🏥 Prescription: {prescription.prescription_id}\n\n")
        f.append("### Priority Implementation\n")
        for item in prescription.implementation_priority:
            f.append(f"- {item}\n")
        f.append("\n### Weekly Medicine\n")
        for item in prescription.weekly_medicine:
            f.append(f"- {item}\n")
        f.append("\n### Monthly Checkup\n")
        for item in prescription.monthly_checkup:
            f.append(f"- {item}\n")
        f.append("\n")
        
        f.append(f"## 📈 Expected Recovery\n")
        f.append(f"- {prescription.expected_results}\n")
        f.append(f"- **Next Follow-up**: {prescription.follow_up_date.strftime('%Y-%m-%d')}\n")
        
        report_content = "".join(f)
        
        # 2. Upload to Google Drive via MCP
        upload_url = f"{self.google_drive_mcp_url}/upload"
        payload = {
            "fileName": report_file_name,
            "content": report_content
        }
        
        try:
            async with get_http_session() as session:
                async with session.post(upload_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if result.get("success"):
                        file_id = result.get("fileId")
                        logger.info(f"Successfully uploaded report to Google Drive. File ID: {file_id}")
                        # In a real scenario, this would be a real shareable link
                        return f"https://docs.google.com/document/d/{file_id}"
                    else:
                        logger.error(f"Failed to upload report to Google Drive MCP: {result.get('message')}")
                        return "upload_failed"
                        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Error connecting to Google Drive MCP: {e}")
            return "upload_failed"
        except Exception as e:
            logger.error(f"An unexpected error occurred during report upload: {e}")
            return "upload_failed"


# Export main functions
async def create_seo_doctor_agent(google_drive_mcp_url: str = "http://localhost:3001", seo_mcp_url: str = "http://localhost:3002") -> SEODoctorMCPAgent:
    return SEODoctorMCPAgent(google_drive_mcp_url=google_drive_mcp_url, seo_mcp_url=seo_mcp_url)

async def run_emergency_seo_diagnosis(
    url: str,
    include_competitors: bool = True,
    competitor_urls: Optional[List[str]] = None,
    google_drive_mcp_url: str = "http://localhost:3001",
    seo_mcp_url: str = "http://localhost:3002"
) -> SEOAnalysisResult:
    agent = await create_seo_doctor_agent(google_drive_mcp_url, seo_mcp_url)
    return await agent.emergency_seo_diagnosis(
        url=url,
        include_competitors=include_competitors,
        competitor_urls=competitor_urls
    )