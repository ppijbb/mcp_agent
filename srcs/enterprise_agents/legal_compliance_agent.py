import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator, QualityRating
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)
from srcs.common.utils import setup_agent_app, save_report


# Configuration
OUTPUT_DIR = "legal_compliance_reports"
COMPANY_NAME = "TechCorp Inc."
JURISDICTION = "United States - Federal and California State"

app = setup_agent_app("legal_compliance_system")


async def main():
    """
    Legal Compliance Agent System
    
    Handles comprehensive legal and compliance operations:
    1. Contract analysis and risk assessment
    2. Regulatory compliance monitoring
    3. Legal document generation
    4. Data privacy and security compliance
    5. Employment law compliance
    6. Intellectual property management
    7. Vendor and partnership agreements
    8. Litigation support and e-discovery
    """
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    app = setup_agent_app("legal_compliance_system")

    async with app.run() as legal_app:
        context = legal_app.context
        logger = legal_app.logger
        
        # Configure servers
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")
        
        # --- LEGAL COMPLIANCE AGENTS ---
        
        # Contract Analyzer
        contract_analyzer_agent = Agent(
            name="contract_analyzer",
            instruction=f"""You are a senior corporate attorney specializing in contract analysis and risk assessment.
            
            Analyze contracts and agreements for {COMPANY_NAME} with focus on:
            
            1. Risk Assessment:
               - Liability exposure and limitations
               - Indemnification clauses
               - Force majeure provisions
               - Termination conditions
               - Penalty and liquidated damages
            
            2. Commercial Terms Review:
               - Payment terms and conditions
               - Pricing and escalation clauses
               - Performance milestones
               - Service level agreements
               - Intellectual property rights
            
            3. Legal Compliance:
               - Regulatory compliance requirements
               - Data protection and privacy
               - Export control restrictions
               - Anti-corruption provisions
               - Employment law compliance
            
            4. Risk Mitigation Recommendations:
               - Suggested contract modifications
               - Alternative clause language
               - Negotiation strategies
               - Deal structure optimization
               - Insurance requirements
            
            Provide detailed risk ratings (High/Medium/Low) with specific justifications.
            Flag any showstopper issues that require immediate attention.
            """,
            server_names=["filesystem", "fetch"],
        )
        
        # Regulatory Compliance Monitor
        compliance_monitor_agent = Agent(
            name="regulatory_compliance_monitor",
            instruction=f"""You are a compliance officer expert in regulatory monitoring and reporting.
            
            Monitor and assess regulatory compliance for {COMPANY_NAME} in {JURISDICTION}:
            
            1. Data Privacy Regulations:
               - GDPR compliance (EU operations)
               - CCPA/CPRA compliance (California)
               - PIPEDA compliance (Canada)
               - Sector-specific privacy laws
               - Cross-border data transfer requirements
            
            2. Securities and Financial Regulations:
               - SOX compliance (if publicly traded)
               - SEC reporting requirements
               - Financial industry regulations
               - Anti-money laundering (AML)
               - Know Your Customer (KYC)
            
            3. Employment and Labor Laws:
               - Equal employment opportunity
               - Wage and hour compliance
               - Workplace safety (OSHA)
               - Benefits administration
               - Workers' compensation
            
            4. Industry-Specific Regulations:
               - Healthcare (HIPAA, FDA)
               - Financial services (FINRA, OCC)
               - Technology (FCC, NIST)
               - Environmental (EPA)
               - Export controls (ITAR, EAR)
            
            5. Compliance Monitoring System:
               - Regular compliance audits
               - Policy updates and training
               - Incident response procedures
               - Regulatory change tracking
               - Reporting and documentation
            
            Create compliance calendars with key deadlines and requirements.
            """,
            server_names=["filesystem", "g-search", "fetch"],
        )
        
        # Legal Document Generator
        document_generator_agent = Agent(
            name="legal_document_generator",
            instruction=f"""You are a corporate legal counsel specializing in legal document drafting.
            
            Generate comprehensive legal documents for {COMPANY_NAME}:
            
            1. Corporate Governance Documents:
               - Board resolutions and minutes
               - Corporate bylaws amendments
               - Shareholder agreements
               - Stock option plans
               - Director and officer insurance
            
            2. Commercial Agreements:
               - Master service agreements
               - Software licensing agreements
               - Non-disclosure agreements
               - Employment contracts
               - Vendor agreements
            
            3. Intellectual Property Documents:
               - Patent applications and assignments
               - Trademark registrations
               - Copyright assignments
               - Technology transfer agreements
               - IP licensing agreements
            
            4. Regulatory Filings:
               - SEC forms and disclosures
               - State corporate filings
               - Regulatory compliance reports
               - Privacy policy updates
               - Terms of service agreements
            
            5. Litigation Support Documents:
               - Demand letters
               - Settlement agreements
               - Discovery requests
               - Protective orders
               - Expert witness agreements
            
            Ensure all documents comply with applicable laws and include:
            - Proper legal language and structure
            - Risk mitigation provisions
               - Dispute resolution clauses
            - Governing law and jurisdiction
            - Amendment and modification procedures
            """,
            server_names=["filesystem"],
        )
        
        # Data Privacy Officer
        privacy_officer_agent = Agent(
            name="data_privacy_officer",
            instruction=f"""You are a certified data protection officer (DPO) specializing in privacy compliance.
            
            Manage data privacy and protection for {COMPANY_NAME}:
            
            1. Privacy Impact Assessments:
               - Data processing activity mapping
               - Risk assessment and mitigation
               - Consent management systems
               - Data retention policies
               - Cross-border transfer mechanisms
            
            2. Privacy Policy Development:
               - Privacy notice requirements
               - Cookie and tracking policies
               - Data subject rights procedures
               - Breach notification processes
               - Vendor data processing agreements
            
            3. Compliance Program Management:
               - Privacy training programs
               - Regular compliance audits
               - Privacy by design implementation
               - Data minimization strategies
               - Record keeping requirements
            
            4. Incident Response Management:
               - Data breach response plans
               - Notification timelines and procedures
               - Regulatory reporting requirements
               - Communication strategies
               - Remediation and improvement plans
            
            5. International Privacy Compliance:
               - Multi-jurisdictional requirements
               - Data localization requirements
               - Standard contractual clauses
               - Adequacy decision monitoring
               - Binding corporate rules
            
            Focus on practical implementation and business risk mitigation.
            """,
            server_names=["filesystem", "fetch"],
        )
        
        # IP and Technology Lawyer
        ip_lawyer_agent = Agent(
            name="ip_technology_lawyer",
            instruction=f"""You are an intellectual property attorney specializing in technology law.
            
            Manage IP and technology legal matters for {COMPANY_NAME}:
            
            1. Patent Strategy and Management:
               - Patent landscape analysis
               - Patent application strategy
               - Patent portfolio management
               - Freedom to operate analysis
               - Patent litigation support
            
            2. Trademark and Brand Protection:
               - Trademark clearance searches
               - Registration and maintenance
               - Brand enforcement strategies
               - Domain name management
               - Anti-counterfeiting measures
            
            3. Technology Transactions:
               - Software licensing negotiations
               - Technology transfer agreements
               - Open source compliance
               - SaaS and cloud agreements
               - API licensing terms
            
            4. Employment IP Issues:
               - Invention assignment agreements
               - Non-compete and non-solicitation
               - Trade secret protection
               - Employee IP training
               - Contractor IP agreements
            
            5. IP Litigation and Enforcement:
               - Infringement analysis
               - Cease and desist letters
               - DMCA takedown procedures
               - Patent dispute resolution
               - Trade secret litigation
            
            Provide strategic IP advice aligned with business objectives.
            """,
            server_names=["filesystem", "g-search"],
        )
        
        # Employment Law Specialist
        employment_lawyer_agent = Agent(
            name="employment_law_specialist",
            instruction=f"""You are an employment law attorney specializing in workplace compliance.
            
            Handle employment law matters for {COMPANY_NAME}:
            
            1. Employment Policies and Procedures:
               - Employee handbook development
               - Anti-harassment and discrimination policies
               - Remote work policies
               - Social media and technology use
               - Performance management procedures
            
            2. Wage and Hour Compliance:
               - FLSA classification analysis
               - Overtime and break requirements
               - State-specific wage laws
               - Independent contractor classification
               - Payroll and timekeeping compliance
            
            3. Workplace Safety and Health:
               - OSHA compliance programs
               - Workplace injury procedures
               - Return-to-work policies
               - Safety training requirements
               - Environmental health programs
            
            4. Employee Relations and Disputes:
               - Disciplinary action procedures
               - Investigation protocols
               - Grievance and complaint handling
               - Accommodation procedures (ADA)
               - Union relations (if applicable)
            
            5. Termination and Separation:
               - Termination procedures and documentation
               - Severance agreement negotiations
               - COBRA and benefits continuation
               - Non-compete enforcement
               - Post-employment obligations
            
            Focus on risk prevention and compliant HR practices.
            """,
            server_names=["filesystem"],
        )
        
        # Legal Quality Evaluator
        legal_evaluator = Agent(
            name="legal_quality_evaluator",
            instruction="""You are a senior partner at a top-tier law firm evaluating legal work quality.
            
            Evaluate legal deliverables based on:
            
            1. Legal Accuracy and Completeness (30%)
               - Correct legal analysis and citations
               - Comprehensive coverage of issues
               - Accurate risk assessment
               - Current law and regulations
            
            2. Professional Standards (25%)
               - Professional writing quality
               - Proper legal formatting
               - Ethical compliance
               - Client confidentiality
            
            3. Business Practicality (25%)
               - Actionable recommendations
               - Business risk mitigation
               - Cost-effective solutions
               - Implementation feasibility
            
            4. Risk Management (20%)
               - Liability exposure analysis
               - Compliance gap identification
               - Preventive measures
               - Documentation quality
            
            Provide EXCELLENT, GOOD, FAIR, or POOR ratings with specific improvement areas.
            Flag any critical legal issues requiring immediate attention.
            """,
        )
        
        # Create quality controller for contracts
        contract_quality_controller = EvaluatorOptimizerLLM(
            optimizer=contract_analyzer_agent,
            evaluator=legal_evaluator,
            llm_factory=GoogleAugmentedLLM,
            min_rating=QualityRating.GOOD,
        )
        
        # --- CREATE ORCHESTRATOR ---
        logger.info(f"Initializing legal compliance workflow for {COMPANY_NAME}")
        
        orchestrator = Orchestrator(
            llm_factory=GoogleAugmentedLLM,
            available_agents=[
                contract_quality_controller,
                compliance_monitor_agent,
                document_generator_agent,
                privacy_officer_agent,
                ip_lawyer_agent,
                employment_lawyer_agent,
            ],
            plan_type="full",
        )
        
        # Define comprehensive legal compliance task
        task = f"""Execute a comprehensive legal compliance assessment and documentation project for {COMPANY_NAME}:

        1. Use the contract_quality_controller to analyze and assess:
           - Current contract templates and agreements
           - Risk exposure and mitigation strategies
           - Commercial terms optimization
           - Legal compliance gaps
           
        2. Use the compliance_monitor_agent to create:
           - Comprehensive regulatory compliance matrix
           - Industry-specific compliance requirements
           - Compliance calendar with key deadlines
           - Monitoring and reporting procedures
           
        3. Use the document_generator_agent to develop:
           - Updated corporate governance documents
           - Standard commercial agreement templates
           - Regulatory filing templates
           - Legal policy and procedure documents
           
        4. Use the privacy_officer_agent to establish:
           - Data privacy compliance program
           - Privacy impact assessment procedures
           - Data breach response protocols
           - International privacy compliance framework
           
        5. Use the ip_lawyer_agent to create:
           - IP strategy and management plan
           - Technology transaction guidelines
           - Patent and trademark protection procedures
           - IP enforcement and litigation protocols
           
        6. Use the employment_lawyer_agent to develop:
           - Employment policy manual
           - Wage and hour compliance procedures
           - Workplace safety and health programs
           - Employee relations and dispute resolution
           
        Save all deliverables in the {OUTPUT_DIR} directory:
        - contract_analysis_report_{timestamp}.md
        - regulatory_compliance_matrix_{timestamp}.md
        - legal_document_templates_{timestamp}/
        - privacy_compliance_program_{timestamp}.md
        - ip_management_plan_{timestamp}.md
        - employment_law_manual_{timestamp}.md
        - legal_compliance_summary_{timestamp}.md
        
        Create a comprehensive legal compliance dashboard showing:
        - Current compliance status
        - Risk assessment summary
        - Action items and priorities
        - Compliance calendar and deadlines
        """
        
        # Execute the workflow
        logger.info("Starting legal compliance workflow")
        try:
            result = await orchestrator.generate_str(
                message=task,
                request_params=RequestParams(model="gemini-2.5-flash-lite")
            )
            
            logger.info("Legal compliance workflow completed successfully")
            logger.info(f"All deliverables saved in {OUTPUT_DIR}/")
            
            # Generate compliance summary report
            summary_path = os.path.join(OUTPUT_DIR, f"legal_compliance_dashboard_{timestamp}.md")
            with open(summary_path, 'w') as f:
                f.write(f"""# Legal Compliance Dashboard - {COMPANY_NAME}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Executive Summary
Comprehensive legal compliance assessment and documentation completed.
All critical legal areas reviewed and documented with actionable recommendations.

### Deliverables Created
1. Contract Analysis Report
2. Regulatory Compliance Matrix
3. Legal Document Templates
4. Privacy Compliance Program
5. IP Management Plan
6. Employment Law Manual

### Next Steps
1. Review all documents with internal legal team
2. Implement recommended policy changes
3. Schedule quarterly compliance reviews
4. Update employee training programs
5. Monitor regulatory changes and updates

### Critical Action Items
- [ ] Review high-risk contract provisions
- [ ] Update privacy policies for new regulations
- [ ] Implement IP protection procedures
- [ ] Train managers on employment law compliance
- [ ] Schedule quarterly legal compliance audits

For detailed information, refer to individual reports in {OUTPUT_DIR}/
""")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during legal compliance workflow execution: {str(e)}")
            return False


if __name__ == "__main__":
    asyncio.run(main()) 