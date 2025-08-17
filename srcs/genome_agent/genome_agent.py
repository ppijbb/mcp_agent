#!/usr/bin/env python3
"""
Genome Information Agent with MCP Integration

This agent leverages multiple MCP servers for:
- Filesystem operations (save/load genome data)
- Web search and research for genomic information
- Browser automation for data collection
- Enhanced genome analysis and planning
- Persistent genome data management
"""

import os
import json
import httpx
import argparse
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from string import Template
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from core.agent.base import BaseAgent
from core.config.loader import load_config


class GenomeDataType(Enum):
    """Types of genome data for analysis"""
    DNA_SEQUENCE = "dna_sequence"
    GENE_EXPRESSION = "gene_expression"
    PROTEIN_SEQUENCE = "protein_sequence"
    VARIANT_DATA = "variant_data"
    PHENOTYPE_DATA = "phenotype_data"
    METABOLIC_PATHWAY = "metabolic_pathway"
    PHYLOGENETIC_DATA = "phylogenetic_data"
    EPIGENETIC_DATA = "epigenetic_data"


class AnalysisType(Enum):
    """Types of genome analysis"""
    SEQUENCE_ALIGNMENT = "sequence_alignment"
    VARIANT_CALLING = "variant_calling"
    GENE_ANNOTATION = "gene_annotation"
    PHYLOGENETIC_ANALYSIS = "phylogenetic_analysis"
    EXPRESSION_ANALYSIS = "expression_analysis"
    PATHWAY_ANALYSIS = "pathway_analysis"
    COMPARATIVE_GENOMICS = "comparative_genomics"
    STRUCTURAL_VARIATION = "structural_variation"


@dataclass
class GenomeData:
    """Genome data structure"""
    data_id: str
    data_type: GenomeDataType
    organism: str
    sequence: str
    metadata: Dict[str, Any]
    source: str
    timestamp: datetime
    quality_score: float = 0.0
    annotations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisRequest:
    """Genome analysis request"""
    request_id: str
    data_ids: List[str]
    analysis_type: AnalysisType
    parameters: Dict[str, Any]
    priority: str = "medium"
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisResult:
    """Genome analysis result"""
    result_id: str
    request_id: str
    analysis_type: AnalysisType
    results: Dict[str, Any]
    confidence_score: float
    processing_time: float
    timestamp: datetime
    recommendations: List[str] = field(default_factory=list)
    visualization_data: Optional[Dict[str, Any]] = None


class GenomeAgentMCP(BaseAgent):
    """
    Enhanced Genome Agent with MCP Integration
    
    This agent provides comprehensive genome analysis capabilities:
    - Data collection from various genomic databases
    - Advanced sequence analysis and annotation
    - Variant calling and interpretation
    - Phylogenetic analysis
    - Integration with external genomic tools
    """

    def __init__(self, 
                 output_dir: str = "genome_analysis_reports",
                 enable_mcp: bool = True,
                 mcp_servers: Optional[Dict[str, str]] = None):
        """
        Initialize the Genome Agent
        
        Args:
            output_dir: Directory for saving analysis reports
            enable_mcp: Whether to enable MCP server connections
            mcp_servers: Dictionary of MCP server configurations
        """
        super().__init__(
            name="GenomeAgentMCP",
            instruction="You are a specialized genome analysis agent with expertise in bioinformatics and genomic data processing."
        )
        
        self.output_dir = output_dir
        self.enable_mcp = enable_mcp
        self.mcp_servers = mcp_servers or {}
        self.mcp_session = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logger()
        
        # Genome data storage
        self.genome_data_store: Dict[str, GenomeData] = {}
        self.analysis_requests: Dict[str, AnalysisRequest] = {}
        self.analysis_results: Dict[str, AnalysisResult] = {}
        
        # Supported databases and tools
        self.supported_databases = [
            "NCBI", "Ensembl", "UCSC", "UniProt", "KEGG", "Reactome",
            "Gene Ontology", "STRING", "BioGRID", "ChEMBL"
        ]
        
        self.supported_tools = [
            "BLAST", "BWA", "GATK", "Samtools", "IGV", "R", "Python"
        ]

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the genome agent"""
        logger = logging.getLogger("GenomeAgentMCP")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    async def _initialize_mcp_connections(self):
        """Initialize connections to MCP servers"""
        if not self.enable_mcp:
            self.logger.info("MCP connections disabled")
            return
        
        try:
            # Create HTTP session for MCP communication
            self.mcp_session = httpx.AsyncClient(timeout=30.0)
            
            # Test connections to configured MCP servers
            for server_name, server_url in self.mcp_servers.items():
                try:
                    response = await self.mcp_session.get(f"{server_url}/health")
                    if response.status_code == 200:
                        self.logger.info(f"Connected to MCP server: {server_name}")
                    else:
                        self.logger.warning(f"Failed to connect to MCP server: {server_name}")
                except Exception as e:
                    self.logger.error(f"Error connecting to MCP server {server_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP connections: {e}")

    async def _connect_filesystem_server(self):
        """Connect to filesystem MCP server for data persistence"""
        if not self.mcp_servers.get("filesystem"):
            self.logger.warning("No filesystem MCP server configured")
            return False
        
        try:
            # Test filesystem server connection
            response = await self.mcp_session.get(
                f"{self.mcp_servers['filesystem']}/tools"
            )
            if response.status_code == 200:
                self.logger.info("Connected to filesystem MCP server")
                return True
            else:
                self.logger.warning("Failed to connect to filesystem MCP server")
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to filesystem server: {e}")
            return False

    async def _connect_search_server(self):
        """Connect to search MCP server for genomic data retrieval"""
        if not self.mcp_servers.get("search"):
            self.logger.warning("No search MCP server configured")
            return False
        
        try:
            # Test search server connection
            response = await self.mcp_session.get(
                f"{self.mcp_servers['search']}/tools"
            )
            if response.status_code == 200:
                self.logger.info("Connected to search MCP server")
                return True
            else:
                self.logger.warning("Failed to connect to search MCP server")
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to search server: {e}")
            return False

    async def _connect_browser_server(self):
        """Connect to browser MCP server for web scraping"""
        if not self.mcp_servers.get("browser"):
            self.logger.warning("No browser MCP server configured")
            return False
        
        try:
            # Test browser server connection
            response = await self.mcp_session.get(
                f"{self.mcp_servers['browser']}/tools"
            )
            if response.status_code == 200:
                self.logger.info("Connected to browser MCP server")
                return True
            else:
                self.logger.warning("Failed to connect to browser MCP server")
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to browser server: {e}")
            return False

    async def _call_mcp_tool(self, session: httpx.AsyncClient, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result"""
        try:
            # This is a simplified MCP tool call
            # In a real implementation, you would use the proper MCP protocol
            response = await session.post(
                f"{self.mcp_servers.get('filesystem', 'http://localhost:3000')}/tools/{tool_name}",
                json=arguments
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"MCP tool call failed: {response.status_code}")
                return {"error": f"Tool call failed with status {response.status_code}"}
                
        except Exception as e:
            self.logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    async def research_genome_context(self, query: str) -> Dict[str, Any]:
        """Research genomic context using available MCP servers"""
        research_data = {
            "query": query,
            "research_results": [],
            "data_sources": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Use search server if available
            if self.mcp_servers.get("search"):
                search_result = await self._call_mcp_tool(
                    self.mcp_session, "search", {"query": query, "max_results": 10}
                )
                if "error" not in search_result:
                    research_data["research_results"].extend(search_result.get("results", []))
                    research_data["data_sources"].append("search_server")
            
            # Use browser server if available
            if self.mcp_servers.get("browser"):
                browser_result = await self._call_mcp_tool(
                    self.mcp_session, "browse", {"url": f"https://www.ncbi.nlm.nih.gov/search/all/?term={query}"}
                )
                if "error" not in browser_result:
                    research_data["research_results"].extend(browser_result.get("content", []))
                    research_data["data_sources"].append("browser_server")
            
            # Add genomic database information
            research_data["genomic_databases"] = self.supported_databases
            research_data["analysis_tools"] = self.supported_tools
            
        except Exception as e:
            self.logger.error(f"Error during genome research: {e}")
            research_data["error"] = str(e)
        
        return research_data

    def create_enhanced_prompt(self, analysis_request: str, available_data: List[str], research_data: Optional[Dict[str, Any]] = None) -> str:
        """Create an enhanced prompt for genome analysis"""
        
        prompt_template = Template("""
You are a specialized genome analysis expert with access to the following resources:

AVAILABLE GENOMIC DATA:
${available_data}

ANALYSIS REQUEST:
${analysis_request}

RESEARCH CONTEXT:
${research_context}

SUPPORTED DATABASES:
${databases}

SUPPORTED TOOLS:
${tools}

Please provide a comprehensive genome analysis plan including:
1. Data preprocessing steps
2. Analysis pipeline design
3. Quality control measures
4. Expected outcomes
5. Potential challenges and solutions
6. Recommendations for further analysis

Focus on:
- Accuracy and reproducibility
- Best practices in bioinformatics
- Integration with existing genomic databases
- Scalability and performance optimization
- Compliance with genomic data standards

Provide specific technical details and command-line examples where applicable.
""")
        
        research_context = "No additional research context available"
        if research_data:
            research_context = json.dumps(research_data, indent=2, default=str)
        
        return prompt_template.substitute(
            available_data="\n".join([f"- {data}" for data in available_data]),
            analysis_request=analysis_request,
            research_context=research_context,
            databases="\n".join([f"- {db}" for db in self.supported_databases]),
            tools="\n".join([f"- {tool}" for tool in self.supported_tools])
        )

    async def generate_enhanced_analysis_plan(self, analysis_request: str, data_ids: Optional[List[str]] = None, enable_research: bool = True) -> Dict[str, Any]:
        """Generate an enhanced genome analysis plan"""
        
        try:
            # Get available data
            available_data = []
            if data_ids:
                available_data = [f"Data ID: {data_id}" for data_id in data_ids]
            else:
                available_data = list(self.genome_data_store.keys())
            
            # Research context if enabled
            research_data = None
            if enable_research:
                research_data = await self.research_genome_context(analysis_request)
            
            # Create enhanced prompt
            enhanced_prompt = self.create_enhanced_prompt(
                analysis_request, available_data, research_data
            )
            
            # Generate analysis plan using LLM
            # This would typically use the orchestrator's LLM
            analysis_plan = {
                "plan_id": f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "analysis_request": analysis_request,
                "available_data": available_data,
                "research_context": research_data,
                "analysis_steps": [
                    "Data validation and quality assessment",
                    "Preprocessing and normalization",
                    "Core analysis execution",
                    "Result validation and interpretation",
                    "Report generation and visualization"
                ],
                "estimated_duration": "2-4 hours",
                "resource_requirements": {
                    "computing_power": "High",
                    "memory": "16GB+",
                    "storage": "100GB+",
                    "software": self.supported_tools
                },
                "quality_metrics": [
                    "Sequence coverage > 30x",
                    "Mapping quality > 30",
                    "Variant quality score > 20",
                    "Statistical significance p < 0.05"
                ],
                "created_at": datetime.now().isoformat()
            }
            
            return analysis_plan
            
        except Exception as e:
            self.logger.error(f"Error generating analysis plan: {e}")
            return {"error": str(e)}

    async def save_genome_data(self, data: GenomeData, filename: Optional[str] = None) -> str:
        """Save genome data using MCP filesystem server"""
        try:
            if not filename:
                filename = f"genome_data_{data.data_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Convert to JSON-serializable format
            data_dict = {
                "data_id": data.data_id,
                "data_type": data.data_type.value,
                "organism": data.organism,
                "sequence": data.sequence,
                "metadata": data.metadata,
                "source": data.source,
                "timestamp": data.timestamp.isoformat(),
                "quality_score": data.quality_score,
                "annotations": data.annotations
            }
            
            # Save locally
            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=2)
            
            # Save to MCP filesystem server if available
            if self.mcp_servers.get("filesystem"):
                await self._call_mcp_tool(
                    self.mcp_session, "write_file",
                    {"path": filepath, "content": json.dumps(data_dict, indent=2)}
                )
            
            self.logger.info(f"Saved genome data to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving genome data: {e}")
            raise

    async def load_genome_data(self, filename: str) -> Optional[GenomeData]:
        """Load genome data from file"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            
            if not os.path.exists(filepath):
                self.logger.warning(f"File not found: {filepath}")
                return None
            
            with open(filepath, 'r') as f:
                data_dict = json.load(f)
            
            # Convert back to GenomeData object
            data = GenomeData(
                data_id=data_dict["data_id"],
                data_type=GenomeDataType(data_dict["data_type"]),
                organism=data_dict["organism"],
                sequence=data_dict["sequence"],
                metadata=data_dict["metadata"],
                source=data_dict["source"],
                timestamp=datetime.fromisoformat(data_dict["timestamp"]),
                quality_score=data_dict["quality_score"],
                annotations=data_dict.get("annotations", [])
            )
            
            self.logger.info(f"Loaded genome data from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading genome data: {e}")
            return None

    async def list_saved_data(self) -> List[Dict[str, Any]]:
        """List all saved genome data files"""
        try:
            files = []
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.output_dir, filename)
                    file_stat = os.stat(filepath)
                    files.append({
                        "filename": filename,
                        "filepath": filepath,
                        "size": file_stat.st_size,
                        "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
            
            return sorted(files, key=lambda x: x["modified"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error listing saved data: {e}")
            return []

    async def execute_analysis_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a genome analysis plan"""
        try:
            execution_result = {
                "plan_id": plan["plan_id"],
                "execution_start": datetime.now().isoformat(),
                "steps_completed": [],
                "results": {},
                "errors": [],
                "execution_time": 0.0
            }
            
            start_time = datetime.now()
            
            # Execute each analysis step
            for step in plan.get("analysis_steps", []):
                try:
                    self.logger.info(f"Executing step: {step}")
                    
                    # Simulate step execution
                    step_result = await self._execute_analysis_step(step, plan)
                    execution_result["steps_completed"].append(step)
                    execution_result["results"][step] = step_result
                    
                except Exception as e:
                    error_msg = f"Error in step '{step}': {str(e)}"
                    execution_result["errors"].append(error_msg)
                    self.logger.error(error_msg)
            
            # Calculate execution time
            execution_result["execution_time"] = (datetime.now() - start_time).total_seconds()
            execution_result["execution_end"] = datetime.now().isoformat()
            
            # Save execution result
            result_filename = f"execution_result_{plan['plan_id']}.json"
            result_filepath = os.path.join(self.output_dir, result_filename)
            
            with open(result_filepath, 'w') as f:
                json.dump(execution_result, f, indent=2, default=str)
            
            self.logger.info(f"Analysis plan execution completed. Results saved to {result_filepath}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Error executing analysis plan: {e}")
            return {"error": str(e)}

    async def _execute_analysis_step(self, step: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single analysis step"""
        # This is a simplified implementation
        # In a real system, this would integrate with actual bioinformatics tools
        
        step_results = {
            "step": step,
            "status": "completed",
            "output": f"Simulated output for {step}",
            "metrics": {
                "processing_time": 30.0,
                "memory_usage": "2GB",
                "cpu_usage": "80%"
            }
        }
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        return step_results

    def pretty_print_analysis_plan(self, plan: Dict[str, Any]):
        """Pretty print an analysis plan"""
        print("\n" + "="*80)
        print("🧬 GENOME ANALYSIS PLAN")
        print("="*80)
        
        print(f"📋 Plan ID: {plan.get('plan_id', 'N/A')}")
        print(f"🎯 Analysis Request: {plan.get('analysis_request', 'N/A')}")
        print(f"📅 Created: {plan.get('created_at', 'N/A')}")
        print(f"⏱️  Estimated Duration: {plan.get('estimated_duration', 'N/A')}")
        
        print("\n📊 AVAILABLE DATA:")
        for data in plan.get('available_data', []):
            print(f"   • {data}")
        
        print("\n🔬 ANALYSIS STEPS:")
        for i, step in enumerate(plan.get('analysis_steps', []), 1):
            print(f"   {i}. {step}")
        
        print("\n💻 RESOURCE REQUIREMENTS:")
        resources = plan.get('resource_requirements', {})
        for resource, value in resources.items():
            print(f"   • {resource.replace('_', ' ').title()}: {value}")
        
        print("\n📈 QUALITY METRICS:")
        for metric in plan.get('quality_metrics', []):
            print(f"   • {metric}")
        
        print("="*80 + "\n")

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.mcp_session:
                await self.mcp_session.aclose()
            self.logger.info("Genome agent cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def run_workflow(self, *args, **kwargs) -> Any:
        """Main workflow for the genome agent"""
        try:
            # Initialize MCP connections
            await self._initialize_mcp_connections()
            
            # Example workflow
            analysis_request = kwargs.get('analysis_request', 'Analyze human genome variants')
            data_ids = kwargs.get('data_ids', [])
            enable_research = kwargs.get('enable_research', True)
            
            # Generate analysis plan
            plan = await self.generate_enhanced_analysis_plan(
                analysis_request, data_ids, enable_research
            )
            
            if "error" in plan:
                return plan
            
            # Display plan
            self.pretty_print_analysis_plan(plan)
            
            # Execute plan if requested
            if kwargs.get('execute_plan', False):
                result = await self.execute_analysis_plan(plan)
                return {
                    "plan": plan,
                    "execution_result": result
                }
            
            return {"plan": plan}
            
        except Exception as e:
            self.logger.error(f"Error in genome agent workflow: {e}")
            return {"error": str(e)}


async def create_genome_agent(output_dir: str = "genome_analysis_reports") -> GenomeAgentMCP:
    """Create a genome agent instance"""
    return GenomeAgentMCP(output_dir=output_dir)


async def run_genome_analysis(
    analysis_request: str,
    data_ids: Optional[List[str]] = None,
    enable_research: bool = True,
    execute_plan: bool = False,
    output_dir: str = "genome_analysis_reports"
) -> Dict[str, Any]:
    """Run a genome analysis workflow"""
    
    agent = await create_genome_agent(output_dir)
    
    try:
        result = await agent.run_workflow(
            analysis_request=analysis_request,
            data_ids=data_ids,
            enable_research=enable_research,
            execute_plan=execute_plan
        )
        return result
    finally:
        await agent.cleanup()


async def main():
    """Main function for testing the genome agent"""
    
    # Example usage
    analysis_request = "Analyze genetic variants in BRCA1 and BRCA2 genes for breast cancer risk assessment"
    
    result = await run_genome_analysis(
        analysis_request=analysis_request,
        enable_research=True,
        execute_plan=False
    )
    
    print("Genome Analysis Result:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
