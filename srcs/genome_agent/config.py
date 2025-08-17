#!/usr/bin/env python3
"""
Configuration for the Genome Agent

This module provides configuration management for genome analysis workflows,
MCP server connections, and genomic database settings.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class GenomeDatabaseType(Enum):
    """Types of genomic databases"""
    SEQUENCE_DATABASE = "sequence_database"
    ANNOTATION_DATABASE = "annotation_database"
    VARIANT_DATABASE = "variant_database"
    EXPRESSION_DATABASE = "expression_database"
    PATHWAY_DATABASE = "pathway_database"
    PHYLOGENETIC_DATABASE = "phylogenetic_database"


class AnalysisToolType(Enum):
    """Types of analysis tools"""
    ALIGNMENT_TOOL = "alignment_tool"
    VARIANT_CALLING_TOOL = "variant_calling_tool"
    ANNOTATION_TOOL = "annotation_tool"
    VISUALIZATION_TOOL = "visualization_tool"
    STATISTICAL_TOOL = "statistical_tool"


@dataclass
class GenomeDatabaseConfig:
    """Configuration for a genomic database"""
    name: str
    type: GenomeDatabaseType
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    retry_count: int = 3
    is_active: bool = True
    description: str = ""
    
    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            env_key = f"{self.name.upper()}_API_KEY"
            self.api_key = os.getenv(env_key)


@dataclass
class AnalysisToolConfig:
    """Configuration for an analysis tool"""
    name: str
    type: AnalysisToolType
    command: str
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_installed: bool = False
    installation_path: Optional[str] = None
    description: str = ""


@dataclass
class MCPServerConfig:
    """Configuration for MCP servers"""
    name: str
    url: str
    port: int
    health_endpoint: str = "/health"
    tools_endpoint: str = "/tools"
    timeout: int = 30
    is_active: bool = True


@dataclass
class GenomeAnalysisConfig:
    """Configuration for genome analysis workflows"""
    default_output_dir: str = "genome_analysis_reports"
    max_concurrent_analyses: int = 5
    default_quality_threshold: float = 0.8
    max_sequence_length: int = 1000000  # 1MB
    supported_organisms: List[str] = field(default_factory=lambda: [
        "Homo sapiens", "Mus musculus", "Drosophila melanogaster",
        "Saccharomyces cerevisiae", "Escherichia coli"
    ])
    default_analysis_parameters: Dict[str, Any] = field(default_factory=lambda: {
        "coverage_threshold": 30,
        "mapping_quality": 30,
        "variant_quality": 20,
        "significance_threshold": 0.05
    })


@dataclass
class GenomeAgentConfig:
    """Main configuration for the Genome Agent"""
    
    # Basic settings
    agent_name: str = "GenomeAgentMCP"
    version: str = "1.0.0"
    environment: str = "development"
    
    # Analysis configuration
    analysis: GenomeAnalysisConfig = field(default_factory=GenomeAnalysisConfig)
    
    # Database configurations
    databases: Dict[str, GenomeDatabaseConfig] = field(default_factory=dict)
    
    # Tool configurations
    tools: Dict[str, AnalysisToolConfig] = field(default_factory=dict)
    
    # MCP server configurations
    mcp_servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_console_logging: bool = True
    
    def __post_init__(self):
        self._setup_default_databases()
        self._setup_default_tools()
        self._setup_default_mcp_servers()
    
    def _setup_default_databases(self):
        """Setup default genomic database configurations"""
        if not self.databases:
            self.databases = {
                "ncbi": GenomeDatabaseConfig(
                    name="NCBI",
                    type=GenomeDatabaseType.SEQUENCE_DATABASE,
                    base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
                    description="National Center for Biotechnology Information"
                ),
                "ensembl": GenomeDatabaseConfig(
                    name="Ensembl",
                    type=GenomeDatabaseType.ANNOTATION_DATABASE,
                    base_url="https://rest.ensembl.org",
                    description="Ensembl genome browser and annotation"
                ),
                "ucsc": GenomeDatabaseConfig(
                    name="UCSC",
                    type=GenomeDatabaseType.ANNOTATION_DATABASE,
                    base_url="https://genome.ucsc.edu/cgi-bin",
                    description="UCSC Genome Browser"
                ),
                "uniprot": GenomeDatabaseConfig(
                    name="UniProt",
                    type=GenomeDatabaseType.ANNOTATION_DATABASE,
                    base_url="https://rest.uniprot.org",
                    description="Universal Protein Resource"
                ),
                "kegg": GenomeDatabaseConfig(
                    name="KEGG",
                    type=GenomeDatabaseType.PATHWAY_DATABASE,
                    base_url="https://rest.kegg.org",
                    description="Kyoto Encyclopedia of Genes and Genomes"
                ),
                "reactome": GenomeDatabaseConfig(
                    name="Reactome",
                    type=GenomeDatabaseType.PATHWAY_DATABASE,
                    base_url="https://reactome.org/ContentService",
                    description="Reactome pathway database"
                ),
                "gene_ontology": GenomeDatabaseConfig(
                    name="Gene Ontology",
                    type=GenomeDatabaseType.ANNOTATION_DATABASE,
                    base_url="http://api.geneontology.org",
                    description="Gene Ontology Consortium"
                ),
                "string": GenomeDatabaseConfig(
                    name="STRING",
                    type=GenomeDatabaseType.ANNOTATION_DATABASE,
                    base_url="https://string-db.org/api",
                    description="STRING protein interaction database"
                ),
                "biogrid": GenomeDatabaseConfig(
                    name="BioGRID",
                    type=GenomeDatabaseType.ANNOTATION_DATABASE,
                    base_url="https://webservice.thebiogrid.org",
                    description="Biological General Repository for Interaction Datasets"
                ),
                "chembl": GenomeDatabaseConfig(
                    name="ChEMBL",
                    type=GenomeDatabaseType.ANNOTATION_DATABASE,
                    base_url="https://www.ebi.ac.uk/chembl/api/data",
                    description="ChEMBL bioactivity database"
                )
            }
    
    def _setup_default_tools(self):
        """Setup default analysis tool configurations"""
        if not self.tools:
            self.tools = {
                "blast": AnalysisToolConfig(
                    name="BLAST",
                    type=AnalysisToolType.ALIGNMENT_TOOL,
                    command="blastn",
                    version="2.13.0+",
                    description="Basic Local Alignment Search Tool"
                ),
                "bwa": AnalysisToolConfig(
                    name="BWA",
                    type=AnalysisToolType.ALIGNMENT_TOOL,
                    command="bwa",
                    version="0.7.17",
                    description="Burrows-Wheeler Aligner"
                ),
                "gatk": AnalysisToolConfig(
                    name="GATK",
                    type=AnalysisToolType.VARIANT_CALLING_TOOL,
                    command="gatk",
                    version="4.2.0.0",
                    description="Genome Analysis Toolkit"
                ),
                "samtools": AnalysisToolConfig(
                    name="Samtools",
                    type=AnalysisToolType.ANNOTATION_TOOL,
                    command="samtools",
                    version="1.15",
                    description="SAM/FASTQ processing utilities"
                ),
                "igv": AnalysisToolConfig(
                    name="IGV",
                    type=AnalysisToolType.VISUALIZATION_TOOL,
                    command="igv",
                    version="2.12.0",
                    description="Integrative Genomics Viewer"
                ),
                "r": AnalysisToolConfig(
                    name="R",
                    type=AnalysisToolType.STATISTICAL_TOOL,
                    command="Rscript",
                    version="4.2.0",
                    description="R Statistical Computing"
                ),
                "python": AnalysisToolConfig(
                    name="Python",
                    type=AnalysisToolType.STATISTICAL_TOOL,
                    command="python3",
                    version="3.8+",
                    description="Python Programming Language"
                )
            }
    
    def _setup_default_mcp_servers(self):
        """Setup default MCP server configurations"""
        if not self.mcp_servers:
            self.mcp_servers = {
                "filesystem": MCPServerConfig(
                    name="filesystem",
                    url="http://localhost",
                    port=3000,
                    description="Filesystem MCP server for data persistence"
                ),
                "search": MCPServerConfig(
                    name="search",
                    url="http://localhost",
                    port=3001,
                    description="Search MCP server for genomic data retrieval"
                ),
                "browser": MCPServerConfig(
                    name="browser",
                    url="http://localhost",
                    port=3002,
                    description="Browser MCP server for web scraping"
                )
            }
    
    def get_database_config(self, name: str) -> Optional[GenomeDatabaseConfig]:
        """Get database configuration by name"""
        return self.databases.get(name.lower())
    
    def get_tool_config(self, name: str) -> Optional[AnalysisToolConfig]:
        """Get tool configuration by name"""
        return self.tools.get(name.lower())
    
    def get_mcp_server_config(self, name: str) -> Optional[MCPServerConfig]:
        """Get MCP server configuration by name"""
        return self.mcp_servers.get(name.lower())
    
    def get_active_databases(self) -> List[GenomeDatabaseConfig]:
        """Get list of active databases"""
        return [db for db in self.databases.values() if db.is_active]
    
    def get_active_tools(self) -> List[AnalysisToolConfig]:
        """Get list of active tools"""
        return [tool for tool in self.tools.values() if tool.is_active]
    
    def get_active_mcp_servers(self) -> List[MCPServerConfig]:
        """Get list of active MCP servers"""
        return [server for server in self.mcp_servers.values() if server.is_active]
    
    def update_database_config(self, name: str, **kwargs):
        """Update database configuration"""
        if name.lower() in self.databases:
            for key, value in kwargs.items():
                if hasattr(self.databases[name.lower()], key):
                    setattr(self.databases[name.lower()], key, value)
    
    def update_tool_config(self, name: str, **kwargs):
        """Update tool configuration"""
        if name.lower() in self.tools:
            for key, value in kwargs.items():
                if hasattr(self.tools[name.lower()], key):
                    setattr(self.tools[name.lower()], key, value)
    
    def update_mcp_server_config(self, name: str, **kwargs):
        """Update MCP server configuration"""
        if name.lower() in self.mcp_servers:
            for key, value in kwargs.items():
                if hasattr(self.mcp_servers[name.lower()], key):
                    setattr(self.mcp_servers[name.lower()], key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "agent_name": self.agent_name,
            "version": self.version,
            "environment": self.environment,
            "analysis": {
                "default_output_dir": self.analysis.default_output_dir,
                "max_concurrent_analyses": self.analysis.max_concurrent_analyses,
                "default_quality_threshold": self.analysis.default_quality_threshold,
                "max_sequence_length": self.analysis.max_sequence_length,
                "supported_organisms": self.analysis.supported_organisms,
                "default_analysis_parameters": self.analysis.default_analysis_parameters
            },
            "databases": {
                name: {
                    "name": db.name,
                    "type": db.type.value,
                    "base_url": db.base_url,
                    "rate_limit": db.rate_limit,
                    "timeout": db.timeout,
                    "retry_count": db.retry_count,
                    "is_active": db.is_active,
                    "description": db.description
                }
                for name, db in self.databases.items()
            },
            "tools": {
                name: {
                    "name": tool.name,
                    "type": tool.type.value,
                    "command": tool.command,
                    "version": tool.version,
                    "parameters": tool.parameters,
                    "is_installed": tool.is_installed,
                    "installation_path": tool.installation_path,
                    "description": tool.description
                }
                for name, tool in self.tools.items()
            },
            "mcp_servers": {
                name: {
                    "name": server.name,
                    "url": server.url,
                    "port": server.port,
                    "health_endpoint": server.health_endpoint,
                    "tools_endpoint": server.tools_endpoint,
                    "timeout": server.timeout,
                    "is_active": server.is_active
                }
                for name, server in self.mcp_servers.items()
            },
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_console_logging": self.enable_console_logging
        }


def get_default_config() -> GenomeAgentConfig:
    """Get default genome agent configuration"""
    return GenomeAgentConfig()


def load_config_from_file(config_path: str) -> GenomeAgentConfig:
    """Load configuration from file"""
    import json
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create config object from file data
        config = GenomeAgentConfig()
        
        # Update with file data
        if "analysis" in config_data:
            for key, value in config_data["analysis"].items():
                if hasattr(config.analysis, key):
                    setattr(config.analysis, key, value)
        
        if "databases" in config_data:
            for name, db_data in config_data["databases"].items():
                if name in config.databases:
                    for key, value in db_data.items():
                        if hasattr(config.databases[name], key):
                            setattr(config.databases[name], key, value)
        
        if "tools" in config_data:
            for name, tool_data in config_data["tools"].items():
                if name in config.tools:
                    for key, value in tool_data.items():
                        if hasattr(config.tools[name], key):
                            setattr(config.tools[name], key, value)
        
        if "mcp_servers" in config_data:
            for name, server_data in config_data["mcp_servers"].items():
                if name in config.mcp_servers:
                    for key, value in server_data.items():
                        if hasattr(config.mcp_servers[name], key):
                            setattr(config.mcp_servers[name], key, value)
        
        return config
        
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return get_default_config()


def save_config_to_file(config: GenomeAgentConfig, config_path: str):
    """Save configuration to file"""
    import json
    
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")


def get_env_config() -> GenomeAgentConfig:
    """Get configuration from environment variables"""
    config = get_default_config()
    
    # Override with environment variables
    if os.getenv("GENOME_AGENT_OUTPUT_DIR"):
        config.analysis.default_output_dir = os.getenv("GENOME_AGENT_OUTPUT_DIR")
    
    if os.getenv("GENOME_AGENT_LOG_LEVEL"):
        config.log_level = os.getenv("GENOME_AGENT_LOG_LEVEL")
    
    if os.getenv("GENOME_AGENT_LOG_FILE"):
        config.log_file = os.getenv("GENOME_AGENT_LOG_FILE")
    
    return config


def get_config(config_path: Optional[str] = None) -> GenomeAgentConfig:
    """Get configuration from file, environment, or defaults"""
    if config_path and os.path.exists(config_path):
        return load_config_from_file(config_path)
    
    return get_env_config()
