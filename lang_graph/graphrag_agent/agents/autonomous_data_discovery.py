"""
Autonomous Data Discovery and Analysis Module

This module implements true GraphRAG capabilities:
- Autonomous data discovery and validation
- Intelligent data analysis and understanding
- Dynamic schema inference
- Context-aware data processing
- Self-directed learning from data patterns
"""

import asyncio
import logging
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import hashlib

from config import AgentConfig
from .llm_processor import LLMProcessor


class DataSourceType(Enum):
    """Types of data sources"""
    CSV = "csv"
    JSON = "json"
    TXT = "txt"
    PDF = "pdf"
    URL = "url"
    DATABASE = "database"
    API = "api"
    UNKNOWN = "unknown"


class DataQualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class DataSource:
    """Represents a discovered data source"""
    source_id: str
    source_type: DataSourceType
    path: str
    size: int
    quality_level: DataQualityLevel
    schema: Dict[str, Any]
    metadata: Dict[str, Any]
    discovered_at: datetime
    last_analyzed: Optional[datetime] = None
    confidence: float = 0.0


@dataclass
class DataInsight:
    """Insight discovered from data analysis"""
    insight_id: str
    insight_type: str
    description: str
    confidence: float
    data_source: str
    evidence: List[str]
    implications: List[str]
    discovered_at: datetime


class AutonomousDataDiscovery:
    """
    Autonomous data discovery and analysis engine
    
    This engine embodies true GraphRAG principles:
    - Automatically discovers and validates data sources
    - Intelligently analyzes data structure and content
    - Infers schemas and relationships autonomously
    - Learns from data patterns and user behavior
    - Adapts analysis strategies based on domain
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_processor = LLMProcessor(config)
        
        # Discovery state
        self.discovered_sources = {}
        self.data_insights = []
        self.analysis_patterns = []
        self.domain_knowledge = {}
        
        # Learning and adaptation
        self.learning_enabled = True
        self.adaptation_threshold = 0.7
        self.quality_threshold = 0.6
        
        self.logger.info("Autonomous Data Discovery Engine initialized")
    
    async def discover_and_analyze_data(self, base_path: str = ".", user_intent: str = "") -> Dict[str, Any]:
        """
        Main entry point for autonomous data discovery and analysis
        
        This method implements the core GraphRAG principle of autonomous data understanding:
        1. Discovers all available data sources
        2. Analyzes data structure and content intelligently
        3. Infers schemas and relationships
        4. Generates insights and recommendations
        5. Learns from patterns for future improvements
        """
        try:
            self.logger.info(f"Starting autonomous data discovery in: {base_path}")
            
            # Step 1: Discover data sources
            sources = await self._discover_data_sources(base_path)
            self.logger.info(f"Discovered {len(sources)} data sources")
            
            # Step 2: Analyze each source autonomously
            analysis_results = []
            for source in sources:
                analysis = await self._analyze_data_source_autonomously(source, user_intent)
                analysis_results.append(analysis)
                self.discovered_sources[source.source_id] = source
            
            # Step 3: Cross-source analysis and relationship discovery
            cross_analysis = await self._perform_cross_source_analysis(analysis_results, user_intent)
            
            # Step 4: Generate comprehensive insights
            insights = await self._generate_data_insights(analysis_results, cross_analysis, user_intent)
            
            # Step 5: Learn from analysis patterns
            await self._learn_from_analysis(analysis_results, insights)
            
            # Step 6: Generate recommendations
            recommendations = await self._generate_recommendations(analysis_results, insights, user_intent)
            
            return {
                "status": "success",
                "discovered_sources": len(sources),
                "analysis_results": analysis_results,
                "cross_analysis": cross_analysis,
                "insights": insights,
                "recommendations": recommendations,
                "data_quality_summary": self._calculate_quality_summary(analysis_results),
                "discovery_confidence": self._calculate_discovery_confidence(analysis_results)
            }
            
        except Exception as e:
            self.logger.error(f"Autonomous data discovery failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "discovered_sources": 0
            }
    
    async def _discover_data_sources(self, base_path: str) -> List[DataSource]:
        """Autonomously discover data sources in the given path"""
        sources = []
        base_path = Path(base_path)
        
        if not base_path.exists():
            self.logger.warning(f"Base path does not exist: {base_path}")
            return sources
        
        # Discover files recursively
        for file_path in base_path.rglob("*"):
            if file_path.is_file():
                source_type = self._infer_source_type(file_path)
                if source_type != DataSourceType.UNKNOWN:
                    source = await self._create_data_source(file_path, source_type)
                    if source:
                        sources.append(source)
        
        # Sort by quality and relevance
        sources.sort(key=lambda x: (x.quality_level.value, x.confidence), reverse=True)
        return sources
    
    def _infer_source_type(self, file_path: Path) -> DataSourceType:
        """Infer data source type from file extension and content"""
        extension = file_path.suffix.lower()
        
        type_mapping = {
            '.csv': DataSourceType.CSV,
            '.json': DataSourceType.JSON,
            '.txt': DataSourceType.TXT,
            '.pdf': DataSourceType.PDF,
            '.xlsx': DataSourceType.CSV,
            '.xls': DataSourceType.CSV,
            '.tsv': DataSourceType.CSV
        }
        
        return type_mapping.get(extension, DataSourceType.UNKNOWN)
    
    async def _create_data_source(self, file_path: Path, source_type: DataSourceType) -> Optional[DataSource]:
        """Create a DataSource object with initial analysis"""
        try:
            # Calculate file size
            size = file_path.stat().st_size
            
            # Generate source ID
            source_id = hashlib.md5(str(file_path).encode()).hexdigest()[:12]
            
            # Initial quality assessment
            quality_level = await self._assess_initial_quality(file_path, source_type)
            
            # Basic schema inference
            schema = await self._infer_basic_schema(file_path, source_type)
            
            # Generate metadata
            metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_size": size,
                "created_time": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "source_type": source_type.value
            }
            
            return DataSource(
                source_id=source_id,
                source_type=source_type,
                path=str(file_path),
                size=size,
                quality_level=quality_level,
                schema=schema,
                metadata=metadata,
                discovered_at=datetime.now(),
                confidence=0.5  # Initial confidence
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create data source for {file_path}: {e}")
            return None
    
    async def _assess_initial_quality(self, file_path: Path, source_type: DataSourceType) -> DataQualityLevel:
        """Assess initial data quality based on file characteristics"""
        try:
            size = file_path.stat().st_size
            
            # Size-based assessment
            if size == 0:
                return DataQualityLevel.UNUSABLE
            elif size < 100:  # Less than 100 bytes
                return DataQualityLevel.POOR
            elif size < 1000:  # Less than 1KB
                return DataQualityLevel.FAIR
            elif size < 100000:  # Less than 100KB
                return DataQualityLevel.GOOD
            else:
                return DataQualityLevel.EXCELLENT
                
        except Exception:
            return DataQualityLevel.POOR
    
    async def _infer_basic_schema(self, file_path: Path, source_type: DataSourceType) -> Dict[str, Any]:
        """Infer basic schema from data source"""
        schema = {
            "columns": [],
            "data_types": {},
            "sample_data": [],
            "row_count": 0,
            "has_header": False
        }
        
        try:
            if source_type == DataSourceType.CSV:
                # Read first few rows to infer schema
                df = pd.read_csv(file_path, nrows=5)
                schema["columns"] = df.columns.tolist()
                schema["data_types"] = df.dtypes.astype(str).to_dict()
                schema["sample_data"] = df.head(3).to_dict('records')
                schema["row_count"] = len(pd.read_csv(file_path, nrows=1000))  # Estimate
                schema["has_header"] = True
                
            elif source_type == DataSourceType.JSON:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        schema["columns"] = list(data[0].keys()) if isinstance(data[0], dict) else []
                        schema["sample_data"] = data[:3]
                        schema["row_count"] = len(data)
                    elif isinstance(data, dict):
                        schema["columns"] = list(data.keys())
                        schema["sample_data"] = [data]
                        schema["row_count"] = 1
                        
        except Exception as e:
            self.logger.warning(f"Failed to infer schema for {file_path}: {e}")
        
        return schema
    
    async def _analyze_data_source_autonomously(self, source: DataSource, user_intent: str = "") -> Dict[str, Any]:
        """Perform autonomous analysis of a data source using LLM"""
        try:
            self.logger.info(f"Analyzing data source: {source.source_id}")
            
            # Load and prepare data for analysis
            data_content = await self._load_data_content(source)
            if not data_content:
                return {"status": "error", "error": "Failed to load data content"}
            
            # Use LLM to analyze data structure and content
            analysis_prompt = self._build_data_analysis_prompt(data_content, source, user_intent)
            llm_response = self.llm_processor._call_llm(analysis_prompt)
            analysis_result = json.loads(llm_response)
            
            # Enhance analysis with statistical insights
            statistical_insights = await self._generate_statistical_insights(source, data_content)
            
            # Combine LLM and statistical analysis
            enhanced_analysis = {
                "source_id": source.source_id,
                "llm_analysis": analysis_result,
                "statistical_insights": statistical_insights,
                "data_quality": self._assess_data_quality(analysis_result, statistical_insights),
                "recommendations": self._generate_source_recommendations(analysis_result, statistical_insights),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Update source with analysis results
            source.last_analyzed = datetime.now()
            source.confidence = analysis_result.get("confidence", 0.5)
            
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze data source {source.source_id}: {e}")
            return {
                "source_id": source.source_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _load_data_content(self, source: DataSource) -> Optional[str]:
        """Load data content for analysis"""
        try:
            if source.source_type == DataSourceType.CSV:
                # Load sample of CSV data
                df = pd.read_csv(source.path, nrows=100)  # First 100 rows
                return df.to_string()
            elif source.source_type == DataSourceType.JSON:
                with open(source.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)[:5000]  # First 5000 chars
            elif source.source_type == DataSourceType.TXT:
                with open(source.path, 'r', encoding='utf-8') as f:
                    return f.read()[:5000]  # First 5000 chars
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to load data content: {e}")
            return None
    
    def _build_data_analysis_prompt(self, data_content: str, source: DataSource, user_intent: str = "") -> str:
        """Build LLM prompt for autonomous data analysis"""
        prompt = f"""
You are an expert data analyst with GraphRAG capabilities. Analyze the following data source autonomously and provide comprehensive insights.

Data Source Information:
- Type: {source.source_type.value}
- Size: {source.size} bytes
- Schema: {json.dumps(source.schema, indent=2)}

Data Content (sample):
{data_content}

User Intent: {user_intent if user_intent else "General data analysis"}

Provide a comprehensive analysis in JSON format:
{{
    "data_understanding": {{
        "primary_purpose": "What this data represents",
        "key_entities": ["main entities in the data"],
        "data_patterns": ["patterns observed in the data"],
        "domain_context": "domain or field this data belongs to"
    }},
    "structure_analysis": {{
        "data_organization": "how the data is organized",
        "key_columns": ["most important columns"],
        "relationships": ["potential relationships between columns"],
        "data_quality_issues": ["any quality issues observed"]
    }},
    "graph_potential": {{
        "entity_candidates": ["entities that could be graph nodes"],
        "relationship_candidates": ["potential relationships between entities"],
        "graph_structure_suggestions": ["suggested graph structure"],
        "visualization_recommendations": ["how to visualize this data"]
    }},
    "insights": {{
        "key_findings": ["main insights from the data"],
        "anomalies": ["unusual patterns or outliers"],
        "trends": ["trends observed in the data"],
        "opportunities": ["opportunities for further analysis"]
    }},
    "recommendations": {{
        "preprocessing_needed": ["data preprocessing steps needed"],
        "analysis_priorities": ["what to analyze first"],
        "graph_generation_strategy": ["how to generate a knowledge graph"],
        "next_steps": ["recommended next steps"]
    }},
    "confidence": 0.0-1.0,
    "complexity_level": "simple|medium|complex"
}}

Consider:
- The data's potential for knowledge graph generation
- Relationships and connections between entities
- Data quality and completeness
- Domain-specific patterns and insights
- User intent and requirements
- GraphRAG best practices
"""
        return prompt
    
    async def _generate_statistical_insights(self, source: DataSource, data_content: str) -> Dict[str, Any]:
        """Generate statistical insights from data"""
        insights = {
            "basic_stats": {},
            "data_distribution": {},
            "quality_metrics": {},
            "patterns": []
        }
        
        try:
            if source.source_type == DataSourceType.CSV:
                df = pd.read_csv(source.path, nrows=1000)
                
                # Basic statistics
                insights["basic_stats"] = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "null_values": df.isnull().sum().to_dict()
                }
                
                # Data distribution
                for col in df.select_dtypes(include=[np.number]).columns:
                    insights["data_distribution"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max())
                    }
                
                # Quality metrics
                insights["quality_metrics"] = {
                    "completeness": (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))),
                    "uniqueness": df.nunique().to_dict(),
                    "duplicate_rows": df.duplicated().sum()
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to generate statistical insights: {e}")
        
        return insights
    
    def _assess_data_quality(self, llm_analysis: Dict[str, Any], statistical_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality"""
        quality_score = 0.5  # Default
        
        # Factor in LLM confidence
        llm_confidence = llm_analysis.get("confidence", 0.5)
        quality_score = (quality_score + llm_confidence) / 2
        
        # Factor in statistical quality metrics
        if "quality_metrics" in statistical_insights:
            completeness = statistical_insights["quality_metrics"].get("completeness", 0.5)
            quality_score = (quality_score + completeness) / 2
        
        # Determine quality level
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.4:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "overall_score": quality_score,
            "quality_level": quality_level,
            "factors": {
                "llm_confidence": llm_confidence,
                "completeness": statistical_insights.get("quality_metrics", {}).get("completeness", 0.5),
                "structure_clarity": llm_analysis.get("structure_analysis", {}).get("data_organization", "unknown")
            }
        }
    
    def _generate_source_recommendations(self, llm_analysis: Dict[str, Any], statistical_insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations for data source"""
        recommendations = []
        
        # Based on LLM analysis
        if "recommendations" in llm_analysis:
            recommendations.extend(llm_analysis["recommendations"].get("preprocessing_needed", []))
            recommendations.extend(llm_analysis["recommendations"].get("next_steps", []))
        
        # Based on statistical insights
        quality_metrics = statistical_insights.get("quality_metrics", {})
        if quality_metrics.get("completeness", 1.0) < 0.8:
            recommendations.append("Consider data cleaning to improve completeness")
        
        if quality_metrics.get("duplicate_rows", 0) > 0:
            recommendations.append("Remove duplicate rows to improve data quality")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _perform_cross_source_analysis(self, analysis_results: List[Dict[str, Any]], user_intent: str = "") -> Dict[str, Any]:
        """Perform cross-source analysis to discover relationships"""
        if len(analysis_results) < 2:
            return {"cross_relationships": [], "insights": []}
        
        try:
            # Combine insights from all sources
            combined_insights = []
            for result in analysis_results:
                if "llm_analysis" in result:
                    combined_insights.append(result["llm_analysis"])
            
            # Use LLM to find cross-source relationships
            cross_analysis_prompt = self._build_cross_analysis_prompt(combined_insights, user_intent)
            llm_response = self.llm_processor._call_llm(cross_analysis_prompt)
            cross_analysis = json.loads(llm_response)
            
            return cross_analysis
            
        except Exception as e:
            self.logger.error(f"Cross-source analysis failed: {e}")
            return {"cross_relationships": [], "insights": []}
    
    def _build_cross_analysis_prompt(self, insights: List[Dict[str, Any]], user_intent: str = "") -> str:
        """Build prompt for cross-source analysis"""
        prompt = f"""
You are an expert at analyzing multiple data sources to discover cross-source relationships and insights.

Data Sources Analysis:
{json.dumps(insights, indent=2)}

User Intent: {user_intent if user_intent else "General cross-source analysis"}

Analyze these data sources and find:
1. Common entities across sources
2. Potential relationships between sources
3. Opportunities for data integration
4. Unified graph structure recommendations

Provide analysis in JSON format:
{{
    "cross_relationships": [
        {{
            "source1": "source_id",
            "source2": "source_id",
            "relationship_type": "type of relationship",
            "common_entities": ["entities found in both sources"],
            "confidence": 0.0-1.0
        }}
    ],
    "integration_opportunities": [
        {{
            "description": "opportunity description",
            "sources_involved": ["source_ids"],
            "potential_benefit": "benefit of integration",
            "complexity": "low|medium|high"
        }}
    ],
    "unified_graph_structure": {{
        "main_entities": ["entities that should be central"],
        "relationship_hierarchy": ["how relationships should be organized"],
        "integration_strategy": "strategy for combining sources"
    }},
    "insights": [
        "key insights from cross-source analysis"
    ],
    "recommendations": [
        "recommendations for data integration and graph generation"
    ]
}}
"""
        return prompt
    
    async def _generate_data_insights(self, analysis_results: List[Dict[str, Any]], cross_analysis: Dict[str, Any], user_intent: str = "") -> List[DataInsight]:
        """Generate comprehensive data insights"""
        insights = []
        
        # Generate insights from individual sources
        for result in analysis_results:
            if "llm_analysis" in result:
                llm_analysis = result["llm_analysis"]
                
                # Key findings insights
                for finding in llm_analysis.get("insights", {}).get("key_findings", []):
                    insight = DataInsight(
                        insight_id=f"insight_{len(insights) + 1}",
                        insight_type="key_finding",
                        description=finding,
                        confidence=llm_analysis.get("confidence", 0.5),
                        data_source=result["source_id"],
                        evidence=[finding],
                        implications=["Further analysis recommended"],
                        discovered_at=datetime.now()
                    )
                    insights.append(insight)
        
        # Generate insights from cross-analysis
        for relationship in cross_analysis.get("cross_relationships", []):
            insight = DataInsight(
                insight_id=f"insight_{len(insights) + 1}",
                insight_type="cross_source_relationship",
                description=f"Relationship between {relationship['source1']} and {relationship['source2']}: {relationship['relationship_type']}",
                confidence=relationship.get("confidence", 0.5),
                data_source="multiple",
                evidence=relationship.get("common_entities", []),
                implications=["Data integration opportunity"],
                discovered_at=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    async def _learn_from_analysis(self, analysis_results: List[Dict[str, Any]], insights: List[DataInsight]):
        """Learn from analysis patterns for future improvements"""
        if not self.learning_enabled:
            return
        
        # Extract patterns from successful analyses
        for result in analysis_results:
            if result.get("status") != "error" and "llm_analysis" in result:
                pattern = {
                    "source_type": result.get("source_type"),
                    "analysis_approach": result.get("llm_analysis", {}).get("structure_analysis", {}),
                    "success_factors": result.get("data_quality", {}).get("factors", {}),
                    "timestamp": datetime.now().isoformat()
                }
                self.analysis_patterns.append(pattern)
        
        # Update domain knowledge
        for insight in insights:
            if insight.insight_type == "key_finding":
                domain = insight.description.split()[0] if insight.description else "general"
                if domain not in self.domain_knowledge:
                    self.domain_knowledge[domain] = []
                self.domain_knowledge[domain].append(insight.description)
    
    async def _generate_recommendations(self, analysis_results: List[Dict[str, Any]], insights: List[DataInsight], user_intent: str = "") -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on data quality
        quality_issues = []
        for result in analysis_results:
            if "data_quality" in result:
                quality = result["data_quality"]
                if quality.get("quality_level") in ["poor", "fair"]:
                    quality_issues.append(f"Improve data quality for {result['source_id']}")
        
        if quality_issues:
            recommendations.extend(quality_issues)
        
        # Based on insights
        high_confidence_insights = [i for i in insights if i.confidence > 0.7]
        if high_confidence_insights:
            recommendations.append(f"Focus on {len(high_confidence_insights)} high-confidence insights for graph generation")
        
        # Based on cross-source relationships
        cross_relationships = [r for r in analysis_results if "cross_analysis" in r]
        if cross_relationships:
            recommendations.append("Consider integrating multiple data sources for richer graph structure")
        
        # Based on user intent
        if user_intent:
            recommendations.append(f"Tailor graph generation to user intent: {user_intent}")
        
        return recommendations
    
    def _calculate_quality_summary(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall data quality summary"""
        if not analysis_results:
            return {"overall_quality": "unknown", "sources_analyzed": 0}
        
        quality_scores = []
        for result in analysis_results:
            if "data_quality" in result:
                quality_scores.append(result["data_quality"].get("overall_score", 0.5))
        
        if not quality_scores:
            return {"overall_quality": "unknown", "sources_analyzed": 0}
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality >= 0.8:
            overall_quality = "excellent"
        elif avg_quality >= 0.6:
            overall_quality = "good"
        elif avg_quality >= 0.4:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        return {
            "overall_quality": overall_quality,
            "average_score": avg_quality,
            "sources_analyzed": len(analysis_results),
            "quality_distribution": {
                "excellent": sum(1 for s in quality_scores if s >= 0.8),
                "good": sum(1 for s in quality_scores if 0.6 <= s < 0.8),
                "fair": sum(1 for s in quality_scores if 0.4 <= s < 0.6),
                "poor": sum(1 for s in quality_scores if s < 0.4)
            }
        }
    
    def _calculate_discovery_confidence(self, analysis_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the discovery and analysis process"""
        if not analysis_results:
            return 0.0
        
        successful_analyses = sum(1 for r in analysis_results if r.get("status") != "error")
        success_rate = successful_analyses / len(analysis_results)
        
        # Factor in data quality
        quality_scores = []
        for result in analysis_results:
            if "data_quality" in result:
                quality_scores.append(result["data_quality"].get("overall_score", 0.5))
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Combine success rate and quality
        confidence = (success_rate + avg_quality) / 2
        return min(1.0, max(0.0, confidence))
    
    async def get_discovery_status(self) -> Dict[str, Any]:
        """Get current status of data discovery"""
        return {
            "discovered_sources": len(self.discovered_sources),
            "data_insights": len(self.data_insights),
            "analysis_patterns": len(self.analysis_patterns),
            "domains_learned": len(self.domain_knowledge),
            "learning_enabled": self.learning_enabled,
            "quality_threshold": self.quality_threshold
        }
