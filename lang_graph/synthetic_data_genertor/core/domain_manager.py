"""
Domain Manager for the Kimi-K2 Agentic Data Synthesis System

Manages domains, scenarios, and their relationships for the synthesis system.
"""

from typing import List, Dict, Any, Optional, Set
from ..models.domain import Domain, Scenario, Criteria, ComplexityLevel, DomainCategory
from ..models.tool import Tool
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DomainManager:
    """
    Manages domains and scenarios for the agentic data synthesis system.
    
    Responsibilities:
    - Domain creation and management
    - Scenario generation and organization
    - Domain relationships and dependencies
    - Domain-specific criteria and evaluation rubrics
    """
    
    def __init__(self):
        self.domains: Dict[str, Domain] = {}
        self.domain_templates: Dict[str, Dict[str, Any]] = {}
        self.domain_relationships: Dict[str, Set[str]] = {}
        self.scenario_templates: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with default domains
        self._initialize_default_domains()
    
    def _initialize_default_domains(self) -> None:
        """Initialize the system with default domains"""
        default_domains = [
            {
                "name": "Technical Support",
                "description": "Technical support and troubleshooting scenarios",
                "category": DomainCategory.TECHNOLOGY,
                "complexity_level": ComplexityLevel.INTERMEDIATE,
                "required_tools": ["system_diagnostics", "log_analysis", "remote_access"],
                "metadata": {
                    "industry": "IT Support",
                    "target_audience": "Technical Support Engineers"
                }
            },
            {
                "name": "Financial Analysis",
                "description": "Financial analysis and reporting scenarios",
                "category": DomainCategory.FINANCE,
                "complexity_level": ComplexityLevel.ADVANCED,
                "required_tools": ["data_analysis", "spreadsheet_tools", "reporting"],
                "metadata": {
                    "industry": "Finance",
                    "target_audience": "Financial Analysts"
                }
            },
            {
                "name": "Educational Content",
                "description": "Educational content creation and tutoring scenarios",
                "category": DomainCategory.EDUCATION,
                "complexity_level": ComplexityLevel.INTERMEDIATE,
                "required_tools": ["content_creation", "assessment_tools", "multimedia"],
                "metadata": {
                    "industry": "Education",
                    "target_audience": "Educators and Tutors"
                }
            },
            {
                "name": "Creative Writing",
                "description": "Creative writing and content creation scenarios",
                "category": DomainCategory.CREATIVE,
                "complexity_level": ComplexityLevel.INTERMEDIATE,
                "required_tools": ["text_generation", "style_analysis", "content_editing"],
                "metadata": {
                    "industry": "Creative",
                    "target_audience": "Writers and Content Creators"
                }
            }
        ]
        
        for domain_config in default_domains:
            self.create_domain(**domain_config)
    
    def create_domain(self, name: str, description: str, category: DomainCategory,
                     complexity_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE,
                     required_tools: List[str] = None, metadata: Dict[str, Any] = None) -> Domain:
        """Create a new domain"""
        domain = Domain(
            name=name,
            description=description,
            category=category,
            complexity_level=complexity_level,
            required_tools=required_tools or [],
            metadata=metadata or {}
        )
        
        self.domains[domain.id] = domain
        logger.info(f"Created domain: {name} (ID: {domain.id})")
        
        return domain
    
    def get_domain(self, domain_id: str) -> Optional[Domain]:
        """Get a domain by ID"""
        return self.domains.get(domain_id)
    
    def get_domain_by_name(self, name: str) -> Optional[Domain]:
        """Get a domain by name"""
        for domain in self.domains.values():
            if domain.name.lower() == name.lower():
                return domain
        return None
    
    def list_domains(self, category: Optional[DomainCategory] = None,
                    complexity_level: Optional[ComplexityLevel] = None) -> List[Domain]:
        """List domains with optional filtering"""
        domains = list(self.domains.values())
        
        if category:
            domains = [d for d in domains if d.category == category]
        
        if complexity_level:
            domains = [d for d in domains if d.complexity_level == complexity_level]
        
        return domains
    
    def update_domain(self, domain_id: str, **kwargs) -> bool:
        """Update a domain"""
        domain = self.get_domain(domain_id)
        if not domain:
            logger.warning(f"Domain not found: {domain_id}")
            return False
        
        for key, value in kwargs.items():
            if hasattr(domain, key):
                setattr(domain, key, value)
        
        domain.updated_at = datetime.utcnow()
        logger.info(f"Updated domain: {domain.name}")
        return True
    
    def delete_domain(self, domain_id: str) -> bool:
        """Delete a domain"""
        if domain_id not in self.domains:
            logger.warning(f"Domain not found: {domain_id}")
            return False
        
        domain_name = self.domains[domain_id].name
        del self.domains[domain_id]
        logger.info(f"Deleted domain: {domain_name}")
        return True
    
    def add_scenario_to_domain(self, domain_id: str, scenario: Scenario) -> bool:
        """Add a scenario to a domain"""
        domain = self.get_domain(domain_id)
        if not domain:
            logger.warning(f"Domain not found: {domain_id}")
            return False
        
        domain.add_scenario(scenario)
        logger.info(f"Added scenario '{scenario.name}' to domain '{domain.name}'")
        return True
    
    def create_scenario(self, domain_id: str, name: str, description: str,
                       steps: List[Dict[str, Any]], expected_outcome: str,
                       difficulty_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE,
                       required_tools: List[str] = None, tags: List[str] = None) -> Optional[Scenario]:
        """Create a new scenario for a domain"""
        domain = self.get_domain(domain_id)
        if not domain:
            logger.warning(f"Domain not found: {domain_id}")
            return None
        
        # Convert step dictionaries to Step objects
        scenario_steps = []
        for i, step_data in enumerate(steps):
            step = Scenario.Step(
                step_number=i + 1,
                description=step_data.get("description", ""),
                expected_action=step_data.get("expected_action", ""),
                required_tools=step_data.get("required_tools", []),
                expected_outcome=step_data.get("expected_outcome", ""),
                difficulty=step_data.get("difficulty", ComplexityLevel.INTERMEDIATE)
            )
            scenario_steps.append(step)
        
        # Create evaluation criteria
        evaluation_criteria = [
            Criteria(
                name="Task Completion",
                description="Whether the agent successfully completed the assigned task",
                weight=0.4,
                evaluation_type="accuracy",
                scoring_scale=5
            ),
            Criteria(
                name="Solution Quality",
                description="Quality and effectiveness of the provided solution",
                weight=0.3,
                evaluation_type="completeness",
                scoring_scale=5
            ),
            Criteria(
                name="Communication",
                description="Clarity and professionalism of communication",
                weight=0.3,
                evaluation_type="user_satisfaction",
                scoring_scale=5
            )
        ]
        
        scenario = Scenario(
            domain_id=domain_id,
            name=name,
            description=description,
            steps=scenario_steps,
            expected_outcome=expected_outcome,
            difficulty_level=difficulty_level,
            evaluation_rubric=evaluation_criteria,
            required_tools=required_tools or [],
            tags=tags or []
        )
        
        domain.add_scenario(scenario)
        logger.info(f"Created scenario '{name}' for domain '{domain.name}'")
        
        return scenario
    
    def get_scenarios_by_domain(self, domain_id: str,
                               difficulty: Optional[ComplexityLevel] = None) -> List[Scenario]:
        """Get scenarios for a specific domain"""
        domain = self.get_domain(domain_id)
        if not domain:
            return []
        
        scenarios = domain.scenarios
        if difficulty:
            scenarios = domain.get_scenarios_by_difficulty(difficulty)
        
        return scenarios
    
    def get_scenarios_by_tools(self, tools: List[str]) -> List[Scenario]:
        """Get scenarios that require specific tools"""
        matching_scenarios = []
        
        for domain in self.domains.values():
            for scenario in domain.scenarios:
                if any(tool in scenario.required_tools for tool in tools):
                    matching_scenarios.append(scenario)
        
        return matching_scenarios
    
    def add_domain_relationship(self, domain_id: str, related_domain_id: str) -> bool:
        """Add a relationship between domains"""
        if domain_id not in self.domains or related_domain_id not in self.domains:
            logger.warning(f"One or both domains not found: {domain_id}, {related_domain_id}")
            return False
        
        if domain_id not in self.domain_relationships:
            self.domain_relationships[domain_id] = set()
        
        self.domain_relationships[domain_id].add(related_domain_id)
        logger.info(f"Added relationship between domains: {domain_id} -> {related_domain_id}")
        return True
    
    def get_related_domains(self, domain_id: str) -> List[Domain]:
        """Get domains related to a specific domain"""
        related_ids = self.domain_relationships.get(domain_id, set())
        return [self.domains[rid] for rid in related_ids if rid in self.domains]
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about all domains"""
        stats = {
            "total_domains": len(self.domains),
            "domains_by_category": {},
            "domains_by_complexity": {},
            "total_scenarios": 0,
            "average_scenarios_per_domain": 0.0
        }
        
        for domain in self.domains.values():
            # Category statistics
            category = domain.category.value
            stats["domains_by_category"][category] = stats["domains_by_category"].get(category, 0) + 1
            
            # Complexity statistics
            complexity = domain.complexity_level.value
            stats["domains_by_complexity"][complexity] = stats["domains_by_complexity"].get(complexity, 0) + 1
            
            # Scenario statistics
            stats["total_scenarios"] += len(domain.scenarios)
        
        if stats["total_domains"] > 0:
            stats["average_scenarios_per_domain"] = stats["total_scenarios"] / stats["total_domains"]
        
        return stats
    
    def export_domain(self, domain_id: str, format: str = "json") -> Optional[str]:
        """Export a domain to specified format"""
        domain = self.get_domain(domain_id)
        if not domain:
            return None
        
        if format.lower() == "json":
            return domain.model_dump_json(indent=2)
        else:
            logger.warning(f"Unsupported export format: {format}")
            return None
    
    def import_domain(self, domain_data: Dict[str, Any]) -> Optional[Domain]:
        """Import a domain from data"""
        try:
            domain = Domain(**domain_data)
            self.domains[domain.id] = domain
            logger.info(f"Imported domain: {domain.name}")
            return domain
        except Exception as e:
            logger.error(f"Failed to import domain: {e}")
            return None
    
    def validate_domain(self, domain_id: str) -> Dict[str, Any]:
        """Validate a domain and its scenarios"""
        domain = self.get_domain(domain_id)
        if not domain:
            return {"valid": False, "errors": ["Domain not found"]}
        
        errors = []
        warnings = []
        
        # Validate domain structure
        if not domain.name:
            errors.append("Domain name is required")
        
        if not domain.description:
            warnings.append("Domain description is empty")
        
        # Validate scenarios
        for scenario in domain.scenarios:
            if not scenario.name:
                errors.append(f"Scenario name is required in scenario {scenario.id}")
            
            if not scenario.steps:
                warnings.append(f"Scenario '{scenario.name}' has no steps")
            
            # Validate steps
            for step in scenario.steps:
                if not step.description:
                    warnings.append(f"Step {step.step_number} in scenario '{scenario.name}' has no description")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "scenario_count": len(domain.scenarios)
        } 