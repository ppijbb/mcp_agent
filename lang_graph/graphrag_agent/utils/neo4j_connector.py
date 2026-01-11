"""
Neo4j GraphDB Connector

This module provides Neo4j integration for GraphRAG Agent:
- Async Neo4j driver connection management
- NetworkX to Neo4j graph conversion
- Neo4j to NetworkX graph conversion
- Graph persistence and retrieval
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import networkx as nx

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, basic_auth
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None
    AsyncDriver = None
    AsyncSession = None
    basic_auth = None


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    max_retry_time: int = 30


class Neo4jConnector:
    """
    Neo4j GraphDB Connector for GraphRAG Agent
    
    Provides async connection management and graph conversion utilities
    """
    
    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j connector
        
        Args:
            config: Neo4j connection configuration
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package is not installed. "
                "Please install it with: pip install neo4j"
            )
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.driver: Optional[AsyncDriver] = None
        self._lock = asyncio.Lock()
    
    async def connect(self) -> bool:
        """
        Connect to Neo4j database
        
        Returns:
            True if connection successful, False otherwise
        """
        async with self._lock:
            if self.driver is not None:
                try:
                    await self.driver.verify_connectivity()
                    return True
                except Exception:
                    # Connection lost, need to reconnect
                    await self._close_driver()
            
            try:
                self.logger.info(f"Connecting to Neo4j at {self.config.uri}")
                
                self.driver = AsyncGraphDatabase.driver(
                    self.config.uri,
                    auth=basic_auth(self.config.username, self.config.password),
                    max_connection_pool_size=self.config.max_connection_pool_size,
                    connection_timeout=self.config.connection_timeout,
                    max_retry_time=self.config.max_retry_time
                )
                
                await self.driver.verify_connectivity()
                self.logger.info("Neo4j connection established successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to connect to Neo4j: {e}")
                self.driver = None
                return False
    
    async def disconnect(self):
        """Disconnect from Neo4j database"""
        async with self._lock:
            await self._close_driver()
    
    async def _close_driver(self):
        """Close the Neo4j driver"""
        if self.driver is not None:
            try:
                await self.driver.close()
                self.logger.info("Neo4j connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing Neo4j connection: {e}")
            finally:
                self.driver = None
    
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Execute a Cypher query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name (defaults to config database)
            
        Returns:
            Tuple of (records, summary)
        """
        if not await self.connect():
            raise ConnectionError("Not connected to Neo4j")
        
        db = database or self.config.database
        
        try:
            async with self.driver.session(database=db) as session:
                result = await session.run(query, parameters or {})
                records = [record async for record in result]
                summary = await result.consume()
                return records, summary
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    async def clear_database(self, database: Optional[str] = None):
        """
        Clear all nodes and relationships from the database
        
        Args:
            database: Database name (defaults to config database)
        """
        query = "MATCH (n) DETACH DELETE n"
        await self.execute_query(query, database=database)
        self.logger.info("Database cleared")
    
    async def save_graph(
        self, 
        graph: nx.Graph, 
        graph_name: Optional[str] = None,
        clear_existing: bool = True,
        database: Optional[str] = None
    ) -> bool:
        """
        Save NetworkX graph to Neo4j
        
        Args:
            graph: NetworkX graph to save
            graph_name: Optional name for the graph
            clear_existing: Whether to clear existing graph data
            database: Database name (defaults to config database)
            
        Returns:
            True if successful, False otherwise
        """
        if not await self.connect():
            return False
        
        try:
            if clear_existing:
                await self.clear_database(database)
            
            # Create nodes
            entity_nodes = []
            text_nodes = []
            
            for node_id, data in graph.nodes(data=True):
                node_type = data.get("type", "entity")
                
                if node_type == "entity":
                    entity_nodes.append((node_id, data))
                elif node_type == "text_unit":
                    text_nodes.append((node_id, data))
            
            # Create Entity nodes
            for node_id, data in entity_nodes:
                name = data.get("name", node_id)
                category = data.get("category", "other")
                importance = data.get("importance", 0.5)
                attributes = data.get("attributes", {})
                confidence = data.get("confidence", 0.5)
                context = data.get("context", "")
                
                # Prepare properties
                properties = {
                    "id": node_id,
                    "name": name,
                    "category": category,
                    "importance": float(importance),
                    "confidence": float(confidence),
                    "context": context
                }
                
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        if isinstance(value, (str, int, float, bool)):
                            properties[f"attr_{key}"] = value
                
                # Create node
                query = """
                CREATE (e:Entity $properties)
                RETURN e
                """
                await self.execute_query(query, {"properties": properties}, database)
            
            # Create TextUnit nodes
            for node_id, data in text_nodes:
                content = data.get("content", "")
                document_id = data.get("document_id", node_id)
                metadata = data.get("metadata", {})
                
                properties = {
                    "id": node_id,
                    "content": content,
                    "document_id": document_id
                }
                
                # Add metadata
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            properties[f"meta_{key}"] = value
                
                query = """
                CREATE (t:TextUnit $properties)
                RETURN t
                """
                await self.execute_query(query, {"properties": properties}, database)
            
            # Create relationships
            for source, target, edge_data in graph.edges(data=True):
                relationship_type = edge_data.get("relationship_type", "RELATES_TO")
                confidence = edge_data.get("confidence", 0.5)
                weight = edge_data.get("weight", 1.0)
                context = edge_data.get("context", "")
                attributes = edge_data.get("attributes", {})
                
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]
                
                source_type = source_data.get("type", "entity")
                target_type = target_data.get("type", "entity")
                
                # Determine source and target labels
                source_label = "Entity" if source_type == "entity" else "TextUnit"
                target_label = "Entity" if target_type == "entity" else "TextUnit"
                
                # Prepare relationship properties
                rel_properties = {
                    "confidence": float(confidence),
                    "weight": float(weight),
                    "context": context
                }
                
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        if isinstance(value, (str, int, float, bool)):
                            rel_properties[f"attr_{key}"] = value
                
                # Create relationship
                query = f"""
                MATCH (a:{source_label} {{id: $source_id}})
                MATCH (b:{target_label} {{id: $target_id}})
                CREATE (a)-[r:{relationship_type} $properties]->(b)
                RETURN r
                """
                
                await self.execute_query(
                    query,
                    {
                        "source_id": source,
                        "target_id": target,
                        "properties": rel_properties
                    },
                    database
                )
            
            # Store graph metadata if provided
            if graph_name:
                graph_metadata = graph.graph.get("construction_plan", {})
                if graph_metadata:
                    query = """
                    CREATE (g:GraphMetadata {
                        name: $name,
                        structure_type: $structure_type,
                        created_at: datetime()
                    })
                    RETURN g
                    """
                    await self.execute_query(
                        query,
                        {
                            "name": graph_name,
                            "structure_type": graph_metadata.get("structure_type", "unknown")
                        },
                        database
                    )
            
            self.logger.info(f"Graph saved to Neo4j: {len(entity_nodes)} entities, {len(text_nodes)} text units, {graph.number_of_edges()} relationships")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save graph to Neo4j: {e}")
            return False
    
    async def load_graph(
        self,
        graph_name: Optional[str] = None,
        database: Optional[str] = None
    ) -> Optional[nx.Graph]:
        """
        Load graph from Neo4j to NetworkX
        
        Args:
            graph_name: Optional graph name filter
            database: Database name (defaults to config database)
            
        Returns:
            NetworkX graph or None if failed
        """
        if not await self.connect():
            return None
        
        try:
            graph = nx.Graph()
            
            # Load Entity nodes
            query = "MATCH (e:Entity) RETURN e"
            records, _ = await self.execute_query(query, database=database)
            
            for record in records:
                node = record["e"]
                node_id = node.get("id", str(node.id))
                properties = dict(node)
                
                graph.add_node(
                    node_id,
                    type="entity",
                    name=properties.get("name", ""),
                    category=properties.get("category", "other"),
                    importance=properties.get("importance", 0.5),
                    confidence=properties.get("confidence", 0.5),
                    context=properties.get("context", ""),
                    attributes={k.replace("attr_", ""): v for k, v in properties.items() if k.startswith("attr_")}
                )
            
            # Load TextUnit nodes
            query = "MATCH (t:TextUnit) RETURN t"
            records, _ = await self.execute_query(query, database=database)
            
            for record in records:
                node = record["t"]
                node_id = node.get("id", str(node.id))
                properties = dict(node)
                
                graph.add_node(
                    node_id,
                    type="text_unit",
                    content=properties.get("content", ""),
                    document_id=properties.get("document_id", node_id),
                    metadata={k.replace("meta_", ""): v for k, v in properties.items() if k.startswith("meta_")}
                )
            
            # Load relationships
            query = """
            MATCH (a)-[r]->(b)
            RETURN a.id as source, b.id as target, type(r) as rel_type, properties(r) as props
            """
            records, _ = await self.execute_query(query, database=database)
            
            for record in records:
                source = record["source"]
                target = record["target"]
                rel_type = record["rel_type"]
                props = record["props"]
                
                graph.add_edge(
                    source,
                    target,
                    relationship_type=rel_type,
                    confidence=props.get("confidence", 0.5),
                    weight=props.get("weight", 1.0),
                    context=props.get("context", ""),
                    attributes={k.replace("attr_", ""): v for k, v in props.items() if k.startswith("attr_")}
                )
            
            self.logger.info(f"Graph loaded from Neo4j: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            self.logger.error(f"Failed to load graph from Neo4j: {e}")
            return None
    
    async def get_graph_stats(self, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the graph in Neo4j
        
        Args:
            database: Database name (defaults to config database)
            
        Returns:
            Dictionary with graph statistics
        """
        if not await self.connect():
            return {}
        
        try:
            stats = {}
            
            # Count nodes by type
            query = "MATCH (n:Entity) RETURN count(n) as count"
            records, _ = await self.execute_query(query, database=database)
            stats["entity_count"] = records[0]["count"] if records else 0
            
            query = "MATCH (n:TextUnit) RETURN count(n) as count"
            records, _ = await self.execute_query(query, database=database)
            stats["text_unit_count"] = records[0]["count"] if records else 0
            
            # Count relationships
            query = "MATCH ()-[r]->() RETURN count(r) as count"
            records, _ = await self.execute_query(query, database=database)
            stats["relationship_count"] = records[0]["count"] if records else 0
            
            # Count relationship types
            query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
            """
            records, _ = await self.execute_query(query, database=database)
            stats["relationship_types"] = {r["rel_type"]: r["count"] for r in records}
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get graph stats: {e}")
            return {}
    
    async def create_indexes(self, database: Optional[str] = None):
        """
        Create indexes for better query performance
        
        Args:
            database: Database name (defaults to config database)
        """
        if not await self.connect():
            return
        
        try:
            indexes = [
                "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX entity_category IF NOT EXISTS FOR (e:Entity) ON (e.category)",
                "CREATE INDEX textunit_id IF NOT EXISTS FOR (t:TextUnit) ON (t.id)",
                "CREATE INDEX textunit_document_id IF NOT EXISTS FOR (t:TextUnit) ON (t.document_id)"
            ]
            
            for index_query in indexes:
                try:
                    await self.execute_query(index_query, database=database)
                    self.logger.debug(f"Created index: {index_query}")
                except Exception as e:
                    self.logger.warning(f"Index creation failed (may already exist): {e}")
            
            # System ontology indexes
            system_indexes = [
                "CREATE INDEX goal_id IF NOT EXISTS FOR (g:Goal) ON (g.id)",
                "CREATE INDEX goal_name IF NOT EXISTS FOR (g:Goal) ON (g.name)",
                "CREATE INDEX goal_status IF NOT EXISTS FOR (g:Goal) ON (g.status)",
                "CREATE INDEX task_id IF NOT EXISTS FOR (t:Task) ON (t.id)",
                "CREATE INDEX task_name IF NOT EXISTS FOR (t:Task) ON (t.name)",
                "CREATE INDEX task_status IF NOT EXISTS FOR (t:Task) ON (t.status)",
                "CREATE INDEX state_id IF NOT EXISTS FOR (s:State) ON (s.id)",
                "CREATE INDEX state_name IF NOT EXISTS FOR (s:State) ON (s.name)",
                "CREATE INDEX resource_id IF NOT EXISTS FOR (r:Resource) ON (r.id)",
                "CREATE INDEX precondition_id IF NOT EXISTS FOR (p:Precondition) ON (p.id)",
                "CREATE INDEX postcondition_id IF NOT EXISTS FOR (p:Postcondition) ON (p.id)"
            ]
            
            for index_query in system_indexes:
                try:
                    await self.execute_query(index_query, database=database)
                    self.logger.debug(f"Created system index: {index_query}")
                except Exception as e:
                    self.logger.warning(f"System index creation failed (may already exist): {e}")
            
            self.logger.info("Indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")
    
    async def save_system_ontology(
        self,
        ontology,
        database: Optional[str] = None
    ) -> bool:
        """
        Save system ontology to Neo4j
        
        Args:
            ontology: SystemOntology instance
            database: Database name (defaults to config database)
            
        Returns:
            True if successful, False otherwise
        """
        if not await self.connect():
            return False
        
        try:
            # Save Goals
            for goal_id, goal in ontology.goals.items():
                properties = {
                    "id": goal.id,
                    "name": goal.name,
                    "description": goal.description,
                    "priority": float(goal.priority),
                    "status": goal.status.value,
                    "created_at": goal.created_at.isoformat(),
                    "achieved_at": goal.achieved_at.isoformat() if goal.achieved_at else None
                }
                
                # Add metadata
                for key, value in goal.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        properties[f"meta_{key}"] = value
                
                query = """
                MERGE (g:Goal {id: $id})
                SET g += $properties
                RETURN g
                """
                await self.execute_query(query, {"id": goal.id, "properties": properties}, database)
            
            # Save Tasks
            for task_id, task in ontology.tasks.items():
                properties = {
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "executable": task.executable,
                    "status": task.status.value,
                    "priority": float(task.priority),
                    "execution_time": task.execution_time,
                    "success_rate": float(task.success_rate),
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }
                
                # Add metadata
                for key, value in task.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        properties[f"meta_{key}"] = value
                
                query = """
                MERGE (t:Task {id: $id})
                SET t += $properties
                RETURN t
                """
                await self.execute_query(query, {"id": task.id, "properties": properties}, database)
            
            # Save Preconditions
            for pre_id, pre in ontology.preconditions.items():
                properties = {
                    "id": pre.id,
                    "description": pre.description,
                    "condition": pre.condition,
                    "satisfied": pre.satisfied,
                    "required_by": pre.required_by,
                    "satisfied_by": pre.satisfied_by
                }
                
                query = """
                MERGE (p:Precondition {id: $id})
                SET p += $properties
                RETURN p
                """
                await self.execute_query(query, {"id": pre.id, "properties": properties}, database)
            
            # Save Postconditions
            for post_id, post in ontology.postconditions.items():
                properties = {
                    "id": post.id,
                    "description": post.description,
                    "condition": post.condition,
                    "achieved": post.achieved,
                    "produced_by": post.produced_by
                }
                
                query = """
                MERGE (p:Postcondition {id: $id})
                SET p += $properties
                RETURN p
                """
                await self.execute_query(query, {"id": post.id, "properties": properties}, database)
            
            # Save States
            for state_id, state in ontology.states.items():
                properties = {
                    "id": state.id,
                    "name": state.name,
                    "state_type": state.state_type.value,
                    "value": str(state.value),
                    "timestamp": state.timestamp.isoformat()
                }
                
                query = """
                MERGE (s:State {id: $id})
                SET s += $properties
                RETURN s
                """
                await self.execute_query(query, {"id": state.id, "properties": properties}, database)
            
            # Save Resources
            for res_id, resource in ontology.resources.items():
                properties = {
                    "id": resource.id,
                    "name": resource.name,
                    "resource_type": resource.resource_type,
                    "availability": float(resource.availability),
                    "capacity": float(resource.capacity),
                    "current_usage": float(resource.current_usage)
                }
                
                query = """
                MERGE (r:Resource {id: $id})
                SET r += $properties
                RETURN r
                """
                await self.execute_query(query, {"id": resource.id, "properties": properties}, database)
            
            # Save Constraints
            for const_id, constraint in ontology.constraints.items():
                properties = {
                    "id": constraint.id,
                    "constraint_type": constraint.constraint_type.value,
                    "condition": constraint.condition,
                    "severity": constraint.severity.value,
                    "violated": constraint.violated
                }
                
                query = """
                MERGE (c:Constraint {id: $id})
                SET c += $properties
                RETURN c
                """
                await self.execute_query(query, {"id": constraint.id, "properties": properties}, database)
            
            # Create relationships: Goal -> SubGoal
            for goal_id, goal in ontology.goals.items():
                parent_goal_id = goal.metadata.get("parent_goal")
                if parent_goal_id and parent_goal_id in ontology.goals:
                    query = """
                    MATCH (parent:Goal {id: $parent_id})
                    MATCH (child:Goal {id: $child_id})
                    MERGE (parent)-[r:HAS_SUBGOAL]->(child)
                    RETURN r
                    """
                    await self.execute_query(
                        query,
                        {"parent_id": parent_goal_id, "child_id": goal_id},
                        database
                    )
            
            # Create relationships: Goal -> Task (ACHIEVED_BY)
            for task_id, task in ontology.tasks.items():
                for goal_id in task.metadata.get("achieves_goals", []):
                    if goal_id in ontology.goals:
                        query = """
                        MATCH (g:Goal {id: $goal_id})
                        MATCH (t:Task {id: $task_id})
                        MERGE (g)-[r:ACHIEVED_BY]->(t)
                        RETURN r
                        """
                        await self.execute_query(
                            query,
                            {"goal_id": goal_id, "task_id": task_id},
                            database
                        )
            
            # Create relationships: Task -> Precondition (REQUIRES)
            for task_id, task in ontology.tasks.items():
                for pre_id in task.preconditions:
                    if pre_id in ontology.preconditions:
                        query = """
                        MATCH (t:Task {id: $task_id})
                        MATCH (p:Precondition {id: $pre_id})
                        MERGE (t)-[r:REQUIRES]->(p)
                        RETURN r
                        """
                        await self.execute_query(
                            query,
                            {"task_id": task_id, "pre_id": pre_id},
                            database
                        )
            
            # Create relationships: Task -> Postcondition (PRODUCES)
            for task_id, task in ontology.tasks.items():
                for post_id in task.postconditions:
                    if post_id in ontology.postconditions:
                        query = """
                        MATCH (t:Task {id: $task_id})
                        MATCH (p:Postcondition {id: $post_id})
                        MERGE (t)-[r:PRODUCES]->(p)
                        RETURN r
                        """
                        await self.execute_query(
                            query,
                            {"task_id": task_id, "post_id": post_id},
                            database
                        )
            
            # Create relationships: Task -> Task (DEPENDS_ON)
            for task_id, task in ontology.tasks.items():
                for dep_id in task.dependencies:
                    if dep_id in ontology.tasks:
                        query = """
                        MATCH (dependent:Task {id: $task_id})
                        MATCH (dependency:Task {id: $dep_id})
                        MERGE (dependent)-[r:DEPENDS_ON]->(dependency)
                        RETURN r
                        """
                        await self.execute_query(
                            query,
                            {"task_id": task_id, "dep_id": dep_id},
                            database
                        )
            
            # Create relationships: Task -> Resource (CONSUMES)
            for task_id, task in ontology.tasks.items():
                for req_id in task.resource_requirements:
                    if req_id in ontology.resource_requirements:
                        req = ontology.resource_requirements[req_id]
                        if req.resource_id in ontology.resources:
                            query = """
                            MATCH (t:Task {id: $task_id})
                            MATCH (r:Resource {id: $resource_id})
                            MERGE (t)-[rel:CONSUMES {amount: $amount}]->(r)
                            RETURN rel
                            """
                            await self.execute_query(
                                query,
                                {
                                    "task_id": task_id,
                                    "resource_id": req.resource_id,
                                    "amount": req.required_amount
                                },
                                database
                            )
            
            # Create relationships: Task -> State (TRANSITIONS_TO)
            for task_id, task in ontology.tasks.items():
                for trans_id in task.state_transitions:
                    if trans_id in ontology.state_transitions:
                        trans = ontology.state_transitions[trans_id]
                        query = """
                        MATCH (t:Task {id: $task_id})
                        MATCH (s:State {id: $state_id})
                        MERGE (t)-[r:TRANSITIONS_TO]->(s)
                        RETURN r
                        """
                        await self.execute_query(
                            query,
                            {"task_id": task_id, "state_id": trans.to_state},
                            database
                        )
            
            # Create relationships: State -> State (PRECEDES)
            for trans_id, trans in ontology.state_transitions.items():
                query = """
                MATCH (from:State {id: $from_id})
                MATCH (to:State {id: $to_id})
                MERGE (from)-[r:PRECEDES]->(to)
                RETURN r
                """
                await self.execute_query(
                    query,
                    {"from_id": trans.from_state, "to_id": trans.to_state},
                    database
                )
            
            # Create relationships: Task -> Constraint (CONSTRAINED_BY)
            for task_id, task in ontology.tasks.items():
                for const_id in task.constraints:
                    if const_id in ontology.constraints:
                        query = """
                        MATCH (t:Task {id: $task_id})
                        MATCH (c:Constraint {id: $const_id})
                        MERGE (t)-[r:CONSTRAINED_BY]->(c)
                        RETURN r
                        """
                        await self.execute_query(
                            query,
                            {"task_id": task_id, "const_id": const_id},
                            database
                        )
            
            self.logger.info(f"System ontology saved to Neo4j: {len(ontology.goals)} goals, {len(ontology.tasks)} tasks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save system ontology to Neo4j: {e}")
            return False
    
    async def load_system_ontology(
        self,
        database: Optional[str] = None
    ):
        """
        Load system ontology from Neo4j
        
        Args:
            database: Database name (defaults to config database)
            
        Returns:
            SystemOntology or None if failed
        """
        if not await self.connect():
            return None
        
        try:
            from models.system_ontology import SystemOntology, Goal, Task, Precondition, Postcondition, State, Resource, Constraint, StateTransition, GoalStatus, TaskStatus, StateType, ConstraintType, ConstraintSeverity
            
            ontology = SystemOntology()
            
            # Load Goals
            query = "MATCH (g:Goal) RETURN g"
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                node = record["g"]
                props = dict(node)
                goal = Goal(
                    id=props["id"],
                    name=props.get("name", ""),
                    description=props.get("description", ""),
                    priority=float(props.get("priority", 0.5)),
                    status=GoalStatus(props.get("status", "pending")),
                    created_at=datetime.fromisoformat(props.get("created_at", datetime.now().isoformat())),
                    achieved_at=datetime.fromisoformat(props["achieved_at"]) if props.get("achieved_at") else None,
                    metadata={k.replace("meta_", ""): v for k, v in props.items() if k.startswith("meta_")}
                )
                ontology.add_goal(goal)
            
            # Load Tasks
            query = "MATCH (t:Task) RETURN t"
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                node = record["t"]
                props = dict(node)
                task = Task(
                    id=props["id"],
                    name=props.get("name", ""),
                    description=props.get("description", ""),
                    executable=props.get("executable", True),
                    status=TaskStatus(props.get("status", "pending")),
                    priority=float(props.get("priority", 0.5)),
                    execution_time=props.get("execution_time"),
                    success_rate=float(props.get("success_rate", 1.0)),
                    metadata={k.replace("meta_", ""): v for k, v in props.items() if k.startswith("meta_")},
                    created_at=datetime.fromisoformat(props.get("created_at", datetime.now().isoformat())),
                    started_at=datetime.fromisoformat(props["started_at"]) if props.get("started_at") else None,
                    completed_at=datetime.fromisoformat(props["completed_at"]) if props.get("completed_at") else None
                )
                ontology.add_task(task)
            
            # Load Preconditions and link to tasks
            query = "MATCH (p:Precondition) RETURN p"
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                node = record["p"]
                props = dict(node)
                pre = Precondition(
                    id=props["id"],
                    description=props.get("description", ""),
                    condition=props.get("condition", ""),
                    satisfied=props.get("satisfied", False),
                    required_by=props.get("required_by"),
                    satisfied_by=props.get("satisfied_by")
                )
                ontology.add_precondition(pre)
                
                # Link to task
                if pre.required_by and pre.required_by in ontology.tasks:
                    if pre.id not in ontology.tasks[pre.required_by].preconditions:
                        ontology.tasks[pre.required_by].preconditions.append(pre.id)
            
            # Load Postconditions and link to tasks
            query = "MATCH (p:Postcondition) RETURN p"
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                node = record["p"]
                props = dict(node)
                post = Postcondition(
                    id=props["id"],
                    description=props.get("description", ""),
                    condition=props.get("condition", ""),
                    achieved=props.get("achieved", False),
                    produced_by=props.get("produced_by")
                )
                ontology.add_postcondition(post)
                
                # Link to task
                if post.produced_by and post.produced_by in ontology.tasks:
                    if post.id not in ontology.tasks[post.produced_by].postconditions:
                        ontology.tasks[post.produced_by].postconditions.append(post.id)
            
            # Load States
            query = "MATCH (s:State) RETURN s"
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                node = record["s"]
                props = dict(node)
                state = State(
                    id=props["id"],
                    name=props.get("name", ""),
                    state_type=StateType(props.get("state_type", "system")),
                    value=props.get("value", ""),
                    timestamp=datetime.fromisoformat(props.get("timestamp", datetime.now().isoformat()))
                )
                ontology.add_state(state)
            
            # Load Resources
            query = "MATCH (r:Resource) RETURN r"
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                node = record["r"]
                props = dict(node)
                resource = Resource(
                    id=props["id"],
                    name=props.get("name", ""),
                    resource_type=props.get("resource_type", "generic"),
                    availability=float(props.get("availability", 1.0)),
                    capacity=float(props.get("capacity", 1.0)),
                    current_usage=float(props.get("current_usage", 0.0))
                )
                ontology.add_resource(resource)
            
            # Load Constraints
            query = "MATCH (c:Constraint) RETURN c"
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                node = record["c"]
                props = dict(node)
                constraint = Constraint(
                    id=props["id"],
                    constraint_type=ConstraintType(props.get("constraint_type", "logical")),
                    condition=props.get("condition", ""),
                    severity=ConstraintSeverity(props.get("severity", "hard")),
                    violated=props.get("violated", False)
                )
                ontology.add_constraint(constraint)
            
            # Load relationships: Task dependencies
            query = """
            MATCH (t1:Task)-[r:DEPENDS_ON]->(t2:Task)
            RETURN t1.id as task_id, t2.id as depends_on_id
            """
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                task_id = record["task_id"]
                dep_id = record["depends_on_id"]
                if task_id in ontology.tasks and dep_id not in ontology.tasks[task_id].dependencies:
                    ontology.tasks[task_id].dependencies.append(dep_id)
            
            # Load relationships: Task constraints
            query = """
            MATCH (t:Task)-[r:CONSTRAINED_BY]->(c:Constraint)
            RETURN t.id as task_id, c.id as constraint_id
            """
            records, _ = await self.execute_query(query, database=database)
            for record in records:
                task_id = record["task_id"]
                const_id = record["constraint_id"]
                if task_id in ontology.tasks and const_id not in ontology.tasks[task_id].constraints:
                    ontology.tasks[task_id].constraints.append(const_id)
            
            self.logger.info(f"System ontology loaded from Neo4j: {len(ontology.goals)} goals, {len(ontology.tasks)} tasks")
            return ontology
            
        except Exception as e:
            self.logger.error(f"Failed to load system ontology from Neo4j: {e}")
            return None

