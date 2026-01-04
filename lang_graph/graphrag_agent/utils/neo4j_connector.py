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
            
            self.logger.info("Indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")

