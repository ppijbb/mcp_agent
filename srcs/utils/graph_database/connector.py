"""Neo4j Graph Database Connector.

This module provides an async connector for Neo4j graph database operations.
"""

import asyncio
from neo4j import AsyncGraphDatabase, basic_auth
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Async Neo4j database connector with thread-safe connection management."""

    _driver = None
    _lock = asyncio.Lock()

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """Initialize the Neo4j connector.

        Args:
            uri: Connection URI for Neo4j database.
            user: Database username.
            password: Database password.
        """
        self._uri = uri
        self._user = user
        self._password = password

    async def connect(self):
        """Establish connection to Neo4j database with thread-safety.

        Raises:
            Exception: If connection fails.
        """
        async with self._lock:
            if not self._driver:
                logger.info(f"Attempting to connect to Neo4j at {self._uri}")
                try:
                    self._driver = AsyncGraphDatabase.driver(self._uri, auth=basic_auth(self._user, self._password))
                    await self._driver.verify_connectivity()
                    logger.info("Neo4j connection successful.")
                except Exception as e:
                    logger.error(f"Failed to connect to Neo4j: {e}")
                    self._driver = None
                    raise

    async def close(self):
        """Close the Neo4j connection with thread-safety."""
        async with self._lock:
            if self._driver:
                await self._driver.close()
                self._driver = None
                logger.info("Neo4j connection closed.")

    async def query(self, query: str, parameters: dict = None, database: str = None):
        """Execute a Cypher query against the Neo4j database.

        Args:
            query: Cypher query string.
            parameters: Optional query parameters.
            database: Optional database name.

        Returns:
            Tuple of (records, summary).
        """
        if not self._driver:
            await self.connect()

        async with self._driver.session(database=database) as session:
            result = await session.run(query, parameters)
            records = [record async for record in result]
            summary = await result.consume()
            return records, summary
