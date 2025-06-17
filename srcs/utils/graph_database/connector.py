from neo4j import GraphDatabase
from srcs.utils.graph_database import config
import asyncio
from neo4j import AsyncGraphDatabase, basic_auth
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnector:
    _driver = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._uri = "bolt://localhost:7687"
        self._user = "neo4j"
        self._password = "password"

    async def connect(self):
        async with self._lock:
            if not self._driver:
                logger.info(f"Attempting to connect to Neo4j at {self._uri}")
                try:
                    self._driver = AsyncGraphDatabase.driver(self._uri, auth=basic_auth(self._user, self._password))
                    await self._driver.verify_connectivity()
                    logger.info("Neo4j connection successful.")
                except Exception as e:
                    logger.error(f"Failed to connect to Neo4j: {e}")
                    self._driver = None # Ensure driver is None on failure
                    raise

    async def close(self):
        async with self._lock:
            if self._driver:
                await self._driver.close()
                self._driver = None
                logger.info("Neo4j connection closed.")

    async def query(self, query, parameters=None, database=None):
        if not self._driver:
            await self.connect()
        
        async with self._driver.session(database=database) as session:
            result = await session.run(query, parameters)
            # Use a list comprehension with async for to handle the async generator
            records = [record async for record in result]
            summary = await result.consume()
            return records, summary

# The global instance is removed to prevent async issues.
# Each part of the application should manage its own connector instance. 