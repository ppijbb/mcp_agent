import uuid
from sqlalchemy import (
    create_engine,
    Column,
    String,
    JSON,
    TIMESTAMP,
    ForeignKey,
    DECIMAL,
    INTEGER,
    BOOLEAN
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class AgentSession(Base):
    __tablename__ = 'agent_sessions'
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False) # In a real app, this would be a FK to a users table
    agent_type = Column(String(50), nullable=False)
    consensus_data = Column(JSON)
    a2a_messages = Column(JSON)
    mcp_responses = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    consensus_logs = relationship("AgentConsensusLog", back_populates="session")
    mcp_logs = relationship("MCPInteractionLog", back_populates="session")

class AgentConsensusLog(Base):
    __tablename__ = 'agent_consensus_logs'
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('agent_sessions.session_id'))
    decision_point = Column(String(100), nullable=False)
    participating_agents = Column(JSON)
    consensus_result = Column(JSON)
    confidence_score = Column(DECIMAL(3, 2))
    processing_time_ms = Column(INTEGER)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    session = relationship("AgentSession", back_populates="consensus_logs")

class MCPInteractionLog(Base):
    __tablename__ = 'mcp_interaction_logs'
    
    interaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('agent_sessions.session_id'))
    server_name = Column(String(50), nullable=False)
    capability_used = Column(String(100), nullable=False)
    request_payload = Column(JSON)
    response_data = Column(JSON)
    response_time_ms = Column(INTEGER)
    success = Column(BOOLEAN, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    session = relationship("AgentSession", back_populates="mcp_logs")

# Example of how to set up the database engine and session
# This part would typically be in a separate database configuration file
def setup_database():
    DATABASE_URL = "postgresql+psycopg2://user:password@localhost/hsp_agent_db"
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal() 