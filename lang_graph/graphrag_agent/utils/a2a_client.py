"""
A2A (Application-to-Application) Protocol Client

This module provides A2A protocol compliant HTTP communication
for agent-to-agent and agent-to-application interactions.
"""

import json
import logging
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class A2AMessage:
    """A2A Protocol Message Structure"""
    message_id: str
    timestamp: str
    source_agent: str
    target_agent: str
    message_type: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    ttl: int = 300  # Time to live in seconds


class A2AClient:
    """A2A Protocol compliant HTTP client"""
    
    def __init__(self, agent_id: str, base_url: str = None):
        self.agent_id = agent_id
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
        # A2A Protocol headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': f'A2A-Agent/{agent_id}',
            'X-A2A-Version': '1.0',
            'X-A2A-Protocol': 'A2A'
        })
    
    def send_message(self, target_agent: str, message_type: str, payload: Dict[str, Any], 
                    correlation_id: str = None, priority: int = 1, ttl: int = 300) -> Dict[str, Any]:
        """Send A2A compliant message"""
        try:
            message = A2AMessage(
                message_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat() + 'Z',
                source_agent=self.agent_id,
                target_agent=target_agent,
                message_type=message_type,
                payload=payload,
                correlation_id=correlation_id,
                priority=priority,
                ttl=ttl
            )
            
            # Convert to dict for JSON serialization
            message_dict = {
                'message_id': message.message_id,
                'timestamp': message.timestamp,
                'source_agent': message.source_agent,
                'target_agent': message.target_agent,
                'message_type': message.message_type,
                'payload': message.payload,
                'correlation_id': message.correlation_id,
                'priority': message.priority,
                'ttl': message.ttl
            }
            
            if self.base_url:
                response = self.session.post(
                    f"{self.base_url}/a2a/message",
                    json=message_dict,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            else:
                # Local processing - return mock response
                self.logger.info(f"A2A Message sent: {message_type} to {target_agent}")
                return {
                    'status': 'success',
                    'message_id': message.message_id,
                    'timestamp': message.timestamp
                }
                
        except Exception as e:
            self.logger.error(f"A2A message send failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def send_graph_request(self, target_agent: str, graph_data: Dict[str, Any], 
                          operation: str = "create") -> Dict[str, Any]:
        """Send graph operation request via A2A protocol"""
        payload = {
            'operation': operation,
            'graph_data': graph_data,
            'request_type': 'graph_operation'
        }
        
        return self.send_message(
            target_agent=target_agent,
            message_type='graph_request',
            payload=payload,
            priority=1
        )
    
    def send_query_request(self, target_agent: str, query: str, 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send query request via A2A protocol"""
        payload = {
            'query': query,
            'context': context or {},
            'request_type': 'query'
        }
        
        return self.send_message(
            target_agent=target_agent,
            message_type='query_request',
            payload=payload,
            priority=2
        )
    
    def send_status_request(self, target_agent: str) -> Dict[str, Any]:
        """Send status request via A2A protocol"""
        payload = {
            'request_type': 'status'
        }
        
        return self.send_message(
            target_agent=target_agent,
            message_type='status_request',
            payload=payload,
            priority=3
        )
    
    def send_visualization_request(self, target_agent: str, graph_data: Dict[str, Any],
                                 format: str = "png") -> Dict[str, Any]:
        """Send visualization request via A2A protocol"""
        payload = {
            'graph_data': graph_data,
            'format': format,
            'request_type': 'visualization'
        }
        
        return self.send_message(
            target_agent=target_agent,
            message_type='visualization_request',
            payload=payload,
            priority=2
        )


class A2AServer:
    """A2A Protocol compliant HTTP server for agent communication"""
    
    def __init__(self, agent_id: str, port: int = 8000):
        self.agent_id = agent_id
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.message_handlers = {}
    
    def register_handler(self, message_type: str, handler_func):
        """Register message handler for specific message type"""
        self.message_handlers[message_type] = handler_func
        self.logger.info(f"Registered handler for message type: {message_type}")
    
    def handle_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle incoming A2A message"""
        try:
            # Check TTL
            message_time = datetime.fromisoformat(message.timestamp.replace('Z', '+00:00'))
            current_time = datetime.utcnow()
            if (current_time - message_time).total_seconds() > message.ttl:
                return {
                    'status': 'error',
                    'error': 'Message TTL expired'
                }
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                return handler(message)
            else:
                return {
                    'status': 'error',
                    'error': f'No handler for message type: {message.message_type}'
                }
                
        except Exception as e:
            self.logger.error(f"Message handling failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def start_server(self):
        """Start A2A server (placeholder for actual server implementation)"""
        self.logger.info(f"A2A Server started for agent {self.agent_id} on port {self.port}")
        # In a real implementation, this would start an HTTP server
        # For now, it's just a placeholder
        return True


# A2A Protocol Constants
A2A_MESSAGE_TYPES = {
    'GRAPH_REQUEST': 'graph_request',
    'QUERY_REQUEST': 'query_request',
    'STATUS_REQUEST': 'status_request',
    'VISUALIZATION_REQUEST': 'visualization_request',
    'HEARTBEAT': 'heartbeat',
    'ERROR': 'error'
}

A2A_PRIORITIES = {
    'HIGH': 1,
    'MEDIUM': 2,
    'LOW': 3
}

A2A_DEFAULT_TTL = 300  # 5 minutes
