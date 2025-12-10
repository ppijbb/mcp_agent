"""
Environment Manager for the Kimi-K2 Agentic Data Synthesis System

Manages simulation environments, resource allocation, and environment state tracking.
"""

from typing import List, Dict, Any, Optional
from ..models.simulation import EnvironmentState
import logging
from datetime import datetime
import uuid
import threading
import time

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """
    Manages simulation environments and resources.
    
    Responsibilities:
    - Virtual environment creation and management
    - Resource allocation and tracking
    - Environment state management
    - Environment isolation and cleanup
    """
    
    def __init__(self):
        self.environments: Dict[str, Dict[str, Any]] = {}
        self.resource_pools: Dict[str, Dict[str, Any]] = {}
        self.environment_states: Dict[str, List[EnvironmentState]] = {}
        self.active_environments: Dict[str, Dict[str, Any]] = {}
        self.max_environments = 50
        self.environment_timeout = 1800  # 30 minutes
        
        # Initialize default resource pools
        self._initialize_resource_pools()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _initialize_resource_pools(self) -> None:
        """Initialize default resource pools"""
        self.resource_pools = {
            "cpu": {
                "total": 8,
                "available": 8,
                "allocated": 0,
                "unit": "cores"
            },
            "memory": {
                "total": 16384,  # 16GB
                "available": 16384,
                "allocated": 0,
                "unit": "MB"
            },
            "gpu": {
                "total": 2,
                "available": 2,
                "allocated": 0,
                "unit": "devices"
            },
            "storage": {
                "total": 1000000,  # 1TB
                "available": 1000000,
                "allocated": 0,
                "unit": "MB"
            }
        }
    
    def create_environment(self, session_id: str, requirements: Dict[str, Any] = None) -> Optional[str]:
        """Create a new simulation environment"""
        if len(self.active_environments) >= self.max_environments:
            logger.warning("Maximum number of environments reached")
            return None
        
        # Check resource availability
        if not self._check_resource_availability(requirements or {}):
            logger.warning("Insufficient resources for environment creation")
            return None
        
        environment_id = str(uuid.uuid4())
        
        # Allocate resources
        allocated_resources = self._allocate_resources(requirements or {})
        
        # Create environment
        environment = {
            "id": environment_id,
            "session_id": session_id,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "status": "active",
            "allocated_resources": allocated_resources,
            "requirements": requirements or {},
            "metadata": {}
        }
        
        self.environments[environment_id] = environment
        self.active_environments[environment_id] = environment
        self.environment_states[environment_id] = []
        
        logger.info(f"Created environment: {environment_id} for session: {session_id}")
        return environment_id
    
    def _check_resource_availability(self, requirements: Dict[str, Any]) -> bool:
        """Check if required resources are available"""
        for resource_type, required_amount in requirements.items():
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                if pool["available"] < required_amount:
                    return False
        return True
    
    def _allocate_resources(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources for an environment"""
        allocated = {}
        
        for resource_type, required_amount in requirements.items():
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                allocated_amount = min(required_amount, pool["available"])
                
                pool["available"] -= allocated_amount
                pool["allocated"] += allocated_amount
                
                allocated[resource_type] = allocated_amount
        
        return allocated
    
    def get_environment(self, environment_id: str) -> Optional[Dict[str, Any]]:
        """Get environment by ID"""
        return self.environments.get(environment_id)
    
    def get_environment_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get environment by session ID"""
        for env in self.environments.values():
            if env["session_id"] == session_id:
                return env
        return None
    
    def update_environment_state(self, environment_id: str, state: EnvironmentState) -> bool:
        """Update environment state"""
        if environment_id not in self.environments:
            logger.warning(f"Environment not found: {environment_id}")
            return False
        
        # Add state to history
        if environment_id not in self.environment_states:
            self.environment_states[environment_id] = []
        
        self.environment_states[environment_id].append(state)
        
        # Update last accessed time
        if environment_id in self.active_environments:
            self.active_environments[environment_id]["last_accessed"] = datetime.utcnow()
        
        logger.debug(f"Updated environment state: {environment_id}")
        return True
    
    def get_environment_states(self, environment_id: str) -> List[EnvironmentState]:
        """Get all states for an environment"""
        return self.environment_states.get(environment_id, [])
    
    def get_current_environment_state(self, environment_id: str) -> Optional[EnvironmentState]:
        """Get current state for an environment"""
        states = self.get_environment_states(environment_id)
        return states[-1] if states else None
    
    def add_environment_variable(self, environment_id: str, key: str, value: Any) -> bool:
        """Add or update environment variable"""
        env = self.get_environment(environment_id)
        if not env:
            return False
        
        if "variables" not in env:
            env["variables"] = {}
        
        env["variables"][key] = value
        env["last_accessed"] = datetime.utcnow()
        
        return True
    
    def get_environment_variable(self, environment_id: str, key: str) -> Optional[Any]:
        """Get environment variable"""
        env = self.get_environment(environment_id)
        if not env or "variables" not in env:
            return None
        
        return env["variables"].get(key)
    
    def list_environments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List environments with optional filtering"""
        environments = list(self.environments.values())
        
        if status:
            environments = [env for env in environments if env["status"] == status]
        
        return environments
    
    def destroy_environment(self, environment_id: str) -> bool:
        """Destroy an environment and free resources"""
        env = self.get_environment(environment_id)
        if not env:
            logger.warning(f"Environment not found: {environment_id}")
            return False
        
        # Free allocated resources
        self._free_resources(env["allocated_resources"])
        
        # Update status
        env["status"] = "destroyed"
        env["destroyed_at"] = datetime.utcnow()
        
        # Remove from active environments
        if environment_id in self.active_environments:
            del self.active_environments[environment_id]
        
        logger.info(f"Destroyed environment: {environment_id}")
        return True
    
    def _free_resources(self, allocated_resources: Dict[str, Any]) -> None:
        """Free allocated resources"""
        for resource_type, allocated_amount in allocated_resources.items():
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                pool["available"] += allocated_amount
                pool["allocated"] -= allocated_amount
    
    def pause_environment(self, environment_id: str) -> bool:
        """Pause an environment"""
        env = self.get_environment(environment_id)
        if not env:
            return False
        
        env["status"] = "paused"
        env["paused_at"] = datetime.utcnow()
        
        if environment_id in self.active_environments:
            del self.active_environments[environment_id]
        
        logger.info(f"Paused environment: {environment_id}")
        return True
    
    def resume_environment(self, environment_id: str) -> bool:
        """Resume a paused environment"""
        env = self.get_environment(environment_id)
        if not env or env["status"] != "paused":
            return False
        
        env["status"] = "active"
        env["resumed_at"] = datetime.utcnow()
        env["last_accessed"] = datetime.utcnow()
        
        self.active_environments[environment_id] = env
        
        logger.info(f"Resumed environment: {environment_id}")
        return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        usage = {}
        
        for resource_type, pool in self.resource_pools.items():
            usage[resource_type] = {
                "total": pool["total"],
                "allocated": pool["allocated"],
                "available": pool["available"],
                "utilization_percent": (pool["allocated"] / pool["total"]) * 100 if pool["total"] > 0 else 0,
                "unit": pool["unit"]
            }
        
        return usage
    
    def get_environment_statistics(self) -> Dict[str, Any]:
        """Get environment statistics"""
        stats = {
            "total_environments": len(self.environments),
            "active_environments": len(self.active_environments),
            "paused_environments": len([env for env in self.environments.values() if env["status"] == "paused"]),
            "destroyed_environments": len([env for env in self.environments.values() if env["status"] == "destroyed"]),
            "average_lifetime": 0.0,
            "resource_utilization": self.get_resource_usage()
        }
        
        # Calculate average lifetime
        if self.environments:
            total_lifetime = 0.0
            count = 0
            
            for env in self.environments.values():
                if "destroyed_at" in env:
                    lifetime = (env["destroyed_at"] - env["created_at"]).total_seconds()
                    total_lifetime += lifetime
                    count += 1
            
            if count > 0:
                stats["average_lifetime"] = total_lifetime / count
        
        return stats
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_expired_environments()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_environments(self) -> None:
        """Clean up expired environments"""
        current_time = datetime.utcnow()
        expired_environments = []
        
        for env_id, env in self.active_environments.items():
            time_since_access = (current_time - env["last_accessed"]).total_seconds()
            if time_since_access > self.environment_timeout:
                expired_environments.append(env_id)
        
        for env_id in expired_environments:
            logger.info(f"Cleaning up expired environment: {env_id}")
            self.destroy_environment(env_id)
    
    def backup_environment_state(self, environment_id: str) -> Optional[Dict[str, Any]]:
        """Create a backup of environment state"""
        env = self.get_environment(environment_id)
        if not env:
            return None
        
        states = self.get_environment_states(environment_id)
        
        backup = {
            "environment": env.copy(),
            "states": [state.model_dump() for state in states],
            "backup_created_at": datetime.utcnow().isoformat()
        }
        
        return backup
    
    def restore_environment_state(self, backup: Dict[str, Any]) -> Optional[str]:
        """Restore environment from backup"""
        try:
            env_data = backup["environment"]
            environment_id = env_data["id"]
            
            # Restore environment
            self.environments[environment_id] = env_data
            
            # Restore states
            states = []
            for state_data in backup["states"]:
                state = EnvironmentState(**state_data)
                states.append(state)
            
            self.environment_states[environment_id] = states
            
            # Add to active environments if status is active
            if env_data["status"] == "active":
                self.active_environments[environment_id] = env_data
            
            logger.info(f"Restored environment from backup: {environment_id}")
            return environment_id
            
        except Exception as e:
            logger.error(f"Failed to restore environment: {e}")
            return None 