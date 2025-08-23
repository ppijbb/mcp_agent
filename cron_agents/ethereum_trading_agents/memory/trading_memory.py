"""
Ethereum Trading Memory

This module provides memory management for trading agents:
1. Trading Context Memory
2. Historical Data Memory
3. Strategy Memory
4. Performance Memory
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum

class MemoryType(Enum):
    """Types of memory storage"""
    TRADING_CONTEXT = "trading_context"
    HISTORICAL_DATA = "historical_data"
    STRATEGY = "strategy"
    PERFORMANCE = "performance"
    USER_PREFERENCES = "user_preferences"

@dataclass
class MemoryItem:
    """Individual memory item"""
    key: str
    value: Any
    memory_type: MemoryType
    timestamp: datetime
    ttl: Optional[int] = None  # Time to live in seconds
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['memory_type'] = self.memory_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['memory_type'] = MemoryType(data['memory_type'])
        return cls(**data)

class TradingMemory:
    """Comprehensive memory management for trading agents"""
    
    def __init__(self, redis_client=None, max_memory_size: int = 10000):
        self.redis_client = redis_client
        self.max_memory_size = max_memory_size
        self.memory_cache = {}
        self.memory_stats = {
            'total_items': 0,
            'memory_types': {mem_type.value: 0 for mem_type in MemoryType},
            'last_cleanup': datetime.now()
        }
        self._setup_memory_structure()
    
    def _setup_memory_structure(self):
        """Initialize memory structure"""
        for mem_type in MemoryType:
            self.memory_cache[mem_type.value] = {}
    
    async def store(
        self, 
        key: str, 
        value: Any, 
        memory_type: MemoryType,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a memory item"""
        
        try:
            # Create memory item
            memory_item = MemoryItem(
                key=key,
                value=value,
                memory_type=memory_type,
                timestamp=datetime.now(),
                ttl=ttl,
                metadata=metadata or {}
            )
            
            # Store in local cache
            self.memory_cache[memory_type.value][key] = memory_item
            
            # Store in Redis if available
            if self.redis_client:
                await self._store_in_redis(memory_item)
            
            # Update statistics
            self._update_stats(memory_type, 1)
            
            # Check memory limits
            await self._check_memory_limits()
            
            return True
            
        except Exception as e:
            print(f"Error storing memory item: {e}")
            return False
    
    async def retrieve(
        self, 
        key: str, 
        memory_type: MemoryType,
        default: Any = None
    ) -> Any:
        """Retrieve a memory item"""
        
        try:
            # Try local cache first
            if key in self.memory_cache[memory_type.value]:
                item = self.memory_cache[memory_type.value][key]
                if not self._is_expired(item):
                    return item.value
            
            # Try Redis if available
            if self.redis_client:
                item = await self._retrieve_from_redis(key, memory_type)
                if item and not self._is_expired(item):
                    # Update local cache
                    self.memory_cache[memory_type.value][key] = item
                    return item.value
            
            return default
            
        except Exception as e:
            print(f"Error retrieving memory item: {e}")
            return default
    
    async def search(
        self, 
        memory_type: MemoryType,
        query: str = None,
        filters: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Any]:
        """Search memory items"""
        
        try:
            results = []
            memory_items = self.memory_cache[memory_type.value].values()
            
            for item in memory_items:
                if self._is_expired(item):
                    continue
                
                # Apply filters
                if filters and not self._matches_filters(item, filters):
                    continue
                
                # Apply query search
                if query and not self._matches_query(item, query):
                    continue
                
                results.append(item.value)
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            print(f"Error searching memory: {e}")
            return []
    
    async def update(
        self, 
        key: str, 
        value: Any, 
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory item"""
        
        try:
            if key in self.memory_cache[memory_type.value]:
                item = self.memory_cache[memory_type.value][key]
                item.value = value
                item.timestamp = datetime.now()
                
                if metadata:
                    item.metadata.update(metadata)
                
                # Update Redis if available
                if self.redis_client:
                    await self._store_in_redis(item)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error updating memory item: {e}")
            return False
    
    async def delete(self, key: str, memory_type: MemoryType) -> bool:
        """Delete a memory item"""
        
        try:
            # Remove from local cache
            if key in self.memory_cache[memory_type.value]:
                del self.memory_cache[memory_type.value][key]
                self._update_stats(memory_type, -1)
            
            # Remove from Redis if available
            if self.redis_client:
                await self._delete_from_redis(key, memory_type)
            
            return True
            
        except Exception as e:
            print(f"Error deleting memory item: {e}")
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memory items"""
        
        try:
            cleaned_count = 0
            
            for mem_type in MemoryType:
                expired_keys = []
                
                for key, item in self.memory_cache[mem_type.value].items():
                    if self._is_expired(item):
                        expired_keys.append(key)
                
                # Remove expired items
                for key in expired_keys:
                    del self.memory_cache[mem_type.value][key]
                    cleaned_count += 1
                
                # Update statistics
                self._update_stats(mem_type, -cleaned_count)
            
            self.memory_stats['last_cleanup'] = datetime.now()
            return cleaned_count
            
        except Exception as e:
            print(f"Error cleaning up expired items: {e}")
            return 0
    
    def _is_expired(self, item: MemoryItem) -> bool:
        """Check if memory item is expired"""
        if item.ttl is None:
            return False
        
        expiry_time = item.timestamp + timedelta(seconds=item.ttl)
        return datetime.now() > expiry_time
    
    def _matches_filters(self, item: MemoryItem, filters: Dict[str, Any]) -> bool:
        """Check if item matches filters"""
        for key, value in filters.items():
            if key in item.metadata:
                if item.metadata[key] != value:
                    return False
            else:
                return False
        return True
    
    def _matches_query(self, item: MemoryItem, query: str) -> bool:
        """Check if item matches search query"""
        query_lower = query.lower()
        
        # Search in key
        if query_lower in item.key.lower():
            return True
        
        # Search in value (if string)
        if isinstance(item.value, str) and query_lower in item.value.lower():
            return True
        
        # Search in metadata
        for key, value in item.metadata.items():
            if isinstance(value, str) and query_lower in value.lower():
                return True
        
        return False
    
    def _update_stats(self, memory_type: MemoryType, change: int):
        """Update memory statistics"""
        self.memory_stats['total_items'] += change
        self.memory_stats['memory_types'][memory_type.value] += change
    
    async def _check_memory_limits(self):
        """Check and enforce memory limits"""
        if self.memory_stats['total_items'] > self.max_memory_size:
            await self.cleanup_expired()
            
            # If still over limit, remove oldest items
            if self.memory_stats['total_items'] > self.max_memory_size:
                await self._remove_oldest_items()
    
    async def _remove_oldest_items(self):
        """Remove oldest memory items to stay under limit"""
        try:
            all_items = []
            
            for mem_type in MemoryType:
                for key, item in self.memory_cache[mem_type.value].items():
                    all_items.append((item.timestamp, mem_type, key))
            
            # Sort by timestamp (oldest first)
            all_items.sort(key=lambda x: x[0])
            
            # Remove oldest items until under limit
            items_to_remove = self.memory_stats['total_items'] - self.max_memory_size + 1000
            
            for i in range(min(items_to_remove, len(all_items))):
                timestamp, mem_type, key = all_items[i]
                await self.delete(key, mem_type)
                
        except Exception as e:
            print(f"Error removing oldest items: {e}")
    
    async def _store_in_redis(self, item: MemoryItem):
        """Store item in Redis"""
        if self.redis_client:
            try:
                key = f"trading_memory:{item.memory_type.value}:{item.key}"
                value = json.dumps(item.to_dict())
                
                if item.ttl:
                    await self.redis_client.setex(key, item.ttl, value)
                else:
                    await self.redis_client.set(key, value)
                    
            except Exception as e:
                print(f"Error storing in Redis: {e}")
    
    async def _retrieve_from_redis(self, key: str, memory_type: MemoryType) -> Optional[MemoryItem]:
        """Retrieve item from Redis"""
        if self.redis_client:
            try:
                redis_key = f"trading_memory:{memory_type.value}:{key}"
                value = await self.redis_client.get(redis_key)
                
                if value:
                    data = json.loads(value)
                    return MemoryItem.from_dict(data)
                    
            except Exception as e:
                print(f"Error retrieving from Redis: {e}")
        
        return None
    
    async def _delete_from_redis(self, key: str, memory_type: MemoryType):
        """Delete item from Redis"""
        if self.redis_client:
            try:
                redis_key = f"trading_memory:{memory_type.value}:{key}"
                await self.redis_client.delete(redis_key)
                
            except Exception as e:
                print(f"Error deleting from Redis: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            **self.memory_stats,
            'memory_usage_percent': (self.memory_stats['total_items'] / self.max_memory_size) * 100
        }
    
    async def export_memory(self, memory_type: Optional[MemoryType] = None) -> Dict[str, Any]:
        """Export memory data"""
        try:
            export_data = {}
            
            if memory_type:
                types_to_export = [memory_type]
            else:
                types_to_export = list(MemoryType)
            
            for mem_type in types_to_export:
                export_data[mem_type.value] = {}
                
                for key, item in self.memory_cache[mem_type.value].items():
                    if not self._is_expired(item):
                        export_data[mem_type.value][key] = item.to_dict()
            
            return export_data
            
        except Exception as e:
            print(f"Error exporting memory: {e}")
            return {}
    
    async def import_memory(self, import_data: Dict[str, Any]) -> int:
        """Import memory data"""
        try:
            imported_count = 0
            
            for mem_type_str, items in import_data.items():
                try:
                    mem_type = MemoryType(mem_type_str)
                    
                    for key, item_data in items.items():
                        item = MemoryItem.from_dict(item_data)
                        await self.store(key, item.value, mem_type, item.ttl, item.metadata)
                        imported_count += 1
                        
                except ValueError:
                    print(f"Invalid memory type: {mem_type_str}")
                    continue
            
            return imported_count
            
        except Exception as e:
            print(f"Error importing memory: {e}")
            return 0
