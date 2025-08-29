"""
Data Manager for Local Researcher

This module provides data storage, retrieval, and management functionality
for research data, reports, and metadata.
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import hashlib
import shutil

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data storage and retrieval for the Local Researcher system."""
    
    def __init__(self, config_manager):
        """Initialize the data manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.base_path = Path(self.config_manager.get("output.directory", "./outputs"))
        self.data_path = Path(self.config_manager.get("database.sqlite.path", "./data/local_researcher.db"))
        self.cache_path = Path(self.config_manager.get("cache.directory", "./cache"))
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize database
        self._init_database()
        
        logger.info("Data Manager initialized")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.base_path,
            self.data_path.parent,
            self.cache_path,
            self.base_path / "reports",
            self.base_path / "data",
            self.base_path / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            # Create research projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research_projects (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    domain TEXT,
                    depth TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Create research results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research_results (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    step_name TEXT,
                    result_data TEXT,
                    execution_time REAL,
                    created_at TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES research_projects (id)
                )
            ''')
            
            # Create reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    report_type TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (project_id) REFERENCES research_projects (id)
                )
            ''')
            
            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def save_research_project(self, project_data: Dict[str, Any]) -> str:
        """Save a new research project.
        
        Args:
            project_data: Project data dictionary
            
        Returns:
            Project ID
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            project_id = project_data.get("id", self._generate_id())
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO research_projects 
                (id, topic, domain, depth, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                project_id,
                project_data.get("topic"),
                project_data.get("domain"),
                project_data.get("depth"),
                project_data.get("status", "pending"),
                now,
                now,
                json.dumps(project_data.get("metadata", {}))
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved research project: {project_id}")
            return project_id
            
        except Exception as e:
            logger.error(f"Failed to save research project: {e}")
            raise
    
    def get_research_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a research project by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            Project data or None if not found
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, topic, domain, depth, status, created_at, updated_at, metadata
                FROM research_projects WHERE id = ?
            ''', (project_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "topic": row[1],
                    "domain": row[2],
                    "depth": row[3],
                    "status": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "metadata": json.loads(row[7]) if row[7] else {}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get research project: {e}")
            return None
    
    def update_research_project(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """Update a research project.
        
        Args:
            project_id: Project ID
            updates: Dictionary of updates
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            # Build update query dynamically
            update_fields = []
            values = []
            
            for key, value in updates.items():
                if key in ["topic", "domain", "depth", "status"]:
                    update_fields.append(f"{key} = ?")
                    values.append(value)
            
            if update_fields:
                update_fields.append("updated_at = ?")
                values.append(datetime.now().isoformat())
                values.append(project_id)
                
                query = f"UPDATE research_projects SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, values)
                
                conn.commit()
                conn.close()
                
                logger.info(f"Updated research project: {project_id}")
                return True
            
            conn.close()
            return False
            
        except Exception as e:
            logger.error(f"Failed to update research project: {e}")
            return False
    
    def save_research_result(self, result_data: Dict[str, Any]) -> str:
        """Save a research result.
        
        Args:
            result_data: Result data dictionary
            
        Returns:
            Result ID
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            result_id = result_data.get("id", self._generate_id())
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO research_results 
                (id, project_id, step_name, result_data, execution_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                result_id,
                result_data.get("project_id"),
                result_data.get("step_name"),
                json.dumps(result_data.get("result_data", {})),
                result_data.get("execution_time", 0.0),
                now
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved research result: {result_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Failed to save research result: {e}")
            raise
    
    def get_research_results(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all results for a research project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of result data
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, step_name, result_data, execution_time, created_at
                FROM research_results WHERE project_id = ?
                ORDER BY created_at ASC
            ''', (project_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "step_name": row[1],
                    "result_data": json.loads(row[2]) if row[2] else {},
                    "execution_time": row[3],
                    "created_at": row[4]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get research results: {e}")
            return []
    
    def save_report(self, report_data: Dict[str, Any]) -> str:
        """Save a research report.
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Report ID
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            report_id = report_data.get("id", self._generate_id())
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO reports 
                (id, project_id, report_type, file_path, file_size, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                report_id,
                report_data.get("project_id"),
                report_data.get("report_type"),
                report_data.get("file_path"),
                report_data.get("file_size", 0),
                now,
                json.dumps(report_data.get("metadata", {}))
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved report: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise
    
    def get_reports(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all reports for a research project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of report data
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, report_type, file_path, file_size, created_at, metadata
                FROM reports WHERE project_id = ?
                ORDER BY created_at DESC
            ''', (project_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            reports = []
            for row in rows:
                reports.append({
                    "id": row[0],
                    "report_type": row[1],
                    "file_path": row[2],
                    "file_size": row[3],
                    "created_at": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {}
                })
            
            return reports
            
        except Exception as e:
            logger.error(f"Failed to get reports: {e}")
            return []
    
    def save_file(self, file_path: str, content: Union[str, bytes], project_id: str = None) -> str:
        """Save a file to the output directory.
        
        Args:
            file_path: Relative file path
            content: File content (string or bytes)
            project_id: Optional project ID for organization
            
        Returns:
            Full file path
        """
        try:
            # Create project-specific directory if project_id is provided
            if project_id:
                target_dir = self.base_path / "reports" / project_id
            else:
                target_dir = self.base_path / "reports"
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure file path is safe
            safe_path = Path(file_path).name
            full_path = target_dir / safe_path
            
            # Write file content
            if isinstance(content, str):
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                with open(full_path, 'wb') as f:
                    f.write(content)
            
            logger.info(f"Saved file: {full_path}")
            return str(full_path)
            
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise
    
    def read_file(self, file_path: str) -> Optional[Union[str, bytes]]:
        """Read a file from the output directory.
        
        Args:
            file_path: File path
            
        Returns:
            File content or None if not found
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return None
            
            # Try to read as text first, then as binary
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(path, 'rb') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file.
        
        Args:
            file_path: File path to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            path = Path(file_path)
            
            if path.exists():
                path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set a cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if set successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            now = datetime.now()
            expires_at = datetime.fromtimestamp(now.timestamp() + ttl)
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (
                key,
                json.dumps(value),
                now.isoformat(),
                expires_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get a cache value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT value, expires_at FROM cache WHERE key = ?
            ''', (key,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                value, expires_at = row
                expires_at = datetime.fromisoformat(expires_at)
                
                if datetime.now() < expires_at:
                    return json.loads(value)
                else:
                    # Remove expired cache entry
                    self.delete_cache(key)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cache: {e}")
            return None
    
    def delete_cache(self, key: str) -> bool:
        """Delete a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")
            return False
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute('DELETE FROM cache WHERE expires_at < ?', (now,))
            
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {deleted_count} expired cache entries")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            return 0
    
    def list_research_projects(self, status_filter: str = None) -> List[Dict[str, Any]]:
        """List all research projects with optional status filtering.
        
        Args:
            status_filter: Optional status filter
            
        Returns:
            List of project data
        """
        try:
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            if status_filter:
                cursor.execute('''
                    SELECT id, topic, domain, depth, status, created_at, updated_at
                    FROM research_projects WHERE status = ?
                    ORDER BY created_at DESC
                ''', (status_filter,))
            else:
                cursor.execute('''
                    SELECT id, topic, domain, depth, status, created_at, updated_at
                    FROM research_projects
                    ORDER BY created_at DESC
                ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            projects = []
            for row in rows:
                projects.append({
                    "id": row[0],
                    "topic": row[1],
                    "domain": row[2],
                    "depth": row[3],
                    "status": row[4],
                    "created_at": row[5],
                    "updated_at": row[6]
                })
            
            return projects
            
        except Exception as e:
            logger.error(f"Failed to list research projects: {e}")
            return []
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old research data.
        
        Args:
            days_old: Age threshold in days
            
        Returns:
            Number of items cleaned up
        """
        try:
            cutoff_date = datetime.fromtimestamp(
                datetime.now().timestamp() - (days_old * 24 * 3600)
            ).isoformat()
            
            conn = sqlite3.connect(self.data_path)
            cursor = conn.cursor()
            
            # Count old projects
            cursor.execute('''
                SELECT COUNT(*) FROM research_projects 
                WHERE created_at < ?
            ''', (cutoff_date,))
            
            old_count = cursor.fetchone()[0]
            
            # Delete old data
            cursor.execute('DELETE FROM research_projects WHERE created_at < ?', (cutoff_date,))
            cursor.execute('DELETE FROM research_results WHERE created_at < ?', (cutoff_date,))
            cursor.execute('DELETE FROM reports WHERE created_at < ?', (cutoff_date,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {old_count} old research projects")
            return old_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def _generate_id(self) -> str:
        """Generate a unique ID.
        
        Returns:
            Unique ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"{timestamp}_{random_suffix}"
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information.
        
        Returns:
            Storage information dictionary
        """
        try:
            # Calculate directory sizes
            base_size = self._get_directory_size(self.base_path)
            cache_size = self._get_directory_size(self.cache_path)
            db_size = self.data_path.stat().st_size if self.data_path.exists() else 0
            
            return {
                "base_directory_size": base_size,
                "cache_directory_size": cache_size,
                "database_size": db_size,
                "total_size": base_size + cache_size + db_size,
                "base_path": str(self.base_path),
                "cache_path": str(self.cache_path),
                "database_path": str(self.data_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {}
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate directory size in bytes.
        
        Args:
            directory: Directory path
            
        Returns:
            Directory size in bytes
        """
        total_size = 0
        
        try:
            for path in directory.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        except Exception as e:
            logger.warning(f"Could not calculate size for {directory}: {e}")
        
        return total_size
