#!/usr/bin/env python3
"""
Task Queue System for Parallel Agent Execution

작업 큐 및 우선순위 관리 시스템.
병렬 실행 가능한 작업 그룹을 식별하고, 의존성을 고려하여 작업을 분배합니다.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TaskQueueItem:
    """작업 큐 아이템."""
    task_id: str
    task: Dict[str, Any]
    priority: int = 0  # 높을수록 우선순위 높음
    dependencies: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()


class TaskQueue:
    """작업 큐 및 우선순위 관리 시스템."""
    
    def __init__(self):
        """초기화."""
        self.tasks: Dict[str, TaskQueueItem] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.completed_tasks: Set[str] = set()
        self.parallel_groups: List[List[str]] = []
        
    def add_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """작업들을 큐에 추가."""
        logger.info(f"Adding {len(tasks)} tasks to queue")
        
        for task in tasks:
            # task_id 또는 id 필드에서 ID 추출
            task_id = task.get('task_id') or task.get('id', '')
            if not task_id:
                # ID가 없으면 생성
                import uuid
                task_id = str(uuid.uuid4())
                task['task_id'] = task_id
                task['id'] = task_id
                logger.warning(f"Generated task_id for task without ID: {task_id}")
            
            # TaskQueueItem 생성
            queue_item = TaskQueueItem(
                task_id=task_id,
                task=task,
                priority=task.get('priority', 0),
                dependencies=task.get('dependencies', [])
            )
            
            self.tasks[task_id] = queue_item
            self.dependency_graph[task_id] = task.get('dependencies', [])
        
        # 병렬 그룹 식별
        self._identify_parallel_groups()
    
    def _identify_parallel_groups(self) -> None:
        """병렬 실행 가능한 작업 그룹 식별."""
        self.parallel_groups = []
        processed = set()
        
        # 의존성 그래프 기반으로 병렬 그룹 생성
        for task_id, dependencies in self.dependency_graph.items():
            if task_id in processed:
                continue
            
            # 의존성이 없거나 모든 의존성이 완료된 작업들로 그룹 생성
            if not dependencies or all(dep in self.completed_tasks for dep in dependencies):
                group = [task_id]
                
                # 다른 독립적인 작업들 찾기
                for other_task_id, other_deps in self.dependency_graph.items():
                    if (other_task_id != task_id and 
                        other_task_id not in processed and
                        (not other_deps or all(dep in self.completed_tasks for dep in other_deps))):
                        group.append(other_task_id)
                
                if len(group) > 1:
                    self.parallel_groups.append(group)
                    processed.update(group)
        
        logger.info(f"Identified {len(self.parallel_groups)} parallel groups")
    
    def get_next_task_group(self, max_group_size: Optional[int] = None) -> Optional[List[str]]:
        """다음 실행 가능한 작업 그룹 반환."""
        # 병렬 그룹이 있으면 반환
        if self.parallel_groups:
            group = self.parallel_groups[0]
            if max_group_size:
                group = group[:max_group_size]
            self.parallel_groups.pop(0)
            
            # 그룹의 모든 작업이 여전히 실행 가능한지 확인
            ready_tasks = [
                task_id for task_id in group
                if task_id not in self.completed_tasks and
                all(dep in self.completed_tasks for dep in self.dependency_graph.get(task_id, []))
            ]
            
            if ready_tasks:
                return ready_tasks
        
        # 병렬 그룹이 없으면 의존성 없는 단일 작업 반환
        for task_id, dependencies in self.dependency_graph.items():
            if (task_id not in self.completed_tasks and
                all(dep in self.completed_tasks for dep in dependencies)):
                return [task_id]
        
        return None
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 정보 반환."""
        if task_id in self.tasks:
            return self.tasks[task_id].task
        return None
    
    def mark_completed(self, task_id: str) -> None:
        """작업 완료 표시."""
        if task_id in self.tasks:
            self.completed_tasks.add(task_id)
            logger.debug(f"Task {task_id} marked as completed")
            
            # 병렬 그룹 재계산
            self._identify_parallel_groups()
    
    def is_completed(self, task_id: str) -> bool:
        """작업 완료 여부 확인."""
        return task_id in self.completed_tasks
    
    def get_remaining_count(self) -> int:
        """남은 작업 수 반환."""
        return len(self.tasks) - len(self.completed_tasks)
    
    def has_pending_tasks(self) -> bool:
        """대기 중인 작업이 있는지 확인."""
        return self.get_remaining_count() > 0
    
    def get_progress(self) -> Dict[str, Any]:
        """작업 진행 상황 반환."""
        total = len(self.tasks)
        completed = len(self.completed_tasks)
        remaining = total - completed
        progress_percentage = (completed / total * 100) if total > 0 else 0.0
        
        return {
            'total_tasks': total,
            'completed_tasks': completed,
            'remaining_tasks': remaining,
            'progress_percentage': progress_percentage,
            'parallel_groups_count': len(self.parallel_groups)
        }

