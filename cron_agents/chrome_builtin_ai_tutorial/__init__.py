"""
Chrome Built-in AI (Gemini Nano) 튜토리얼.

다른 agent에서 사용 시:
  sys.path.insert(0, "/path/to/cron_agents")
  from chrome_builtin_ai_tutorial.runner import run_builtin_ai_task_sync, run_builtin_ai_task
"""

from chrome_builtin_ai_tutorial.runner import (
    run_builtin_ai_task,
    run_builtin_ai_task_sync,
)

__all__ = ["run_builtin_ai_task", "run_builtin_ai_task_sync"]
