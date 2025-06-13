"""
Drone Scout - Enterprise Drone Control Agent

Advanced autonomous drone fleet management with natural language task processing.

Features:
- 🚁 Multi-drone coordination and fleet management
- 🎯 Natural language task definition (Korean/English)
- 📊 Real-time monitoring and progress tracking
- 🛡️ Safety systems and emergency protocols
- 🔌 Multi-provider hardware support (DJI, Parrot, ArduPilot)
- 📈 Advanced analytics and reporting

Usage:
    from drone_scout import DroneControlAgent
    
    agent = DroneControlAgent()
    result = await agent.execute_task("농장 A구역 작물 상태 점검해줘")
"""

from .drone_control_agent import main

__version__ = "2.0.0"
__author__ = "MCP Agent Team"
__all__ = ["main"] 