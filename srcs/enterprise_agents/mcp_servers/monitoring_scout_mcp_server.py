#!/usr/bin/env python3
"""
Real System Monitoring MCP Server for AIOps
Provides actual system metrics using psutil library
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monitoring-scout")

class SystemMonitor:
    """Real system monitoring using psutil"""
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get detailed CPU information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            return {
                "cpu_percent_total": psutil.cpu_percent(interval=0.1),
                "cpu_percent_per_core": cpu_percent,
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_frequency": {
                    "current": cpu_freq.current if cpu_freq else None,
                    "min": cpu_freq.min if cpu_freq else None,
                    "max": cpu_freq.max if cpu_freq else None,
                } if cpu_freq else None,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            }
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get detailed memory information"""
        try:
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()
            
            return {
                "virtual_memory": {
                    "total": virtual_mem.total,
                    "available": virtual_mem.available,
                    "used": virtual_mem.used,
                    "free": virtual_mem.free,
                    "percent": virtual_mem.percent,
                    "total_gb": round(virtual_mem.total / (1024**3), 2),
                    "used_gb": round(virtual_mem.used / (1024**3), 2),
                    "available_gb": round(virtual_mem.available / (1024**3), 2),
                },
                "swap_memory": {
                    "total": swap_mem.total,
                    "used": swap_mem.used,
                    "free": swap_mem.free,
                    "percent": swap_mem.percent,
                    "total_gb": round(swap_mem.total / (1024**3), 2),
                    "used_gb": round(swap_mem.used / (1024**3), 2),
                }
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_disk_info() -> Dict[str, Any]:
        """Get disk usage information"""
        try:
            disk_usage = {}
            disk_partitions = psutil.disk_partitions()
            
            for partition in disk_partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        "device": partition.device,
                        "fstype": partition.fstype,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": round((usage.used / usage.total) * 100, 2),
                        "total_gb": round(usage.total / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                    }
                except PermissionError:
                    # Skip partitions we can't access
                    continue
            
            return {"disk_partitions": disk_usage}
        except Exception as e:
            logger.error(f"Error getting disk info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_network_info() -> Dict[str, Any]:
        """Get network interface information"""
        try:
            net_io = psutil.net_io_counters(pernic=True)
            net_connections = len(psutil.net_connections())
            
            network_info = {
                "total_connections": net_connections,
                "interfaces": {}
            }
            
            for interface, stats in net_io.items():
                network_info["interfaces"][interface] = {
                    "bytes_sent": stats.bytes_sent,
                    "bytes_recv": stats.bytes_recv,
                    "packets_sent": stats.packets_sent,
                    "packets_recv": stats.packets_recv,
                    "errin": stats.errin,
                    "errout": stats.errout,
                    "dropin": stats.dropin,
                    "dropout": stats.dropout,
                }
            
            return network_info
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_top_processes(limit: int = 10) -> List[Dict[str, Any]]:
        """Get top processes by CPU and memory usage"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info', 'status', 'create_time']):
                try:
                    proc_info = proc.info
                    proc_info['memory_mb'] = round(proc_info['memory_info'].rss / (1024*1024), 2) if proc_info['memory_info'] else 0
                    proc_info['cpu_percent'] = proc.cpu_percent()
                    processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Sort by CPU usage
            processes_by_cpu = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:limit]
            # Sort by memory usage
            processes_by_memory = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:limit]
            
            return {
                "top_by_cpu": processes_by_cpu,
                "top_by_memory": processes_by_memory,
                "total_processes": len(processes)
            }
        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get general system information"""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            return {
                "hostname": psutil.os.uname().nodename if hasattr(psutil.os, 'uname') else "unknown",
                "platform": psutil.os.name,
                "boot_time": boot_time.isoformat(),
                "uptime_seconds": (datetime.now() - boot_time).total_seconds(),
                "users": [{"name": user.name, "terminal": user.terminal, "host": user.host} 
                         for user in psutil.users()],
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}

# Initialize the MCP server
server = Server("monitoring-scout")

@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available monitoring tools"""
    return ListToolsResult(
        tools=[
            Tool(
                name="get_system_snapshot",
                description="Get a comprehensive system snapshot including CPU, memory, and top processes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_processes": {
                            "type": "boolean",
                            "description": "Whether to include top processes information",
                            "default": True
                        },
                        "process_limit": {
                            "type": "integer",
                            "description": "Number of top processes to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        }
                    }
                }
            ),
            Tool(
                name="diagnose_high_cpu",
                description="Diagnose high CPU usage and provide recommendations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "threshold": {
                            "type": "number",
                            "description": "CPU usage threshold to consider as high",
                            "default": 80.0,
                            "minimum": 0.0,
                            "maximum": 100.0
                        }
                    }
                }
            )
        ]
    )

@server.call_tool()
async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls"""
    try:
        monitor = SystemMonitor()
        
        if request.name == "get_system_snapshot":
            # Get comprehensive system snapshot
            args = request.arguments or {}
            include_processes = args.get("include_processes", True)
            process_limit = args.get("process_limit", 10)
            
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "cpu": monitor.get_cpu_info(),
                "memory": monitor.get_memory_info(),
            }
            
            if include_processes:
                snapshot["processes"] = monitor.get_top_processes(process_limit)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"System Snapshot:\n{json.dumps(snapshot, indent=2)}"
                )]
            )
        
        elif request.name == "diagnose_high_cpu":
            args = request.arguments or {}
            threshold = args.get("threshold", 80.0)
            
            # Get current CPU usage
            cpu_info = monitor.get_cpu_info()
            processes = monitor.get_top_processes(15)
            
            diagnosis = {
                "timestamp": datetime.now().isoformat(),
                "threshold": threshold,
                "current_cpu_usage": cpu_info.get("cpu_percent_total", 0),
                "is_high_cpu": cpu_info.get("cpu_percent_total", 0) > threshold,
                "cpu_details": cpu_info,
                "top_cpu_processes": processes.get("top_by_cpu", [])[:5],
                "recommendations": []
            }
            
            # Generate recommendations
            if diagnosis["is_high_cpu"]:
                top_process = processes.get("top_by_cpu", [{}])[0] if processes.get("top_by_cpu") else {}
                if top_process:
                    diagnosis["recommendations"].append(
                        f"High CPU process detected: {top_process.get('name', 'unknown')} "
                        f"(PID: {top_process.get('pid', 'unknown')}) using "
                        f"{top_process.get('cpu_percent', 0):.1f}% CPU"
                    )
                
                diagnosis["recommendations"].extend([
                    "Consider investigating the top CPU-consuming processes",
                    "Check for runaway processes or infinite loops",
                    "Monitor system for sustained high CPU usage",
                    "Consider process restart or system reboot if necessary"
                ])
            else:
                diagnosis["recommendations"].append("CPU usage is within normal range")
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"CPU Diagnosis:\n{json.dumps(diagnosis, indent=2)}"
                )]
            )
        
        else:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Unknown tool: {request.name}"
                )],
                isError=True
            )
    
    except Exception as e:
        logger.error(f"Error in tool call {request.name}: {e}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Error executing {request.name}: {str(e)}"
            )],
            isError=True
        )

async def main():
    """Run the monitoring scout MCP server"""
    logger.info("Starting Monitoring Scout MCP Server...")
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="monitoring-scout",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 