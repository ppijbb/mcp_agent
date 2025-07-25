"""
Kubernetes Monitor MCP Server

Kubernetes 클러스터 모니터링을 위한 MCP 서버입니다.
Gemini CLI와 연동하여 K8s 상태를 실시간으로 분석하고 보고합니다.
"""

import asyncio
import json
import subprocess
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.server.stdio


@dataclass
class K8sResource:
    """Kubernetes 리소스 정보"""
    name: str
    namespace: str
    kind: str
    status: str
    age: str
    details: Dict[str, Any]


class K8sMonitorServer:
    """Kubernetes 모니터링 MCP 서버"""
    
    def __init__(self):
        self.server = Server("k8s-monitor")
        self._register_tools()
    
    def _register_tools(self):
        """MCP 도구들 등록"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """사용 가능한 도구 목록 반환"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="get_cluster_status",
                        description="Kubernetes 클러스터 전체 상태 조회",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "namespace": {
                                    "type": "string",
                                    "description": "특정 네임스페이스 (선택사항)"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="get_pod_status",
                        description="Pod 상태 및 로그 조회",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "namespace": {
                                    "type": "string",
                                    "description": "네임스페이스"
                                },
                                "pod_name": {
                                    "type": "string", 
                                    "description": "Pod 이름 (선택사항)"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="get_node_metrics",
                        description="노드 리소스 사용률 조회",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "node_name": {
                                    "type": "string",
                                    "description": "노드 이름 (선택사항)"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="analyze_cluster_health",
                        description="클러스터 건강도 분석 및 예측",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "analysis_type": {
                                    "type": "string",
                                    "enum": ["performance", "security", "capacity"],
                                    "description": "분석 유형"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="get_deployment_status",
                        description="Deployment 상태 및 롤아웃 정보 조회",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "namespace": {
                                    "type": "string",
                                    "description": "네임스페이스"
                                },
                                "deployment_name": {
                                    "type": "string",
                                    "description": "Deployment 이름 (선택사항)"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="monitor_events",
                        description="클러스터 이벤트 모니터링",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "namespace": {
                                    "type": "string",
                                    "description": "네임스페이스 (선택사항)"
                                },
                                "event_type": {
                                    "type": "string",
                                    "enum": ["Warning", "Normal"],
                                    "description": "이벤트 유형 (선택사항)"
                                }
                            }
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """도구 실행"""
            
            if name == "get_cluster_status":
                return await self._get_cluster_status(arguments)
            elif name == "get_pod_status":
                return await self._get_pod_status(arguments)
            elif name == "get_node_metrics":
                return await self._get_node_metrics(arguments)
            elif name == "analyze_cluster_health":
                return await self._analyze_cluster_health(arguments)
            elif name == "get_deployment_status":
                return await self._get_deployment_status(arguments)
            elif name == "monitor_events":
                return await self._monitor_events(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def _get_cluster_status(self, args: Dict[str, Any]) -> CallToolResult:
        """클러스터 상태 조회"""
        namespace = args.get("namespace", "all")
        
        try:
            # kubectl 명령어 실행
            if namespace == "all":
                cmd = ["kubectl", "get", "all", "--all-namespaces", "-o", "wide"]
            else:
                cmd = ["kubectl", "get", "all", "-n", namespace, "-o", "wide"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # 상태 분석
                analysis = self._analyze_cluster_output(result.stdout)
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"## Kubernetes Cluster Status\n\n"
                                 f"**Namespace:** {namespace}\n\n"
                                 f"**Raw Output:**\n```\n{result.stdout}\n```\n\n"
                                 f"**Analysis:**\n{analysis}"
                        )
                    ]
                )
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"❌ Error getting cluster status: {result.stderr}"
                        )
                    ]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ Exception: {str(e)}"
                    )
                ]
            )
    
    async def _get_pod_status(self, args: Dict[str, Any]) -> CallToolResult:
        """Pod 상태 조회"""
        namespace = args.get("namespace", "default")
        pod_name = args.get("pod_name")
        
        try:
            if pod_name:
                # 특정 Pod 조회
                cmd = ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "wide"]
                logs_cmd = ["kubectl", "logs", pod_name, "-n", namespace, "--tail=50"]
            else:
                # 네임스페이스의 모든 Pod 조회
                cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "wide"]
                logs_cmd = None
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = f"## Pod Status - {namespace}\n\n"
                output += f"**Status:**\n```\n{result.stdout}\n```\n\n"
                
                # 로그 조회 (특정 Pod인 경우)
                if pod_name and logs_cmd:
                    logs_result = subprocess.run(logs_cmd, capture_output=True, text=True, timeout=30)
                    if logs_result.returncode == 0:
                        output += f"**Recent Logs:**\n```\n{logs_result.stdout}\n```\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"❌ Error getting pod status: {result.stderr}"
                        )
                    ]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ Exception: {str(e)}"
                    )
                ]
            )
    
    async def _get_node_metrics(self, args: Dict[str, Any]) -> CallToolResult:
        """노드 메트릭 조회"""
        node_name = args.get("node_name")
        
        try:
            if node_name:
                cmd = ["kubectl", "top", "node", node_name]
            else:
                cmd = ["kubectl", "top", "nodes"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # 메트릭 분석
                analysis = self._analyze_node_metrics(result.stdout)
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"## Node Metrics\n\n"
                                 f"**Raw Output:**\n```\n{result.stdout}\n```\n\n"
                                 f"**Analysis:**\n{analysis}"
                        )
                    ]
                )
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"❌ Error getting node metrics: {result.stderr}"
                        )
                    ]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ Exception: {str(e)}"
                    )
                ]
            )
    
    async def _analyze_cluster_health(self, args: Dict[str, Any]) -> CallToolResult:
        """클러스터 건강도 분석"""
        analysis_type = args.get("analysis_type", "performance")
        
        try:
            # 종합적인 클러스터 정보 수집
            cluster_info = await self._collect_cluster_info()
            
            # 분석 유형별 처리
            if analysis_type == "performance":
                analysis = self._analyze_performance(cluster_info)
            elif analysis_type == "security":
                analysis = self._analyze_security(cluster_info)
            elif analysis_type == "capacity":
                analysis = self._analyze_capacity(cluster_info)
            else:
                analysis = "Unknown analysis type"
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"## Cluster Health Analysis - {analysis_type.title()}\n\n{analysis}"
                    )
                ]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ Exception during analysis: {str(e)}"
                    )
                ]
            )
    
    async def _get_deployment_status(self, args: Dict[str, Any]) -> CallToolResult:
        """Deployment 상태 조회"""
        namespace = args.get("namespace", "default")
        deployment_name = args.get("deployment_name")
        
        try:
            if deployment_name:
                cmd = ["kubectl", "get", "deployment", deployment_name, "-n", namespace, "-o", "wide"]
                rollout_cmd = ["kubectl", "rollout", "status", "deployment", deployment_name, "-n", namespace]
            else:
                cmd = ["kubectl", "get", "deployments", "-n", namespace, "-o", "wide"]
                rollout_cmd = None
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = f"## Deployment Status - {namespace}\n\n"
                output += f"**Status:**\n```\n{result.stdout}\n```\n\n"
                
                # 롤아웃 상태 조회
                if deployment_name and rollout_cmd:
                    rollout_result = subprocess.run(rollout_cmd, capture_output=True, text=True, timeout=30)
                    if rollout_result.returncode == 0:
                        output += f"**Rollout Status:**\n```\n{rollout_result.stdout}\n```\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"❌ Error getting deployment status: {result.stderr}"
                        )
                    ]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ Exception: {str(e)}"
                    )
                ]
            )
    
    async def _monitor_events(self, args: Dict[str, Any]) -> CallToolResult:
        """이벤트 모니터링"""
        namespace = args.get("namespace")
        event_type = args.get("event_type")
        
        try:
            cmd = ["kubectl", "get", "events", "--sort-by='.lastTimestamp'"]
            
            if namespace:
                cmd.extend(["-n", namespace])
            else:
                cmd.append("--all-namespaces")
            
            if event_type:
                cmd.extend(["--field-selector", f"type={event_type}"])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # 이벤트 분석
                analysis = self._analyze_events(result.stdout)
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"## Cluster Events\n\n"
                                 f"**Events:**\n```\n{result.stdout}\n```\n\n"
                                 f"**Analysis:**\n{analysis}"
                        )
                    ]
                )
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"❌ Error getting events: {result.stderr}"
                        )
                    ]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ Exception: {str(e)}"
                    )
                ]
            )
    
    def _analyze_cluster_output(self, output: str) -> str:
        """클러스터 출력 분석"""
        lines = output.strip().split('\n')
        if len(lines) <= 1:
            return "No resources found"
        
        # 헤더 제거
        data_lines = lines[1:]
        
        # 상태별 카운트
        status_counts = {}
        namespace_counts = {}
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 4:
                namespace = parts[0]
                name = parts[1]
                status = parts[3]
                
                status_counts[status] = status_counts.get(status, 0) + 1
                namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1
        
        analysis = f"**Resource Summary:**\n"
        analysis += f"- Total Resources: {len(data_lines)}\n"
        analysis += f"- Namespaces: {len(namespace_counts)}\n\n"
        
        analysis += f"**Status Breakdown:**\n"
        for status, count in status_counts.items():
            analysis += f"- {status}: {count}\n"
        
        analysis += f"\n**Namespace Breakdown:**\n"
        for namespace, count in namespace_counts.items():
            analysis += f"- {namespace}: {count}\n"
        
        return analysis
    
    def _analyze_node_metrics(self, output: str) -> str:
        """노드 메트릭 분석"""
        lines = output.strip().split('\n')
        if len(lines) <= 1:
            return "No node metrics available"
        
        # 헤더 제거
        data_lines = lines[1:]
        
        analysis = f"**Node Metrics Summary:**\n"
        analysis += f"- Total Nodes: {len(data_lines)}\n\n"
        
        total_cpu_percent = 0
        total_memory_percent = 0
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 5:
                node_name = parts[0]
                cpu_percent = parts[1]
                memory_percent = parts[3]
                
                analysis += f"- {node_name}: CPU {cpu_percent}, Memory {memory_percent}\n"
                
                # 퍼센트 추출 (예: "100m" -> 0.1, "1Gi" -> 1.0)
                try:
                    if 'm' in cpu_percent:
                        cpu_val = float(cpu_percent.replace('m', '')) / 1000
                    else:
                        cpu_val = float(cpu_percent)
                    total_cpu_percent += cpu_val
                except:
                    pass
                
                try:
                    if 'Mi' in memory_percent:
                        mem_val = float(memory_percent.replace('Mi', '')) / 1024
                    elif 'Gi' in memory_percent:
                        mem_val = float(memory_percent.replace('Gi', ''))
                    else:
                        mem_val = float(memory_percent)
                    total_memory_percent += mem_val
                except:
                    pass
        
        if len(data_lines) > 0:
            avg_cpu = total_cpu_percent / len(data_lines)
            avg_memory = total_memory_percent / len(data_lines)
            analysis += f"\n**Averages:**\n"
            analysis += f"- CPU: {avg_cpu:.2f} cores\n"
            analysis += f"- Memory: {avg_memory:.2f} Gi\n"
        
        return analysis
    
    async def _collect_cluster_info(self) -> Dict[str, Any]:
        """클러스터 정보 수집"""
        info = {}
        
        # 노드 정보
        try:
            result = subprocess.run(["kubectl", "get", "nodes", "-o", "wide"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info["nodes"] = result.stdout
        except:
            info["nodes"] = "Error getting node info"
        
        # Pod 정보
        try:
            result = subprocess.run(["kubectl", "get", "pods", "--all-namespaces", "-o", "wide"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info["pods"] = result.stdout
        except:
            info["pods"] = "Error getting pod info"
        
        # 이벤트 정보
        try:
            result = subprocess.run(["kubectl", "get", "events", "--all-namespaces", "--sort-by='.lastTimestamp'"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info["events"] = result.stdout
        except:
            info["events"] = "Error getting event info"
        
        return info
    
    def _analyze_performance(self, cluster_info: Dict[str, Any]) -> str:
        """성능 분석"""
        analysis = "## Performance Analysis\n\n"
        
        # 노드 분석
        if "nodes" in cluster_info:
            nodes_output = cluster_info["nodes"]
            lines = nodes_output.strip().split('\n')
            if len(lines) > 1:
                ready_nodes = sum(1 for line in lines[1:] if "Ready" in line)
                total_nodes = len(lines) - 1
                analysis += f"**Node Health:** {ready_nodes}/{total_nodes} nodes ready\n\n"
        
        # Pod 분석
        if "pods" in cluster_info:
            pods_output = cluster_info["pods"]
            lines = pods_output.strip().split('\n')
            if len(lines) > 1:
                running_pods = sum(1 for line in lines[1:] if "Running" in line)
                total_pods = len(lines) - 1
                analysis += f"**Pod Health:** {running_pods}/{total_pods} pods running\n\n"
        
        # 이벤트 분석
        if "events" in cluster_info:
            events_output = cluster_info["events"]
            warning_count = events_output.count("Warning")
            analysis += f"**Recent Warnings:** {warning_count} warning events\n\n"
        
        return analysis
    
    def _analyze_security(self, cluster_info: Dict[str, Any]) -> str:
        """보안 분석"""
        analysis = "## Security Analysis\n\n"
        
        # 기본 보안 체크
        analysis += "**Security Checks:**\n"
        
        # RBAC 체크
        try:
            result = subprocess.run(["kubectl", "get", "clusterroles"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                analysis += "- ✅ ClusterRoles configured\n"
            else:
                analysis += "- ⚠️ ClusterRoles not accessible\n"
        except:
            analysis += "- ❌ Cannot check ClusterRoles\n"
        
        # 네트워크 정책 체크
        try:
            result = subprocess.run(["kubectl", "get", "networkpolicies", "--all-namespaces"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    analysis += f"- ✅ NetworkPolicies found: {len(lines)-1}\n"
                else:
                    analysis += "- ⚠️ No NetworkPolicies configured\n"
            else:
                analysis += "- ⚠️ NetworkPolicies not accessible\n"
        except:
            analysis += "- ❌ Cannot check NetworkPolicies\n"
        
        return analysis
    
    def _analyze_capacity(self, cluster_info: Dict[str, Any]) -> str:
        """용량 분석"""
        analysis = "## Capacity Analysis\n\n"
        
        # 노드 용량 분석
        try:
            result = subprocess.run(["kubectl", "top", "nodes"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    analysis += "**Node Resource Usage:**\n"
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 5:
                            node_name = parts[0]
                            cpu_usage = parts[1]
                            cpu_capacity = parts[2]
                            memory_usage = parts[3]
                            memory_capacity = parts[4]
                            analysis += f"- {node_name}: CPU {cpu_usage}/{cpu_capacity}, Memory {memory_usage}/{memory_capacity}\n"
        except:
            analysis += "❌ Cannot get node metrics\n"
        
        return analysis
    
    def _analyze_events(self, events_output: str) -> str:
        """이벤트 분석"""
        lines = events_output.strip().split('\n')
        if len(lines) <= 1:
            return "No events found"
        
        # 헤더 제거
        data_lines = lines[1:]
        
        analysis = f"**Event Summary:**\n"
        analysis += f"- Total Events: {len(data_lines)}\n"
        
        # 이벤트 유형별 카운트
        event_types = {}
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 3:
                event_type = parts[2]
                event_types[event_type] = event_types.get(event_type, 0) + 1
        
        analysis += f"\n**Event Types:**\n"
        for event_type, count in event_types.items():
            analysis += f"- {event_type}: {count}\n"
        
        # 최근 경고 이벤트
        warning_events = [line for line in data_lines if "Warning" in line]
        if warning_events:
            analysis += f"\n**Recent Warnings:**\n"
            for event in warning_events[:5]:  # 최근 5개만
                analysis += f"- {event}\n"
        
        return analysis


async def main():
    """MCP 서버 실행"""
    server = K8sMonitorServer()
    
    # stdio 서버로 실행
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.get_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main()) 