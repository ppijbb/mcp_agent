#!/usr/bin/env python3
"""
DevOps Productivity MCP Agent
================================
Production-level DevOps assistant with API integrations:
- GitHub API for repositories, PRs, issues
- Prometheus API for metrics
"""

import asyncio
import os
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin

# MCP Agent imports
from mcp_agent.workflows.llm.google_augmented_llm import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams


@dataclass
class APIConfig:
    """API configuration with environment variables"""
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    prometheus_url: str = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


class GitHubClient:
    """GitHub API client with proper authentication and error handling"""
    
    def __init__(self, token: str):
        if not token:
            raise ValueError("GitHub token is required")
            
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DevOps-Assistant-Agent"
        }
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated GitHub API request with rate limit handling"""
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        # Check rate limit before making request
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= 5:
            reset_time = datetime.fromtimestamp(self.rate_limit_reset)
            current_time = datetime.now()
            if reset_time > current_time:
                wait_seconds = (reset_time - current_time).total_seconds() + 1
                print(f"Rate limit low ({self.rate_limit_remaining}), waiting {wait_seconds:.1f}s")
                await asyncio.sleep(wait_seconds)
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(method, url, headers=self.headers, **kwargs) as response:
                    # Update rate limit info
                    self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 1000))
                    self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
                    
                    if response.status == 204:  # No content
                        return {}
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        if response.status == 403 and 'rate limit' in error_text.lower():
                            reset_time = datetime.fromtimestamp(self.rate_limit_reset)
                            raise Exception(f"GitHub API rate limit exceeded. Resets at {reset_time.isoformat()}")
                        raise Exception(f"GitHub API error {response.status}: {error_text}")
                    
                    return await response.json()
            except aiohttp.ClientError as e:
                raise Exception(f"Network error when calling GitHub API: {str(e)}")
    
    async def get_repositories(self, org: Optional[str] = None) -> List[Dict]:
        """Get repositories with pagination support"""
        if org:
            endpoint = f"/orgs/{org}/repos"
        else:
            endpoint = "/user/repos"
        
        params = {"per_page": 100, "page": 1}
        all_repos = []
        
        while True:
            repos = await self._request("GET", endpoint, params=params)
            if not repos:
                break
                
            all_repos.extend(repos)
            
            if len(repos) < 100:
                break
                
            params["page"] += 1
            
        return all_repos
    
    async def get_pull_requests(self, owner: str, repo: str, state: str = "open") -> List[Dict]:
        """Get pull requests with pagination and state filter"""
        endpoint = f"/repos/{owner}/{repo}/pulls"
        params = {"state": state, "per_page": 100, "page": 1}
        
        all_prs = []
        
        while True:
            prs = await self._request("GET", endpoint, params=params)
            if not prs:
                break
                
            all_prs.extend(prs)
            
            if len(prs) < 100:
                break
                
            params["page"] += 1
            
        return all_prs
    
    async def get_workflow_runs(self, owner: str, repo: str, branch: Optional[str] = None) -> List[Dict]:
        """Get GitHub Actions workflow runs with optional branch filter"""
        endpoint = f"/repos/{owner}/{repo}/actions/runs"
        params = {"per_page": 30}
        
        if branch:
            params["branch"] = branch
            
        result = await self._request("GET", endpoint, params=params)
        return result.get("workflow_runs", [])


class PrometheusClient:
    """Prometheus API client for metrics retrieval"""
    
    def __init__(self, url: str):
        self.url = url.rstrip('/')
    
    async def _request(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make Prometheus API request with error handling"""
        url = f"{self.url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise Exception(f"Prometheus API error {response.status}: {error_text}")
                    return await response.json()
            except aiohttp.ClientError as e:
                raise Exception(f"Network error when calling Prometheus API: {str(e)}")
    
    async def query(self, query: str) -> Dict[str, Any]:
        """Execute Prometheus query"""
        params = {"query": query}
        return await self._request("/api/v1/query", params)
    
    async def get_cpu_usage(self) -> Optional[float]:
        """Get CPU usage percentage"""
        try:
            query = '100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
            result = await self.query(query)
            
            if result["status"] == "success" and result["data"]["result"]:
                return float(result["data"]["result"][0]["value"][1])
            return None
        except Exception:
            return None


class DevOpsAssistantAgent:
    """Production DevOps Assistant with API integrations"""
    
    def __init__(self):
        self.config = APIConfig()
        
        # Initialize API clients
        self.github = None
        if self.config.github_token:
            try:
                self.github = GitHubClient(self.config.github_token)
            except ValueError as e:
                print(f"Warning: {str(e)}")
                
        self.prometheus = PrometheusClient(self.config.prometheus_url)
        
        # Initialize MCP Agent with Google LLM
        self.model_name = "gemini-2.5-flash-lite-preview-0607"
        self.llm = GoogleAugmentedLLM(
            model_name=self.model_name,
            api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        
        # Define capabilities
        self.capabilities = {
            "github_analysis": "GitHub repository analysis",
            "ci_cd_monitoring": "CI/CD pipeline monitoring", 
            "infrastructure_health": "Infrastructure health monitoring"
        }
    
    async def analyze_github_repositories(self, org: str) -> Dict[str, Any]:
        """Analyze GitHub repositories with comprehensive metrics"""
        if not self.github:
            return {"error": "GitHub token not configured or invalid"}
        
        try:
            # Fetch repositories with error handling
            try:
                repos = await self.github.get_repositories(org=org)
            except Exception as e:
                return {"error": f"Failed to fetch repositories: {str(e)}"}
            
            # Process repository data
            analysis = {
                "total_repositories": len(repos),
                "languages": {},
                "stars_total": 0,
                "forks_total": 0,
                "repositories": []
            }
            
            # Extract and aggregate repository information
            for repo in repos:
                # Skip archived repositories
                if repo.get("archived", False):
                    continue
                    
                repo_info = {
                    "name": repo.get("name", ""),
                    "description": repo.get("description", ""),
                    "language": repo.get("language", "Unknown"),
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "open_issues": repo.get("open_issues_count", 0),
                    "updated_at": repo.get("updated_at", "")
                }
                
                analysis["repositories"].append(repo_info)
                analysis["stars_total"] += repo_info["stars"]
                analysis["forks_total"] += repo_info["forks"]
                
                if repo_info["language"]:
                    analysis["languages"][repo_info["language"]] = analysis["languages"].get(repo_info["language"], 0) + 1
            
            # Sort repositories by stars for convenience
            analysis["repositories"].sort(key=lambda x: x["stars"], reverse=True)
            
            # Add timestamp
            analysis["timestamp"] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            return {"error": f"Repository analysis failed: {str(e)}"}
    
    async def monitor_ci_cd_pipelines(self, owner: str, repo: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """Monitor CI/CD pipelines with detailed status information"""
        if not self.github:
            return {"error": "GitHub token not configured or invalid"}
        
        try:
            # Fetch workflow runs with error handling
            try:
                workflow_runs = await self.github.get_workflow_runs(owner, repo, branch)
            except Exception as e:
                return {"error": f"Failed to fetch workflow runs: {str(e)}"}
            
            # Process workflow data
            pipeline_status = {
                "total_runs": len(workflow_runs),
                "success_count": 0,
                "failure_count": 0,
                "in_progress_count": 0,
                "recent_runs": []
            }
            
            # Extract and categorize workflow information
            for run in workflow_runs[:10]:  # Last 10 runs for recent activity
                run_info = {
                    "id": run.get("id", 0),
                    "name": run.get("name", ""),
                    "status": run.get("status", ""),
                    "conclusion": run.get("conclusion", ""),
                    "created_at": run.get("created_at", ""),
                    "updated_at": run.get("updated_at", ""),
                    "branch": run.get("head_branch", ""),
                    "commit_sha": run.get("head_sha", "")[:8] if run.get("head_sha") else ""
                }
                
                pipeline_status["recent_runs"].append(run_info)
                
                # Count by status
                if run.get("conclusion") == "success":
                    pipeline_status["success_count"] += 1
                elif run.get("conclusion") == "failure":
                    pipeline_status["failure_count"] += 1
                elif run.get("status") == "in_progress":
                    pipeline_status["in_progress_count"] += 1
            
            # Calculate success rate
            total_completed = pipeline_status["success_count"] + pipeline_status["failure_count"]
            if total_completed > 0:
                pipeline_status["success_rate"] = round((pipeline_status["success_count"] / total_completed) * 100, 1)
            else:
                pipeline_status["success_rate"] = 0
                
            # Add timestamp
            pipeline_status["timestamp"] = datetime.now().isoformat()
            
            return pipeline_status
            
        except Exception as e:
            return {"error": f"CI/CD monitoring failed: {str(e)}"}
    
    async def check_infrastructure_health(self) -> Dict[str, Any]:
        """Check infrastructure health with available metrics"""
        try:
            # Initialize health data structure
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {},
                "overall_status": "unknown"
            }
            
            # Get CPU usage if Prometheus is available
            try:
                cpu_usage = await self.prometheus.get_cpu_usage()
                if cpu_usage is not None:
                    health_data["metrics"]["cpu_usage_percent"] = round(cpu_usage, 2)
                    
                    # Determine status based on CPU
                    if cpu_usage < 50:
                        health_data["overall_status"] = "healthy"
                    elif cpu_usage < 80:
                        health_data["overall_status"] = "warning"
                    else:
                        health_data["overall_status"] = "critical"
                else:
                    health_data["metrics"]["cpu_usage_percent"] = "unavailable"
                    health_data["overall_status"] = "unknown"
            except Exception as e:
                health_data["metrics"]["error"] = f"Failed to get metrics: {str(e)}"
            
            return health_data
            
        except Exception as e:
            return {"error": f"Infrastructure health check failed: {str(e)}"}
    
    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process DevOps request using LLM for intent classification"""
        try:
            # Create prompt for LLM to classify the request
            prompt = f"""
            Analyze this DevOps request and determine the appropriate action:
            
            Request: "{request}"
            
            Available actions:
            - github_analysis: analyze GitHub repositories
            - ci_cd_monitoring: monitor CI/CD pipelines  
            - infrastructure_health: check infrastructure health
            
            Respond with JSON containing:
            - action: one of the available actions
            - parameters: any parameters needed (org, owner, repo, branch)
            """
            
            # Configure request parameters
            params = RequestParams(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            # Call LLM to determine action
            try:
                llm_response = await self.llm.process_request(params)
                response_data = json.loads(llm_response.response)
            except Exception as e:
                return {
                    "error": f"Failed to process request with LLM: {str(e)}",
                    "request": request,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract action and parameters
            action = response_data.get("action")
            parameters = response_data.get("parameters", {})
            
            # Execute the determined action
            if action == "github_analysis":
                if "org" not in parameters:
                    return {"error": "Missing required parameter: org"}
                result = await self.analyze_github_repositories(**parameters)
            elif action == "ci_cd_monitoring":
                if "owner" not in parameters or "repo" not in parameters:
                    return {"error": "Missing required parameters: owner and repo"}
                result = await self.monitor_ci_cd_pipelines(**parameters)
            elif action == "infrastructure_health":
                result = await self.check_infrastructure_health()
            else:
                result = {"error": f"Unknown action: {action}"}
            
            # Return comprehensive response
            return {
                "request": request,
                "action": action,
                "parameters": parameters,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse LLM response",
                "request": request,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": f"Request processing failed: {str(e)}",
                "request": request,
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Test the DevOps assistant with proper error handling"""
    agent = DevOpsAssistantAgent()
    
    # Check if GitHub token is configured
    if not agent.github:
        print("⚠️ Warning: GitHub token not configured. GitHub features will be unavailable.")
    
    # Test with sample requests
    test_requests = [
        "Check the health of our infrastructure",
        "Analyze repositories in microsoft organization"
    ]
    
    for request in test_requests:
        print(f"\n🔄 Processing: {request}")
        try:
            result = await agent.process_request(request)
            print(f"✅ Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 