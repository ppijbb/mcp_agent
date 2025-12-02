"""
Cloud Resources Simulator

GitHub, AWS, Kubernetes 리소스 상태 시뮬레이션.
실제 API 응답 형식과 상태 전이 패턴을 모방합니다.
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

from srcs.common.simulation_utils import (
    StateMachine, State, ProbabilityDistributions, PatternGenerator
)


class GitHubSimulator:
    """GitHub API 응답 시뮬레이션"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.prob_dist = ProbabilityDistributions(seed)
    
    def get_repositories(self, username: str = "organization") -> List[Dict[str, Any]]:
        """저장소 목록 시뮬레이션"""
        num_repos = self.rng.randint(5, 20)
        repos = []
        
        repo_names = [
            "web-app", "api-service", "mobile-app", "data-pipeline",
            "ml-model", "infrastructure", "documentation", "testing",
            "monitoring", "analytics", "auth-service", "payment-gateway"
        ]
        
        for i in range(num_repos):
            repo_name = self.rng.choice(repo_names) if i < len(repo_names) else f"repo-{i}"
            created_at = datetime.now() - timedelta(days=self.rng.randint(30, 1000))
            updated_at = datetime.now() - timedelta(days=self.rng.randint(0, 30))
            
            # 언어 분포
            languages = ["Python", "JavaScript", "TypeScript", "Go", "Java", "Rust"]
            language = self.rng.choice(languages)
            
            # 스타 수
            stars = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.LOG_NORMAL,
                mean=math.log(10), std=1.5
            )
            stars = max(0, int(stars))
            
            # 포크 수
            forks = int(stars * self.rng.uniform(0.1, 0.3))
            
            # 이슈 수
            open_issues = self.rng.randint(0, 20)
            
            repo = {
                "id": self.rng.randint(100000, 999999),
                "name": repo_name,
                "full_name": f"{username}/{repo_name}",
                "description": f"Repository for {repo_name} project",
                "private": self.rng.random() < 0.3,
                "fork": False,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "pushed_at": updated_at.isoformat(),
                "language": language,
                "stargazers_count": stars,
                "forks_count": forks,
                "open_issues_count": open_issues,
                "default_branch": "main",
                "archived": self.rng.random() < 0.1,
                "disabled": False
            }
            repos.append(repo)
        
        return repos
    
    def get_pull_requests(self, repo: str, state: str = "open") -> List[Dict[str, Any]]:
        """Pull Request 목록 시뮬레이션"""
        num_prs = self.rng.randint(0, 10) if state == "open" else self.rng.randint(10, 50)
        prs = []
        
        for i in range(num_prs):
            created_at = datetime.now() - timedelta(days=self.rng.randint(0, 30))
            updated_at = created_at + timedelta(days=self.rng.randint(0, 7))
            
            # PR 상태
            if state == "open":
                pr_state = "open"
            else:
                pr_state = self.rng.choice(["closed", "merged"])
            
            pr = {
                "id": self.rng.randint(1000, 9999),
                "number": self.rng.randint(1, 1000),
                "state": pr_state,
                "title": f"Fix issue #{self.rng.randint(1, 100)}",
                "body": f"Description of pull request #{i+1}",
                "user": {
                    "login": f"user-{self.rng.randint(1, 10)}",
                    "avatar_url": f"https://avatar.example.com/{i}"
                },
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "merged_at": updated_at.isoformat() if pr_state == "merged" else None,
                "draft": self.rng.random() < 0.2,
                "mergeable": self.rng.random() < 0.8,
                "additions": self.rng.randint(10, 500),
                "deletions": self.rng.randint(5, 200),
                "changed_files": self.rng.randint(1, 20)
            }
            prs.append(pr)
        
        return prs
    
    def get_workflow_runs(self, repo: str) -> List[Dict[str, Any]]:
        """GitHub Actions 워크플로우 실행 시뮬레이션"""
        num_runs = self.rng.randint(10, 50)
        runs = []
        
        workflow_names = ["CI", "CD", "Tests", "Lint", "Build", "Deploy"]
        
        for i in range(num_runs):
            created_at = datetime.now() - timedelta(hours=self.rng.randint(0, 168))
            started_at = created_at + timedelta(minutes=self.rng.randint(1, 5))
            
            # 실행 상태
            status = self.rng.choice(["completed", "in_progress", "queued"])
            conclusion = None
            if status == "completed":
                conclusion = self.rng.choice(["success", "failure", "cancelled"])
            
            # 실행 시간
            if conclusion == "success":
                duration_seconds = self.rng.randint(30, 300)
            else:
                duration_seconds = self.rng.randint(10, 120)
            
            completed_at = started_at + timedelta(seconds=duration_seconds) if status == "completed" else None
            
            run = {
                "id": self.rng.randint(100000, 999999),
                "name": self.rng.choice(workflow_names),
                "status": status,
                "conclusion": conclusion,
                "workflow_id": self.rng.randint(1000, 9999),
                "created_at": created_at.isoformat(),
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat() if completed_at else None,
                "duration_seconds": duration_seconds if status == "completed" else None
            }
            runs.append(run)
        
        return runs


class AWSSimulator:
    """AWS 리소스 상태 시뮬레이션"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.prob_dist = ProbabilityDistributions(seed)
    
    def get_ec2_instances(self) -> List[Dict[str, Any]]:
        """EC2 인스턴스 목록 시뮬레이션"""
        num_instances = self.rng.randint(5, 20)
        instances = []
        
        instance_types = ["t3.medium", "t3.large", "m5.large", "m5.xlarge", "c5.xlarge"]
        states = ["running", "stopped", "stopping", "pending", "terminated"]
        
        for i in range(num_instances):
            instance_id = f"i-{''.join([str(self.rng.randint(0, 9)) for _ in range(17))])"
            state = self.rng.choice(states)
            
            # 상태별 생성 시간
            if state == "running":
                launch_time = datetime.now() - timedelta(days=self.rng.randint(1, 365))
            elif state == "stopped":
                launch_time = datetime.now() - timedelta(days=self.rng.randint(1, 365))
                stop_time = datetime.now() - timedelta(hours=self.rng.randint(1, 48))
            else:
                launch_time = datetime.now() - timedelta(hours=self.rng.randint(0, 24))
                stop_time = None
            
            instance = {
                "InstanceId": instance_id,
                "InstanceType": self.rng.choice(instance_types),
                "State": {
                    "Name": state,
                    "Code": {"running": 16, "stopped": 80, "stopping": 64, "pending": 0, "terminated": 48}.get(state, 0)
                },
                "LaunchTime": launch_time.isoformat(),
                "PrivateIpAddress": f"10.0.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}",
                "PublicIpAddress": f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}" if state == "running" else None,
                "Tags": [
                    {"Key": "Name", "Value": f"instance-{i+1}"},
                    {"Key": "Environment", "Value": self.rng.choice(["production", "staging", "development"])}
                ]
            }
            instances.append(instance)
        
        return instances
    
    def get_s3_buckets(self) -> List[Dict[str, Any]]:
        """S3 버킷 목록 시뮬레이션"""
        num_buckets = self.rng.randint(3, 10)
        buckets = []
        
        for i in range(num_buckets):
            bucket_name = f"bucket-{self.rng.randint(1000, 9999)}-{uuid.uuid4().hex[:8]}"
            created_at = datetime.now() - timedelta(days=self.rng.randint(30, 1000))
            
            # 버킷 크기 (GB)
            size_gb = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.LOG_NORMAL,
                mean=math.log(10), std=2.0
            )
            size_gb = max(0.1, size_gb)
            
            # 객체 수
            num_objects = int(size_gb * 1000)  # 대략적 변환
            
            bucket = {
                "Name": bucket_name,
                "CreationDate": created_at.isoformat(),
                "SizeGB": round(size_gb, 2),
                "NumberOfObjects": num_objects,
                "Region": self.rng.choice(["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"])
            }
            buckets.append(bucket)
        
        return buckets
    
    def get_lambda_functions(self) -> List[Dict[str, Any]]:
        """Lambda 함수 목록 시뮬레이션"""
        num_functions = self.rng.randint(5, 15)
        functions = []
        
        for i in range(num_functions):
            function_name = f"function-{self.rng.randint(1, 100)}"
            created_at = datetime.now() - timedelta(days=self.rng.randint(1, 365))
            last_modified = datetime.now() - timedelta(hours=self.rng.randint(0, 168))
            
            # 런타임
            runtime = self.rng.choice(["python3.9", "python3.10", "nodejs18.x", "go1.x", "java11"])
            
            # 메모리 설정
            memory_mb = self.rng.choice([128, 256, 512, 1024, 2048])
            
            # 타임아웃
            timeout_seconds = self.rng.choice([30, 60, 120, 300, 900])
            
            function = {
                "FunctionName": function_name,
                "Runtime": runtime,
                "Role": f"arn:aws:iam::123456789012:role/lambda-role",
                "Handler": "index.handler",
                "CodeSize": self.rng.randint(1000000, 50000000),
                "Description": f"Lambda function {function_name}",
                "Timeout": timeout_seconds,
                "MemorySize": memory_mb,
                "LastModified": last_modified.isoformat(),
                "CodeSha256": uuid.uuid4().hex,
                "Version": "$LATEST",
                "State": "Active",
                "StateReason": None
            }
            functions.append(function)
        
        return functions


class KubernetesSimulator:
    """Kubernetes 리소스 상태 시뮬레이션"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.prob_dist = ProbabilityDistributions(seed)
    
    def get_pods(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """Pod 목록 시뮬레이션"""
        num_pods = self.rng.randint(10, 50)
        pods = []
        
        pod_names = [
            "web-app", "api-server", "database", "cache", "worker",
            "scheduler", "monitor", "logger", "auth", "gateway"
        ]
        
        for i in range(num_pods):
            pod_name = f"{self.rng.choice(pod_names)}-{self.rng.randint(1, 100)}-{uuid.uuid4().hex[:8]}"
            
            # Pod 상태
            phase = self.rng.choice(["Running", "Pending", "Succeeded", "Failed", "Unknown"])
            
            # 생성 시간
            created_at = datetime.now() - timedelta(minutes=self.rng.randint(0, 1440))
            
            # 리소스 요청/제한
            cpu_request = f"{self.rng.choice([100, 200, 500, 1000])}m"
            memory_request = f"{self.rng.choice([128, 256, 512, 1024])}Mi"
            cpu_limit = f"{int(cpu_request[:-1]) * 2}m"
            memory_limit = f"{int(memory_request[:-2]) * 2}Mi"
            
            # 노드 할당
            node_name = f"node-{self.rng.randint(1, 5)}"
            
            pod = {
                "metadata": {
                    "name": pod_name,
                    "namespace": namespace,
                    "creationTimestamp": created_at.isoformat(),
                    "labels": {
                        "app": pod_name.split("-")[0],
                        "version": f"v{self.rng.randint(1, 10)}"
                    }
                },
                "status": {
                    "phase": phase,
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "True" if phase == "Running" else "False",
                            "lastTransitionTime": created_at.isoformat()
                        }
                    ],
                    "containerStatuses": [
                        {
                            "name": "main",
                            "ready": phase == "Running",
                            "restartCount": self.rng.randint(0, 5),
                            "state": {
                                "running": {"startedAt": created_at.isoformat()} if phase == "Running" else None
                            }
                        }
                    ]
                },
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "resources": {
                                "requests": {
                                    "cpu": cpu_request,
                                    "memory": memory_request
                                },
                                "limits": {
                                    "cpu": cpu_limit,
                                    "memory": memory_limit
                                }
                            }
                        }
                    ],
                    "nodeName": node_name
                }
            }
            pods.append(pod)
        
        return pods
    
    def get_services(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """Service 목록 시뮬레이션"""
        num_services = self.rng.randint(5, 15)
        services = []
        
        service_types = ["ClusterIP", "NodePort", "LoadBalancer"]
        
        for i in range(num_services):
            service_name = f"service-{self.rng.randint(1, 100)}"
            service_type = self.rng.choice(service_types)
            
            service = {
                "metadata": {
                    "name": service_name,
                    "namespace": namespace,
                    "creationTimestamp": (datetime.now() - timedelta(days=self.rng.randint(1, 365))).isoformat()
                },
                "spec": {
                    "type": service_type,
                    "ports": [
                        {
                            "port": self.rng.choice([80, 443, 8080, 3000]),
                            "targetPort": self.rng.choice([80, 443, 8080, 3000]),
                            "protocol": "TCP"
                        }
                    ],
                    "selector": {
                        "app": service_name
                    }
                },
                "status": {
                    "loadBalancer": {} if service_type == "LoadBalancer" else None
                }
            }
            services.append(service)
        
        return services
    
    def get_deployments(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """Deployment 목록 시뮬레이션"""
        num_deployments = self.rng.randint(5, 15)
        deployments = []
        
        for i in range(num_deployments):
            deployment_name = f"deployment-{self.rng.randint(1, 100)}"
            replicas = self.rng.randint(1, 10)
            ready_replicas = self.rng.randint(0, replicas)
            available_replicas = self.rng.randint(0, ready_replicas)
            
            deployment = {
                "metadata": {
                    "name": deployment_name,
                    "namespace": namespace,
                    "creationTimestamp": (datetime.now() - timedelta(days=self.rng.randint(1, 365))).isoformat()
                },
                "spec": {
                    "replicas": replicas,
                    "selector": {
                        "matchLabels": {
                            "app": deployment_name
                        }
                    }
                },
                "status": {
                    "replicas": replicas,
                    "readyReplicas": ready_replicas,
                    "availableReplicas": available_replicas,
                    "unavailableReplicas": replicas - available_replicas
                }
            }
            deployments.append(deployment)
        
        return deployments


class CICDPipelineSimulator:
    """CI/CD 파이프라인 실행 시뮬레이션"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.prob_dist = ProbabilityDistributions(seed)
    
    def simulate_pipeline_run(
        self,
        pipeline_name: str,
        stages: List[str] = None
    ) -> Dict[str, Any]:
        """
        파이프라인 실행 시뮬레이션
        
        Args:
            pipeline_name: 파이프라인 이름
            stages: 스테이지 목록
        """
        if stages is None:
            stages = ["build", "test", "deploy"]
        
        pipeline_id = f"pipeline-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now()
        
        # 전체 파이프라인 상태
        overall_status = self.rng.choice(["success", "failed", "running"])
        
        # 스테이지별 실행
        stage_results = []
        current_time = started_at
        
        for i, stage in enumerate(stages):
            # 스테이지 실행 시간
            duration_seconds = self.rng.randint(30, 600)
            
            # 스테이지 성공 확률
            if overall_status == "success":
                stage_status = "success"
            elif overall_status == "failed":
                # 이전 스테이지가 실패하면 이후도 실패
                if i > 0 and stage_results[-1]["status"] == "failed":
                    stage_status = "skipped"
                else:
                    stage_status = "failed" if self.rng.random() < 0.3 else "success"
            else:
                # 실행 중
                if i < len(stages) - 1:
                    stage_status = "success"
                else:
                    stage_status = "running"
            
            stage_started = current_time
            stage_completed = current_time + timedelta(seconds=duration_seconds) if stage_status != "running" else None
            
            stage_result = {
                "name": stage,
                "status": stage_status,
                "started_at": stage_started.isoformat(),
                "completed_at": stage_completed.isoformat() if stage_completed else None,
                "duration_seconds": duration_seconds if stage_completed else None,
                "steps": self._generate_steps(stage, stage_status, duration_seconds)
            }
            stage_results.append(stage_result)
            
            current_time = stage_completed if stage_completed else current_time + timedelta(seconds=duration_seconds)
        
        pipeline = {
            "id": pipeline_id,
            "name": pipeline_name,
            "status": overall_status,
            "started_at": started_at.isoformat(),
            "completed_at": current_time.isoformat() if overall_status != "running" else None,
            "stages": stage_results,
            "duration_seconds": (current_time - started_at).total_seconds() if overall_status != "running" else None
        }
        
        return pipeline
    
    def _generate_steps(self, stage: str, status: str, duration: int) -> List[Dict[str, Any]]:
        """스테이지 내 스텝 생성"""
        step_names = {
            "build": ["install-dependencies", "compile", "package"],
            "test": ["unit-tests", "integration-tests", "coverage"],
            "deploy": ["build-image", "push-image", "deploy-to-staging", "smoke-tests"]
        }
        
        steps = step_names.get(stage, ["step-1", "step-2"])
        step_results = []
        
        step_duration = duration / len(steps)
        current_time = datetime.now()
        
        for step in steps:
            step_status = status if status != "running" else "running"
            
            step_result = {
                "name": step,
                "status": step_status,
                "duration_seconds": step_duration if step_status != "running" else None
            }
            step_results.append(step_result)
        
        return step_results








