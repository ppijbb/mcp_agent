"""
비용 최적화 Agent
================

클라우드 비용 최적화 및 관리를 담당하는 Agent
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

@dataclass
class CostData:
    """비용 데이터"""
    service: str
    cost: float
    currency: str = "USD"
    period: str = "monthly"
    timestamp: datetime = None

@dataclass
class ResourceUsage:
    """리소스 사용량"""
    resource_type: str
    current_usage: float
    allocated: float
    utilization_percent: float
    cost_per_hour: float

@dataclass
class OptimizationRecommendation:
    """최적화 권장사항"""
    type: str
    description: str
    potential_savings: float
    implementation_effort: str  # low, medium, high
    priority: str  # low, medium, high, critical

class CostOptimizerAgent:
    """mcp_agent 기반 비용 최적화 Agent"""
    
    def __init__(self, output_dir: str = "cost_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # mcp_agent App 초기화
        self.app = MCPApp(
            name="cost_optimizer",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="cost_optimizer",
            instruction="클라우드 비용 최적화 및 관리를 담당하는 전문 Agent입니다.",
            server_names=["cost-mcp", "k8s-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        # 비용 데이터
        self.cost_history: List[CostData] = []
        self.resource_usage: Dict[str, ResourceUsage] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
    async def start_cost_monitoring(self):
        """비용 모니터링 시작"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info("Starting cost monitoring")
                
                # 1. 현재 비용 분석
                await self._analyze_current_costs(context)
                
                # 2. 리소스 사용량 분석
                await self._analyze_resource_usage(context)
                
                # 3. 최적화 기회 식별
                await self._identify_optimization_opportunities(context)
                
                # 4. 실시간 비용 모니터링 시작
                await self._start_real_time_cost_monitoring(context)
                
                logger.info("Cost monitoring started successfully")
                
        except Exception as e:
            logger.error(f"Failed to start cost monitoring: {e}")
            raise
    
    async def _analyze_current_costs(self, context):
        """현재 비용 분석"""
        logger = context.logger
        
        # 클라우드 서비스별 비용 수집
        aws_costs = await self._get_aws_costs(context)
        gcp_costs = await self._get_gcp_costs(context)
        azure_costs = await self._get_azure_costs(context)
        
        # Kubernetes 리소스 비용
        k8s_costs = await self._get_kubernetes_costs(context)
        
        # 비용 데이터 저장
        all_costs = aws_costs + gcp_costs + azure_costs + k8s_costs
        self.cost_history.extend(all_costs)
        
        total_cost = sum(cost.cost for cost in all_costs)
        logger.info(f"Total monthly cost: ${total_cost:.2f}")
    
    async def _analyze_resource_usage(self, context):
        """리소스 사용량 분석"""
        logger = context.logger
        
        # CPU 사용량 분석
        cpu_usage = await self._analyze_cpu_usage(context)
        self.resource_usage["cpu"] = cpu_usage
        
        # 메모리 사용량 분석
        memory_usage = await self._analyze_memory_usage(context)
        self.resource_usage["memory"] = memory_usage
        
        # 스토리지 사용량 분석
        storage_usage = await self._analyze_storage_usage(context)
        self.resource_usage["storage"] = storage_usage
        
        # 네트워크 사용량 분석
        network_usage = await self._analyze_network_usage(context)
        self.resource_usage["network"] = network_usage
        
        logger.info("Resource usage analysis completed")
    
    async def _identify_optimization_opportunities(self, context):
        """최적화 기회 식별"""
        logger = context.logger
        
        recommendations = []
        
        # 1. 미사용 리소스 식별
        unused_resources = await self._find_unused_resources(context)
        for resource in unused_resources:
            recommendations.append(OptimizationRecommendation(
                type="unused_resource",
                description=f"Remove unused {resource['type']}: {resource['name']}",
                potential_savings=resource['monthly_cost'],
                implementation_effort="low",
                priority="medium"
            ))
        
        # 2. 오버프로비저닝 리소스 식별
        overprovisioned = await self._find_overprovisioned_resources(context)
        for resource in overprovisioned:
            recommendations.append(OptimizationRecommendation(
                type="overprovisioned",
                description=f"Downsize {resource['type']}: {resource['name']}",
                potential_savings=resource['potential_savings'],
                implementation_effort="medium",
                priority="high"
            ))
        
        # 3. 예약 인스턴스 기회 식별
        reservation_opportunities = await self._find_reservation_opportunities(context)
        for opportunity in reservation_opportunities:
            recommendations.append(OptimizationRecommendation(
                type="reserved_instance",
                description=f"Purchase reserved instance for {opportunity['service']}",
                potential_savings=opportunity['savings'],
                implementation_effort="low",
                priority="high"
            ))
        
        # 4. 스팟 인스턴스 기회 식별
        spot_opportunities = await self._find_spot_opportunities(context)
        for opportunity in spot_opportunities:
            recommendations.append(OptimizationRecommendation(
                type="spot_instance",
                description=f"Use spot instances for {opportunity['workload']}",
                potential_savings=opportunity['savings'],
                implementation_effort="medium",
                priority="medium"
            ))
        
        self.optimization_recommendations = recommendations
        
        total_savings = sum(rec.potential_savings for rec in recommendations)
        logger.info(f"Identified optimization opportunities with potential savings: ${total_savings:.2f}")
    
    async def _start_real_time_cost_monitoring(self, context):
        """실시간 비용 모니터링 시작"""
        logger = context.logger
        
        # 주기적 비용 모니터링 태스크 시작
        asyncio.create_task(self._periodic_cost_monitoring(context))
        
        logger.info("Real-time cost monitoring started")
    
    async def _periodic_cost_monitoring(self, context):
        """주기적 비용 모니터링"""
        logger = context.logger
        
        while True:
            try:
                # 실시간 비용 체크
                current_costs = await self._get_current_costs(context)
                
                # 예산 초과 체크
                budget_alerts = await self._check_budget_limits(current_costs, context)
                
                # 비용 스파이크 감지
                cost_spikes = await self._detect_cost_spikes(context)
                
                # 알림 전송
                for alert in budget_alerts:
                    await self._send_cost_alert(alert, context)
                
                for spike in cost_spikes:
                    await self._send_cost_spike_alert(spike, context)
                
                logger.debug("Cost monitoring cycle completed")
                
                # 1시간 대기
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Cost monitoring error: {e}")
                await asyncio.sleep(7200)  # 에러 시 2시간 대기
    
    async def _find_unused_resources(self, context) -> List[Dict[str, Any]]:
        """미사용 리소스 찾기"""
        return [
            {
                "type": "EC2 instance",
                "name": "i-1234567890abcdef0",
                "monthly_cost": 45.60,
                "unused_days": 30
            },
            {
                "type": "EBS volume",
                "name": "vol-0987654321fedcba0",
                "monthly_cost": 12.50,
                "unused_days": 45
            }
        ]
    
    async def _find_overprovisioned_resources(self, context) -> List[Dict[str, Any]]:
        """오버프로비저닝 리소스 찾기"""
        return [
            {
                "type": "EC2 instance",
                "name": "i-abcdef1234567890",
                "current_size": "m5.xlarge",
                "recommended_size": "m5.large",
                "potential_savings": 120.00
            }
        ]
    
    async def _find_reservation_opportunities(self, context) -> List[Dict[str, Any]]:
        """예약 인스턴스 기회 찾기"""
        return [
            {
                "service": "EC2 m5.large",
                "current_cost": 85.00,
                "reserved_cost": 45.00,
                "savings": 40.00,
                "commitment_period": "1 year"
            }
        ]
    
    async def _find_spot_opportunities(self, context) -> List[Dict[str, Any]]:
        """스팟 인스턴스 기회 찾기"""
        return [
            {
                "workload": "batch-processing",
                "current_cost": 60.00,
                "spot_cost": 15.00,
                "savings": 45.00,
                "availability": "high"
            }
        ]
    
    async def _get_current_costs(self, context) -> List[CostData]:
        """현재 비용 조회"""
        return [
            CostData("EC2", 450.00, timestamp=datetime.now()),
            CostData("EBS", 120.00, timestamp=datetime.now()),
            CostData("RDS", 200.00, timestamp=datetime.now()),
            CostData("S3", 50.00, timestamp=datetime.now())
        ]
    
    async def _check_budget_limits(self, costs: List[CostData], context) -> List[Dict[str, Any]]:
        """예산 한도 체크"""
        alerts = []
        total_cost = sum(cost.cost for cost in costs)
        
        # 월 예산 한도 체크
        monthly_budget = 1000.00
        if total_cost > monthly_budget:
            alerts.append({
                "type": "budget_exceeded",
                "message": f"Monthly budget exceeded: ${total_cost:.2f} > ${monthly_budget}",
                "severity": "high"
            })
        
        # 일일 예산 한도 체크
        daily_budget = 50.00
        daily_cost = total_cost / 30  # 간단한 계산
        if daily_cost > daily_budget:
            alerts.append({
                "type": "daily_budget_exceeded",
                "message": f"Daily budget exceeded: ${daily_cost:.2f} > ${daily_budget}",
                "severity": "medium"
            })
        
        return alerts
    
    async def _detect_cost_spikes(self, context) -> List[Dict[str, Any]]:
        """비용 스파이크 감지"""
        # 실제로는 과거 데이터와 비교하여 스파이크 감지
        return []  # 시뮬레이션
    
    async def _send_cost_alert(self, alert: Dict[str, Any], context):
        """비용 알림 전송"""
        logger = context.logger
        logger.warning(f"COST ALERT: {alert['message']}")
        print(f"💰 COST ALERT: {alert['message']}")
    
    async def _send_cost_spike_alert(self, spike: Dict[str, Any], context):
        """비용 스파이크 알림 전송"""
        logger = context.logger
        logger.warning(f"COST SPIKE: {spike['message']}")
        print(f"📈 COST SPIKE: {spike['message']}")
    
    # 시뮬레이션 메서드들
    async def _get_aws_costs(self, context) -> List[CostData]:
        """AWS 비용 조회"""
        return [
            CostData("EC2", 450.00, timestamp=datetime.now()),
            CostData("EBS", 120.00, timestamp=datetime.now()),
            CostData("RDS", 200.00, timestamp=datetime.now())
        ]
    
    async def _get_gcp_costs(self, context) -> List[CostData]:
        """GCP 비용 조회"""
        return [
            CostData("Compute Engine", 300.00, timestamp=datetime.now()),
            CostData("Cloud Storage", 80.00, timestamp=datetime.now())
        ]
    
    async def _get_azure_costs(self, context) -> List[CostData]:
        """Azure 비용 조회"""
        return [
            CostData("Virtual Machines", 250.00, timestamp=datetime.now()),
            CostData("Blob Storage", 60.00, timestamp=datetime.now())
        ]
    
    async def _get_kubernetes_costs(self, context) -> List[CostData]:
        """Kubernetes 비용 조회"""
        return [
            CostData("EKS", 150.00, timestamp=datetime.now()),
            CostData("GKE", 100.00, timestamp=datetime.now())
        ]
    
    async def _analyze_cpu_usage(self, context) -> ResourceUsage:
        """CPU 사용량 분석"""
        return ResourceUsage(
            resource_type="cpu",
            current_usage=45.2,
            allocated=100.0,
            utilization_percent=45.2,
            cost_per_hour=0.05
        )
    
    async def _analyze_memory_usage(self, context) -> ResourceUsage:
        """메모리 사용량 분석"""
        return ResourceUsage(
            resource_type="memory",
            current_usage=62.8,
            allocated=100.0,
            utilization_percent=62.8,
            cost_per_hour=0.03
        )
    
    async def _analyze_storage_usage(self, context) -> ResourceUsage:
        """스토리지 사용량 분석"""
        return ResourceUsage(
            resource_type="storage",
            current_usage=75.5,
            allocated=100.0,
            utilization_percent=75.5,
            cost_per_hour=0.02
        )
    
    async def _analyze_network_usage(self, context) -> ResourceUsage:
        """네트워크 사용량 분석"""
        return ResourceUsage(
            resource_type="network",
            current_usage=35.6,
            allocated=100.0,
            utilization_percent=35.6,
            cost_per_hour=0.01
        )
    
    async def get_cost_history(self) -> List[CostData]:
        """비용 히스토리 조회"""
        return self.cost_history
    
    async def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """최적화 권장사항 조회"""
        return self.optimization_recommendations
    
    async def get_resource_usage(self) -> Dict[str, ResourceUsage]:
        """리소스 사용량 조회"""
        return self.resource_usage
    
    async def generate_cost_report(self):
        """비용 보고서 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"cost_report_{timestamp}.json")
        
        total_cost = sum(cost.cost for cost in self.cost_history)
        total_savings = sum(rec.potential_savings for rec in self.optimization_recommendations)
        
        report_data = {
            "timestamp": timestamp,
            "total_cost": total_cost,
            "potential_savings": total_savings,
            "cost_history": [asdict(cost) for cost in self.cost_history],
            "resource_usage": {k: asdict(v) for k, v in self.resource_usage.items()},
            "recommendations": [asdict(rec) for rec in self.optimization_recommendations]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Cost report saved to: {report_file}")

# 사용 예시
async def main():
    """사용 예시"""
    agent = CostOptimizerAgent()
    
    # 비용 모니터링 시작
    await agent.start_cost_monitoring()
    
    # 10초 대기
    await asyncio.sleep(10)
    
    # 최적화 권장사항 조회
    recommendations = await agent.get_optimization_recommendations()
    print(f"Optimization recommendations: {len(recommendations)}")
    
    # 비용 보고서 생성
    await agent.generate_cost_report()

if __name__ == "__main__":
    asyncio.run(main()) 