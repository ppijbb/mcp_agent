#!/usr/bin/env python3
"""
Enhanced Goal Setter Agent with MCP Integration

This agent now leverages multiple MCP servers for:
- Filesystem operations (save/load goal plans)
- Web search and research
- Browser automation for data collection
- Enhanced goal validation and planning
- Persistent goal management
"""

import os
import json
import httpx
import argparse
import asyncio
import logging
from typing import Dict, Any, List, Optional
from string import Template
from datetime import datetime
from pathlib import Path

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP packages not available. Running in limited mode.")

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class MCPGoalSetterAgent:
    """Enhanced Goal Setter Agent with MCP Integration"""

    def __init__(self,
                 output_dir: str = "goal_plans",
                 enable_mcp: bool = True,
                 mcp_servers: Optional[Dict[str, str]] = None):
        """
        Initialize the MCP Goal Setter Agent

        Args:
            output_dir: Directory to save goal plans
            enable_mcp: Whether to enable MCP functionality
            mcp_servers: Dictionary of MCP server configurations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_mcp = enable_mcp and MCP_AVAILABLE
        self.mcp_servers = mcp_servers or {}

        # MCP client sessions
        self.filesystem_session: Optional[ClientSession] = None
        self.search_session: Optional[ClientSession] = None
        self.browser_session: Optional[ClientSession] = None

        # Available agents for goal planning
        self.available_agents = [
            "CodeReviewAgent",
            "DocumentationAgent",
            "PerformanceAgent",
            "SecurityAgent",
            "KubernetesAgent",
            "TravelScoutAgent",
            "BusinessStrategyAgent",
            "DataAnalysisAgent",
            "MLOpsAgent"
        ]

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize MCP connections if enabled
        if self.enable_mcp:
            asyncio.create_task(self._initialize_mcp_connections())

    async def _initialize_mcp_connections(self):
        """Initialize connections to MCP servers"""
        try:
            # Initialize filesystem MCP server
            if "filesystem" in self.mcp_servers:
                await self._connect_filesystem_server()

            # Initialize search MCP server
            if "search" in self.mcp_servers:
                await self._connect_search_server()

            # Initialize browser MCP server
            if "browser" in self.mcp_servers:
                await self._connect_browser_server()

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP connections: {e}")

    async def _connect_filesystem_server(self):
        """Connect to filesystem MCP server"""
        try:
            server_config = self.mcp_servers["filesystem"]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )

            context = stdio_client(server_params)
            receive_stream, write_stream = await context.__aenter__()
            self.filesystem_session = ClientSession(receive_stream, write_stream)

            self.logger.info("✅ Connected to filesystem MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to filesystem server: {e}")

    async def _connect_search_server(self):
        """Connect to search MCP server"""
        try:
            server_config = self.mcp_servers["search"]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )

            context = stdio_client(server_params)
            receive_stream, write_stream = await context.__aenter__()
            self.search_session = ClientSession(receive_stream, write_stream)

            self.logger.info("✅ Connected to search MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to search server: {e}")

    async def _connect_browser_server(self):
        """Connect to browser MCP server"""
        try:
            server_config = self.mcp_servers["browser"]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )

            context = stdio_client(server_params)
            receive_stream, write_stream = await context.__aenter__()
            self.browser_session = ClientSession(receive_stream, write_stream)

            self.logger.info("✅ Connected to browser MCP server")
        except Exception as e:
            self.logger.error(f"Failed to connect to browser server: {e}")

    async def _call_mcp_tool(self, session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result"""
        try:
            # List available tools first
            tools_response = await session.call_tool("list_tools", {})
            tools = tools_response.content[0].text if tools_response.content else "[]"

            # Call the specific tool
            result = await session.call_tool(tool_name, arguments)
            return json.loads(result.content[0].text) if result.content else {}
        except Exception as e:
            self.logger.error(f"Failed to call MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    async def research_goal_context(self, goal: str) -> Dict[str, Any]:
        """Research the goal context using MCP search and browser tools"""
        research_data = {
            "goal": goal,
            "search_results": [],
            "web_data": [],
            "related_concepts": [],
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Use search MCP server if available
            if self.search_session:
                search_result = await self._call_mcp_tool(
                    self.search_session,
                    "search_web",
                    {"query": f"{goal} best practices strategies", "count": 5}
                )
                if "error" not in search_result:
                    research_data["search_results"] = search_result.get("results", [])

            # Use browser MCP server if available
            if self.browser_session:
                # Navigate to relevant research sites
                browser_result = await self._call_mcp_tool(
                    self.browser_session,
                    "navigate",
                    {"url": "https://www.google.com/search?q=" + goal.replace(" ", "+")}
                )
                if "error" not in browser_result:
                    research_data["web_data"].append(browser_result)

        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            research_data["error"] = str(e)

        return research_data

    def create_enhanced_prompt(self, high_level_goal: str, available_agents: List[str], research_data: Optional[Dict[str, Any]] = None) -> str:
        """Create an enhanced prompt with research context and MCP integration"""
        agents_joined = ", ".join(available_agents)
        default_agent = available_agents[0] if available_agents else "GeneralAgent"

        # Include research context if available
        research_context = ""
        if research_data and research_data.get("search_results"):
            research_context = f"\n📚 Research Context:\n"
            for i, result in enumerate(research_data["search_results"][:3], 1):
                research_context += f"{i}. {result.get('title', 'N/A')}: {result.get('snippet', 'N/A')}\n"

        tmpl = Template(
            """
역할: 수석 전략 기획 에이전트 (MCP 통합). 다음 상위 목표를 구체적이고 실행 가능한 계획으로 분해하라.
원칙: 명확성, 간결성, 결정성. 불필요한 문장/사족/사과 금지. 출력은 오직 JSON만.

상위 목표: "$high_level_goal"

사용 가능 에이전트(정확한 이름만 사용, 임의 생성 금지): $agents_joined
$research_context

요구 사항:
1) 2~4개의 독립적이며 측정 가능한 하위 목표(sub_goal)를 도출하라(SMART).
2) 각 sub_goal에 KPI 1~2개를 정의하라. KPI는 name/metric/target/data_source를 포함한다.
3) 각 sub_goal에 대한 실행 계획(action_plan) 2~5개를 정의하라.
   - 각 action은 action_item, suggested_agent(반드시 위 목록 중 하나), due_days(1~30 정수),
     acceptance_criteria(검증 기준), dependencies(선택, action_item 참조 리스트)를 포함한다.
4) 각 sub_goal에 risks(선택, 최대 3개)를 기술하라.
5) 전체 성공 기준(overall_success_criteria)을 간결히 제시하라.
6) MCP 도구 활용 계획(mcp_tool_usage)을 포함하라.
7) 모든 내용은 한국어로 작성하라.

출력은 아래 JSON 스키마를 오직 그대로 충족하는 단일 JSON 객체로만 반환하라. 마크다운/코드펜스/설명 금지.
{
  "original_goal": "$high_level_goal",
  "research_context": "$research_context",
  "decomposed_plan": [
    {
      "sub_goal": "구체적이며 측정 가능한 하위 목표",
      "rationale": "왜 중요한지",
      "priority": "high|medium|low",
      "kpis": [
        {
          "name": "KPI 이름",
          "metric": "측정 방법",
          "target": "목표치",
          "data_source": "데이터 출처"
        }
      ],
      "action_plan": [
        {
          "action_item": "구체적 작업",
          "suggested_agent": "$default_agent",
          "due_days": 7,
          "acceptance_criteria": "완료 판정 기준",
          "dependencies": ["선행 작업 이름"]
        }
      ],
      "risks": ["위험 1", "위험 2"]
    }
  ],
  "mcp_tool_usage": {
    "filesystem_operations": ["저장할 파일들"],
    "search_queries": ["추가 검색 쿼리"],
    "browser_automation": ["자동화할 웹 작업"]
  },
  "overall_success_criteria": "전반적 성공 기준"
}
"""
        )
        return tmpl.substitute(
            high_level_goal=high_level_goal,
            agents_joined=agents_joined,
            default_agent=default_agent,
            research_context=research_context
        )

    async def generate_enhanced_goal_plan(self, goal: str, agents: Optional[List[str]] = None, enable_research: bool = True) -> Dict[str, Any]:
        """Generate an enhanced goal plan with MCP integration"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        agents = agents or self.available_agents

        # Research phase using MCP tools
        research_data = None
        if enable_research and self.enable_mcp:
            self.logger.info("🔍 Researching goal context using MCP tools...")
            research_data = await self.research_goal_context(goal)

        # Create enhanced prompt
        prompt = self.create_enhanced_prompt(goal, agents, research_data)

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-5-mini-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(OPENAI_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"OpenAI API request failed: {response.text}")

        message_content = response.json()["choices"][0]["message"]["content"]
        try:
            plan = json.loads(message_content)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode JSON from LLM response:\n{message_content}")
            raise

        # Validate and enhance the plan
        self._validate_enhanced_plan(plan, agents)

        # Add metadata
        plan["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "mcp_enabled": self.enable_mcp,
            "research_data": research_data,
            "version": "2.0"
        }

        return plan

    def _validate_enhanced_plan(self, plan: Dict[str, Any], allowed_agents: List[str]) -> None:
        """Validate the enhanced plan structure"""
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a JSON object.")

        required_fields = ["original_goal", "decomposed_plan", "overall_success_criteria"]
        for field in required_fields:
            if not plan.get(field):
                raise ValueError(f"Missing required field: {field}")

        decomposed = plan.get("decomposed_plan")
        if not isinstance(decomposed, list) or not (2 <= len(decomposed) <= 4):
            raise ValueError("'decomposed_plan' must be a list with 2~4 items.")

        for idx, sub in enumerate(decomposed, start=1):
            if not isinstance(sub, dict):
                raise ValueError(f"sub_goal[{idx}] must be an object.")

            required_sub_fields = ["sub_goal", "rationale", "priority", "kpis", "action_plan"]
            for key in required_sub_fields:
                if key not in sub:
                    raise ValueError(f"sub_goal[{idx}] missing '{key}'.")

            if sub["priority"] not in ("high", "medium", "low"):
                raise ValueError(f"sub_goal[{idx}].priority must be one of high|medium|low.")

            # Validate KPIs
            kpis = sub.get("kpis", [])
            if not isinstance(kpis, list) or not (1 <= len(kpis) <= 2):
                raise ValueError(f"sub_goal[{idx}].kpis must contain 1~2 items.")

            for k_i, kpi in enumerate(kpis, start=1):
                required_kpi_fields = ["name", "metric", "target", "data_source"]
                if not all(k in kpi for k in required_kpi_fields):
                    raise ValueError(f"sub_goal[{idx}].kpis[{k_i}] missing required fields.")

            # Validate action plan
            actions = sub.get("action_plan", [])
            if not isinstance(actions, list) or not (2 <= len(actions) <= 5):
                raise ValueError(f"sub_goal[{idx}].action_plan must contain 2~5 items.")

            for a_i, act in enumerate(actions, start=1):
                required_action_fields = ["action_item", "suggested_agent", "due_days", "acceptance_criteria"]
                for key in required_action_fields:
                    if key not in act:
                        raise ValueError(f"sub_goal[{idx}].action_plan[{a_i}] missing '{key}'.")

                if act["suggested_agent"] not in allowed_agents:
                    raise ValueError(
                        f"sub_goal[{idx}].action_plan[{a_i}].suggested_agent must be one of {allowed_agents}."
                    )

                if not isinstance(act["due_days"], int) or not (1 <= act["due_days"] <= 30):
                    raise ValueError(f"sub_goal[{idx}].action_plan[{a_i}].due_days must be an integer 1~30.")

    async def save_goal_plan(self, plan: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save the goal plan using MCP filesystem server or local filesystem"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            goal_name = plan.get("original_goal", "unknown_goal").replace(" ", "_")[:30]
            filename = f"goal_plan_{goal_name}_{timestamp}.json"

        file_path = self.output_dir / filename

        try:
            # Try to use MCP filesystem server if available
            if self.filesystem_session and self.enable_mcp:
                result = await self._call_mcp_tool(
                    self.filesystem_session,
                    "write_file",
                    {
                        "file_path": str(file_path),
                        "content": json.dumps(plan, ensure_ascii=False, indent=2),
                        "encoding": "utf-8"
                    }
                )
                if "error" not in result:
                    self.logger.info(f"✅ Goal plan saved via MCP filesystem: {file_path}")
                    return str(file_path)

            # Fallback to local filesystem
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)

            self.logger.info(f"✅ Goal plan saved locally: {file_path}")
            return str(file_path)

        except Exception as e:
            self.logger.error(f"Failed to save goal plan: {e}")
            raise

    async def load_goal_plan(self, filename: str) -> Dict[str, Any]:
        """Load a goal plan using MCP filesystem server or local filesystem"""
        file_path = self.output_dir / filename

        try:
            # Try to use MCP filesystem server if available
            if self.filesystem_session and self.enable_mcp:
                result = await self._call_mcp_tool(
                    self.filesystem_session,
                    "read_file",
                    {
                        "file_path": str(file_path),
                        "encoding": "utf-8"
                    }
                )
                if "error" not in result:
                    return json.loads(result.get("content", "{}"))

            # Fallback to local filesystem
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load goal plan: {e}")
            raise

    async def list_saved_plans(self) -> List[Dict[str, Any]]:
        """List all saved goal plans using MCP filesystem server or local filesystem"""
        try:
            # Try to use MCP filesystem server if available
            if self.filesystem_session and self.enable_mcp:
                result = await self._call_mcp_tool(
                    self.filesystem_session,
                    "list_directory",
                    {"path": str(self.output_dir)}
                )
                if "error" not in result:
                    files = result.get("files", [])
                    return [{"name": f["name"], "path": f["path"]} for f in files if f["name"].endswith(".json")]

            # Fallback to local filesystem
            plans = []
            for file_path in self.output_dir.glob("*.json"):
                plans.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
            return plans

        except Exception as e:
            self.logger.error(f"Failed to list saved plans: {e}")
            return []

    async def execute_mcp_tool_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the MCP tool usage plan from the goal plan"""
        execution_results = {
            "plan_id": plan.get("metadata", {}).get("generated_at", "unknown"),
            "execution_start": datetime.now().isoformat(),
            "results": {},
            "errors": []
        }

        mcp_tool_usage = plan.get("mcp_tool_usage", {})

        try:
            # Execute filesystem operations
            if "filesystem_operations" in mcp_tool_usage and self.filesystem_session:
                for operation in mcp_tool_usage["filesystem_operations"]:
                    try:
                        result = await self._call_mcp_tool(
                            self.filesystem_session,
                            "write_file",
                            {
                                "file_path": f"execution_logs/{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                "content": f"Executed: {operation}\nTimestamp: {datetime.now().isoformat()}",
                                "encoding": "utf-8"
                            }
                        )
                        execution_results["results"][f"filesystem_{operation}"] = result
                    except Exception as e:
                        execution_results["errors"].append(f"Filesystem operation failed: {e}")

            # Execute search queries
            if "search_queries" in mcp_tool_usage and self.search_session:
                for query in mcp_tool_usage["search_queries"]:
                    try:
                        result = await self._call_mcp_tool(
                            self.search_session,
                            "search_web",
                            {"query": query, "count": 3}
                        )
                        execution_results["results"][f"search_{query}"] = result
                    except Exception as e:
                        execution_results["errors"].append(f"Search query failed: {e}")

            # Execute browser automation
            if "browser_automation" in mcp_tool_usage and self.browser_session:
                for automation in mcp_tool_usage["browser_automation"]:
                    try:
                        # This would be customized based on the specific automation task
                        result = await self._call_mcp_tool(
                            self.browser_session,
                            "navigate",
                            {"url": "https://www.google.com"}
                        )
                        execution_results["results"][f"browser_{automation}"] = result
                    except Exception as e:
                        execution_results["errors"].append(f"Browser automation failed: {e}")

        except Exception as e:
            execution_results["errors"].append(f"General execution error: {e}")

        execution_results["execution_end"] = datetime.now().isoformat()
        return execution_results

    def pretty_print_enhanced_plan(self, plan: Dict[str, Any]):
        """Print the enhanced plan in a human-readable format"""
        print("=" * 80)
        print(f"🎯 Enhanced Goal Plan: {plan.get('original_goal')}")
        print("=" * 80)

        # Print metadata
        metadata = plan.get("metadata", {})
        if metadata:
            print(f"\n📊 Metadata:")
            print(f"  - Generated: {metadata.get('generated_at', 'N/A')}")
            print(f"  - MCP Enabled: {metadata.get('mcp_enabled', False)}")
            print(f"  - Version: {metadata.get('version', 'N/A')}")

        # Print research context
        research_context = plan.get("research_context")
        if research_context:
            print(f"\n🔍 Research Context:")
            print(f"  {research_context}")

        # Print decomposed plan
        for i, sub_plan in enumerate(plan.get('decomposed_plan', []), 1):
            print(f"\n📋 Sub-Goal {i}: {sub_plan.get('sub_goal')}")
            print(f"  - Rationale: {sub_plan.get('rationale')}")
            print(f"  - Priority: {sub_plan.get('priority')}")

            print("  - KPIs:")
            for kpi in sub_plan.get('kpis', []):
                print(f"    - {kpi.get('name')}: {kpi.get('metric')} → {kpi.get('target')}")

            print("  - Action Plan:")
            for action in sub_plan.get('action_plan', []):
                print(f"    - Task: {action.get('action_item')}")
                print(f"      -> Agent: [{action.get('suggested_agent')}]")
                print(f"      -> Due: {action.get('due_days')} days")
                print(f"      -> Criteria: {action.get('acceptance_criteria')}")

        # Print MCP tool usage
        mcp_tool_usage = plan.get('mcp_tool_usage', {})
        if mcp_tool_usage:
            print(f"\n🔧 MCP Tool Usage Plan:")
            for tool_type, operations in mcp_tool_usage.items():
                print(f"  - {tool_type}: {operations}")

        print(f"\n🎯 Overall Success Criteria: {plan.get('overall_success_criteria')}")
        print("\n" + "=" * 80)

    async def cleanup(self):
        """Clean up MCP connections"""
        try:
            if self.filesystem_session:
                await self.filesystem_session.close()
            if self.search_session:
                await self.search_session.close()
            if self.browser_session:
                await self.browser_session.close()
            self.logger.info("✅ MCP connections cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup MCP connections: {e}")


async def main():
    """Main function for CLI execution"""
    parser = argparse.ArgumentParser(description="Enhanced MCP Goal-Setting Agent")
    parser.add_argument("--goal", required=True, help="The high-level goal to be decomposed")
    parser.add_argument("--output-dir", default="goal_plans", help="Directory to save goal plans")
    parser.add_argument("--enable-mcp", action="store_true", help="Enable MCP functionality")
    parser.add_argument("--no-research", action="store_true", help="Disable goal research phase")
    parser.add_argument("--list-plans", action="store_true", help="List existing goal plans")
    parser.add_argument("--load-plan", help="Load a specific goal plan by filename")

    args = parser.parse_args()

    # MCP server configurations
    mcp_servers = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "env": {"ALLOWED_PATHS": f"{args.output_dir},reports/,data/"}
        },
        "search": {
            "command": "npx",
            "args": ["-y", "g-search-mcp"],
            "env": {}
        },
        "browser": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
            "env": {}
        }
    }

    agent = MCPGoalSetterAgent(
        output_dir=args.output_dir,
        enable_mcp=args.enable_mcp,
        mcp_servers=mcp_servers if args.enable_mcp else {}
    )

    try:
        if args.list_plans:
            print("📋 Existing Goal Plans:")
            plans = await agent.list_saved_plans()
            for plan in plans:
                print(f"  - {plan['name']}")
            return

        if args.load_plan:
            print(f"📖 Loading goal plan: {args.load_plan}")
            plan = await agent.load_goal_plan(args.load_plan)
            agent.pretty_print_enhanced_plan(plan)
            return

        print(f"🧠 Decomposing goal: \"{args.goal}\"...")
        if args.enable_mcp:
            print("🔧 MCP integration enabled - using enhanced capabilities")

        plan = await agent.generate_enhanced_goal_plan(
            args.goal,
            enable_research=not args.no_research
        )

        print("\n✅ Successfully generated an enhanced strategic plan!")
        agent.pretty_print_enhanced_plan(plan)

        # Save the plan
        filename = await agent.save_goal_plan(plan)
        print(f"\n💾 Goal plan saved to: {filename}")

        # Execute MCP tool plan if MCP is enabled
        if args.enable_mcp and plan.get("mcp_tool_usage"):
            print("\n🔧 Executing MCP tool usage plan...")
            execution_results = await agent.execute_mcp_tool_plan(plan)
            print(f"✅ MCP execution completed with {len(execution_results.get('errors', []))} errors")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
