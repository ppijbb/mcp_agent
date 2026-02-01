import json
from srcs.common.templates import EnterpriseAgentTemplate
from srcs.common.config import get_llm, get_memory, get_tool_manager, get_comm_manager


class UltraAgenticLLMAgent(EnterpriseAgentTemplate):
    """
    MCP Agent 라이브러리 기반 초 agentic LLM 에이전트
    - LLM이 모든 의사결정/계획/상호작용/학습의 중심
    - Persistent state, goal-driven planning, tool 사용, multi-agent 협력, self-reflection
    """
    def __init__(self, agent_id, goal, tools=None, peers=None):
        super().__init__(agent_name=agent_id, business_scope=goal)
        self.goal = goal
        self.llm = get_llm()
        self.memory = get_memory(agent_id)
        self.tools = get_tool_manager(tools or [])
        self.comm = get_comm_manager(peers or [])
        self.state = {"status": "idle", "plan": None, "history": []}

    def perceive(self, observation):
        self.memory.store_observation(observation)
        self.state["last_obs"] = observation

    def decide_and_plan(self):
        prompt = self.build_prompt()
        llm_output = self.llm(prompt)
        plan, action, comms = self.parse_llm_output(llm_output)
        self.state["plan"] = plan
        return action, comms

    def act(self, action):
        result = self.tools.execute(action)
        self.memory.store_action(action, result)
        return result

    def communicate(self, comms):
        for peer, msg in comms.items():
            self.comm.send(peer, msg)
        self.memory.store_comm(comms)

    def reflect_and_learn(self, feedback):
        reflection_prompt = self.build_reflection_prompt(feedback)
        insights = self.llm(reflection_prompt)
        self.memory.store_reflection(insights)
        # 정책/프롬프트/기억 업데이트 로직 추가 가능

    def run(self, observation):
        self.perceive(observation)
        action, comms = self.decide_and_plan()
        result = self.act(action)
        self.communicate(comms)
        self.reflect_and_learn(result)

    def build_prompt(self):
        return f"""
You are an autonomous agent responsible for real-time anomaly detection and collaborative response.\nGoal: {self.goal}\nCurrent state: {json.dumps(self.state)}\nRecent memory: {self.memory.recall_recent()}\nObservation: {self.state.get('last_obs')}\nPeers: {self.comm.list_peers()}\nAvailable tools: {self.tools.list()}\nWhat is your next plan?\n- What action will you take (tool/API/code)?\n- Will you communicate with any peer agent? If so, what message?\n- How do you reflect on your last action/result?\nOutput in JSON: {{ \"plan\": ..., \"action\": ..., \"comms\": ... }}\n"""

    def parse_llm_output(self, output):
        try:
            parsed = json.loads(output)
            return parsed.get("plan"), parsed.get("action"), parsed.get("comms", {})
        except Exception:
            return None, None, {}

    def build_reflection_prompt(self, feedback):
        return f"""
Reflect on your last action and its result.\nFeedback: {feedback}\nCurrent state: {json.dumps(self.state)}\nWhat did you learn? How can you improve your next plan?\n"""
