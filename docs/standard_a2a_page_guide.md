# 표준 A2A Page 사용 가이드

## 개요

모든 pages에서 agent를 호출하는 방식을 표준화하여 일관성과 유지보수성을 확보합니다.

## 마이그레이션 현황

### 완료된 Pages (11개)
- ✅ swarm.py
- ✅ parallel.py
- ✅ self_evolving_swarm.py
- ✅ customer_white_hacking.py
- ✅ real_estate_agent.py
- ✅ smart_shopping_assistant.py
- ✅ product_innovation.py
- ✅ seo_doctor.py
- ✅ supply_chain.py
- ✅ revenue_operations.py
- ✅ ultra_agentic_llm.py

### 자동 마이그레이션 스크립트

남은 pages는 `scripts/migrate_pages_to_standard_a2a.py` 스크립트를 사용하여 자동으로 마이그레이션할 수 있습니다:

```bash
python scripts/migrate_pages_to_standard_a2a.py
```

**주의**: 자동 마이그레이션 스크립트는 기본적인 패턴만 변경합니다. 복잡한 구조의 pages는 수동으로 마이그레이션해야 할 수 있습니다.

## 표준화된 구성 요소

### 1. 표준 헬퍼 함수 (`srcs/common/standard_a2a_page_helper.py`)

#### `create_standard_agent_metadata()`
표준화된 agent_metadata 생성

```python
from srcs.common.standard_a2a_page_helper import create_standard_agent_metadata
from srcs.common.agent_interface import AgentType

metadata = create_standard_agent_metadata(
    agent_id="my_agent",
    agent_name="My Agent",
    entry_point="srcs.my_module.run_agent",
    agent_type=AgentType.MCP_AGENT,
    capabilities=["capability1", "capability2"],
    description="Agent description"
)
```

#### `create_standard_input_data()`
Agent 타입에 따라 표준화된 input_data 생성

```python
from srcs.common.standard_a2a_page_helper import create_standard_input_data

# MCP Agent (클래스 기반)
input_data = create_standard_input_data(
    agent_type=AgentType.MCP_AGENT,
    entry_point="srcs.my_module.run_agent",
    class_name="MyAgentRunner",
    method_name="run_task",
    result_json_path=str(result_json_path),
    task="my task",
    param1="value1"
)

# MCP Agent (함수 기반)
input_data = create_standard_input_data(
    agent_type=AgentType.MCP_AGENT,
    entry_point="srcs.my_module.run_agent",
    method_name="main",
    result_json_path=str(result_json_path),
    task="my task"
)

# LangGraph Agent
input_data = create_standard_input_data(
    agent_type=AgentType.LANGGRAPH_AGENT,
    entry_point="lang_graph.my_agent",
    result_json_path=str(result_json_path),
    messages=[{"role": "user", "content": "query"}],
    query="my query"
)
```

#### `execute_standard_agent_via_a2a()`
표준화된 방식으로 A2A를 통해 agent 실행

```python
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a

result = execute_standard_agent_via_a2a(
    placeholder=st.empty(),
    agent_id="my_agent",
    agent_name="My Agent",
    entry_point="srcs.my_module.run_agent",
    agent_type=AgentType.MCP_AGENT,
    capabilities=["capability1"],
    description="Agent description",
    input_params={"task": "my task"},
    class_name="MyAgentRunner",  # 클래스 기반인 경우
    method_name="run_task",      # 클래스/함수 기반인 경우
    result_json_path=result_json_path
)
```

#### `process_standard_agent_result()`
표준화된 방식으로 결과 처리

```python
from srcs.common.standard_a2a_page_helper import process_standard_agent_result

processed = process_standard_agent_result(result, "my_agent")

if processed["success"] and processed["has_data"]:
    display_results(processed["data"])
else:
    st.error(f"Error: {processed.get('error')}")
```

### 2. 표준 Page 템플릿 (`srcs/common/standard_a2a_page_template.py`)

#### `create_standard_a2a_page()`
완전한 표준화된 page 생성

```python
from srcs.common.standard_a2a_page_template import create_standard_a2a_page
from srcs.common.agent_interface import AgentType

def display_results(result_data):
    st.json(result_data)

def main():
    create_standard_a2a_page(
        agent_id="my_agent",
        agent_name="My Agent",
        page_icon="🤖",
        page_type="my_agent",
        title="My Agent",
        subtitle="My Agent Description",
        entry_point="srcs.my_module.run_agent",
        agent_type=AgentType.MCP_AGENT,
        capabilities=["capability1", "capability2"],
        description="My Agent Description",
        form_fields=[
            {
                "type": "text_area",
                "key": "task",
                "label": "작업 설명",
                "default": "",
                "height": 150,
                "help": "작업을 설명하세요",
                "required": True
            },
            {
                "type": "slider",
                "key": "count",
                "label": "개수",
                "min_value": 1,
                "max_value": 10,
                "default": 5
            }
        ],
        display_results_func=display_results,
        result_category="my_category"
    )
```

#### `create_simple_a2a_page()`
간단한 page 생성 (최소 설정)

```python
from srcs.common.standard_a2a_page_template import create_simple_a2a_page

def main():
    create_simple_a2a_page(
        agent_id="my_agent",
        agent_name="My Agent",
        page_icon="🤖",
        entry_point="srcs.my_module.run_agent",
        form_config={
            "fields": [
                {"type": "text_area", "key": "task", "label": "Task"}
            ],
            "result_category": "my_category"
        }
    )
```

## 마이그레이션 가이드

### 기존 코드 (비표준)

```python
agent_metadata = {
    "agent_id": "my_agent",
    "agent_name": "My Agent",
    "entry_point": "srcs.my_module.run_agent",
    "agent_type": "mcp_agent",  # 문자열 사용
    "capabilities": ["cap1"],
    "description": "Description"
}

input_data = {
    "task": task,
    "result_json_path": str(result_json_path)
}

result = run_agent_via_a2a(
    placeholder=placeholder,
    agent_metadata=agent_metadata,
    input_data=input_data,
    result_json_path=result_json_path,
    use_a2a=True
)

if result and "data" in result:
    display_results(result["data"])
```

### 표준화된 코드

#### 방법 1: 표준 헬퍼 함수 사용

```python
from srcs.common.standard_a2a_page_helper import (
    execute_standard_agent_via_a2a,
    process_standard_agent_result
)
from srcs.common.agent_interface import AgentType

result = execute_standard_agent_via_a2a(
    placeholder=placeholder,
    agent_id="my_agent",
    agent_name="My Agent",
    entry_point="srcs.my_module.run_agent",
    agent_type=AgentType.MCP_AGENT,  # enum 사용
    capabilities=["cap1"],
    description="Description",
    input_params={"task": task},
    result_json_path=result_json_path
)

processed = process_standard_agent_result(result, "my_agent")
if processed["success"] and processed["has_data"]:
    display_results(processed["data"])
```

#### 방법 2: 표준 템플릿 사용 (권장)

```python
from srcs.common.standard_a2a_page_template import create_standard_a2a_page
from srcs.common.agent_interface import AgentType

def display_results(result_data):
    st.json(result_data)

def main():
    create_standard_a2a_page(
        agent_id="my_agent",
        agent_name="My Agent",
        page_icon="🤖",
        page_type="my_agent",
        title="My Agent",
        subtitle="Description",
        entry_point="srcs.my_module.run_agent",
        agent_type=AgentType.MCP_AGENT,
        capabilities=["cap1"],
        description="Description",
        form_fields=[
            {
                "type": "text_area",
                "key": "task",
                "label": "Task",
                "required": True
            }
        ],
        display_results_func=display_results,
        result_category="my_category"
    )
```

## Agent 타입별 패턴

### MCP Agent (클래스 기반)

```python
# Runner 클래스 필요
class MyAgentRunner:
    async def run_task(self, task: str, **kwargs):
        # 실행 로직
        return {"result": "..."}

# Page에서 사용
execute_standard_agent_via_a2a(
    ...,
    class_name="MyAgentRunner",
    method_name="run_task",
    input_params={"task": "..."}
)
```

### MCP Agent (함수 기반)

```python
# 함수만 필요
async def run_agent(task: str, **kwargs):
    # 실행 로직
    return {"result": "..."}

# Page에서 사용
execute_standard_agent_via_a2a(
    ...,
    method_name="run_agent",
    input_params={"task": "..."}
)
```

### LangGraph Agent

```python
# LangGraph app 필요
# lang_graph/my_agent/app.py
app = create_agent_app(...)

# Page에서 사용
execute_standard_agent_via_a2a(
    ...,
    agent_type=AgentType.LANGGRAPH_AGENT,
    input_params={
        "messages": [{"role": "user", "content": "query"}],
        "query": "query"
    }
)
```

### SparkleForge Agent

```python
execute_standard_agent_via_a2a(
    ...,
    agent_type=AgentType.SPARKLEFORGE_AGENT,
    input_params={
        "query": "query",
        "context": {...}
    }
)
```

## 필수 사항

1. **AgentType enum 사용**: 항상 `AgentType.MCP_AGENT` 형식 사용 (문자열 금지)
2. **표준 헬퍼 함수 사용**: 직접 `run_agent_via_a2a` 호출 대신 표준 함수 사용
3. **결과 처리 표준화**: `process_standard_agent_result` 사용
4. **일관된 에러 처리**: 표준화된 에러 메시지 형식

## 선택 사항

1. **표준 템플릿 사용**: `create_standard_a2a_page` 사용 시 자동으로 표준 패턴 적용
2. **결과 카테고리 지정**: `result_category` 지정 시 자동으로 최신 결과 표시
3. **커스텀 결과 표시**: `display_results_func` 제공 시 커스텀 UI 가능

## 예제

완전한 예제는 `pages/swarm.py`를 참고하세요.

