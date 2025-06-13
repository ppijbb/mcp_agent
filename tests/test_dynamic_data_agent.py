import asyncio
from typing import List, Dict

import pytest

# Import the agent under test
import srcs.urban_hive.dynamic_data_agent as _module  # type: ignore

# ---------------------------------------------------------------------------
# Helper stubs / fixtures
# ---------------------------------------------------------------------------

class _DummyExternalManager:
    """Minimal stub for external_data_manager used by DynamicDataAgent."""

    async def get_districts(self, region: str = "seoul") -> List[str]:
        return ["강남구", "서초구", "송파구"]

    async def get_community_data(self) -> Dict:
        return {
            "members": [
                {
                    "name": "테스트회원",
                    "age": 30,
                    "interests": ["fitness"],
                    "location": "강남구",
                    "activity_level": "high",
                    "expertise_areas": ["python"],
                }
            ],
            "groups": [
                {
                    "name": "테스트그룹",
                    "type": "fitness",
                    "members": 10,
                    "location": "강남구",
                    "schedule": "매일 06:00",
                    "skill_level": "beginner",
                }
            ],
        }

    async def get_resource_data(self) -> Dict:
        return {
            "available": [
                {
                    "name": "전동드릴",
                    "category": "item",
                    "owner": "테스트회원",
                    "location": "서초구",
                    "condition": "양호",
                    "rental_price": 0,
                    "delivery_available": False,
                    "available_until": "2099-01-01T00:00:00",
                }
            ],
            "requests": [
                {
                    "name": "사다리",
                    "category": "item",
                    "requester": "요청자",
                    "location": "송파구",
                    "urgency": "보통",
                    "max_rental_price": 0,
                    "needed_by": "2099-01-02T00:00:00",
                }
            ],
        }


@pytest.fixture(autouse=True)
def _patch_external_manager(monkeypatch):
    """Patch dynamic_data_agent to use dummy external manager during tests."""

    # Replace the real external manager with our stub
    _module.dynamic_data_agent.external_manager = _DummyExternalManager()  # type: ignore
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_dynamic_districts():
    districts = await _module.dynamic_data_agent.get_dynamic_districts("seoul")
    assert districts == ["강남구", "서초구", "송파구"]


@pytest.mark.asyncio
async def test_get_dynamic_community_members():
    members = await _module.dynamic_data_agent.get_dynamic_community_members()
    assert len(members) == 1
    assert members[0]["name"] == "테스트회원"


@pytest.mark.asyncio
async def test_get_dynamic_resources_available():
    resources = await _module.dynamic_data_agent.get_dynamic_resources("available")
    assert resources and resources[0]["name"] == "전동드릴"


@pytest.mark.asyncio
async def test_get_dynamic_resources_requests():
    requests = await _module.dynamic_data_agent.get_dynamic_resources("requests")
    assert requests and requests[0]["name"] == "사다리" 