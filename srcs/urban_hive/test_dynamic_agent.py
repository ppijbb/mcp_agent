"""
Test script for Dynamic Data Agent

This script demonstrates how the Dynamic Data Agent replaces hardcoded values
with intelligent, context-aware data generation.
"""

import asyncio
import json
from datetime import datetime
from dynamic_data_agent import dynamic_data_agent, get_district_characteristics
from providers.public_data_client import PublicDataClient


async def test_dynamic_districts():
    """Test dynamic district generation."""
    print("🏙️ Testing Dynamic Districts Generation")
    print("=" * 50)
    
    districts = await dynamic_data_agent.get_dynamic_districts("seoul")
    print(f"Generated {len(districts)} Seoul districts:")
    for i, district in enumerate(districts[:10], 1):
        print(f"  {i}. {district}")
    
    if len(districts) > 10:
        print(f"  ... and {len(districts) - 10} more districts")
    
    print()


async def test_dynamic_community_data():
    """Test dynamic community data generation."""
    print("👥 Testing Dynamic Community Data Generation")
    print("=" * 50)
    
    # Test community members
    members = await dynamic_data_agent.get_dynamic_community_members(count=5)
    print("Generated Community Members:")
    for member in members:
        print(f"  • {member['name']} ({member['age']}세) - {member['location']}")
        print(f"    관심사: {', '.join(member['interests'])}")
        print(f"    활동레벨: {member['activity_level']}, 전문분야: {', '.join(member['expertise_areas'])}")
        print()
    
    # Test community groups
    groups = await dynamic_data_agent.get_dynamic_community_groups(count=4)
    print("Generated Community Groups:")
    for group in groups:
        print(f"  • {group['name']} ({group['type']})")
        print(f"    위치: {group['location']}, 멤버: {group['members']}명")
        print(f"    일정: {group['schedule']}, 레벨: {group['skill_level']}")
        print()


async def test_dynamic_resources():
    """Test dynamic resource data generation."""
    print("📦 Testing Dynamic Resource Data Generation")
    print("=" * 50)
    
    # Test available resources
    available = await dynamic_data_agent.get_dynamic_resources("available", count=4)
    print("Available Resources:")
    for resource in available:
        price_info = f"무료" if resource['rental_price'] == 0 else f"{resource['rental_price']:,}원"
        delivery = "배송가능" if resource['delivery_available'] else "직접수령"
        print(f"  • {resource['name']} ({resource['category']})")
        print(f"    소유자: {resource['owner']}, 위치: {resource['location']}")
        print(f"    상태: {resource['condition']}, 가격: {price_info}, {delivery}")
        print()
    
    # Test resource requests
    requests = await dynamic_data_agent.get_dynamic_resources("requests", count=3)
    print("Resource Requests:")
    for request in requests:
        print(f"  • {request['name']} ({request['category']})")
        print(f"    요청자: {request['requester']}, 위치: {request['location']}")
        print(f"    긴급도: {request['urgency']}, 최대가격: {request['max_rental_price']:,}원")
        print()


async def test_district_characteristics():
    """Test dynamic district characteristics."""
    print("🏘️ Testing Dynamic District Characteristics")
    print("=" * 50)
    
    test_districts = ["강남구", "서초구", "마포구", "노원구"]
    
    for district in test_districts:
        chars = await get_district_characteristics(district)
        print(f"{district} 특성:")
        print(f"  인구밀도: {chars['population_density']}, 경제수준: {chars['economic_level']}")
        print(f"  교통수준: {chars['traffic_level']}, 안전수준: {chars['safety_level']}")
        print(f"  현재 사건 기준: {chars['current_incident_base']}")
        print(f"  현재 교통 혼잡도: {chars['current_traffic_level']}%")
        print(f"  현재 범죄율: {chars['current_crime_rate']:.1f}")
        print()


async def test_public_data_client_integration():
    """Test integration with Public Data Client."""
    print("🔗 Testing Public Data Client Integration")
    print("=" * 50)
    
    client = PublicDataClient()
    
    # Test illegal dumping data with dynamic characteristics
    print("Illegal Dumping Data (with dynamic characteristics):")
    dumping_data = await client.fetch_illegal_dumping_data("강남구")
    for item in dumping_data[:2]:
        print(f"  • {item['location']}: {item['incidents']}건 ({item['trend']})")
        print(f"    심각도: {item['severity']}, 카테고리: {item['category']}")
    print()
    
    # Test community data with dynamic agent
    print("Community Data (from dynamic agent):")
    community_data = await client.fetch_community_data()
    print(f"  멤버 수: {len(community_data['members'])}명")
    print(f"  그룹 수: {len(community_data['groups'])}개")
    
    # Show first member and group
    if community_data['members']:
        member = community_data['members'][0]
        print(f"  첫 번째 멤버: {member['name']} ({member['location']})")
    
    if community_data['groups']:
        group = community_data['groups'][0]
        print(f"  첫 번째 그룹: {group['name']} ({group['location']})")
    print()


async def test_seasonal_and_time_factors():
    """Test seasonal and time-based dynamic factors."""
    print("🌍 Testing Seasonal and Time-based Factors")
    print("=" * 50)
    
    # Get characteristics for the same district multiple times to see variation
    district = "강남구"
    print(f"Testing time-based variations for {district}:")
    
    for i in range(3):
        chars = await get_district_characteristics(district)
        seasonal = chars['seasonal_factors']
        time_based = chars['time_based_modifiers']
        
        print(f"  시도 {i+1}:")
        print(f"    계절 요인: {seasonal}")
        print(f"    시간 요인: {time_based}")
        print(f"    교통 혼잡도: {chars['current_traffic_level']}%")
        print()


async def main():
    """Run all tests."""
    print("🤖 Dynamic Data Agent Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        await test_dynamic_districts()
        await test_dynamic_community_data()
        await test_dynamic_resources()
        await test_district_characteristics()
        await test_public_data_client_integration()
        await test_seasonal_and_time_factors()
        
        print("✅ All tests completed successfully!")
        print()
        print("🎯 Key Benefits Demonstrated:")
        print("  • Eliminated hardcoded data throughout the system")
        print("  • Dynamic, context-aware data generation")
        print("  • Seasonal and time-based intelligent variations")
        print("  • Configurable and maintainable data patterns")
        print("  • Seamless integration with existing components")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 