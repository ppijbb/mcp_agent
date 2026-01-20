"""
HobbyList Integration Test
전체 시스템 통합 테스트
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime


async def test_mcp_servers():
    """MCP Server 테스트"""
    print("=" * 60)
    print("TEST 1: MCP Servers")
    print("=" * 60)
    
    try:
        from mcp.web_search_mcp import WebSearchMCPServer
        from mcp.market_search_mcp import MarketSearchMCPServer
        from mcp.event_search_mcp import EventSearchMCPServer
        from mcp.location_search_mcp import LocationSearchMCPServer
        
        web_mcp = WebSearchMCPServer()
        market_mcp = MarketSearchMCPServer()
        event_mcp = EventSearchMCPServer()
        location_mcp = LocationSearchMCPServer()
        
        # Test web search
        result = await web_mcp.search_hobby_trends(category="운동")
        print(f"✅ Web Search: {result['success']}")
        
        # Test market search
        result = await market_mcp.search_discounts(hobby="등산")
        print(f"✅ Market Search: {result['success']}")
        
        # Test event search
        result = await event_mcp.search_events(hobby="요리")
        print(f"✅ Event Search: {result['success']}")
        
        # Test location search
        result = await location_mcp.search_nearby_shops(hobby="등산", location="서울 강남구")
        print(f"✅ Location Search: {result['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP Server Test Failed: {e}")
        return False


async def test_sales_agents():
    """Sales Agent 테스트"""
    print("\n" + "=" * 60)
    print("TEST 2: Sales Agents")
    print("=" * 60)
    
    try:
        from autogen.sales_agents import (
            sales_manager,
            ticket_agent,
            starter_pack_agent,
            order_agent,
            bulk_agent
        )
        
        # Test Ticket Agent
        result = await ticket_agent.search_tickets(hobby="등산")
        print(f"✅ Ticket Search: {result['success']}")
        
        # Test Starter Pack Agent
        result = await starter_pack_agent.get_starter_pack(hobby="등산")
        print(f"✅ Starter Pack: {result['success']}")
        
        # Test recommended packs
        result = await starter_pack_agent.get_recommended_packs(
            user_profile={"interests": ["등산", "야외활동"]}
        )
        print(f"✅ Recommended Packs: {result['success']}")
        
        # Test Bulk Agent
        result = await bulk_agent.search_bulk_deals(product_name="등산화", quantity=50)
        print(f"✅ Bulk Deals: {result['success']}")
        
        # Test Order Agent
        order_result = await order_agent.create_order(
            user_id="test_user",
            products=[{"product_id": "TKT-001", "name": "테스트 티켓", "price": 50000}]
        )
        print(f"✅ Order Create: {order_result['success']}")
        
        # Test payment
        if order_result['success']:
            payment_result = await order_agent.process_payment(
                order_id=order_result['data']['order_id'],
                payment_method="card",
                payment_details={"card_no": "****-****-****-1234"}
            )
            print(f"✅ Payment Process: {payment_result['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sales Agent Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_sales_bridge():
    """Sales A2A Bridge 테스트"""
    print("\n" + "=" * 60)
    print("TEST 3: Sales A2A Bridge")
    print("=" * 60)
    
    try:
        from bridge.sales_bridge import sales_a2a_bridge
        
        # Test ticket recommendation request
        result = await sales_a2a_bridge.request_ticket_recommendation(
            hobby="등산",
            user_profile={"location": "서울"}
        )
        print(f"✅ Ticket Recommendation: {result['success']}")
        
        # Test starter pack request
        result = await sales_a2a_bridge.request_starter_pack(
            hobby="등산",
            user_profile={"interests": ["운동", "야외"]}
        )
        print(f"✅ Starter Pack Request: {result['success']}")
        
        # Test price quote
        result = await sales_a2a_bridge.get_price_quote(
            hobby="등산",
            item_type="pack"
        )
        print(f"✅ Price Quote: {result['success']}")
        
        # Check message history
        history = sales_a2a_bridge.get_message_history()
        print(f"✅ Message History: {len(history)} messages")
        
        return True
        
    except Exception as e:
        print(f"❌ Sales Bridge Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration_scenario():
    """통합 시나리오 테스트"""
    print("\n" + "=" * 60)
    print("TEST 4: Integration Scenario")
    print("=" * 60)
    
    try:
        from bridge.sales_bridge import sales_a2a_bridge
        from autogen.sales_agents import order_agent, bulk_agent
        
        # Scenario: User explores "등산" hobby and wants to purchase
        
        print("\n📍 Step 1: User explores hiking (등산)")
        result = await sales_a2a_bridge.request_ticket_recommendation(
            hobby="등산",
            user_profile={"location": "서울"}
        )
        print(f"   Found {len(result.get('data', {}).get('tickets', []))} tickets")
        
        print("\n📦 Step 2: User wants starter pack")
        result = await sales_a2a_bridge.request_starter_pack(
            hobby="등산",
            user_profile={"interests": ["등산"]}
        )
        pack = result.get('data', {}).get('recommended_packs', [])
        if pack:
            print(f"   Recommended: {pack[0].get('name', 'N/A')}")
            print(f"   Price: ₩{pack[0].get('pack_price', 0):,}")
        
        print("\n💰 Step 3: Check bulk pricing")
        result = await bulk_agent.search_bulk_deals(product_name="등산화", quantity=100)
        deals = result.get('data', {}).get('deals', [])
        if deals:
            best_deal = min(deals, key=lambda x: x.get('unit_price', 0))
            print(f"   Best bulk price: ₩{best_deal.get('unit_price', 0):,}/개")
        
        print("\n🛒 Step 4: Create sample order")
        order_result = await order_agent.create_order(
            user_id="test_user_001",
            products=[
                {"product_id": "PACK-001", "name": "등산 입문 키트", "price": 99000}
            ]
        )
        print(f"   Order ID: {order_result['data'].get('order_id', 'N/A')}")
        
        print("\n✅ Integration Scenario Completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration Scenario Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 테스트 실행"""
    print("\n" + "🎯" * 25)
    print("HOBYLIST AGENT INTEGRATION TEST")
    print("🎯" * 25)
    
    results = {}
    
    # Run all tests
    results["MCP Servers"] = await test_mcp_servers()
    results["Sales Agents"] = await test_sales_agents()
    results["Sales Bridge"] = await test_sales_bridge()
    results["Integration"] = await test_integration_scenario()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n{'=' * 60}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
