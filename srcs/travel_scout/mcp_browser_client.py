#!/usr/bin/env python3
"""
MCP Browser Client for Travel Search

Uses MCP Browser Use server to control browser in incognito mode
for travel data collection without price manipulation.
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import AsyncExitStack

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .travel_utils import TravelSearchUtils
from .scrapers import BookingComScraper, GoogleFlightsScraper

logger = logging.getLogger(__name__)


class MCPBrowserClient:
    """MCP Browser client for travel search automation"""
    
    def __init__(self, config: Optional[Dict] = None, debug: bool = False):
        self.config = config or {}
        self.debug = debug  # When True: keep browser visible and capture screenshots
        self.session = None
        self.exit_stack = AsyncExitStack()
        self._setup_browser_config()
    
    def _setup_browser_config(self):
        """Setup browser configuration for incognito mode"""
        # MCP Browser Use environment variables for incognito browsing
        # Override headless/keep-open based on debug flag
        headless_val = 'false' if self.debug else 'true'
        keep_open_val = 'true' if self.debug else 'false'
        
        # GPU 가속 비활성화 및 샌드박스 옵션 추가 (WSL 렌더링 문제 해결)
        # https://github.com/puppeteer/puppeteer/blob/main/docs/troubleshooting.md#setting-up-chrome-linux-sandbox
        browser_args = [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-gpu',
            '--disable-dev-shm-usage'
        ]
        
        browser_env = {
            # LLM Configuration (required)
            'MCP_LLM_PROVIDER': 'vertexai',
            'MCP_LLM_MODEL_NAME': 'gemini-2.0-flash-lite',
            
            # Browser Configuration for Incognito Mode
            'MCP_BROWSER_HEADLESS': headless_val,
            'MCP_BROWSER_DISABLE_SECURITY': 'false',
            'MCP_BROWSER_USER_DATA_DIR': '',  # Empty for fresh sessions
            'MCP_BROWSER_KEEP_OPEN': keep_open_val,
            'MCP_BROWSER_LAUNCH_ARGS': ",".join(browser_args),
            
            # Agent Tool Configuration
            'MCP_AGENT_TOOL_MAX_STEPS': '20',
            'MCP_AGENT_TOOL_MAX_ACTIONS_PER_STEP': '5',
            'MCP_AGENT_TOOL_USE_VISION': 'true',
            
            # Paths for temporary data
            'MCP_PATHS_DOWNLOADS': './tmp/downloads',
            'MCP_AGENT_TOOL_HISTORY_PATH': './tmp/history',
            
            # Server Configuration
            'MCP_SERVER_LOGGING_LEVEL': 'INFO',
            'MCP_SERVER_ANONYMIZED_TELEMETRY': 'false'
        }
        
        # Update environment
        for key, value in browser_env.items():
            if not os.getenv(key):
                os.environ[key] = value
        
        # Create directories
        os.makedirs('./tmp/downloads', exist_ok=True)
        if self.debug:
            os.makedirs('./tmp/debug_screenshots', exist_ok=True)
            # Add screenshot path to env
            os.environ['MCP_PATHS_SCREENSHOTS'] = os.path.abspath('./tmp/debug_screenshots')
        os.makedirs('./tmp/history', exist_ok=True)
    
    async def connect_to_mcp_server(self):
        """Connect to MCP Browser Use server"""
        try:
            # Base command
            args = ["-y", "@modelcontextprotocol/server-puppeteer"]
            
            # Add debug arguments if needed
            if self.debug:
                debug_args = [
                    '--headless=false',
                    '--launch-arg=--no-sandbox',
                    '--launch-arg=--disable-setuid-sandbox',
                    '--launch-arg=--disable-gpu',
                    '--launch-arg=--disable-dev-shm-usage',
                ]
                args.extend(debug_args)

            # MCP Browser Use server parameters
            server_params = StdioServerParameters(
                command="npx",
                args=args,
                env=dict(os.environ)  # Pass current environment with MCP config
            )
            
            # Connect to MCP server
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            # Initialize the session
            await self.session.initialize()
            
            # List available tools
            tools_response = await self.session.list_tools()
            available_tools = [tool.name for tool in tools_response.tools]
            logger.info(f"🔗 Connected to MCP Browser Use server with tools: {available_tools}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP Browser Use server: {e}")
            return False
    
    async def search_hotels_incognito(self, destination: str, check_in: str, check_out: str) -> List[Dict]:
        """Search for hotels using MCP browser in incognito mode"""
        if not self.session:
            connected = await self.connect_to_mcp_server()
            if not connected:
                raise ConnectionError("MCP Browser connection failed. Cannot search for hotels.")
        
        logger.info(f"🏨 Searching hotels for {destination} ({check_in} to {check_out})")
        
        try:
            # Step 1: Navigate to Booking.com
            logger.info("🌐 Navigating to Booking.com...")
            await self.session.call_tool(
                "puppeteer_navigate",
                arguments={"url": "https://www.booking.com"}
            )
            await self._debug_screenshot("after_navigate_booking")
            
            # Step 2: Fill in destination
            logger.info(f"📍 Entering destination: {destination}")
            await self.session.call_tool(
                "puppeteer_fill",
                arguments={
                    "selector": "input[name='ss'], input[placeholder*='destination'], #ss",
                    "value": destination
                }
            )
            
            # Step 3: Fill in check-in date
            logger.info(f"📅 Setting check-in date: {check_in}")
            await self.session.call_tool(
                "puppeteer_fill",
                arguments={
                    "selector": "input[name='checkin'], input[data-placeholder*='check-in']",
                    "value": check_in
                }
            )
            
            # Step 4: Fill in check-out date  
            logger.info(f"📅 Setting check-out date: {check_out}")
            await self.session.call_tool(
                "puppeteer_fill",
                arguments={
                    "selector": "input[name='checkout'], input[data-placeholder*='check-out']",
                    "value": check_out
                }
            )
            
            # Step 5: Click search button
            logger.info("🔍 Clicking search button...")
            await self.session.call_tool(
                "puppeteer_click",
                arguments={
                    "selector": "button[type='submit'], .sb-searchbox__button, button:contains('Search')"
                }
            )
            await self._debug_screenshot("after_click_search_button")
            
            # Step 6: Wait for results to load and extract data
            logger.info("⏳ Waiting for search results...")
            await asyncio.sleep(3)  # Give time for results to load
            await self._debug_screenshot("after_wait_results")
            
            # Step 7: Extract hotel data from results page
            extract_script = """
            () => {
                const hotels = [];
                const hotelElements = document.querySelectorAll('[data-testid="property-card"], .sr_property_block, .property_card, [data-testid="property-card-desktop"]');
                
                hotelElements.forEach(element => {
                    const nameEl = element.querySelector('h3, h4, .sr-hotel__name, [data-testid="title"], a[data-testid="title-link"]');
                    const priceEl = element.querySelector('.bui-price-display__value, .sr_price_wrap, [data-testid="price-and-discounted-price"], .prco-valign-middle-helper');
                    const ratingEl = element.querySelector('.bui-rating__title, .bui-review-score__badge, [data-testid="review-score"], .ac78a73c96');
                    const locationEl = element.querySelector('.sr_card_address, [data-testid="address"], .f419a93f12');
                    
                    if (nameEl && nameEl.textContent.trim()) {
                        hotels.push({
                            name: nameEl.textContent.trim(),
                            price: priceEl?.textContent?.trim() || 'N/A',
                            rating: ratingEl?.textContent?.trim() || 'N/A',
                            location: locationEl?.textContent?.trim() || ''
                        });
                    }
                });
                
                return hotels.slice(0, 10);
            }
            """
            
            extract_result = await self.session.call_tool(
                "puppeteer_evaluate", 
                arguments={"script": extract_script}
            )
            await self._debug_screenshot("after_extract_hotels")
            
            # Parse extracted data
            hotels = []
            if hasattr(extract_result, 'content') and extract_result.content:
                result_text = extract_result.content[0].text if extract_result.content else ""
                if result_text.strip():
                    try:
                        extracted_data = json.loads(result_text)
                        for item in extracted_data:
                            if item.get('name'):  # Only include hotels with actual names
                                hotel = {
                                    'name': item.get('name'),
                                    'price': item.get('price', 'N/A'),
                                    'price_numeric': TravelSearchUtils.extract_price_from_text(item.get('price', '0')),
                                    'rating': item.get('rating', 'N/A'),
                                    'rating_numeric': TravelSearchUtils.extract_rating_from_text(item.get('rating', '0')),
                                    'location': item.get('location', destination),
                                    'platform': 'booking.com',
                                    'source': 'Real Booking.com Scraping',
                                    'quality_score': TravelSearchUtils.calculate_hotel_quality_score(item),
                                    'meets_quality_criteria': TravelSearchUtils.extract_rating_from_text(item.get('rating', '0')) >= 4.0
                                }
                                hotels.append(hotel)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse extracted hotel data")
            
            if hotels:
                logger.info(f"✅ Found {len(hotels)} real hotels from Booking.com")
                return hotels
            else:
                logger.warning("No real hotels found, using minimal fallback")
                return [{
                    'name': f'No hotels found for {destination}',
                    'price': 'N/A',
                    'price_numeric': 0,
                    'rating': 'N/A', 
                    'rating_numeric': 0,
                    'location': destination,
                    'platform': 'booking.com',
                    'source': 'Search Failed - No Results',
                    'quality_score': 0,
                    'meets_quality_criteria': False
                }]
                
        except Exception as e:
            logger.error(f"Hotel search failed: {e}")
            return [{
                'name': f'Hotel search error for {destination}',
                'price': 'Error',
                'price_numeric': 0,
                'rating': 'Error',
                'rating_numeric': 0,
                'location': destination,
                'platform': 'booking.com',
                'source': f'Error: {str(e)}',
                'quality_score': 0,
                'meets_quality_criteria': False
            }]
    
    async def search_flights_incognito(self, origin: str, destination: str, departure_date: str, return_date: str) -> List[Dict]:
        """Search for flights using MCP browser in incognito mode"""
        if not self.session:
            connected = await self.connect_to_mcp_server()
            if not connected:
                raise ConnectionError("MCP Browser connection failed. Cannot search for flights.")

        logger.info(f"✈️ Searching flights {origin} -> {destination} ({departure_date} to {return_date})")
        
        try:
            # Step 1: Navigate to Google Flights
            logger.info("🌐 Navigating to Google Flights...")
            await self.session.call_tool(
                "puppeteer_navigate",
                arguments={"url": "https://www.google.com/travel/flights"}
            )
            await self._debug_screenshot("after_navigate_flights")
            
            # Step 2: Fill origin
            logger.info(f"🛫 Entering origin: {origin}")
            await self.session.call_tool(
                "puppeteer_fill",
                arguments={
                    "selector": "input[placeholder*='Where from'], input[aria-label*='Where from']",
                    "value": origin
                }
            )
            
            # Step 3: Fill destination
            logger.info(f"🛬 Entering destination: {destination}")
            await self.session.call_tool(
                "puppeteer_fill",
                arguments={
                    "selector": "input[placeholder*='Where to'], input[aria-label*='Where to']",
                    "value": destination
                }
            )
            
            # Step 4: Set departure date
            logger.info(f"📅 Setting departure date: {departure_date}")
            await self.session.call_tool(
                "puppeteer_fill",
                arguments={
                    "selector": "input[placeholder*='Departure'], input[aria-label*='Departure']",
                    "value": departure_date
                }
            )
            
            # Step 5: Set return date
            logger.info(f"📅 Setting return date: {return_date}")
            await self.session.call_tool(
                "puppeteer_fill",
                arguments={
                    "selector": "input[placeholder*='Return'], input[aria-label*='Return']",
                    "value": return_date
                }
            )
            
            # Step 6: Click search
            logger.info("🔍 Searching flights...")
            await self.session.call_tool(
                "puppeteer_click",
                arguments={
                    "selector": "button[aria-label*='Search'], button:contains('Search')"
                }
            )
            await self._debug_screenshot("after_click_flight_search")
            
            # Step 7: Wait for results
            logger.info("⏳ Waiting for flight results...")
            await asyncio.sleep(5)  # Flights take longer to load
            await self._debug_screenshot("after_wait_flight_results")
            
            # Step 8: Extract flight data
            extract_script = """
            () => {
                const flights = [];
                const flightElements = document.querySelectorAll('[role="listitem"], .gws-flights-results__result-item, .flight-item');
                
                flightElements.forEach(element => {
                    const airlineEl = element.querySelector('[data-testid="airline-name"], .airline-name, .carrier-name');
                    const priceEl = element.querySelector('[data-testid="price"], .price, .fare-price');
                    const durationEl = element.querySelector('[data-testid="duration"], .duration, .flight-duration');
                    const timeEl = element.querySelector('[data-testid="departure-time"], .departure-time, .time');
                    
                    if (airlineEl && airlineEl.textContent.trim()) {
                        flights.push({
                            airline: airlineEl.textContent.trim(),
                            price: priceEl?.textContent?.trim() || 'N/A',
                            duration: durationEl?.textContent?.trim() || 'N/A',
                            departure_time: timeEl?.textContent?.trim() || 'N/A'
                        });
                    }
                });
                
                return flights.slice(0, 10);
            }
            """
            
            extract_result = await self.session.call_tool(
                "puppeteer_evaluate",
                arguments={"script": extract_script}
            )
            await self._debug_screenshot("after_extract_flights")
            
            # Parse extracted data
            flights = []
            if hasattr(extract_result, 'content') and extract_result.content:
                result_text = extract_result.content[0].text if extract_result.content else ""
                if result_text.strip():
                    try:
                        extracted_data = json.loads(result_text)
                        for item in extracted_data:
                            if item.get('airline'):  # Only include flights with actual airline names
                                flight = {
                                    'airline': item.get('airline'),
                                    'price': item.get('price', 'N/A'),
                                    'price_numeric': TravelSearchUtils.extract_price_from_text(item.get('price', '0')),
                                    'duration': item.get('duration', 'N/A'),
                                    'departure_time': item.get('departure_time', 'N/A'),
                                    'platform': 'google_flights',
                                    'source': 'Real Google Flights Scraping',
                                    'route': f'{origin} → {destination}',
                                    'quality_score': TravelSearchUtils.calculate_flight_quality_score(item),
                                    'meets_quality_criteria': item.get('airline', '') in ['Korean Air', 'Asiana Airlines', 'Delta', 'Emirates']
                                }
                                flights.append(flight)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse extracted flight data")
            
            if flights:
                logger.info(f"✅ Found {len(flights)} real flights from Google Flights")
                return flights
            else:
                logger.warning("No real flights found, using minimal fallback")
                return [{
                    'airline': f'No flights found for {origin} → {destination}',
                    'price': 'N/A',
                    'price_numeric': 0,
                    'duration': 'N/A',
                    'departure_time': 'N/A',
                    'platform': 'google_flights',
                    'source': 'Search Failed - No Results',
                    'route': f'{origin} → {destination}',
                    'quality_score': 0,
                    'meets_quality_criteria': False
                }]
                
        except Exception as e:
            logger.error(f"Flight search failed: {e}")
            return [{
                'airline': f'Flight search error',
                'price': 'Error',
                'price_numeric': 0,
                'duration': 'Error',
                'departure_time': 'Error',
                'platform': 'google_flights',
                'source': f'Error: {str(e)}',
                'route': f'{origin} → {destination}',
                'quality_score': 0,
                'meets_quality_criteria': False
            }]
    
    async def cleanup(self):
        """Cleanup MCP browser resources"""
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
            logger.info("🧹 MCP browser resources cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    # ---------------------------------------------------------------------
    # Debug helper
    # ---------------------------------------------------------------------
    async def _debug_screenshot(self, name: str):
        """Capture screenshot if debug mode is enabled"""
        if not self.debug or not self.session:
            return
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_name = f"{timestamp}_{name}"
            
            # Call the screenshot tool and get the response
            response = await self.session.call_tool(
                "puppeteer_screenshot",
                arguments={
                    "name": screenshot_name,
                    "fullPage": True,
                    "encoded": True,  # Request base64 data URI
                }
            )

            # The screenshot is returned as a base64 data URI in text content.
            screenshot_data_uri = None
            if response.content:
                for content_item in response.content:
                    if hasattr(content_item, 'text') and content_item.text.startswith('data:image/png;base64,'):
                        screenshot_data_uri = content_item.text
                        break

            if screenshot_data_uri:
                # Create directory if it doesn't exist
                screenshot_dir = "./tmp/debug_screenshots"
                os.makedirs(screenshot_dir, exist_ok=True)
                
                # Decode and save the screenshot
                file_path = os.path.join(screenshot_dir, f"{screenshot_name}.png")
                base64_data = screenshot_data_uri.split(',')[1]
                image_data = base64.b64decode(base64_data)
                
                with open(file_path, "wb") as f:
                    f.write(image_data)
                logger.info(f"[DEBUG] Screenshot saved to {file_path}")
            else:
                logger.warning(f"[DEBUG] Screenshot data URI not found for '{name}'")

        except Exception as e:
            logger.warning(f"[DEBUG] Failed to capture screenshot ({name}): {e}")


class TravelMCPManager:
    """Manages MCP client and travel search orchestration"""
    def __init__(self):
        self.mcp_client = MCPBrowserClient(debug=True) # Enable debug mode for screenshots
        self.search_history = []
    
    async def search_travel_options(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search travel options using MCP browser with incognito mode"""
        search_start_time = time.time()
        
        try:
            logger.info(f"🔍 Starting MCP travel search for {search_params.get('destination')}")
            
            # Ensure MCP connection
            if not self.mcp_client.session:
                await self.mcp_client.connect_to_mcp_server()
            
            booking_scraper = BookingComScraper(self.mcp_client)
            flights_scraper = GoogleFlightsScraper(self.mcp_client)
            
            # Parallel searches
            search_tasks = []
            
            # Hotel search
            hotel_task = booking_scraper.search(
                search_params['destination'],
                search_params['check_in'],
                search_params['check_out']
            )
            search_tasks.append(hotel_task)
            
            # Flight search (if origin provided)
            if search_params.get('origin'):
                flight_task = flights_scraper.search(
                    search_params['origin'],
                    search_params['destination'],
                    search_params['departure_date'],
                    search_params['return_date']
                )
                search_tasks.append(flight_task)
            
            # Execute searches
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results
            hotels = search_results[0] if not isinstance(search_results[0], Exception) else []
            flights = search_results[1] if len(search_results) > 1 and not isinstance(search_results[1], Exception) else []
            
            # Generate analysis
            analysis = self._analyze_results(hotels, flights, search_params.get('quality_criteria', {}))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(hotels, flights, search_params)
            
            # Build final results
            search_duration = time.time() - search_start_time
            
            final_results = {
                "search_id": f"mcp_search_{int(time.time())}",
                "status": "completed",
                "hotels": hotels,
                "flights": flights,
                "analysis": analysis,
                "recommendations": recommendations,
                "performance": {
                    "total_duration": search_duration,
                    "platforms_searched": 2,
                    "hotels_found": len(hotels),
                    "flights_found": len(flights),
                    "method": "MCP Browser Use (Incognito Mode)",
                    "mcp_connected": self.mcp_client.session is not None
                },
                "search_params": search_params,
                "quality_criteria": search_params.get('quality_criteria', {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store history
            self.search_history.append(final_results)
            
            logger.info(f"✅ MCP search completed in {search_duration:.2f}s: {len(hotels)} hotels, {len(flights)} flights")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ MCP travel search failed: {e}")
            return {
                "search_id": f"failed_mcp_search_{int(time.time())}",
                "status": "failed",
                "error": str(e),
                "hotels": [],
                "flights": [],
                "performance": {"total_duration": time.time() - search_start_time},
                "search_params": search_params
            }
        finally:
            # Cleanup happens when the manager is destroyed
            pass
    
    def _analyze_results(self, hotels: List[Dict], flights: List[Dict], quality_criteria: Dict) -> Dict:
        """Analyze search results"""
        analysis = {}
        
        # Hotel analysis
        if hotels:
            hotel_prices = [h.get('price_numeric', 0) for h in hotels if h.get('price_numeric', 0) != float('inf')]
            hotel_ratings = [h.get('rating_numeric', 0) for h in hotels if h.get('rating_numeric', 0) > 0]
            
            analysis['hotel_analysis'] = {
                'total_found': len(hotels),
                'average_rating': sum(hotel_ratings) / len(hotel_ratings) if hotel_ratings else 0,
                'price_range': {
                    'min': min(hotel_prices) if hotel_prices else 0,
                    'max': max(hotel_prices) if hotel_prices else 0,
                    'average': sum(hotel_prices) / len(hotel_prices) if hotel_prices else 0
                },
                'quality_hotels_count': len([h for h in hotels if h.get('meets_quality_criteria', False)]),
                'mcp_results': len([h for h in hotels if h.get('source') == 'MCP Browser Use']),
                'fallback_results': len([h for h in hotels if 'Fallback' in h.get('source', '')])
            }
        
        # Flight analysis
        if flights:
            flight_prices = [f.get('price_numeric', 0) for f in flights if f.get('price_numeric', 0) != float('inf')]
            airlines_found = list(set([f.get('airline', '') for f in flights if f.get('airline')]))
            
            analysis['flight_analysis'] = {
                'total_found': len(flights),
                'airlines_found': airlines_found,
                'price_range': {
                    'min': min(flight_prices) if flight_prices else 0,
                    'max': max(flight_prices) if flight_prices else 0,
                    'average': sum(flight_prices) / len(flight_prices) if flight_prices else 0
                },
                'quality_flights_count': len([f for f in flights if f.get('meets_quality_criteria', False)]),
                'mcp_results': len([f for f in flights if f.get('source') == 'MCP Browser Use']),
                'fallback_results': len([f for f in flights if 'Fallback' in f.get('source', '')])
            }
        
        return analysis
    
    def _generate_recommendations(self, hotels: List[Dict], flights: List[Dict], search_params: Dict) -> Dict:
        """Generate recommendations based on search results"""
        recommendations = {}
        
        # Best value recommendations
        if hotels:
            best_value_hotels = sorted(hotels, key=lambda x: (x.get('price_numeric', float('inf')), -x.get('rating_numeric', 0)))
            if best_value_hotels:
                recommendations['best_hotel'] = best_value_hotels[0]
        
        if flights:
            best_value_flights = sorted(flights, key=lambda x: x.get('price_numeric', float('inf')))
            if best_value_flights:
                recommendations['best_flight'] = best_value_flights[0]
        
        # Budget vs luxury options
        if hotels:
            budget_hotels = [h for h in hotels if h.get('price_numeric', float('inf')) < 150]
            luxury_hotels = [h for h in hotels if h.get('price_numeric', 0) > 300 and h.get('rating_numeric', 0) >= 4.5]
            
            recommendations['budget_options'] = {
                'hotel': budget_hotels[0] if budget_hotels else None
            }
            
            recommendations['luxury_options'] = {
                'hotel': luxury_hotels[0] if luxury_hotels else None
            }
        
        # Booking strategy
        recommendations['booking_strategy'] = [
            "MCP 브라우저로 시크릿 모드 검색 완료",
            "가격 조작 없는 실제 시장 가격 확인",
            "실시간 데이터 수집으로 최신 정보 제공",
            "예약 전 최신 가격 재확인 권장",
            "취소 가능한 옵션 선택 고려"
        ]
        
        # Total cost estimate
        if recommendations.get('best_hotel') and recommendations.get('best_flight'):
            hotel_price = recommendations['best_hotel'].get('price_numeric', 0)
            flight_price = recommendations['best_flight'].get('price_numeric', 0)
            
            try:
                check_in = datetime.strptime(search_params['check_in'], '%Y-%m-%d')
                check_out = datetime.strptime(search_params['check_out'], '%Y-%m-%d')
                nights = (check_out - check_in).days
            except:
                nights = 3
            
            recommendations['total_trip_cost_estimate'] = {
                'hotel_per_night': hotel_price,
                'hotel_total': hotel_price * nights,
                'flight_total': flight_price,
                'grand_total': hotel_price * nights + flight_price,
                'nights': nights,
                'currency': 'USD'
            }
        
        return recommendations
    
    def get_search_history(self) -> List[Dict]:
        """Get search history"""
        return self.search_history
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.mcp_client.cleanup() 