#!/usr/bin/env python3
"""
Travel Scout Agent

A specialized agent that uses incognito/private browsing mode to search for 
the best value accommodations and flights to target destinations.

Key Features:
- Uses incognito mode to prevent cache interference with pricing
- Searches for high-quality accommodations with excellent reviews
- Finds cost-effective flights while maintaining quality standards
- Always prioritizes the lowest prices meeting quality criteria
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from ..common import *


class TravelScoutAgent(BasicAgentTemplate):
    """Travel search agent using incognito mode for unbiased price discovery"""
    
    def __init__(self, destination: str = None, check_in: str = None, check_out: str = None, 
                 departure_date: str = None, return_date: str = None, origin: str = None):
        super().__init__(
            agent_name="travel_scout",
            task_description="Search for best value travel deals using incognito browsing"
        )
        
        # Travel search parameters
        self.destination = destination or "Seoul, South Korea"
        self.check_in = check_in or self._get_default_checkin()
        self.check_out = check_out or self._get_default_checkout()
        self.departure_date = departure_date or self._get_default_departure()
        self.return_date = return_date or self._get_default_return()
        self.origin = origin or "Seoul, South Korea"
        
        # Quality thresholds
        self.min_hotel_rating = 4.0
        self.min_review_count = 100
        self.min_flight_rating = 4.0
        
        # Search results storage
        self.search_results = {
            "hotels": [],
            "flights": [],
            "timestamp": datetime.now().isoformat(),
            "search_params": {
                "destination": self.destination,
                "check_in": self.check_in,
                "check_out": self.check_out,
                "departure_date": self.departure_date,
                "return_date": self.return_date,
                "origin": self.origin
            }
        }
        
    def _get_default_checkin(self) -> str:
        """Get default check-in date (30 days from now)"""
        return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    def _get_default_checkout(self) -> str:
        """Get default check-out date (33 days from now)"""
        return (datetime.now() + timedelta(days=33)).strftime("%Y-%m-%d")
    
    def _get_default_departure(self) -> str:
        """Get default departure date (30 days from now)"""
        return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    def _get_default_return(self) -> str:
        """Get default return date (37 days from now)"""
        return (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d")

    def create_agents(self):
        """Create specialized travel search agents"""
        return [
            # Hotel Search Agent
            Agent(
                name="hotel_scout",
                instruction=f"""You are a hotel search specialist using incognito browsing to find the best accommodation deals.

                Search Parameters:
                - Destination: {self.destination}
                - Check-in: {self.check_in}
                - Check-out: {self.check_out}
                - Minimum Rating: {self.min_hotel_rating}/5.0
                - Minimum Reviews: {self.min_review_count}

                Your mission:
                1. ALWAYS use incognito/private browsing mode to prevent price manipulation
                2. Search multiple hotel booking sites (Booking.com, Hotels.com, Expedia, Agoda)
                3. Focus on hotels with ratings ≥ {self.min_hotel_rating} and reviews ≥ {self.min_review_count}
                4. Prioritize value for money - best quality at lowest price
                5. Extract detailed information: name, price, rating, review count, amenities, location
                6. Compare prices across platforms for the same hotel
                7. Document search methodology and cache-prevention measures

                Quality Standards:
                - Only consider hotels with excellent reviews (4.0+ rating)
                - Minimum {self.min_review_count} reviews for reliability
                - Look for recent positive reviews
                - Consider location convenience and safety
                - Check for hidden fees or additional charges

                Output Format:
                - Hotel name and exact location
                - Nightly rate and total cost
                - Rating and number of reviews
                - Key amenities and features
                - Booking platform and direct comparison
                - Value assessment and recommendation reasoning
                """,
                server_names=["mcp_playwright-mcp-server"]
            ),
            
            # Flight Search Agent
            Agent(
                name="flight_scout",
                instruction=f"""You are a flight search specialist using incognito browsing to find the best flight deals.

                Search Parameters:
                - Origin: {self.origin}
                - Destination: {self.destination}
                - Departure: {self.departure_date}
                - Return: {self.return_date}
                - Minimum Rating: {self.min_flight_rating}/5.0

                Your mission:
                1. ALWAYS use incognito/private browsing mode to avoid dynamic pricing
                2. Search multiple flight booking sites (Kayak, Expedia, Skyscanner, Google Flights)
                3. Focus on airlines with good ratings and reliable service
                4. Find the best balance of price, convenience, and quality
                5. Check for nearby airports and flexible dates for better deals
                6. Extract comprehensive flight information and compare options
                7. Document incognito browsing usage and price consistency

                Quality Standards:
                - Prefer airlines with ratings ≥ {self.min_flight_rating}
                - Consider flight duration and layover times
                - Check baggage policies and additional fees
                - Verify airline reliability and punctuality
                - Consider flight times and convenience

                Output Format:
                - Airline and flight numbers
                - Departure/arrival times and airports
                - Total flight time and layovers
                - Price breakdown (base fare + taxes + fees)
                - Airline rating and service quality
                - Booking platform comparison
                - Value assessment and recommendation
                """,
                server_names=["mcp_playwright-mcp-server"]
            ),
            
            # Price Comparison Agent
            Agent(
                name="price_analyzer",
                instruction=f"""You are a price analysis specialist ensuring we get the absolute best deals.

                Your responsibilities:
                1. Analyze all hotel and flight options found by other agents
                2. Cross-reference prices across different platforms
                3. Calculate total trip cost including all fees
                4. Identify the best value combinations (hotel + flight packages)
                5. Flag any suspicious pricing or potential scams
                6. Recommend final booking strategy

                Analysis Criteria:
                - Total cost optimization
                - Quality-to-price ratio assessment
                - Hidden fee detection
                - Seasonal pricing patterns
                - Platform reliability and booking protection
                - Cancellation policy comparison

                Output Format:
                - Ranked list of best value combinations
                - Total trip cost breakdown
                - Savings opportunities identified
                - Risk assessment for each option
                - Final recommendation with reasoning
                - Booking timeline and strategy
                """,
                server_names=DEFAULT_SERVERS
            )
        ]

    def create_evaluator(self):
        """Create travel search quality evaluator"""
        return Agent(
            name="travel_scout_evaluator",
            instruction=f"""You are a travel booking expert evaluating the quality of travel search results.

            Evaluation Criteria:

            1. INCOGNITO BROWSING COMPLIANCE (25%)
            - Verified use of private/incognito mode
            - Evidence of cache prevention measures
            - Consistency in pricing across sessions
            - Documentation of browsing methodology

            2. ACCOMMODATION QUALITY (25%)
            - Hotels meet minimum rating threshold ({self.min_hotel_rating}+)
            - Sufficient review count ({self.min_review_count}+)
            - Recent positive reviews analysis
            - Location safety and convenience
            - Amenity value assessment

            3. FLIGHT QUALITY (25%)
            - Airlines meet quality standards
            - Reasonable flight times and connections
            - Transparent pricing with all fees included
            - Baggage policy consideration
            - Schedule convenience

            4. PRICE OPTIMIZATION (25%)
            - Comprehensive platform comparison
            - Best value identification
            - Hidden fee detection
            - Total cost accuracy
            - Savings maximization

            Rate as EXCELLENT, GOOD, FAIR, or POOR with specific feedback.
            Highlight any missing elements or quality concerns.
            Provide actionable recommendations for improvement.
            """,
        )

    def define_task(self):
        """Define the travel search task"""
        return f"""Execute comprehensive travel search for the following trip:

        TRIP DETAILS:
        - Destination: {self.destination}
        - Hotel Stay: {self.check_in} to {self.check_out}
        - Flight Route: {self.origin} ↔ {self.destination}
        - Travel Dates: {self.departure_date} to {self.return_date}

        REQUIREMENTS:
        1. Use INCOGNITO/PRIVATE browsing mode exclusively
        2. Search high-quality accommodations (4.0+ rating, 100+ reviews)
        3. Find reliable flights with good airline ratings
        4. Prioritize best value (quality + lowest price)
        5. Compare multiple booking platforms
        6. Provide detailed analysis and recommendations

        DELIVERABLES:
        - Comprehensive hotel options analysis
        - Flight alternatives with pricing breakdown
        - Total trip cost calculations
        - Best value recommendations
        - Booking strategy and timeline
        - Incognito browsing verification report

        Begin the search process immediately and ensure all browsing is done in incognito mode to prevent price manipulation.
        """

    async def run_travel_search(self):
        """Main method to run the travel search"""
        return await self.run()

    def save_results(self):
        """Save search results to file"""
        results_file = self.output_dir / f"travel_search_results_{self.timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.search_results, f, indent=2, ensure_ascii=False)
        
        print(f"🎯 Search results saved to: {results_file}")

    def create_summary(self):
        """Create travel search executive summary"""
        summary_data = {
            "destination": self.destination,
            "travel_dates": f"{self.departure_date} to {self.return_date}",
            "accommodation_dates": f"{self.check_in} to {self.check_out}",
            "search_methodology": "Incognito browsing mode for unbiased pricing",
            "quality_standards": f"Hotels: {self.min_hotel_rating}+ rating, {self.min_review_count}+ reviews",
            "platforms_searched": "Multiple booking platforms for comprehensive comparison"
        }
        
        create_executive_summary(
            output_dir=self.output_dir,
            agent_name=self.agent_name,
            company_name="Travel Scout",
            timestamp=self.timestamp,
            summary_data=summary_data
        )

    def create_kpis(self):
        """Create travel search KPIs"""
        kpi_structure = {
            "Search Quality": {
                "Incognito Compliance": "100%",
                "Platform Coverage": "4+ booking sites",
                "Quality Threshold": f"{self.min_hotel_rating}+ rating hotels",
                "Review Reliability": f"{self.min_review_count}+ reviews"
            },
            "Cost Optimization": {
                "Price Comparison": "Multi-platform analysis",
                "Hidden Fee Detection": "Comprehensive fee audit",
                "Value Optimization": "Quality-to-price ratio maximization",
                "Savings Identification": "Best deal discovery"
            },
            "Service Quality": {
                "Hotel Standards": f"{self.min_hotel_rating}+ rating requirement",
                "Flight Standards": f"{self.min_flight_rating}+ airline rating",
                "Location Quality": "Safety and convenience assessed",
                "Review Recency": "Recent feedback prioritized"
            }
        }
        
        create_kpi_template(
            output_dir=self.output_dir,
            agent_name=self.agent_name,
            kpi_structure=kpi_structure,
            timestamp=self.timestamp
        )


def main():
    """Main execution function"""
    print("🧳 Travel Scout Agent - Incognito Mode Travel Search")
    print("=" * 50)
    
    # Get user input for travel parameters
    destination = input("Enter destination (default: Seoul, South Korea): ").strip()
    origin = input("Enter origin city (default: Seoul, South Korea): ").strip()
    
    # Date inputs with validation
    try:
        departure = input("Enter departure date (YYYY-MM-DD, default: 30 days from now): ").strip()
        return_date = input("Enter return date (YYYY-MM-DD, default: 37 days from now): ").strip()
        check_in = input("Enter hotel check-in date (YYYY-MM-DD, default: 30 days from now): ").strip()
        check_out = input("Enter hotel check-out date (YYYY-MM-DD, default: 33 days from now): ").strip()
    except:
        departure = return_date = check_in = check_out = None
    
    # Create and run the agent
    agent = TravelScoutAgent(
        destination=destination or None,
        origin=origin or None,
        departure_date=departure or None,
        return_date=return_date or None,
        check_in=check_in or None,
        check_out=check_out or None
    )
    
    print(f"\n🎯 Searching for travel deals to: {agent.destination}")
    print(f"📅 Travel dates: {agent.departure_date} to {agent.return_date}")
    print(f"🏨 Hotel stay: {agent.check_in} to {agent.check_out}")
    print(f"🔒 Using INCOGNITO mode to prevent price manipulation")
    print("\nStarting search...")
    
    # Run the travel search
    asyncio.run(agent.run_travel_search())


if __name__ == "__main__":
    main() 