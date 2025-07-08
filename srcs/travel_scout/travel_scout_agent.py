#!/usr/bin/env python3
"""
Travel Scout Agent - MCP-Agent Implementation

A travel search agent using the mcp_agent framework for consistent
integration with the MCP ecosystem.
"""
import asyncio
import os
from typing import Dict, Any
from srcs.core.agent.base import BaseAgent, AgentContext, async_memoize
from srcs.core.errors import WorkflowError
from .mcp_browser_client import MCPBrowserClient
from .scrapers import BookingComScraper, GoogleFlightsScraper
from . import utils as travel_utils

class TravelScoutAgent(BaseAgent):
    """
    A travel search agent using the mcp_agent framework for consistent
    integration with the MCP ecosystem.
    """

    def __init__(self):
        super().__init__("travel_scout_agent")

    @async_memoize
    async def _search_hotels(self, browser_client: MCPBrowserClient, destination: str, check_in: str, check_out: str, guests: int = 2) -> Dict[str, Any]:
        """Uses the browser client to search for hotels."""
        self.logger.info(f"🏨 Searching for hotels in {destination} from {check_in} to {check_out}")
        scraper = BookingComScraper(browser_client, self.logger)
        return await scraper.scrape(destination, check_in, check_out, guests)

    @async_memoize
    async def _search_flights(self, browser_client: MCPBrowserClient, origin: str, destination: str, departure_date: str, return_date: str = None) -> Dict[str, Any]:
        """Uses the browser client to search for flights."""
        self.logger.info(f"✈️ Searching for flights from {origin} to {destination} on {departure_date}")
        scraper = GoogleFlightsScraper(browser_client, self.logger)
        return await scraper.scrape(origin, destination, departure_date, return_date)

    async def run_workflow(self, context: AgentContext):
        """
        Runs the travel search workflow based on parameters from the context.
        """
        search_params = context.get("search_params", {})
        destination = search_params.get("destination")
        check_in = search_params.get("check_in")
        check_out = search_params.get("check_out")
        origin = search_params.get("origin")
        departure_date = search_params.get("departure_date")
        return_date = search_params.get("return_date")

        if not all([destination, check_in, check_out, origin, departure_date]):
            raise WorkflowError("Missing required search parameters in the context.")

        self.logger.info(f"Starting travel scout workflow for destination: {destination}")

        # Use a real browser for this agent
        browser_client = MCPBrowserClient(
            headless=self.settings.get("browser.headless", False),
            logger=self.logger
        )
        await browser_client.launch()

        try:
            # Perform searches in parallel
            hotel_task = self._search_hotels(browser_client, destination, check_in, check_out)
            flight_task = self._search_flights(browser_client, origin, destination, departure_date, return_date)
            
            hotel_results, flight_results = await asyncio.gather(hotel_task, flight_task)

            results = {
                "hotels": hotel_results,
                "flights": flight_results,
                "recommendations": {},
                "analysis": {}
            }

            self.logger.info("Generating travel report...")
            report_content = travel_utils.generate_travel_report_content(results, search_params)
            
            reports_dir = self.settings.get("reporting.reports_dir", "reports")
            report_path = travel_utils.save_travel_report(
                report_content, 
                f"{destination}_travel_plan",
                reports_dir=os.path.join(reports_dir, 'travel_scout')
            )

            self.logger.info(f"✅ Travel search completed. Report saved to {report_path}")

            context.set("results", results)
            context.set("report_path", report_path)

        except Exception as e:
            raise WorkflowError(f"An error occurred during the travel search workflow: {e}") from e
        finally:
            if browser_client.is_running():
                await browser_client.close()
