#!/usr/bin/env python3
"""
Travel Scout Agent - streamlined MCP integration

Provides a thin orchestration layer over `MCPBrowserClient` and concrete
scrapers so the CLI runner can call a consistent API.
"""
import logging
from typing import Dict, Any, Optional
from .mcp_browser_client import MCPBrowserClient
from .scrapers import BookingComScraper, GoogleFlightsScraper

# Re-export commonly used helper functions so that UI code can import directly
from .utils import load_destination_options, load_origin_options  # noqa: F401

__all__ = [
    "TravelScoutAgent",
    "load_destination_options",
    "load_origin_options",
]


class TravelScoutAgent:
    """Orchestrates travel searches using MCP-controlled browser."""

    def __init__(self, browser_client: MCPBrowserClient):
        self.browser_client = browser_client
        self.logger = logging.getLogger(__name__)

    async def search_hotels(
        self,
        destination: str,
        check_in: str,
        check_out: str,
        guests: int = 2,
    ) -> Dict[str, Any]:
        """Run a hotel search via Booking.com scraper and return structured results."""
        scraper = BookingComScraper(self.browser_client)
        search_params: Dict[str, Any] = {
            "destination": destination,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
        }
        return await scraper.search(search_params)

    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a flight search via Google Flights scraper and return structured results."""
        scraper = GoogleFlightsScraper(self.browser_client)
        search_params: Dict[str, Any] = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
        }
        return await scraper.search(search_params)

    async def cleanup(self) -> None:
        """Release browser resources held by the underlying MCP session."""
        await self.browser_client.cleanup()
