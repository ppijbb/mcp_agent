#!/usr/bin/env python3
"""
Enhanced Browser Manager for Local Researcher

This module provides robust browser automation capabilities optimized for
CLI, background, and Streamlit environments with comprehensive error handling.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import markdownify

# Browser automation imports with fallback
try:
    from browser_use import Browser as BrowserUseBrowser
    from browser_use import BrowserConfig
    from browser_use.browser.context import BrowserContext, BrowserContextConfig
    from browser_use.dom.service import DomService
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    BrowserUseBrowser = None
    BrowserConfig = None
    BrowserContext = None
    BrowserContextConfig = None
    DomService = None

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("browser_manager", log_level="INFO")


class BrowserManager:
    """Enhanced browser manager with robust error handling and fallback mechanisms."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the browser manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Browser components
        self.browser: Optional[BrowserUseBrowser] = None
        self.browser_context: Optional[BrowserContext] = None
        self.dom_service: Optional[DomService] = None
        self.browser_lock = asyncio.Lock()
        
        # Environment detection
        self.is_cli = not hasattr(sys, 'ps1') and not hasattr(sys, 'getwindowsversion')
        self.is_streamlit = 'streamlit' in sys.modules
        self.is_background = os.getenv('BACKGROUND_MODE', 'false').lower() == 'true'
        
        # Browser status
        self.browser_available = False
        self.fallback_mode = False
        
        logger.info(f"Browser Manager initialized - CLI: {self.is_cli}, Streamlit: {self.is_streamlit}, Background: {self.is_background}")
    
    async def initialize_browser(self) -> bool:
        """Initialize browser with enhanced error handling and environment detection."""
        try:
            if not BROWSER_USE_AVAILABLE:
                logger.warning("browser-use package not available. Using fallback mode.")
                self.fallback_mode = True
                return False
            
            async with self.browser_lock:
                if self.browser is None:
                    browser_config = self.config_manager.get_browser_config()
                    browser_config_kwargs = self._get_optimized_browser_config(browser_config)
                    
                    # Initialize browser with retry mechanism
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))
                            logger.info(f"Browser initialized successfully (attempt {attempt + 1})")
                            break
                        except Exception as e:
                            logger.warning(f"Browser initialization attempt {attempt + 1} failed: {e}")
                            if attempt == max_retries - 1:
                                logger.error("All browser initialization attempts failed. Using fallback mode.")
                                self.fallback_mode = True
                                return False
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                if self.browser_context is None:
                    context_config = self._get_optimized_context_config()
                    self.browser_context = await self.browser.new_context(context_config)
                    self.dom_service = DomService(await self.browser_context.get_current_page())
                    logger.info("Browser context initialized successfully")
                
                self.browser_available = True
                return True
                
        except Exception as e:
            logger.error(f"Browser initialization failed: {e}")
            self.fallback_mode = True
            return False
    
    def _get_optimized_browser_config(self, browser_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized browser configuration for different environments."""
        # Default to headless for CLI/background/Streamlit environments
        headless = browser_config.get("headless", self.is_cli or self.is_streamlit or self.is_background)
        
        config = {
            "headless": headless,
            "disable_security": browser_config.get("disable_security", True),
            "disable_images": browser_config.get("disable_images", True),
            "disable_javascript": browser_config.get("disable_javascript", False),
            "disable_css": browser_config.get("disable_css", False),
            "disable_plugins": browser_config.get("disable_plugins", True),
            "disable_extensions": browser_config.get("disable_extensions", True),
            "disable_dev_shm_usage": True,
            "no_sandbox": True,
            "disable_gpu": True,
            "disable_web_security": True,
            "disable_features": "VizDisplayCompositor",
            "window_size": browser_config.get("window_size", (1920, 1080)),
            "user_agent": browser_config.get("user_agent", 
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        }
        
        # Streamlit-specific optimizations
        if self.is_streamlit:
            config.update({
                "disable_images": True,
                "disable_css": True,
                "disable_plugins": True,
                "disable_extensions": True,
                "disable_gpu": True,
                "disable_dev_shm_usage": True,
                "no_sandbox": True,
                "headless": True
            })
        
        logger.info(f"Browser config optimized for environment: CLI={self.is_cli}, Streamlit={self.is_streamlit}, Background={self.is_background}")
        return config
    
    def _get_optimized_context_config(self) -> BrowserContextConfig:
        """Get optimized browser context configuration."""
        return BrowserContextConfig(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
        )
    
    async def navigate_and_extract(self, url: str, extraction_goal: str, llm=None) -> Dict[str, Any]:
        """Navigate to URL and extract content with fallback mechanisms.
        
        Args:
            url: URL to navigate to
            extraction_goal: Specific goal for content extraction
            llm: LLM instance for content processing
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Try browser automation first
            if self.browser_available and not self.fallback_mode:
                return await self._browser_extract(url, extraction_goal, llm)
            else:
                # Use fallback method
                return await self._requests_extract(url, extraction_goal, llm)
                
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {
                "success": False,
                "url": url,
                "extraction_goal": extraction_goal,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _browser_extract(self, url: str, extraction_goal: str, llm=None) -> Dict[str, Any]:
        """Extract content using browser automation."""
        try:
            if not self.browser_available:
                raise Exception("Browser not available")
            
            context = await self._ensure_browser_initialized()
            if context is None:
                raise Exception("Browser context not available")
            
            page = await context.get_current_page()
            
            # Enhanced navigation with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await page.goto(url, timeout=30000)
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    break
                except Exception as e:
                    logger.warning(f"Browser navigation attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
            
            # Extract content
            content = markdownify.markdownify(await page.content())
            max_content_length = self.config_manager.get_browser_config().get("max_content_length", 2000)
            
            # Process content with LLM if available
            if llm:
                extracted_data = await self._process_content_with_llm(content, extraction_goal, llm, max_content_length)
            else:
                extracted_data = {"raw_content": content[:max_content_length]}
            
            return {
                "success": True,
                "url": url,
                "extraction_goal": extraction_goal,
                "extracted_data": extracted_data,
                "content_length": len(content),
                "method": "browser",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Browser extraction failed: {e}")
            # Fallback to requests
            return await self._requests_extract(url, extraction_goal, llm)
    
    async def _requests_extract(self, url: str, extraction_goal: str, llm=None) -> Dict[str, Any]:
        """Extract content using requests as fallback."""
        try:
            logger.info(f"Using requests fallback for URL: {url}")
            
            # Enhanced requests configuration
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Make request with timeout and retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                    response.raise_for_status()
                    break
                except Exception as e:
                    logger.warning(f"Requests attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
            
            # Parse content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            max_content_length = self.config_manager.get_browser_config().get("max_content_length", 2000)
            
            # Process content with LLM if available
            if llm:
                extracted_data = await self._process_content_with_llm(text_content, extraction_goal, llm, max_content_length)
            else:
                extracted_data = {"raw_content": text_content[:max_content_length]}
            
            return {
                "success": True,
                "url": url,
                "extraction_goal": extraction_goal,
                "extracted_data": extracted_data,
                "content_length": len(text_content),
                "method": "requests",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Requests extraction failed: {e}")
            return {
                "success": False,
                "url": url,
                "extraction_goal": extraction_goal,
                "error": str(e),
                "method": "requests",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _process_content_with_llm(self, content: str, extraction_goal: str, llm, max_content_length: int) -> Dict[str, Any]:
        """Process content using LLM."""
        try:
            prompt = f"""
            Your task is to extract content from a webpage based on a specific goal.
            Extract all relevant information around this goal from the page.
            If the goal is vague, provide a comprehensive summary.
            Respond in JSON format.
            
            Extraction goal: {extraction_goal}
            
            Page content:
            {content[:max_content_length]}
            """
            
            response = await asyncio.to_thread(llm.generate_content, prompt)
            
            # Parse LLM response
            try:
                extracted_data = json.loads(response.text)
            except json.JSONDecodeError:
                extracted_data = {
                    "extracted_content": {
                        "text": response.text,
                        "metadata": {
                            "extraction_goal": extraction_goal
                        }
                    }
                }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return {"raw_content": content[:max_content_length], "error": str(e)}
    
    async def _ensure_browser_initialized(self) -> Optional[BrowserContext]:
        """Ensure browser is initialized."""
        if not self.browser_available or self.fallback_mode:
            return None
        
        try:
            if self.browser_context is None:
                await self.initialize_browser()
            
            return self.browser_context
            
        except Exception as e:
            logger.error(f"Browser context initialization failed: {e}")
            return None
    
    async def search_and_extract(self, query: str, extraction_goal: str, max_results: int = 3, llm=None) -> List[Dict[str, Any]]:
        """Perform web search and extract content from results."""
        try:
            # First perform web search
            search_results = await self._perform_web_search(query, max_results)
            
            if not search_results:
                return []
            
            extracted_results = []
            
            for result in search_results[:max_results]:
                try:
                    # Extract content from each search result
                    extraction_result = await self.navigate_and_extract(
                        result.get('url', ''), 
                        extraction_goal,
                        llm
                    )
                    
                    if extraction_result.get('success'):
                        extraction_result['search_result'] = result
                        extracted_results.append(extraction_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract from {result.get('url', '')}: {e}")
                    continue
            
            return extracted_results
            
        except Exception as e:
            logger.error(f"Search and extract failed: {e}")
            return []
    
    async def _perform_web_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform web search using available search APIs."""
        try:
            # This would be implemented to use actual search APIs
            # For now, return mock results
            return [
                {
                    "title": f"Search result for: {query}",
                    "url": f"https://example.com/search?q={query}",
                    "snippet": f"This is a search result for {query}"
                }
            ]
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def cleanup(self):
        """Clean up browser resources."""
        try:
            async with self.browser_lock:
                if self.browser_context:
                    await self.browser_context.close()
                    self.browser_context = None
                    self.dom_service = None
                
                if self.browser:
                    await self.browser.close()
                    self.browser = None
                
                self.browser_available = False
                logger.info("Browser resources cleaned up")
                
        except Exception as e:
            logger.error(f"Browser cleanup failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get browser manager status."""
        return {
            "browser_available": self.browser_available,
            "fallback_mode": self.fallback_mode,
            "is_cli": self.is_cli,
            "is_streamlit": self.is_streamlit,
            "is_background": self.is_background,
            "browser_use_available": BROWSER_USE_AVAILABLE
        }
