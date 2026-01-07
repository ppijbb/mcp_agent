"""
Content Generation Agent

Automatically generates AI content, publishes to multiple platforms, and monetizes through ads/affiliates.
"""

import logging
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

import aiohttp

from ...core.orchestrator import BaseAgent
from ...core.ledger import Ledger

logger = logging.getLogger(__name__)


class ContentGenerationAgent(BaseAgent):
    """
    Content Generation Agent
    
    Automatically:
    - Generates high-quality content using LLM (Gemini)
    - Publishes to multiple platforms (Medium, WordPress, etc.)
    - Creates eBooks and uploads to Amazon KDP
    - Generates images for content
    - Tracks revenue from ads and affiliate links
    """
    
    def __init__(self, name: str, config: Dict[str, Any], ledger: Ledger):
        super().__init__(name, config, ledger)
        self.config_detail = config.get('config', {})
        self.platforms = self.config_detail.get('platforms', [])
        self.topics = self.config_detail.get('topics', [])
        self.llm_provider = self.config_detail.get('llm_provider', 'gemini')
        self.image_generation = self.config_detail.get('image_generation', False)
        
        # LLM API key
        self.llm_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        self.llm_model = self.config_detail.get('llm_model', 'gemini-2.5-flash-lite')
        
        # Platform credentials
        self.medium_token = os.getenv('MEDIUM_ACCESS_TOKEN')
        self.wordpress_url = self.config_detail.get('wordpress_url', '')
        self.wordpress_username = os.getenv('WORDPRESS_USERNAME', '')
        self.wordpress_password = os.getenv('WORDPRESS_PASSWORD', '')
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.generated_content: List[Dict[str, Any]] = []
    
    async def initialize(self) -> bool:
        """Initialize content generation agent."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
            
            # Validate LLM API key
            if not self.llm_api_key:
                logger.warning("LLM API key not found. Content generation will be limited.")
            
            logger.info(f"Content Generation Agent initialized with {len(self.platforms)} platforms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Content Generation Agent: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown agent."""
        await super().shutdown()
        if self.session:
            await self.session.close()
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute content generation cycle.
        
        Returns:
            Execution result with estimated revenue
        """
        try:
            self._running = True
            
            # Generate content
            content = await self._generate_content()
            
            if not content:
                logger.warning("No content generated")
                return {
                    'success': True,
                    'income': 0.0,
                    'description': 'No content generated in this cycle'
                }
            
            # Generate images if enabled
            if self.image_generation:
                await self._generate_images(content)
            
            # Publish to platforms
            published_count = 0
            for platform in self.platforms:
                try:
                    success = await self._publish_to_platform(platform, content)
                    if success:
                        published_count += 1
                except Exception as e:
                    logger.error(f"Failed to publish to {platform}: {e}")
            
            # Create eBook if configured
            ebook_revenue = 0.0
            if self.config_detail.get('ebook_enabled', False):
                ebook_revenue = await self._create_ebook(content)
            
            # Estimate revenue
            estimated_revenue = self._estimate_revenue(content, published_count)
            total_revenue = estimated_revenue + ebook_revenue
            
            # Record revenue
            if total_revenue > 0:
                self.ledger.record_transaction(
                    agent_name=self.name,
                    transaction_type='income',
                    amount=total_revenue,
                    description=f"Content generation revenue: {published_count} articles published",
                    metadata={
                        'content_count': len(content),
                        'platforms': self.platforms,
                        'published_count': published_count,
                        'ebook_revenue': ebook_revenue
                    }
                )
            
            result = {
                'success': True,
                'income': total_revenue,
                'description': f"Content Generation: {len(content)} articles, ${total_revenue:.2f} estimated revenue",
                'metadata': {
                    'content_generated': len(content),
                    'published_count': published_count,
                    'platforms': self.platforms
                }
            }
            
            logger.info(
                f"Content Generation Agent executed: "
                f"{len(content)} articles, {published_count} published, ${total_revenue:.2f} revenue"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing Content Generation Agent: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'income': 0.0
            }
        finally:
            self._running = False
    
    async def _generate_content(self) -> List[Dict[str, Any]]:
        """
        Generate content using LLM.
        
        Returns:
            List of generated content dictionaries
        """
        if not self.llm_api_key:
            logger.warning("LLM API key not available, using placeholder content")
            return self._generate_placeholder_content()
        
        content_list = []
        
        # Generate content for each topic
        topics_to_use = self.topics if self.topics else [
            "Technology Trends",
            "Personal Finance",
            "Productivity Tips",
            "Health and Wellness"
        ]
        
        for topic in topics_to_use[:3]:  # Limit to 3 topics per cycle
            try:
                article = await self._generate_article(topic)
                if article:
                    content_list.append(article)
            except Exception as e:
                logger.error(f"Failed to generate content for topic '{topic}': {e}")
        
        return content_list
    
    async def _generate_article(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Generate a single article using Gemini API.
        
        Args:
            topic: Article topic
            
        Returns:
            Article dictionary with title, content, etc.
        """
        if not self.llm_api_key:
            return None
        
        try:
            # Use Gemini API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent"
            headers = {
                'Content-Type': 'application/json',
            }
            params = {
                'key': self.llm_api_key
            }
            
            prompt = f"""Write a comprehensive, SEO-optimized blog article about "{topic}".

Requirements:
- Title: Engaging and SEO-friendly
- Introduction: Hook the reader
- Body: 800-1200 words with clear sections
- Conclusion: Actionable takeaways
- Format: Markdown

Make it valuable, well-researched, and engaging for readers."""

            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            async with self.session.post(url, headers=headers, params=params, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Gemini API error: {response.status} - {error_text}")
                    return None
                
                data = await response.json()
                
                # Extract generated text
                if 'candidates' in data and len(data['candidates']) > 0:
                    generated_text = data['candidates'][0]['content']['parts'][0]['text']
                    
                    # Parse title and content
                    lines = generated_text.split('\n')
                    title = topic
                    content = generated_text
                    
                    # Try to extract title from first line
                    if lines and lines[0].startswith('#'):
                        title = lines[0].lstrip('#').strip()
                        content = '\n'.join(lines[1:])
                    
                    article = {
                        'title': title,
                        'content': content,
                        'topic': topic,
                        'word_count': len(content.split()),
                        'generated_at': datetime.now().isoformat(),
                        'platforms': self.platforms.copy()
                    }
                    
                    self.generated_content.append(article)
                    logger.info(f"Generated article: {title} ({article['word_count']} words)")
                    
                    return article
                else:
                    logger.warning(f"No content generated for topic: {topic}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error generating article for topic '{topic}': {e}")
            return None
    
    def _generate_placeholder_content(self) -> List[Dict[str, Any]]:
        """Generate placeholder content when LLM is not available."""
        return [{
            'title': 'Sample Article',
            'content': '# Sample Article\n\nThis is placeholder content. Configure LLM API key for actual generation.',
            'topic': 'General',
            'word_count': 10,
            'generated_at': datetime.now().isoformat(),
            'platforms': []
        }]
    
    async def _generate_images(self, content: List[Dict[str, Any]]):
        """Generate images for content (placeholder)."""
        # TODO: Implement image generation using DALL-E, Stable Diffusion, or similar
        logger.info(f"Image generation requested for {len(content)} articles (not yet implemented)")
    
    async def _publish_to_platform(self, platform: str, content: List[Dict[str, Any]]) -> bool:
        """
        Publish content to a specific platform.
        
        Args:
            platform: Platform name ('medium', 'wordpress', etc.)
            content: List of content to publish
            
        Returns:
            Success status
        """
        if not content:
            return False
        
        try:
            if platform.lower() == 'medium':
                return await self._publish_to_medium(content[0])
            elif platform.lower() == 'wordpress':
                return await self._publish_to_wordpress(content[0])
            else:
                logger.warning(f"Unknown platform: {platform}")
                return False
        except Exception as e:
            logger.error(f"Error publishing to {platform}: {e}")
            return False
    
    async def _publish_to_medium(self, article: Dict[str, Any]) -> bool:
        """Publish article to Medium."""
        if not self.medium_token:
            logger.warning("Medium access token not configured")
            return False
        
        try:
            # Medium API endpoint
            url = "https://api.medium.com/v1/me"
            headers = {
                'Authorization': f'Bearer {self.medium_token}',
                'Content-Type': 'application/json'
            }
            
            # Get user info first
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Medium API error: {response.status}")
                    return False
                
                user_data = await response.json()
                user_id = user_data['data']['id']
            
            # Create post
            post_url = f"https://api.medium.com/v1/users/{user_id}/posts"
            post_data = {
                'title': article['title'],
                'contentFormat': 'markdown',
                'content': article['content'],
                'publishStatus': 'public'
            }
            
            async with self.session.post(post_url, headers=headers, json=post_data) as response:
                if response.status in [201, 200]:
                    logger.info(f"Published to Medium: {article['title']}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to publish to Medium: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error publishing to Medium: {e}")
            return False
    
    async def _publish_to_wordpress(self, article: Dict[str, Any]) -> bool:
        """Publish article to WordPress."""
        if not self.wordpress_url or not self.wordpress_username or not self.wordpress_password:
            logger.warning("WordPress credentials not configured")
            return False
        
        try:
            # WordPress REST API
            url = f"{self.wordpress_url.rstrip('/')}/wp-json/wp/v2/posts"
            
            # Basic auth
            auth = aiohttp.BasicAuth(self.wordpress_username, self.wordpress_password)
            
            post_data = {
                'title': article['title'],
                'content': article['content'],
                'status': 'publish',
                'format': 'standard'
            }
            
            async with self.session.post(url, auth=auth, json=post_data) as response:
                if response.status in [201, 200]:
                    logger.info(f"Published to WordPress: {article['title']}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to publish to WordPress: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error publishing to WordPress: {e}")
            return False
    
    async def _create_ebook(self, content: List[Dict[str, Any]]) -> float:
        """
        Create eBook from content and prepare for Amazon KDP.
        
        Args:
            content: List of articles to compile into eBook
            
        Returns:
            Estimated revenue from eBook sales
        """
        # TODO: Implement eBook creation and KDP upload
        logger.info(f"eBook creation requested for {len(content)} articles (not yet implemented)")
        return 0.0
    
    def _estimate_revenue(self, content: List[Dict[str, Any]], published_count: int) -> float:
        """
        Estimate revenue from content.
        
        Args:
            content: Generated content
            published_count: Number of published articles
            
        Returns:
            Estimated revenue in USD
        """
        if published_count == 0:
            return 0.0
        
        # Simple estimation:
        # - Medium: $0.01-0.05 per view, assume 100 views per article
        # - WordPress: Ad revenue + affiliate, assume $0.10-0.50 per article per day
        # - Average: $2-5 per article per month
        
        revenue_per_article = 3.0  # $3 per article per month
        monthly_revenue = published_count * revenue_per_article
        
        # Convert to daily estimate (divide by 30)
        daily_revenue = monthly_revenue / 30.0
        
        return daily_revenue

