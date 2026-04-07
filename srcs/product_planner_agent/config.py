"""Configuration settings for the Product Planner Agent."""

from typing import Optional
from pydantic_settings import BaseSettings


class ProductPlannerSettings(BaseSettings):
    """Settings for the Product Planner Agent."""

    notion_api_key: Optional[str] = None
    notion_database_id: Optional[str] = None
    default_output_format: str = "markdown"
    max_iterations: int = 10
    enable_notion_sync: bool = False

    class Config:
        env_prefix = "PRODUCT_PLANNER_"


settings = ProductPlannerSettings()
