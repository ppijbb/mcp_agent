from fastapi import FastAPI
from typing import List, Dict
from .public_data_client import public_data_client

app = FastAPI(
    title="Urban Hive Data Provider",
    description="Provides real urban data for Urban Hive agents via public data APIs.",
    version="0.2.0",
)

# --- Real Data Integration ---
# This provider now fetches data from public APIs via the PublicDataClient
# No more hardcoded data - all information comes from actual urban data sources

# --- API Endpoints ---


@app.get("/resources/available", response_model=List[Dict])
async def get_available_resources():
    """Retrieves all available resources from public data sources."""
    resource_data = await public_data_client.fetch_resource_data()
    return resource_data["available"]


@app.get("/resources/requests", response_model=List[Dict])
async def get_resource_requests():
    """Retrieves all resource requests from public data sources."""
    resource_data = await public_data_client.fetch_resource_data()
    return resource_data["requests"]


@app.get("/community/members", response_model=List[Dict])
async def get_community_members():
    """Retrieves all community members from public data sources."""
    community_data = await public_data_client.fetch_community_data()
    return community_data["members"]


@app.get("/community/groups", response_model=List[Dict])
async def get_available_groups():
    """Retrieves all available community groups from public data sources."""
    community_data = await public_data_client.fetch_community_data()
    return community_data["groups"]


@app.get("/urban-data/illegal-dumping", response_model=List[Dict])
async def get_illegal_dumping_data():
    """Retrieves data on illegal dumping incidents from public data sources."""
    return await public_data_client.fetch_illegal_dumping_data()


@app.get("/urban-data/traffic", response_model=List[Dict])
async def get_traffic_data():
    """Retrieves traffic congestion and accident data from public data sources."""
    return await public_data_client.fetch_traffic_data()


@app.get("/urban-data/safety", response_model=List[Dict])
async def get_safety_data():
    """Retrieves public safety and crime statistics from public data sources."""
    return await public_data_client.fetch_safety_data()


@app.get("/")
async def root():
    return {"message": "Urban Hive Data Provider is running (v0.2.0) - Now powered by real public data APIs!"}

# To run this provider:
# uvicorn srcs.urban_hive.providers.urban_hive_provider:app --reload --port 8001
