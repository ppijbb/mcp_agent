"""
MCP Tools for Local Researcher
"""

from typing import Any, Dict, List, Optional
from mcp import types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import asyncio
import json
import logging

# Import MCP server functions
from mcp_server import (
    mcp_start_research,
    mcp_get_research_status,
    mcp_list_research,
    mcp_cancel_research,
    mcp_get_report_content,
    mcp_search_web,
    mcp_generate_report
)

logger = logging.getLogger(__name__)

# Create MCP server
server = Server("local-researcher")

@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="start_research",
            description="Start a new research project on a given topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research topic to investigate"
                    },
                    "domain": {
                        "type": "string",
                        "description": "The domain of research (e.g., technology, science, business)",
                        "default": "general"
                    },
                    "depth": {
                        "type": "string",
                        "description": "The depth of research (basic, comprehensive)",
                        "default": "basic"
                    }
                },
                "required": ["topic"]
            }
        ),
        types.Tool(
            name="get_research_status",
            description="Get the current status of a research project",
            inputSchema={
                "type": "object",
                "properties": {
                    "research_id": {
                        "type": "string",
                        "description": "The ID of the research project"
                    }
                },
                "required": ["research_id"]
            }
        ),
        types.Tool(
            name="list_research",
            description="List all research projects",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="cancel_research",
            description="Cancel a running research project",
            inputSchema={
                "type": "object",
                "properties": {
                    "research_id": {
                        "type": "string",
                        "description": "The ID of the research project to cancel"
                    }
                },
                "required": ["research_id"]
            }
        ),
        types.Tool(
            name="get_report_content",
            description="Get the content of a completed research report",
            inputSchema={
                "type": "object",
                "properties": {
                    "research_id": {
                        "type": "string",
                        "description": "The ID of the research project"
                    }
                },
                "required": ["research_id"]
            }
        ),
        types.Tool(
            name="search_web",
            description="Perform a web search on a given query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="generate_report",
            description="Generate a report from given content",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic of the report"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to include in the report"
                    },
                    "format": {
                        "type": "string",
                        "description": "The format of the report (markdown, html, pdf)",
                        "default": "markdown"
                    }
                },
                "required": ["topic", "content"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls."""
    try:
        if name == "start_research":
            topic = arguments.get("topic")
            domain = arguments.get("domain", "general")
            depth = arguments.get("depth", "basic")
            
            result = await mcp_start_research(topic, domain, depth)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        
        elif name == "get_research_status":
            research_id = arguments.get("research_id")
            result = await mcp_get_research_status(research_id)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        
        elif name == "list_research":
            result = await mcp_list_research()
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        
        elif name == "cancel_research":
            research_id = arguments.get("research_id")
            result = await mcp_cancel_research(research_id)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        
        elif name == "get_report_content":
            research_id = arguments.get("research_id")
            result = await mcp_get_report_content(research_id)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        
        elif name == "search_web":
            query = arguments.get("query")
            max_results = arguments.get("max_results", 5)
            result = await mcp_search_web(query, max_results)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        
        elif name == "generate_report":
            topic = arguments.get("topic")
            content = arguments.get("content")
            format = arguments.get("format", "markdown")
            result = await mcp_generate_report(topic, content, format)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        
        else:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Unknown tool: {name}"
                }, indent=2, ensure_ascii=False)
            )]
    
    except Exception as e:
        logger.error(f"Error in tool call {name}: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2, ensure_ascii=False)
        )]

async def main():
    """Main function to run the MCP server."""
    # Run the server using stdio
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="local-researcher",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
