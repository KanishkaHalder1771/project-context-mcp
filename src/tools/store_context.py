"""
Store Context MCP Tool
"""

from mcp import types
from ..services.context_manager import ContextManager


def register_store_context_tool(server, context_manager=None):
    """Register the store_context MCP tool"""
    
    # Get singleton instance of ContextManager
    if context_manager is None:
        context_manager = ContextManager()
    
    # Register the tool definition
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="store_context",
                description="Store context about software architecture discussions into the knowledge graph. This tool is used to capture and store information about software components, technologies, architectural decisions, and their relationships from ongoing conversations about software design and architecture.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Cypher query to execute for storing context in the knowledge graph"
                        }
                    },
                    "required": ["query"]
                }
            )
        ]
    
    # Register the tool call handler
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle tool calls"""
        if name == "store_context":
            query = arguments.get("query", "")
            
            if not query or not query.strip():
                return [types.TextContent(
                    type="text",
                    text='{"success": false, "message": "Query cannot be empty"}'
                )]
            
            # Use the ContextManager to query the graph
            result = await context_manager.store_context(query.strip())
            
            import json
            return [types.TextContent(
                type="text", 
                text=json.dumps(result, indent=2)
            )]
        
        return [types.TextContent(
            type="text",
            text=f'{"success": false, "message": "Unknown tool: {name}"}'
        )] 