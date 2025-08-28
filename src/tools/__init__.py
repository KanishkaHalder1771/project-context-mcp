"""
MCP Tools Registration
"""

from mcp import types
import json


def register_all_tools(server, context_manager):
    """Register all MCP tools with centralized handlers"""
    print("📝 Registering MCP tools...")
    
    # Add initialization handler
    @server.initialize()
    async def handle_initialize():
        """Handle MCP initialization"""
        print("🔗 MCP client connected and initialized")
        return {}
    
    # Single list_tools handler for all tools
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List all available tools"""
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
    
    # Single call_tool handler for all tools
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle all tool calls"""
        if name == "store_context":
            query = arguments.get("query", "")
            
            if not query or not query.strip():
                return [types.TextContent(
                    type="text",
                    text='{"success": false, "message": "Query cannot be empty"}'
                )]
            
            # Use the ContextManager to query the graph
            result = await context_manager.store_context(query.strip())
            
            return [types.TextContent(
                type="text", 
                text=json.dumps(result, indent=2)
            )]
        
        return [types.TextContent(
            type="text",
            text=json.dumps({"success": False, "message": f"Unknown tool: {name}"}, indent=2)
        )]
    
    print("✅ All MCP tools registered successfully") 