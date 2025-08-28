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
                description="Store context about software architecture discussions into a database. Give a blob of text with details about the software, it will be stored in as Software service components, techonology and architecture decisions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Text describing the software, techonology and architecture decisions."
                        }
                    },
                    "required": ["context"]
                }
            )
        ]
    
    # Register the tool call handler
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle tool calls"""
        if name == "store_context":
            context = arguments.get("context", "")
            
            if not context or not context.strip():
                return [types.TextContent(
                    type="text",
                    text='{"success": false, "message": "Context cannot be empty"}'
                )]
            
            # Use the ContextManager to store the context
            result = await context_manager.store_context(context.strip())
            
            import json
            return [types.TextContent(
                type="text", 
                text=json.dumps(result, indent=2)
            )]
        
        return [types.TextContent(
            type="text",
            text=f'{"success": false, "message": "Unknown tool: {name}"}'
        )] 