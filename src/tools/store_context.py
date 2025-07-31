"""
Store Context MCP Tool
"""

# from mcp.server import Server
from ..services.context_manager import ContextManager


def register_store_context_tool(server, context_manager=None):
    """Register the store_context MCP tool"""
    
    # Get singleton instance of ContextManager
    if context_manager is None:
        context_manager = ContextManager()
    
    @server.tool("store_context")
    async def store_context(content: str) -> dict:
        if not content or not content.strip():
            return {
                "success": False,
                "message": "Content cannot be empty"
            }
        
        # Use the singleton GraphBuilder through ContextManager
        result = await context_manager.store_context(content.strip())
        
        return result 