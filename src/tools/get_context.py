"""
Get Context MCP Tool
"""

# from mcp.server import Server
from ..services.context_manager import ContextManager


def register_get_context_tool(server, context_manager=None):
    """Register the get_context MCP tool"""
    
    # Get singleton instance of ContextManager
    if context_manager is None:
        context_manager = ContextManager()
    
    @server.tool("get_context")
    async def get_context(query: str) -> dict:
        """Retrieve contexts using semantic search - dummy implementation"""
        return {
            "success": True,
            "message": f"get_context not implemented yet - searched for: {query}",
            "results": []
        } 