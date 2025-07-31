"""
Get Related Contexts MCP Tool
"""

# from mcp.server import Server
from ..services.context_manager import ContextManager


def register_get_related_tool(server, context_manager=None):
    """Register the get_related_contexts MCP tool"""
    
    # Get singleton instance of ContextManager
    if context_manager is None:
        context_manager = ContextManager()
    
    @server.tool("get_related_contexts")
    async def get_related_contexts(query: str) -> dict:
        """Find contexts related to a query - dummy implementation"""
        return {
            "success": True,
            "message": f"get_related_contexts not implemented yet - searched for: {query}",
            "results": []
        } 