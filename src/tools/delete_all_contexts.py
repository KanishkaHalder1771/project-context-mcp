"""
Delete All Contexts MCP Tool
"""

# from mcp.server import Server
from ..services.context_manager import ContextManager


def register_delete_all_contexts_tool(server, context_manager=None):
    """Register the delete_all_contexts MCP tool"""
    
    # Get singleton instance of ContextManager
    if context_manager is None:
        context_manager = ContextManager()
    
    @server.tool("delete_all_contexts")
    async def delete_all_contexts() -> dict:
        """Nuclear option - delete all stored contexts - dummy implementation"""
        return {
            "success": True,
            "message": "delete_all_contexts not implemented yet"
        } 