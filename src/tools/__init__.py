"""
MCP Tools Registration
"""

from .store_context import register_store_context_tool
from .get_context import register_get_context_tool
from .get_related import register_get_related_tool
from .delete_all_contexts import register_delete_all_contexts_tool


def register_all_tools(server, context_manager):
    """Register all 4 simplified MCP tools"""
    print("📝 Registering store_context tool...")
    register_store_context_tool(server, context_manager)
    
    print("🔍 Registering get_context tool...")
    register_get_context_tool(server, context_manager)
    
    print("🔗 Registering get_related_contexts tool...")
    register_get_related_tool(server, context_manager)
    
    print("🗑️ Registering delete_all_contexts tool...")
    register_delete_all_contexts_tool(server, context_manager)
    
    print("✅ All MCP tools registered successfully") 