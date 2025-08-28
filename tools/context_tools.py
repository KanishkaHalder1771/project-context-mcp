# tools/context_tools.py

from server import mcp
from src.services.context_manager import ContextManager
import json

# Initialize context manager
context_manager = ContextManager()

@mcp.tool()
def store_context(query: str) -> str:
    """
    Store context about software architecture discussions into the knowledge graph.
    This tool is used to capture and store information about software components,
    technologies, architectural decisions, and their relationships from ongoing
    conversations about software design and architecture.
    
    Args:
        query: Cypher query to execute for storing context in the knowledge graph
        
    Returns:
        JSON string containing operation results and metadata
    """
    if not query or not query.strip():
        return json.dumps({
            "success": False,
            "message": "Query cannot be empty"
        }, indent=2)
    
    # Use the ContextManager to query the graph (synchronous call)
    import asyncio
    result = asyncio.run(context_manager.store_context(query.strip()))
    
    return json.dumps(result, indent=2)
