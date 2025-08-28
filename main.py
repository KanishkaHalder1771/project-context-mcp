from fastmcp import FastMCP
from src.services.context_manager import ContextManager
import json
import asyncio

# create server
mcp = FastMCP("project-context")

# Initialize context manager
context_manager = ContextManager()

# add store_context tool
@mcp.tool()
async def store_context(context: str) -> str:
    """
    Store context about software architecture discussions into a database. Give a blob of text with details about the software, it will be stored in as Software service components, techonology and architecture decisions.
    
    Args:
        context: Text describing the software, techonology and architecture decisions.
        
    Returns:
        JSON string containing operation results and metadata
    """
    if not context or not context.strip():
        return json.dumps({
            "success": False,
            "message": "context cannot be empty"
        }, indent=2)

    print(f"🔍 ContextManager: Executing graph query: {context[:1000]}...")
    
    # Use the ContextManager to query the graph (async call)
    try:
        result = await context_manager.store_context(context.strip())
        return json.dumps({"success": True}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "message": str(e)}, indent=2)

# add get_context tool
@mcp.tool()
async def get_context(query: str) -> str:
    """
    Get context by answering a query based on the stored project documentation.
    
    Args:
        query: Question about the project context and architecture.
        
    Returns:
        Answer based on the stored project documentation
    """
    if not query or not query.strip():
        return "Query cannot be empty"
    
    # Use the ContextManager to answer the query
    try:
        answer = await context_manager.answer_query(query.strip())
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Run with increased timeout for long-running operations
    mcp.run(
        transport="http", 
        host="127.0.0.1", 
        port=8000,
        # Pass uvicorn-specific timeout configurations
        uvicorn_config={
            "timeout_keep_alive": 300,  # 5 minutes keep-alive timeout
            "timeout_graceful_shutdown": 60,  # 1 minute graceful shutdown
            "access_log": True,
            "timeout_notify": 300,  # Timeout for worker notification
        }
    )



# from fastmcp import FastMCP

# mcp = FastMCP("MyServer HTTP")

# @mcp.tool
# def hello(name: str) -> str:
#     return f"Hello, {name}!"

# if __name__ == "__main__":
#     # Start an HTTP server on port 8000
#     mcp.run(transport="http", host="127.0.0.1", port=8000)