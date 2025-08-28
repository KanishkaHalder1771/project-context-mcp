# #!/usr/bin/env python3
# """
# Project Context MCP Server - Main Entry Point

# A smart context management system for coding projects that uses
# knowledge graphs and vector databases for intelligent storage and retrieval.
# """

# import sys
# import os

# # Add the project root to the path so we can import server
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from server import mcp

# # Import tools so they get registered via decorators
# import tools.context_tools

# # Entry point to run the server
# if __name__ == "__main__":
#     print("🚀 Project Context MCP Server - Starting with FastMCP...")
#     mcp.run() 