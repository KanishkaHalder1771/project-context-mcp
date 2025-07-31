#!/usr/bin/env python3
"""
Project Context MCP Server - Main Entry Point

A smart context management system for coding projects that uses
knowledge graphs and vector databases for intelligent storage and retrieval.
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .services.context_manager import ContextManager
from .tools import register_all_tools


async def main():
    """Main entry point for the MCP server"""
    print("🚀 Project Context MCP Server - Starting...")
    
    # Initialize the context manager (singleton GraphBuilder)
    context_manager = ContextManager()
    print("✅ ContextManager initialized")
    
    # Create MCP server
    server = Server("project-context")
    print("✅ MCP Server created")
    
    # Register all tools
    register_all_tools(server, context_manager)
    print("✅ MCP Tools registered")
    
    # Start the server with stdio transport
    print("🔌 Starting MCP server with stdio transport...")
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1])


if __name__ == "__main__":
    asyncio.run(main()) 