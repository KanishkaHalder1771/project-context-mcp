#!/usr/bin/env python3
"""
Simple MCP client to test the server
"""

import asyncio
import json
import subprocess
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    """Test the MCP server by connecting as a client"""
    print("🔍 Testing MCP Server...")
    
    try:
        # Start the MCP server process
        server_process = subprocess.Popen(
            ["uv", "run", "python", "-m", "src.main"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        print("✅ MCP Server process started")
        
        # Connect to the server
        async with stdio_client(server_process) as (read, write):
            async with ClientSession(read, write) as session:
                print("✅ Connected to MCP server")
                
                # Initialize the session
                await session.initialize()
                print("✅ Session initialized")
                
                # List available tools
                tools_result = await session.list_tools()
                print(f"📋 Available tools: {len(tools_result.tools)}")
                
                for tool in tools_result.tools:
                    print(f"  - {tool.name}: {tool.description[:100]}...")
                
                # Test the store_context tool
                if tools_result.tools:
                    print("\n🧪 Testing store_context tool...")
                    
                    test_query = "MATCH (n) RETURN count(n) as node_count"
                    
                    result = await session.call_tool(
                        "store_context",
                        {"query": test_query}
                    )
                    
                    print("📤 Tool call result:")
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(f"  {content.text}")
                
                print("✅ MCP server test completed successfully!")
                
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'server_process' in locals():
            server_process.terminate()
            server_process.wait()
            print("🛑 Server process terminated")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
