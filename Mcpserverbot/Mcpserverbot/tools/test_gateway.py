
import asyncio
import os
import json
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def test_gateway():
    url = "http://localhost:8000/sse"
    print(f"Connecting to {url}...")
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Connected! Listing tools...")
                tools = await session.list_tools()
                print(f"Found {len(tools.tools)} tools.")
                for t in tools.tools:
                    print(f" - {t.name}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_gateway())
