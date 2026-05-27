import asyncio
import nest_asyncio
import traceback
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

nest_asyncio.apply()

async def test_gateway():
    url = "http://localhost:8000/sse"
    print(f"Connecting to {url}...")
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ CONNECTION SUCCESS!")
                tools = await session.list_tools()
                print(f"📊 {len(tools.tools)} tools found.")
    except Exception:
        print("❌ Connection failed Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gateway())
