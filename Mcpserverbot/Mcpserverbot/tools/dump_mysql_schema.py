import asyncio
import json
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def main():
    url = "http://localhost:8000/sse"
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool("mysql_get_schema", {})
                schema = json.loads(res.content[0].text)
                target_tables = [t for t in schema.keys() if any(k in t.lower() for k in ["emp", "salary", "pay", "staff", "wage"])]
                if not target_tables:
                    print("No obvious salary tables found. Printing keys:")
                    print(list(schema.keys()))
                else:
                    for t in target_tables:
                        print(f"\n--- TABLE: {t} ---")
                        print(json.dumps(schema[t], indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
