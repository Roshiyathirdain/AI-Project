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
                for table, cols in schema.items():
                    col_names = [c.get("Field", "") for c in cols]
                    print(f"{table}: {', '.join(col_names)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
