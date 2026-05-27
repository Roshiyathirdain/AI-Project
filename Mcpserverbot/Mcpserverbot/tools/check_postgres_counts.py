
import asyncio
import json
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import os

async def check_postgres():
    base_dir = r"c:\Users\roshi\Downloads\Mcpserverbot"
    python_exe = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    params = StdioServerParameters(
        command=python_exe,
        args=[os.path.join(base_dir, "mcp_postgres_server.py")],
        env=os.environ.copy()
    )
    
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Count rows in each table
            tables_resp = await session.call_tool("list_tables", {})
            tables = json.loads(tables_resp.content[0].text)
            
            counts = {}
            for t in tables:
                tname = f"{t['table_schema']}.{t['table_name']}"
                try:
                    res = await session.call_tool("query_database", {"sql": f"SELECT COUNT(*) FROM {tname}"})
                    count_val = json.loads(res.content[0].text)[0]["count"]
                    counts[tname] = count_val
                except Exception as e:
                    counts[tname] = f"Error: {e}"
            
            with open("postgres_counts.json", "w") as f:
                json.dump(counts, f, indent=2)
            print("Done. Results in postgres_counts.json")

if __name__ == "__main__":
    asyncio.run(check_postgres())
