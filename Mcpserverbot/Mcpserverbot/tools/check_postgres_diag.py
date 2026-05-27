
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
            
            # List tools
            tools_resp = await session.list_tools()
            
            # Get schema
            schema_resp = await session.call_tool("get_schema", {})
            
            # Check tables
            tables_resp = await session.call_tool("list_tables", {})
            
            # Results
            results = {
                "tools": [t.name for t in tools_resp.tools],
                "schema": json.loads(schema_resp.content[0].text),
                "tables": json.loads(tables_resp.content[0].text)
            }
            
            with open("postgres_diag_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Done. Results in postgres_diag_results.json")

if __name__ == "__main__":
    asyncio.run(check_postgres())
