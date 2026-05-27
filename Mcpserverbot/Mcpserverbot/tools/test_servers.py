"""
test_servers.py
Run with: python test_servers.py
Tests that MySQL and PostgreSQL MCP servers start up and list their tools correctly.
"""
import asyncio
import os
import sys

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv()

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

SERVERS = {
    "MySQL      ": os.path.join(BASE_DIR, "mcp_mysql_server.py"),
    "PostgreSQL ": os.path.join(BASE_DIR, "mcp_postgres_server.py"),
    "Local Files": os.path.join(BASE_DIR, "mcp_local_file_server.py"),
    "SharePoint ": os.path.join(BASE_DIR, "mcp_sharepoint_server.py"),
}

async def test_server(label: str, script: str):
    params = StdioServerParameters(
        command=PYTHON,
        args=[script],
        env=os.environ.copy()
    )
    try:
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                tool_names = [t.name for t in tools_resp.tools]
                print(f"  [OK]   {label}: {len(tool_names)} tools -> {tool_names}")
    except Exception as e:
        print(f"  [FAIL] {label}: FAILED -> {e}")

async def main():
    print("\n" + "="*60)
    print("  MCP Server Tool Discovery Test")
    print("="*60)
    for label, script in SERVERS.items():
        if os.path.exists(script):
            await test_server(label, script)
        else:
            print(f"  [WARN] {label}: Script not found -> {script}")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
