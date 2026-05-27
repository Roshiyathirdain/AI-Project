# diagnostic.py
import os
import json
import asyncio
import pymysql
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

load_dotenv()

async def check_step(name, coro):
    print(f"[*] Checking {name}...", end=" ", flush=True)
    try:
        result = await coro
        print("[OK] SUCCESS")
        return result
    except Exception as e:
        print(f"[FAIL] FAILED\n   Error: {e}")
        return None

async def run_diagnostics():
    print("=== MCP System Diagnostic Tool ===\n")

    # 1. Check MySQL Connection
    async def run_mysql_check():
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB")
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        conn.close()
        return "Connected"

    await check_step("MySQL Database Connection", run_mysql_check())

    # 2. Check Gateway Configuration
    gateway_config = os.getenv("MCP_SERVERS")
    if gateway_config:
        print("[*] Checking Gateway Config in .env... [OK] FOUND")
    else:
        print("[*] Checking Gateway Config in .env... [FAIL] NOT FOUND")

    # 3. Test Local Connectivity to Sub-Servers
    python_exe = "c:/Users/roshi/Downloads/Mcpserverbot/.venv/Scripts/python.exe"
    
    async def test_server(name, script):
        params = StdioServerParameters(command=python_exe, args=[script], env=os.environ.copy())
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return len(tools.tools)

    await check_step("MySQL MCP Server", test_server("MySQL", "c:/Users/roshi/Downloads/Mcpserverbot/mcp_mysql_server.py"))
    await check_step("Local Files MCP Server", test_server("Files", "c:/Users/roshi/Downloads/Mcpserverbot/mcp_local_file_server.py"))
    
    # 4. Test Gateway Connectivity
    await check_step("Enterprise Gateway", test_server("Gateway", "c:/Users/roshi/Downloads/Mcpserverbot/mcp_gateway.py"))

    print("\n[!] Recommendation:")
    print("- If MySQL failed: Check your .env credentials.")
    print("- If a Server failed: Ensure the path to .venv/Scripts/python.exe is correct.")
    print("- To start the app: Use 'streamlit run app.py'")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
