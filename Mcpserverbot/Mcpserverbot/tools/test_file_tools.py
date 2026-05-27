# test_file_tools.py
import asyncio
import os
import json
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def test_local_files():
    print("Testing Local File Server directly...")
    
    # Path to the python executable in the venv
    python_exe = "c:/Users/roshi/Downloads/Mcpserverbot/.venv/Scripts/python.exe"
    server_script = "c:/Users/roshi/Downloads/Mcpserverbot/mcp_local_file_server.py"
    
    params = StdioServerParameters(
        command=python_exe,
        args=[server_script],
        env=os.environ.copy()
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 1. List tools
                print("\n--- Listing Tools ---")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"- {tool.name}: {tool.description}")
                
                # 2. Test list_files
                print("\n--- Testing list_files ---")
                res = await session.call_tool("list_files", {})
                print(f"Result: {res.content[0].text}")
                
                # 3. Test read_csv
                print("\n--- Testing read_csv (employees.csv) ---")
                res = await session.call_tool("read_csv", {"filename": "employees.csv"})
                print(f"Result: {res.content[0].text[:200]}...")
                
                # 4. Test read_document
                print("\n--- Testing read_document (roadmap.md) ---")
                res = await session.call_tool("read_document", {"filename": "roadmap.md"})
                print(f"Result: {res.content[0].text}")

                # 5. Test read_excel
                print("\n--- Testing read_excel (projects.xlsx) ---")
                res = await session.call_tool("read_excel", {"filename": "projects.xlsx"})
                print(f"Result: {res.content[0].text}")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_local_files())
