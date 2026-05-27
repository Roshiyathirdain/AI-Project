from mcp_engine import get_all_mcp_tools_sync
import json

tools = get_all_mcp_tools_sync()
print(f"Found {len(tools)} tools.")
if tools:
    first_tool = tools[0]
    print(f"Sample tool: {first_tool['name']} from {first_tool['server']}")
else:
    print("NO TOOLS FOUND")
