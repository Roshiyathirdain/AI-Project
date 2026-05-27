import os
import json
from mcp_engine import get_all_mcp_tools_sync

tools = get_all_mcp_tools_sync()
print(f"Total tools found: {len(tools)}")
for t in tools:
    print(f"- Server: {t['server']}, Tool: {t['name']}")
