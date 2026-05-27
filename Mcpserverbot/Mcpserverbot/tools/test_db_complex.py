import json
from mcp_engine import mcp_answer, get_all_mcp_tools_sync

tools = get_all_mcp_tools_sync()
question = "Total expenses for regions in Oman"

print(f"Testing Complex Question: {question}")
res = mcp_answer(question, tools, "qwen3.5:cloud", data_source="Postgres")

print(f"SQL: {res.get('sql')}")
print(f"Answer: {res.get('answer')}")
if res.get('data'):
    print(f"Data: {res['data']}")
else:
    print("No data returned.")
