import json
from mcp_engine import mcp_answer, get_all_mcp_tools_sync

tools = get_all_mcp_tools_sync()
question = "Top customers by total sales"
print(f"Testing question: {question}")
# Use a common Ollama model if possible, or just the default
# In the history it was using qwen3.5:cloud (which mapping to qwen2.5:latest)
res = mcp_answer(question, tools, "qwen3.5:cloud", data_source="Postgres")

print("\n--- RESULT ---")
print(f"SQL: {res.get('sql')}")
print(f"Answer: {res.get('answer')}")
if res.get('data'):
    print(f"Data Rows: {len(res['data'])}")
    print(f"Sample: {res['data'][0]}")
else:
    print("No data returned.")
