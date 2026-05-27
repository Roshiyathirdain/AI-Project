import json
from mcp_engine import mcp_answer, get_all_mcp_tools_sync

tools = get_all_mcp_tools_sync()
questions = ["Top customers by total sales", "Total stock quantity", "Sales trend by month"]

for question in questions:
    print(f"\nTesting question: {question}")
    res = mcp_answer(question, tools, "qwen3.5:cloud", data_source="Postgres")

    print(f"SQL: {res.get('sql')}")
    print(f"Answer: {res.get('answer')}")
    if res.get('data'):
        print(f"Data Rows: {len(res['data'])}")
        if isinstance(res['data'], list) and len(res['data']) > 0:
            print(f"Sample: {res['data'][0]}")
