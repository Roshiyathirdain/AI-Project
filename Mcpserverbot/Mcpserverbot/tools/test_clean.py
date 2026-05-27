
import re

def _clean_sql(raw: str) -> str:
    # Remove <think> blocks (DeepSeek)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    
    # Strip markdown code fences
    raw = re.sub(r"```(?:sql)?\s*(.*?)\s*```", r"\1", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # Pre-strip common LLM prefixes (case-insensitive)
    prefixes = ["Here is the SQL:", "The SQL query is:", "Based on the schema:", "Given your request:"]
    temp_raw = raw
    for p in prefixes:
        if temp_raw.lower().startswith(p.lower()):
            temp_raw = temp_raw[len(p):].strip()
            break
    
    # Find SELECT keyword and stop at common explanation keywords
    m = re.search(r"(SELECT\s+.*)", temp_raw, re.IGNORECASE | re.DOTALL)
    sql = m.group(1).strip() if m else temp_raw.strip()
    
    # Remove surrounding quotes
    if (sql.startswith('"') and sql.endswith('"')) or (sql.startswith("'") and sql.endswith("'")):
        sql = sql[1:-1].strip()
        
    # Aggrsessively cut off after the first statement or explanation block
    stop_words = [
        "Explanation:", "Note:", "This query", "Here is", "\n--", "\n/*", 
        "Given the corrected", "does not exist", "The table", "I have corrected",
        "Please note", "According to", "Should you need", "Let me know"
    ]
    for stop in stop_words:
        idx = sql.lower().find(stop.lower())
        if idx != -1:
            sql = sql[:idx].strip()
            
    # Take only the first statement (up to semicolon)
    sql = sql.split(";")[0].strip()
    return sql

test_cases = [
    "SELECT * FROM users;",
    "Sure! Here is the SQL: SELECT * FROM users",
    "```sql\nSELECT name FROM employees\n```",
    "SELECT * FROM orders. Note: this table doesn't have prices.",
    "Based on the schema: SELECT hire_date FROM staff. The table employees does not exist in the schema.",
    "I have corrected your query: SELECT * FROM customers. Given the corrected understanding of your database..."
]

for i, tc in enumerate(test_cases):
    print(f"Test {i+1}: '{tc}'")
    print(f"Result: '{_clean_sql(tc)}'")
    print("-" * 20)
