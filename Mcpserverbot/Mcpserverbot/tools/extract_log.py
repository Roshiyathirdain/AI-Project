import re

with open('engine_debug.log', 'r', encoding='utf-8') as f:
    content = f.read()

# Find last "Attempt X SQL"
matches = re.findall(r'Attempt \d SQL: (.*?)(?=\[|$)', content, re.DOTALL)
if matches:
    sql = matches[-1].strip()
    print("LAST_SQL_START")
    print(sql)
    print("LAST_SQL_END")

# Find last error
errors = re.findall(r'SQL Execution Error: (.*?)(?=\[|$)', content, re.DOTALL)
if errors:
    err = errors[-1].strip()
    print("LAST_ERROR_START")
    print(err)
    print("LAST_ERROR_END")
