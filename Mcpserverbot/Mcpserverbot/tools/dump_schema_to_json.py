import pymysql
import json
import os
from dotenv import load_dotenv

load_dotenv()

conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DB", "bigcompanydb"),
    cursorclass=pymysql.cursors.DictCursor
)

schema = {}

try:
    with conn.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        for table_dict in tables:
            table_name = list(table_dict.values())[0]
            cursor.execute(f"DESCRIBE `{table_name}`")
            columns = cursor.fetchall()
            schema[table_name] = columns
            
    with open("full_schema.json", "w") as f:
        json.dump(schema, f, indent=4)
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
