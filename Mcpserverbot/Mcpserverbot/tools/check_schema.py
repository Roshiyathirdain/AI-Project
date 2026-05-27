import pymysql
import json
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASS = os.getenv("MYSQL_PASSWORD", "")
DB_NAME = os.getenv("MYSQL_DB", "bigcompanydb")

def get_schema():
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            schema_info = []
            for table_dict in tables:
                table_name = list(table_dict.values())[0]
                cursor.execute(f"DESCRIBE `{table_name}`")
                columns = cursor.fetchall()
                cols_str = ", ".join([f"{c['Field']} ({c['Type']})" for c in columns])
                
                try:
                    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 1")
                    samples = cursor.fetchall()
                    sample_str = f" | Sample: {json.dumps(samples, default=str)}" if samples else ""
                except:
                    sample_str = ""
                
                schema_info.append(f"Table `{table_name}`: {cols_str}{sample_str}")
            return "\n".join(schema_info)
    finally:
        conn.close()

if __name__ == "__main__":
    schema = get_schema()
    print(f"Total Schema Length: {len(schema)} characters")
    print("-" * 50)
    print(schema[:1000] + "...")
