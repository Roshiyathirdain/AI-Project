import pymysql
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv

load_dotenv()

# MySQL Connection
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "bigcompanydb"),
}

# Postgres Connection
# Using 'postgres' DB initially to create the target DB if needed
PG_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "dbname": os.getenv("POSTGRES_DB", "your_db"),
}

def migrate():
    print("🚀 Starting Automated Migration: MySQL -> PostgreSQL")
    
    # 1. Ensure target database exists
    target_db = PG_CONFIG["dbname"]
    try:
        # Connect to default 'postgres' to create the target DB
        admin_conn = psycopg2.connect(
            host=PG_CONFIG["host"],
            port=PG_CONFIG["port"],
            user=PG_CONFIG["user"],
            password=PG_CONFIG["password"],
            dbname="postgres"
        )
        admin_conn.autocommit = True
        with admin_conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            if not cur.fetchone():
                print(f"🔹 Creating target database: {target_db}")
                cur.execute(f'CREATE DATABASE "{target_db}"')
            else:
                print(f"🔹 Target database {target_db} already exists.")
        admin_conn.close()
    except Exception as e:
        print(f"⚠️ Postgres Connection/Setup Error: {e}")
        print("Waiting for Postgres to be fully ready...")
        return

    try:
        my_conn = pymysql.connect(**MYSQL_CONFIG, cursorclass=pymysql.cursors.DictCursor)
        pg_conn = psycopg2.connect(
            host=PG_CONFIG["host"],
            port=PG_CONFIG["port"],
            user=PG_CONFIG["user"],
            password=PG_CONFIG["password"],
            dbname=target_db
        )
        pg_conn.autocommit = True
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    try:
        with my_conn.cursor() as my_cur, pg_conn.cursor() as pg_cur:
            # Get Tables
            my_cur.execute("SHOW TABLES")
            tables = [list(t.values())[0] for t in my_cur.fetchall()]
            
            for table in tables:
                print(f"--- Processing Table: {table} ---")
                
                # Get Schema
                my_cur.execute(f"DESCRIBE `{table}`")
                cols = my_cur.fetchall()
                
                # Build Postgres CREATE TABLE
                pg_cols = []
                for col in cols:
                    name = col['Field']
                    m_type = col['Type'].lower()
                    
                    # Type Mapping
                    p_type = "TEXT"
                    if "int" in m_type: p_type = "INTEGER"
                    if "decimal" in m_type: p_type = m_type
                    if "varchar" in m_type: p_type = m_type
                    if "text" in m_type: p_type = "TEXT"
                    if "date" == m_type: p_type = "DATE"
                    if "time" == m_type: p_type = "TIME"
                    if "timestamp" in m_type or "datetime" in m_type: p_type = "TIMESTAMP"
                    if "enum" in m_type: p_type = "VARCHAR(50)"
                    if "tinyint(1)" in m_type: p_type = "BOOLEAN"
                    
                    if col['Key'] == 'PRI':
                        if "int" in m_type: p_type = "SERIAL PRIMARY KEY"
                        else: p_type += " PRIMARY KEY"
                    
                    pg_cols.append(f'"{name}" {p_type}')
                
                # Drop and Create
                pg_cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
                pg_cur.execute(f'CREATE TABLE "{table}" ({", ".join(pg_cols)})')
                print(f"✅ Created table {table}")
                
                # Migrate Data
                my_cur.execute(f"SELECT * FROM `{table}`")
                rows = my_cur.fetchall()
                if rows:
                    columns = rows[0].keys()
                    col_names = ", ".join([f'"{c}"' for c in columns])
                    placeholders = ", ".join(["%s"] * len(columns))
                    
                    # Detect which columns are boolean in our pg_cols mapping
                    bool_cols = []
                    for col_def in pg_cols:
                        c_name = col_def.split('"')[1]
                        if "BOOLEAN" in col_def:
                            bool_cols.append(c_name)

                    insert_query = f'INSERT INTO "{table}" ({col_names}) VALUES ({placeholders})'
                    
                    data_to_insert = []
                    for row in rows:
                        vals = []
                        for c in columns:
                            v = row[c]
                            if c in bool_cols:
                                v = bool(v) if v is not None else None
                            vals.append(v)
                        data_to_insert.append(tuple(vals))
                    
                    pg_cur.executemany(insert_query, data_to_insert)
                    print(f"✅ Migrated {len(rows)} records to {table}")
                else:
                    print(f"ℹ️ Table {table} is empty.")

        print("\n🎉 Migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration Error: {e}")
    finally:
        my_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    migrate()
