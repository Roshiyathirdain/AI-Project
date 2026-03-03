# mcp_postgres_server.py
# Open-source style PostgreSQL MCP Server
# Based on: https://github.com/crystaldba/postgres-mcp (Postgres MCP Pro patterns)
# Python 3.10+ compatible, uses psycopg2
#
# Tools exposed:
#   - list_schemas       : List all schemas in the database
#   - list_tables        : List all tables (optionally filtered by schema)
#   - get_schema         : Full schema dump (tables + columns + types)
#   - describe_table     : Describe a single table's structure
#   - execute_sql        : Execute any SQL query
#   - get_table_data     : Read rows from a table with optional limit
#   - analyze_db_health  : Basic database health statistics

import asyncio
import os
import json
import logging
from dotenv import load_dotenv
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server.stdio import stdio_server

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PG-MCP] %(levelname)s: %(message)s")
logger = logging.getLogger("postgres-mcp-server")

# ── Database config from environment ──────────────────────────────────────────
PG_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "dbname":   os.getenv("POSTGRES_DB", "postgres"),
}

# Also support DATABASE_URI format (used by crystaldba/postgres-mcp)
DATABASE_URI = os.getenv("DATABASE_URI", "")

# ── Helper ─────────────────────────────────────────────────────────────────────
def get_connection():
    import psycopg2
    import psycopg2.extras
    if DATABASE_URI:
        return psycopg2.connect(DATABASE_URI)
    return psycopg2.connect(**PG_CONFIG)

def run_query(sql: str, params=None, read_only=False):
    """Execute a SQL statement and return results as a list of dicts."""
    import psycopg2.extras
    conn = get_connection()
    try:
        if read_only:
            conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            sql_upper = sql.strip().upper()
            if any(sql_upper.startswith(kw) for kw in ["SELECT", "SHOW", "EXPLAIN", "WITH"]):
                rows = cur.fetchall()
                return [dict(r) for r in rows]
            else:
                if not read_only:
                    conn.commit()
                return [{"affected_rows": cur.rowcount, "status": "OK"}]
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# ── MCP Server ─────────────────────────────────────────────────────────────────
server = Server("postgres-mcp-server")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """Expose each table as an MCP resource."""
    try:
        sql = """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
        """
        tables = run_query(sql)
        resources = []
        for row in tables:
            schema = row["table_schema"]
            tname  = row["table_name"]
            resources.append(
                types.Resource(
                    uri=f"mcp://postgres/{schema}/{tname}",
                    name=f"{schema}.{tname}",
                    description=f"PostgreSQL table: {schema}.{tname}",
                    mimeType="application/json",
                )
            )
        return resources
    except Exception as e:
        logger.error(f"list_resources error: {e}")
        return []

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read data from a table resource."""
    if not uri.startswith("mcp://postgres/"):
        return "Error: Unknown resource URI"
    parts = uri.replace("mcp://postgres/", "").split("/")
    if len(parts) < 2:
        return "Error: Invalid URI format"
    schema, table = parts[0], parts[1]
    try:
        rows = run_query(f'SELECT * FROM "{schema}"."{table}" LIMIT 100')
        return json.dumps(rows, default=str, indent=2)
    except Exception as e:
        return f"Error reading table {schema}.{table}: {e}"

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_schemas",
            description="List all schemas in the PostgreSQL database.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="list_tables",
            description="List all tables in the PostgreSQL database, optionally filtered by schema name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "string",
                        "description": "Optional schema name to filter tables (e.g. 'public')."
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_schema",
            description="Get the full database schema: all tables with columns, data types, nullability, and primary keys.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="describe_table",
            description="Describe the structure of a specific PostgreSQL table (columns, types, keys).",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table name (use 'schema.table' format for non-public schemas)."
                    }
                },
                "required": ["table_name"]
            }
        ),
        types.Tool(
            name="execute_sql",
            description="Execute a SQL query against the PostgreSQL database. Supports SELECT, INSERT, UPDATE, DELETE, CREATE.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL statement to execute."
                    }
                },
                "required": ["sql"]
            }
        ),
        types.Tool(
            name="query_database",
            description="Run a SQL SELECT query against the PostgreSQL database and return results as JSON.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A SELECT SQL statement to execute."
                    }
                },
                "required": ["sql"]
            }
        ),
        types.Tool(
            name="get_table_data",
            description="Retrieve rows from a PostgreSQL table with an optional row limit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table name (use 'schema.table' or just 'table')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return (default: 50).",
                        "default": 50
                    }
                },
                "required": ["table_name"]
            }
        ),
        types.Tool(
            name="analyze_db_health",
            description="Get key PostgreSQL database health metrics: table sizes, index usage, connection count, and bloat indicators.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        # ── list_schemas ─────────────────────────────────────────────────────
        if name == "list_schemas":
            rows = run_query(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast') "
                "ORDER BY schema_name"
            )
            return [types.TextContent(type="text", text=json.dumps([r["schema_name"] for r in rows], indent=2))]

        # ── list_tables ──────────────────────────────────────────────────────
        elif name == "list_tables":
            schema_filter = arguments.get("schema", None)
            sql = """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                  AND table_type = 'BASE TABLE'
            """
            if schema_filter:
                sql += f" AND table_schema = '{schema_filter}'"
            sql += " ORDER BY table_schema, table_name"
            rows = run_query(sql)
            return [types.TextContent(type="text", text=json.dumps(rows, default=str, indent=2))]

        # ── get_schema ───────────────────────────────────────────────────────
        elif name == "get_schema":
            sql = """
                SELECT
                    c.table_schema,
                    c.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    CASE WHEN kcu.column_name IS NOT NULL THEN 'YES' ELSE 'NO' END AS is_primary_key
                FROM information_schema.columns c
                LEFT JOIN information_schema.key_column_usage kcu
                    ON c.table_schema = kcu.table_schema
                    AND c.table_name = kcu.table_name
                    AND c.column_name = kcu.column_name
                    AND kcu.constraint_name IN (
                        SELECT constraint_name FROM information_schema.table_constraints
                        WHERE constraint_type = 'PRIMARY KEY'
                    )
                WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY c.table_schema, c.table_name, c.ordinal_position
            """
            rows = run_query(sql)
            # Group by table
            schema = {}
            for row in rows:
                key = f"{row['table_schema']}.{row['table_name']}"
                if key not in schema:
                    schema[key] = []
                schema[key].append({
                    "column": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"],
                    "default": row["column_default"],
                    "primary_key": row["is_primary_key"],
                })
            return [types.TextContent(type="text", text=json.dumps(schema, default=str, indent=2))]

        # ── describe_table ───────────────────────────────────────────────────
        elif name == "describe_table":
            table_ref = arguments.get("table_name", "")
            if not table_ref:
                return [types.TextContent(type="text", text="ERROR: table_name is required")]
            parts = table_ref.split(".")
            schema_name = parts[0] if len(parts) == 2 else "public"
            table_name  = parts[1] if len(parts) == 2 else parts[0]
            sql = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """
            rows = run_query(sql, (schema_name, table_name))
            return [types.TextContent(type="text", text=json.dumps(rows, default=str, indent=2))]

        # ── execute_sql / query_database ─────────────────────────────────────
        elif name in ("execute_sql", "query_database"):
            sql = arguments.get("sql", "").strip()
            if not sql:
                return [types.TextContent(type="text", text="ERROR: sql is required")]
            rows = run_query(sql)
            return [types.TextContent(type="text", text=json.dumps(rows, default=str, indent=2))]

        # ── get_table_data ───────────────────────────────────────────────────
        elif name == "get_table_data":
            table_ref = arguments.get("table_name", "")
            limit = int(arguments.get("limit", 50))
            if not table_ref:
                return [types.TextContent(type="text", text="ERROR: table_name is required")]
            parts = table_ref.split(".")
            if len(parts) == 2:
                full_table = f'"{parts[0]}"."{parts[1]}"'
            else:
                full_table = f'"{parts[0]}"'
            rows = run_query(f"SELECT * FROM {full_table} LIMIT {limit}")
            return [types.TextContent(type="text", text=json.dumps(rows, default=str, indent=2))]

        # ── analyze_db_health ────────────────────────────────────────────────
        elif name == "analyze_db_health":
            health = {}

            # Active connections
            conn_rows = run_query(
                "SELECT count(*) AS total_connections, "
                "sum(CASE WHEN state='active' THEN 1 ELSE 0 END) AS active "
                "FROM pg_stat_activity"
            )
            health["connections"] = conn_rows[0] if conn_rows else {}

            # Table sizes
            size_rows = run_query(
                """
                SELECT schemaname || '.' || relname AS table_name,
                       pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
                       pg_size_pretty(pg_relation_size(relid)) AS table_size,
                       n_live_tup AS live_rows,
                       n_dead_tup AS dead_rows
                FROM pg_stat_user_tables
                ORDER BY pg_total_relation_size(relid) DESC
                LIMIT 20
                """
            )
            health["largest_tables"] = size_rows

            # Index usage
            idx_rows = run_query(
                """
                SELECT schemaname || '.' || relname AS table_name,
                       idx_scan AS index_scans,
                       seq_scan AS sequential_scans
                FROM pg_stat_user_tables
                WHERE seq_scan > 0
                ORDER BY seq_scan DESC
                LIMIT 10
                """
            )
            health["index_usage"] = idx_rows

            # DB size
            db_size = run_query("SELECT pg_size_pretty(pg_database_size(current_database())) AS database_size")
            health["database_size"] = db_size[0] if db_size else {}

            return [types.TextContent(type="text", text=json.dumps(health, default=str, indent=2))]

        else:
            return [types.TextContent(type="text", text=f"ERROR: Unknown tool '{name}'")]

    except Exception as e:
        logger.error(f"Tool '{name}' error: {e}")
        return [types.TextContent(type="text", text=f"ERROR: {e}")]

# ── Entry point ────────────────────────────────────────────────────────────────
async def main():
    db_info = DATABASE_URI if DATABASE_URI else f"{PG_CONFIG['dbname']}@{PG_CONFIG['host']}"
    logger.info(f"Starting PostgreSQL MCP Server | DB: {db_info}")
    async with stdio_server() as (read, write):
        await server.run(
            read, write,
            InitializationOptions(
                server_name="postgres-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
