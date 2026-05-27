# mcp_mysql_server.py
# Open-source style MySQL MCP Server
# Based on: https://github.com/designcomputer/mysql_mcp_server
# Python 3.10+ compatible, uses pymysql
#
# Tools exposed:
#   - list_tables        : List all tables in the database
#   - get_schema         : Get full schema (all tables + columns)
#   - describe_table     : Describe a single table's columns
#   - query_database     : Execute any SQL query (SELECT/INSERT/UPDATE/DELETE)
#   - get_table_data     : Read rows from a table with optional limit

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MySQL-MCP] %(levelname)s: %(message)s")
logger = logging.getLogger("mysql-mcp-server")

# ── Database config from environment ──────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB") or os.getenv("MYSQL_DATABASE", ""),
    "charset":  "utf8mb4",
}

# ── Helper ─────────────────────────────────────────────────────────────────────
def get_connection():
    import pymysql
    import pymysql.cursors
    config = DB_CONFIG.copy()
    config["cursorclass"] = pymysql.cursors.DictCursor
    return pymysql.connect(**config)

def run_query(sql: str, params=None):
    """Execute a SQL statement and return results as a list of dicts."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if sql.strip().upper().startswith("SELECT") or sql.strip().upper().startswith("SHOW") or sql.strip().upper().startswith("DESCRIBE"):
                rows = cur.fetchall()
                return rows
            else:
                conn.commit()
                return [{"affected_rows": cur.rowcount}]
    finally:
        conn.close()

# ── MCP Server ─────────────────────────────────────────────────────────────────
server = Server("mysql-mcp-server")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """Expose each table as an MCP resource."""
    try:
        tables = run_query("SHOW TABLES")
        resources = []
        for row in tables:
            table_name = list(row.values())[0]
            resources.append(
                types.Resource(
                    uri=f"mcp://mysql/table/{table_name}",
                    name=table_name,
                    description=f"MySQL table: {table_name}",
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
    if not uri.startswith("mcp://mysql/table/"):
        return "Error: Unknown resource URI"
    table_name = uri.replace("mcp://mysql/table/", "")
    try:
        rows = run_query(f"SELECT * FROM `{table_name}` LIMIT 100")
        return json.dumps(rows, default=str, indent=2)
    except Exception as e:
        return f"Error reading table {table_name}: {e}"

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_tables",
            description="List all tables available in the MySQL database.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_schema",
            description="Get the full database schema: all tables with their columns, types, keys, and nullability.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="describe_table",
            description="Describe the structure (columns, types, keys) of a specific MySQL table.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to describe."
                    }
                },
                "required": ["table_name"]
            }
        ),
        types.Tool(
            name="query_database",
            description="Execute a SQL query against the MySQL database. Supports SELECT, INSERT, UPDATE, DELETE.",
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
            name="get_table_data",
            description="Retrieve rows from a MySQL table with an optional row limit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to read from."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default: 50).",
                        "default": 50
                    }
                },
                "required": ["table_name"]
            }
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        # ── list_tables ──────────────────────────────────────────────────────
        if name == "list_tables":
            rows = run_query("SHOW TABLES")
            tables = [list(r.values())[0] for r in rows]
            return [types.TextContent(type="text", text=json.dumps(tables, indent=2))]

        # ── get_schema ───────────────────────────────────────────────────────
        elif name == "get_schema":
            tables = run_query("SHOW TABLES")
            schema = {}
            for row in tables:
                table_name = list(row.values())[0]
                cols = run_query(f"DESCRIBE `{table_name}`")
                schema[table_name] = cols
            return [types.TextContent(type="text", text=json.dumps(schema, default=str, indent=2))]

        # ── describe_table ───────────────────────────────────────────────────
        elif name == "describe_table":
            table_name = arguments.get("table_name", "")
            if not table_name:
                return [types.TextContent(type="text", text="ERROR: table_name is required")]
            cols = run_query(f"DESCRIBE `{table_name}`")
            return [types.TextContent(type="text", text=json.dumps(cols, default=str, indent=2))]

        # ── query_database ───────────────────────────────────────────────────
        elif name == "query_database":
            sql = arguments.get("sql", "").strip()
            if not sql:
                return [types.TextContent(type="text", text="ERROR: sql is required")]
            rows = run_query(sql)
            return [types.TextContent(type="text", text=json.dumps(rows, default=str, indent=2))]

        # ── get_table_data ───────────────────────────────────────────────────
        elif name == "get_table_data":
            table_name = arguments.get("table_name", "")
            limit = int(arguments.get("limit", 50))
            if not table_name:
                return [types.TextContent(type="text", text="ERROR: table_name is required")]
            rows = run_query(f"SELECT * FROM `{table_name}` LIMIT {limit}")
            return [types.TextContent(type="text", text=json.dumps(rows, default=str, indent=2))]

        else:
            return [types.TextContent(type="text", text=f"ERROR: Unknown tool '{name}'")]

    except Exception as e:
        logger.error(f"Tool '{name}' error: {e}")
        return [types.TextContent(type="text", text=f"ERROR: {e}")]

# ── Entry point ────────────────────────────────────────────────────────────────
async def main():
    logger.info(f"Starting MySQL MCP Server | DB: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    async with stdio_server() as (read, write):
        await server.run(
            read, write,
            InitializationOptions(
                server_name="mysql-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
