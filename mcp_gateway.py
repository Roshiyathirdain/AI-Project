# mcp_gateway.py
import asyncio
import os
import json
import logging
from typing import Dict, List, Any, Optional
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server.stdio import stdio_server
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from dotenv import load_dotenv

load_dotenv()

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, filename="logs/gateway.log", filemode="a",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp-gateway")

class Connector:
    def __init__(self, name: str, command: str, args: List[str]):
        self.name = name
        self.command = command
        self.args = args
        self.session: Optional[ClientSession] = None
        self._cleanup_ctx = None

    async def connect(self):
        try:
            params = StdioServerParameters(command=self.command, args=self.args, env=os.environ.copy())
            self._cleanup_ctx = stdio_client(params)
            read, write = await self._cleanup_ctx.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            await self.session.initialize()
            logger.info(f"Connected to connector: {self.name}")
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            raise

    async def disconnect(self):
        if self.session:
            try: await self.session.__aexit__(None, None, None)
            except: pass
        if self._cleanup_ctx:
            try: await self._cleanup_ctx.__aexit__(None, None, None)
            except: pass

from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn

# Dynamically detect base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

CONNECTOR_CONFIGS = {
    "mysql": {
        "command": PYTHON_EXE,
        "args": [os.path.join(BASE_DIR, "mcp_mysql_server.py")]
    },
    "local_files": {
        "command": PYTHON_EXE,
        "args": [os.path.join(BASE_DIR, "mcp_local_file_server.py")]
    },
    "postgres": {
        "command": PYTHON_EXE,
        "args": [os.path.join(BASE_DIR, "mcp_postgres_server.py")]
    },
    "sharepoint": {
        "command": PYTHON_EXE,
        "args": [os.path.join(BASE_DIR, "mcp_sharepoint_server.py")]
    }
}

gateway = Server("enterprise-mcp-gateway")
connectors: Dict[str, Connector] = {}

# ... (Previous tool/resource/prompt handlers remain exactly the same) ...
# I will repeat them here to ensure the file is complete and valid.

@gateway.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    all_tools = []
    async def get_tools(name, connector):
        if not connector.session: return []
        try:
            tools_resp = await connector.session.list_tools()
            for tool in tools_resp.tools:
                tool.name = f"{name}_{tool.name}"
                tool.description = f"[{name.upper()}] {tool.description}"
            return tools_resp.tools
        except Exception as e:
            logger.error(f"Error listing tools for {name}: {e}")
            return []
    results = await asyncio.gather(*(get_tools(name, conn) for name, conn in connectors.items()))
    for tools in results: all_tools.extend(tools)
    return all_tools

@gateway.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    conn_name = None
    tool_name = None
    for prefix in sorted(connectors.keys(), key=len, reverse=True):
        if name.startswith(prefix + "_"):
            conn_name = prefix
            tool_name = name[len(prefix) + 1:]
            break
    if not conn_name:
        return [types.TextContent(type="text", text=f"Error: Could not resolve connector for tool '{name}'.")]
    connector = connectors.get(conn_name)
    if not connector or not connector.session:
        return [types.TextContent(type="text", text=f"Error: Connector '{conn_name}' not active")]
    try:
        result = await connector.session.call_tool(tool_name, arguments or {})
        return result.content
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {e}")]

@gateway.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    all_resources = []
    async def get_res(name, connector):
        if not connector.session: return []
        try:
            res_resp = await connector.session.list_resources()
            for res in res_resp.resources:
                res.uri = f"mcp://{name}/{res.uri.replace('mcp://', '')}"
                res.name = f"[{name.upper()}] {res.name}"
            return res_resp.resources
        except Exception as e:
            logger.error(f"Error listing resources for {name}: {e}")
            return []
    results = await asyncio.gather(*(get_res(name, conn) for name, conn in connectors.items()))
    for resources in results: all_resources.extend(resources)
    return all_resources

@gateway.read_resource()
async def handle_read_resource(uri: str) -> str:
    if not uri.startswith("mcp://"): return "Error: Invalid URI"
    parts = uri.replace("mcp://", "").split("/", 1)
    if len(parts) < 2: return "Error: Path too short"
    conn_name, sub_uri = parts[0], parts[1]
    if conn_name in connectors and connectors[conn_name].session:
        result = await connectors[conn_name].session.read_resource(f"mcp://{sub_uri}")
        return result.contents[0].text
    return "Error: Connector not found"

@gateway.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    all_prompts = []
    for name, connector in connectors.items():
        if not connector.session: continue
        try:
            p_resp = await connector.session.list_prompts()
            for p in p_resp.prompts:
                p.name = f"{name}_{p.name}"
                all_prompts.append(p)
        except Exception as e:
            logger.error(f"Error listing prompts for {name}: {e}")
    return all_prompts

# ─── ASGI SERVER ──────────────────────────────────────────────────────────────
sse = SseServerTransport("/messages")

async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await start_connectors()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                for conn in connectors.values():
                    await conn.disconnect()
                await send({"type": "lifespan.shutdown.complete"})
                return

    if scope["type"] == "http":
        path = scope["path"]
        if path == "/sse":
            async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await gateway.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="enterprise-mcp-gateway",
                        server_version="1.2.0",
                        capabilities=gateway.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        elif path == "/messages":
            await sse.handle_post_message(scope, receive, send)
        else:
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")],
            })
            await send({
                "type": "http.response.body",
                "body": b"Not Found",
            })

async def start_connectors():
    async def init_connector(name, config):
        conn = Connector(name, config["command"], config["args"])
        try:
            await conn.connect()
            return name, conn
        except Exception as e:
            logger.error(f"Failed to initialize connector {name}: {e}")
            return None

    tasks = [init_connector(name, config) for name, config in CONNECTOR_CONFIGS.items()]
    results = await asyncio.gather(*tasks)
    for res in results:
        if res:
            name, conn = res
            connectors[name] = conn
    logger.info(f"Gateway connected to {len(connectors)} internal servers.")

if __name__ == "__main__":
    logger.info("🚀 Glimpse AI Gateway launching on http://localhost:8000/sse")
    uvicorn.run(app, host="0.0.0.0", port=8000)
