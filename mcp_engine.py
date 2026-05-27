# mcp_engine.py
import os
import json
import re
import asyncio
import requests
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
import nest_asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

nest_asyncio.apply()
load_dotenv(override=True)

def log_debug(msg):
    os.makedirs("logs", exist_ok=True)
    with open("logs/engine_debug.log", "a", encoding="utf-8") as f:
        f.write(f"[{time.ctime()}] {msg}\n")

# ------------------------------------------------
# Configuration
# ------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
# Force localhost to avoid 0.0.0.0 invalid URL errors from system env
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"
if "0.0.0.0" in OLLAMA_HOST:
    OLLAMA_HOST = OLLAMA_HOST.replace("0.0.0.0", "localhost")

# MCP Server Config
try:
    mcp_servers_raw = os.getenv("MCP_SERVERS", "{}")
    if mcp_servers_raw.startswith("'") and mcp_servers_raw.endswith("'"):
        mcp_servers_raw = mcp_servers_raw[1:-1]
    MCP_SERVERS_CONFIG = json.loads(mcp_servers_raw)
except Exception as e:
    print(f"Error parsing MCP_SERVERS: {e}")
    MCP_SERVERS_CONFIG = {}

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# ------------------------------------------------
# MCP Client Logic (Now using HTTP SSE)
# ------------------------------------------------
async def _call_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, Any] = None) -> str:
    if server_name not in MCP_SERVERS_CONFIG:
        return f"ERROR: Server {server_name} not found in config."
    
    config = MCP_SERVERS_CONFIG[server_name]
    # If the config contains 'url', use SSE. Otherwise, fallback logic (though user requested HTTP)
    url = config.get("url", "http://localhost:8000/sse")
    
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments or {})
                texts = [c.text for c in result.content if hasattr(c, "text")]
                return "\n".join(texts)
    except Exception as e:
        return f"ERROR calling Streamable HTTP tool: {e}"

def call_mcp_tool_sync(server_name: str, tool_name: str, arguments: Dict[str, Any] = None) -> str:
    loop = asyncio.get_event_loop()
    # We use loop.run_until_complete which works with nest_asyncio
    return loop.run_until_complete(_call_mcp_tool(server_name, tool_name, arguments))

# ------------------------------------------------
# Z.ai LLM call (Production)
# ------------------------------------------------
def _zai_chat(prompt: str, model: str = None, timeout: int = 300) -> str:
    """Unified LLM call function supporting both Z.ai and local Ollama."""
    model = model or LLM_MODEL
    
    # -- BRANCH 1: Z.ai (Cloud) -- 
    # Use Z.ai if model explicitly says ":cloud" and we have a key
    if ZAI_API_KEY and (":cloud" in model.lower() or ":" not in model):
        url = "https://api.z.ai/api/paas/v4/chat/completions"
        headers = {
            "Authorization": f"Bearer {ZAI_API_KEY}",
            "Content-Type": "application/json"
        }
        target_model = model.replace(":cloud", "")
        if "flash" in model.lower() or model == "glm-4":
            target_model = "glm-4.7"
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            log_debug(f"Calling Z.ai API: {target_model}")
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                raise RuntimeError("🚨 **Z.ai Account Balance Empty**")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log_debug(f"Z.ai call failed (falling back to Ollama if available): {e}")

    # -- BRANCH 2: Ollama (Local) --
    ollama_model = model if ":" in model and ":cloud" not in model.lower() else "phi3:latest"
    if model.endswith(":cloud"): ollama_model = "phi3:latest"
    
    url = OLLAMA_HOST.rstrip("/") + "/api/generate"
    try:
        log_debug(f"Calling Local Ollama: {ollama_model}")
        resp = requests.post(url, json={"model": ollama_model, "prompt": prompt, "stream": False}, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        log_debug(f"Ollama call failed: {e}")
        raise RuntimeError(f"Intelligence Error: {e}")

    raise RuntimeError("No LLM provider available (Missing Z.ai Key and Ollama Offline).")

# ------------------------------------------------
# Tool List
# ------------------------------------------------
SCHEMA_CACHE = {}

def clear_schema_cache():
    """Manually clear the cached database schemas."""
    global SCHEMA_CACHE
    log_debug("Clearing SCHEMA_CACHE.")
    SCHEMA_CACHE = {}

def get_all_mcp_tools_sync() -> List[Dict[str, Any]]:
    # When we refresh tools, it's a good signal to also refresh schema
    clear_schema_cache()
    all_tools = []
    for server_name in MCP_SERVERS_CONFIG:
        try:
            log_debug(f"Fetching tools from server: {server_name}")
            def _get_tools():
                async def _async_get():
                    config = MCP_SERVERS_CONFIG[server_name]
                    url = config.get("url", "http://localhost:8000/sse")
                    async with sse_client(url) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools_result = await session.list_tools()
                            return [
                                {
                                    "server": server_name,
                                    "name": t.name,
                                    "description": t.description,
                                    "input_schema": t.inputSchema,
                                }
                                for t in tools_result.tools
                            ]
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_async_get())
            all_tools.extend(_get_tools())
        except Exception as e:
            log_debug(f"Error fetching tools from {server_name}: {e}")
    return all_tools

# ------------------------------------------------
# SQL Cleaning helper
# ------------------------------------------------
def _clean_sql(raw: str) -> str:
    """Strip LLM chatter, markdown fences, and explanations from a raw SQL string."""
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
    # Final check: Must start with SELECT
    if not sql.upper().startswith("SELECT"):
        # If it doesn't start with SELECT after all this, it might be junk
        pass
    
    # PRODUCTION FIX: Prevent "too minimal" queries (e.g. only selecting ID)
    if len(sql.split(",")) < 2 and "*" not in sql and "JOIN" not in sql.upper() and "LIMIT" not in sql.upper():
        words = sql.split()
        if len(words) >= 4 and words[0].upper() == "SELECT" and words[2].upper() == "FROM":
            table = words[3].strip("; ")
            sql = f"SELECT * FROM {table}"
            
    # CRITICAL FIX for Python DB Drivers: Escape % to %%
    # This prevents "not enough arguments for format string" when using LIKE '%...%'
    if "%" in sql and "%%" not in sql:
        sql = sql.replace("%", "%%")
        
    return sql

# ------------------------------------------------
# Heuristic-based intent detection  (NO LLM call)
# ------------------------------------------------
def _detect_intent(question: str, data_source: str):
    """
    Returns: ("sql", None) | ("tool", tool_keyword) | ("general", None)
    Uses ZERO LLM calls – pure keyword matching for speed.
    """
    q = question.lower()

    # ── 1. Greetings / meta
    greet_kw = ["hello", "hi ", "hey ", "what can you", "help me", "who are you",
                 "capabilities", "what data", "is connected", "how do you"]
    if any(k in q for k in greet_kw):
        return "general", None

    # ── 2. UI override forces SQL
    if data_source == "MySQL":
        return "sql", None
    if data_source == "Postgres":
        return "sql", None

    # ── 3. SharePoint keywords
    sp_kw = ["sharepoint", "microsoft", "office 365", "teams", "onedrive"]
    if data_source == "SharePoint" or any(k in q for k in sp_kw):
        return "tool", "sharepoint"

    # ── 4. Local file keywords
    file_kw = ["csv", "excel", "xlsx", ".xls", "local file", "document", "roadmap",
                "list files", "read file", "file in data"]
    if data_source == "Local Files" or any(k in q for k in file_kw):
        return "tool", "local_files"

    # ── 5. Postgres explicit
    if any(k in q for k in ["postgres", "postgresql", "cluster", "pg "]):
        return "tool", "postgres"

    # ── 6. General SQL / DB keywords
    sql_kw = [
        "show", "list", "get", "find", "employee", "salary", "payroll", "user",
        "attendance", "product", "sales", "customer", "order", "record", "table",
        "department", "project", "branch", "invoice", "report", "total", "count",
        "average", "highest", "lowest", "max", "min", "how many", "which",
    ]
    if any(k in q for k in sql_kw):
        return "sql", None

    # ── 7. Default – try SQL
    return "sql", None

# ------------------------------------------------
# Main answer function
# ------------------------------------------------
def mcp_answer(question: str, tools: List[Dict[str, Any]], llm_model: str, data_source: str = "Auto") -> Dict[str, Any]:
    global SCHEMA_CACHE

    intent, tool_keyword = _detect_intent(question, data_source)
    log_debug(f"Intent: {intent}, Tool: {tool_keyword}, Source: {data_source}, Q: {question}")

    # ======================================================
    # BRANCH A – General conversation (no LLM SQL, no tools)
    # ======================================================
    if intent == "general":
        sources = "MySQL Database, PostgreSQL Cluster, SharePoint (Microsoft), Local Files (CSV/Excel/Documents)"
        answer = (
            f"Hi! I am your Enterprise Intelligence BI Engine. I can answer questions using your data from:\n\n"
            f"• {sources}\n\n"
            f"Try asking: *'Show all employees'*, *'What is the total payroll?'*, or *'List files in SharePoint'*."
        )
        return {"answer": answer, "data": None, "sql": "General Conversation", "visual_hint": {"show": False}}

    # ======================================================
    # BRANCH B – Specialised Tool Call (SharePoint, Local Files)
    # ======================================================
    if intent == "tool":
        try:
            # ─── LOCAL FILES smart routing ───────────────────────────────
            if tool_keyword == "local_files":
                q_lower = question.lower()
                data_dir = os.path.join(os.path.dirname(__file__), "data")

                # ── Step 1: Try to find a specific file named in the question ──
                target_file = None
                if os.path.exists(data_dir):
                    # Sort files by modification time (most recent first)
                    all_data_files = sorted(
                        [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))],
                        key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
                        reverse=True
                    )

                    # Check if any file name (or stem without extension) appears in the question
                    for fname in all_data_files:
                        stem = fname.rsplit(".", 1)[0].lower()
                        if fname.lower() in q_lower or stem in q_lower:
                            target_file = fname
                            break

                # ── Step 2: Auto-detect document intent without explicit file name ──
                doc_intent_kw = [
                    "this report", "the report", "this document", "the document",
                    "this file", "the file", "this pdf", "the pdf",
                    "title", "topic", "summary", "summarize", "summarise",
                    "what is", "what does", "tell me", "say", "read", "open",
                    "content", "describe", "explain", "analyse", "analyze",
                    "introduction", "conclusion", "assignment", "assessment"
                ]
                is_doc_intent = any(k in q_lower for k in doc_intent_kw)

                list_intent_kw = ["list", "show files", "what files", "available files", "all files"]
                is_list_intent = any(k in q_lower for k in list_intent_kw)

                if not target_file and is_doc_intent and not is_list_intent:
                    # Auto-pick the most recently uploaded file
                    if os.path.exists(data_dir):
                        candidates = sorted(
                            [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))],
                            key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
                            reverse=True
                        )
                        if candidates:
                            target_file = candidates[0]
                            log_debug(f"Auto-selected most recent file: {target_file}")

                # ── Step 3: Choose the right read tool ──
                if target_file:
                    ext = target_file.rsplit(".", 1)[-1].lower() if "." in target_file else ""
                    if ext == "pdf":
                        tool_name_to_use = "local_files_read_pdf"
                    elif ext in ("xlsx", "xls"):
                        tool_name_to_use = "local_files_read_excel"
                    elif ext == "csv":
                        tool_name_to_use = "local_files_read_csv"
                    elif ext in ("docx", "doc"):
                        tool_name_to_use = "local_files_read_word"
                    else:
                        tool_name_to_use = "local_files_read_document"
                    args = {"filename": target_file}
                else:
                    # Explicit list request or no clue — just list files
                    tool_name_to_use = "local_files_list_files"
                    args = {}


                tool_obj = next((t for t in tools if t["name"] == tool_name_to_use), None)
                if not tool_obj:
                    # Fallback: any local_files tool
                    tool_obj = next((t for t in tools if t["name"].startswith("local_files_")), None)

                if not tool_obj:
                    return {"answer": "❌ Local Files connector not active. Ensure the gateway is running.", "data": None, "visual_hint": {"show": False}}

                log_debug(f"Local Files call: {tool_obj['name']} args={args}")
                result_text = call_mcp_tool_sync(tool_obj["server"], tool_obj["name"], args)
                log_debug(f"Local Files result (300): {result_text[:300]}")

                if result_text.startswith("Error"):
                    return {"answer": f"❌ File Error: {result_text}", "data": None, "visual_hint": {"show": False}}

                # For text content (PDF, Word, doc) return as readable text
                if "read_pdf" in tool_obj["name"] or "read_word" in tool_obj["name"] or "read_document" in tool_obj["name"]:
                    summary = f"📄 **Content of `{target_file}`:**\n\n{result_text[:3000]}"
                    if len(result_text) > 3000:
                        summary += f"\n\n*... ({len(result_text) - 3000} more characters)*"
                    return {"answer": summary, "data": None, "sql": f"Tool: {tool_obj['name']}({target_file})", "visual_hint": {"show": True, "type": "Auto"}}

                # For tabular data (CSV, Excel, list)
                data = None
                try:
                    data = json.loads(result_text)
                except Exception:
                    pass

                if not result_text.strip() or result_text.strip() in ("[]", "{}", '{"files": []}'):
                    return {"answer": "📁 No files found in the local data folder. Upload files using the sidebar.", "data": None, "visual_hint": {"show": False}}

                # List of files
                if isinstance(data, dict) and "files" in data:
                    files = data["files"]
                    if not files:
                        return {"answer": "📁 The data folder is empty. Upload files using the sidebar.", "data": None, "visual_hint": {"show": False}}
                    summary = f"📁 **{len(files)} file(s) in local store:**\n\n"
                    for f in files:
                        ext = f.rsplit(".", 1)[-1].lower() if "." in f else "?"
                        icons = {"csv": "📊", "xlsx": "📗", "xls": "📗", "pdf": "📕",
                                 "docx": "📘", "doc": "📘", "txt": "📄", "md": "📝", "json": "🗄️"}
                        summary += f"{icons.get(ext, '📎')} `{f}`\n"
                    return {"answer": summary, "data": None, "sql": "Tool: list_files", "visual_hint": {"show": False}}

                # Tabular data from CSV/Excel
                if isinstance(data, list) and data:
                    return {
                        "answer": f"📊 **`{target_file}`** — {len(data)} rows loaded.",
                        "data": data,
                        "sql": f"Tool: {tool_obj['name']}({target_file})",
                        "visual_hint": {"show": True, "type": "Auto"}
                    }

                return {"answer": f"Result:\n\n{result_text[:2000]}", "data": None, "visual_hint": {"show": True, "type": "Auto"}}

            # ─── SHAREPOINT routing ───────────────────────────────────────
            elif tool_keyword == "sharepoint":
                tool_obj = next((t for t in tools if "sharepoint" in t["name"] and "list" in t["name"]), None)
                if not tool_obj:
                    tool_obj = next((t for t in tools if "sharepoint" in t["name"]), None)
                if not tool_obj:
                    return {"answer": "❌ SharePoint connector not active. Check your .env credentials.", "data": None, "visual_hint": {"show": True, "type": "Auto"}}

                args = {}
                props = (tool_obj.get("input_schema") or {}).get("properties", {})
                if "query" in props:
                    args = {"query": question}

                log_debug(f"SharePoint call: {tool_obj['name']} args={args}")
                result_text = call_mcp_tool_sync(tool_obj["server"], tool_obj["name"], args)
                log_debug(f"SharePoint result (300): {result_text[:300]}")

                if result_text.startswith("Error") or "SharePoint Error" in result_text:
                    return {"answer": f"❌ SharePoint Error: {result_text}", "data": None, "visual_hint": {"show": True, "type": "Auto"}}

                data = None
                try:
                    data = json.loads(result_text)
                except Exception:
                    pass

                items = (data or {}).get("value", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                if items:
                    summary = f"Found **{len(items)} item(s)** in SharePoint:\n\n"
                    for item in items[:15]:
                        if isinstance(item, dict):
                            name = item.get("name") or item.get("displayName") or str(item)
                            summary += f"- 📄 {name}\n"
                    if len(items) > 15:
                        summary += f"\n*... and {len(items)-15} more*"
                else:
                    summary = f"SharePoint result:\n\n{result_text[:1000]}"

                return {"answer": summary, "data": data, "sql": f"Tool: {tool_obj['name']}", "visual_hint": {"show": True, "type": "Auto"}}

            # ─── POSTGRES schema routing ──────────────────────────────────
            elif tool_keyword == "postgres":
                tool_obj = next((t for t in tools if t["name"].startswith("postgres_")), None)
                if not tool_obj:
                    return {"answer": "❌ PostgreSQL connector not active.", "data": None, "visual_hint": {"show": False}}
                result_text = call_mcp_tool_sync(tool_obj["server"], tool_obj["name"], {})
                return {"answer": f"PostgreSQL result:\n\n{result_text[:2000]}", "data": None, "visual_hint": {"show": True, "type": "Auto"}}

        except Exception as e:
            log_debug(f"Tool call failed: {e}")
            return {"answer": f"❌ Tool Error: {e}", "data": None, "visual_hint": {"show": True, "type": "Auto"}}

    # ======================================================
    # BRANCH C – SQL Query (MySQL or Postgres)
    # ======================================================

    # Pick correct tools based on data_source selection
    target = data_source.lower() if data_source in ["MySQL", "Postgres"] else "postgres"
    schema_tool = next(
        (t for t in tools if "get_schema" in t["name"] and t["name"].lower().startswith(target)), None
    )
    query_tool = next(
        (t for t in tools if "query_database" in t["name"] and t["name"].lower().startswith(target)), None
    )
    # Fallback to any available
    if not schema_tool:
        schema_tool = next((t for t in tools if "get_schema" in t["name"]), None)
    if not query_tool:
        query_tool = next((t for t in tools if "query_database" in t["name"]), None)

    if not query_tool:
        return {"answer": f"❌ No database tools found for **{data_source}**. Check your active connectors.", "data": None, "visual_hint": {"show": False}}

    # ── Fetch schema (cached, with file fallback) ──
    cache_key = f"schema_{query_tool['name']}"
    try:
        if cache_key not in SCHEMA_CACHE:
            log_debug(f"Fetching fresh schema via: {schema_tool['name'] if schema_tool else 'FALLBACK'}")
            schema_text = ""
            if schema_tool:
                schema_text = call_mcp_tool_sync(schema_tool["server"], schema_tool["name"])
            
            # Fallback to local full_schema.json if tool fails or is missing
            if not schema_text or schema_text.startswith("ERROR") or schema_text.startswith("❌"):
                local_path = os.path.join(os.path.dirname(__file__), "full_schema.json")
                if os.path.exists(local_path):
                    log_debug("Live schema fetch failed. Using full_schema.json fallback.")
                    with open(local_path, "r") as f:
                        schema_text = f.read()
                else:
                    return {"answer": f"❌ Could not connect to database and no local schema found: {schema_text}", "data": None, "visual_hint": {"show": False}}
            
            SCHEMA_CACHE[cache_key] = schema_text
        else:
            log_debug("Using cached schema.")
            schema_text = SCHEMA_CACHE[cache_key]
    except Exception as e:
        # Final attempt at local fallback
        local_path = os.path.join(os.path.dirname(__file__), "full_schema.json")
        if os.path.exists(local_path):
            with open(local_path, "r") as f:
                schema_text = f.read()
            SCHEMA_CACHE[cache_key] = schema_text
        else:
            return {"answer": f"❌ Database Error: {e}", "data": None, "visual_hint": {"show": False}}

    # ── Build prioritized compact schema ──
    try:
        schema_obj = json.loads(schema_text)
        if isinstance(schema_obj, dict):
            # Scoring tables by relevance to the question
            q_keywords = [k.lower() for k in question.split() if len(k) > 2]
            scored_tables = []
            
            for tbl, cols in schema_obj.items():
                score = 0
                tbl_low = tbl.lower()
                # Exact match
                if tbl_low in q_keywords: score += 50
                # Partial match
                if any(k in tbl_low for k in q_keywords): score += 20
                
                # Check column names for matches
                if isinstance(cols, list):
                    col_names = [str(c.get("Field") or c.get("column") or c.get("column_name", "")).lower() for c in cols]
                    if any(k in " ".join(col_names) for k in q_keywords):
                        score += 15
                
                scored_tables.append((score, tbl, cols))
            
            # Sort by score (highest first), then build lines
            scored_tables.sort(key=lambda x: x[0], reverse=True)
            
            compact_lines = []
            for _, tbl, cols in scored_tables[:12]: # Take top 12 most relevant tables (optimized for local LLM)
                if isinstance(cols, list):
                    col_names = [c.get("Field") or c.get("column") or c.get("column_name", "") for c in cols]
                    col_names = [c for c in col_names if c]
                    compact_lines.append(f"  {tbl}: {', '.join(col_names[:25])}")
                else:
                    compact_lines.append(f"  {tbl}")
            
            compact_schema = "\n".join(compact_lines)
        else:
            compact_schema = schema_text[:1500]
    except Exception as e:
        log_debug(f"Schema priority sort failed: {e}")
        compact_schema = schema_text[:1200]

    # ── FAST PATH: Keyword-to-SQL bypass (no LLM needed) ──
    # Build a table name lookup from the schema (handles public.tablename format)
    fast_sql = None
    try:
        if isinstance(schema_obj, dict):
            # Build table name map: short_name -> full_qualified_name
            table_map = {}
            for tbl_key in schema_obj.keys():
                # Handle both "tablename" and "schema.tablename" format
                short = tbl_key.split(".")[-1].lower()
                table_map[short] = tbl_key

            q_lower = question.lower()

            # Common fast patterns
            FAST_PATTERNS = [
                # 1. SPECIFIC BI PATTERNS (TOP / TRENDS) - Priority over simple aggregates
                (["top customers", "best customers", "highest sales by customer"],
                 ["v_customer_sales", "customer_sales", "customer"], lambda t: f"SELECT * FROM {t} ORDER BY total_sales DESC LIMIT 10"),
                (["stock quantity", "total stock", "inventory count", "qty on hand"],
                 ["inventory", "stock"], lambda t: f"SELECT SUM(qty_on_hand) AS total_stock FROM {t}"),
                (["top selling", "best selling", "popular products"],
                 ["v_sales_by_month", "sales", "product"], lambda t: f"SELECT * FROM {t} LIMIT 10"),
                (["monthly sales", "sales trend", "sales by month"],
                 ["v_sales_by_month", "sales"], lambda t: f"SELECT * FROM {t} ORDER BY month DESC LIMIT 12"),
                (["total expenses", "expense by region", "expenses summary"],
                 ["expenses", "region"], lambda t: f"SELECT r.name, SUM(e.amount) as total_expenses FROM public.expenses e JOIN public.regions r ON e.region_id = r.region_id GROUP BY r.name"),
                (["total revenue", "revenue by region", "sales summary"],
                 ["v_sales_by_region", "sales", "region"], lambda t: f"SELECT * FROM {t}"),
                
                # 2. LIST PATTERNS (Generic)
                (["show all", "list all", "get all", "all employees", "show employees", "list employees"],
                 ["employee"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all products", "show products", "list products"],
                 ["product"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all customers", "show customers", "list customers"],
                 ["customer"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all users", "show users", "list users"],
                 ["user"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all orders", "show orders", "list orders"],
                 ["sales_order", "order"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all invoices", "show invoices"],
                 ["invoice"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all payroll", "payroll list"],
                 ["payroll"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all departments", "show departments"],
                 ["department"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all projects", "show projects"],
                 ["project"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all suppliers", "show suppliers"],
                 ["supplier"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all sales", "show sales"],
                 ["sales"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all inventory", "show inventory"],
                 ["inventory"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all", "list all", "get all", "all warehouses"],
                 ["warehouse"], lambda t: f"SELECT * FROM {t} LIMIT 100"),

                # 3. AGGREGATE PATTERNS (FALLBACKS)
                (["total sales", "total revenue", "sum of sales"],
                 ["sales"], lambda t: f"SELECT SUM(total_amount) AS total_sales FROM {t}"),
                (["total payroll", "total salary", "payroll total", "salary total"],
                 ["payroll"], lambda t: f"SELECT SUM(net_pay) AS total_payroll FROM {t}"),
                (["how many employees", "count employees", "employee count", "number of employees"],
                 ["employee"], lambda t: f"SELECT COUNT(*) AS employee_count FROM {t}"),
                (["how many customers", "count customers", "customer count"],
                 ["customer"], lambda t: f"SELECT COUNT(*) AS customer_count FROM {t}"),
                (["highest salary", "top salary", "highest paid", "max salary", "top earner"],
                 ["employee", "payroll"], lambda t: f"SELECT * FROM {t} ORDER BY basic_salary DESC LIMIT 10"),
            ]

            for trigger_phrases, table_hints, sql_gen in FAST_PATTERNS:
                if any(phrase in q_lower for phrase in trigger_phrases):
                    # Find matching table
                    for hint in table_hints:
                        match = next((full for short, full in table_map.items() if hint in short), None)
                        if match:
                            fast_sql = sql_gen(match)
                            log_debug(f"FAST PATH SQL: {fast_sql}")
                            break
                if fast_sql:
                    break
    except Exception as e:
        log_debug(f"Fast path error (ignoring): {e}")
        fast_sql = None

    # ── Build MINIMAL SQL prompt with stricter rules ──
    db_type = "PostgreSQL" if "postgres" in query_tool["name"].lower() else "MySQL"
    sql_prompt = f"""You are a senior {db_type} expert. Your ONLY job is to output a single SQL SELECT query.

DATABASE SCHEMA (use ONLY these table names and column names):
{compact_schema}

USER'S QUESTION: {question}

RULES (follow ALL of them):
1. Output ONLY the raw SQL query string — nothing else. No explanation, no markdown, no backtick.
2. The query MUST start with SELECT.
3. NEVER use column names that are not in the schema above.
4. To show people/employees/records: SELECT name, email, department, role, designation, salary or SELECT * FROM the correct table with LIMIT 100.
5. NEVER select columns like 'created_by', 'updated_at', 'version', 'status' for people queries — those are metadata.
6. If the question is about "all" records or "show me", use SELECT * FROM table_name LIMIT 50.
7. If the question uses "highest", "maximum", use ORDER BY column DESC LIMIT 10.
8. If the question uses "count" or "how many", use SELECT COUNT(*) or GROUP BY.

SQL:"""

    sql = ""
    result_text = ""
    error_to_retry = None

    # Try fast path first (no LLM needed)
    if fast_sql:
        try:
            result_text = call_mcp_tool_sync(query_tool["server"], query_tool["name"], {"sql": fast_sql})
            if not result_text.startswith("ERROR"):
                sql = fast_sql
                log_debug(f"Fast path query successful: {fast_sql}")
            else:
                log_debug(f"Fast path failed ({result_text[:100]}), falling back to LLM.")
                fast_sql = None  # fall through to LLM
        except Exception as e:
            log_debug(f"Fast path exception: {e}, falling back to LLM.")
            fast_sql = None

    if not fast_sql:
        for attempt in range(2):
            try:
                if error_to_retry:
                    # ── Correction Phase: Forced SQL-ONLY ──
                    retry_prompt = f"""[CRITICAL: SQL-ONLY MODE] 
Your previous {db_type} query failed: '{error_to_retry}'

Target Schema:
{compact_schema}

Original request: {question}

INSTRUCTION: 
1. Fix the error. 
2. Return ONLY the raw SQL. 
3. DO NOT include any text, explanations, notes, or markdown. 
4. Start directly with SELECT."""
                    raw_sql = _zai_chat(retry_prompt, llm_model, timeout=180)
                else:
                    raw_sql = _zai_chat(sql_prompt, llm_model, timeout=180)

                temp_sql = _clean_sql(raw_sql)
                log_debug(f"Attempt {attempt+1} Cleaned SQL: {temp_sql}")

                if "SELECT" not in temp_sql.upper():
                    log_debug("No SELECT found in response.")
                    error_to_retry = "Response did not contain a valid SELECT statement."
                    continue

                result_text = call_mcp_tool_sync(query_tool["server"], query_tool["name"], {"sql": temp_sql})

                if not result_text.startswith("ERROR"):
                    sql = temp_sql
                    log_debug("Query successful.")
                    break
                else:
                    log_debug(f"SQL Error: {result_text}")
                    error_to_retry = result_text
            except Exception as e:
                log_debug(f"SQL Exception: {e}")
                error_to_retry = str(e)

    if not sql:
        friendly_err = error_to_retry
        if "format string" in str(error_to_retry).lower():
            friendly_err = "The query could not be processed due to a character format issue. This usually means the keyword you searched for was not found in any related tables."
        elif "no such table" in str(error_to_retry).lower() or "not exist" in str(error_to_retry).lower():
            friendly_err = "The requested information or category does not exist in the current database schema."
            
        return {
            "answer": f"❌ **Query Not Found in Database**\n\n{friendly_err}",
            "data": None,
            "visual_hint": {"show": True, "type": "Auto"}
        }

    # ── Parse result ──
    data = None
    try:
        data = json.loads(result_text)
    except Exception:
        pass

    # Default summary in case logic below doesn't catch it
    summary = "No matching records were found."
    if isinstance(data, list) and data:
        summary = f"Found {len(data)} results matching your query."
    else:
        summary = "Query completed, but no matching information was found in the database."

    # ── Final Layer: AI Summarization & Visual Strategy ──
    visual_hint = {"show": False}
    if data and len(data) > 0:
        try:
            # Sample data for context
            sample = data[:10]
            summary_prompt = f"""User Question: {question}
Data Summary ({len(data)} total records): {json.dumps(sample)}

INSTRUCTION: You are a Senior Data Scientist at a top-tier SaaS company.
1. Provide a professional, executive summary (max 3 sentences). 
   - Highlight key trends or data points.
   - Be objective and concise.
2. CRITICAL - Decide on Visualization (CHART):
   - YES if: The data has multiple entries (2+) and involves any numeric comparison, trend, or category breakdown.
   - NO only if: The result is a single number or purely textual without table structure.
   - DEFAULT TO YES: We want to WOW the user with PowerBI-style dashboards.

RESPONSE FORMAT (Strictly match):
SUMMARY: [Your executive answer]
CHART: [YES/NO] | [Type: Bar/Line/Pie] | [X_Column] | [Y_Column] | [Chart Title]
"""
            
            ai_res = _zai_chat(summary_prompt, llm_model, timeout=120)
            log_debug(f"AI Strategy Response: {ai_res}")
            
            # Parse AI Response
            if "SUMMARY:" in ai_res:
                summary_part = ai_res.split("SUMMARY:")[1].split("CHART:")[0].strip()
                if len(summary_part) > 5:
                    summary = summary_part
            
            if "CHART:" in ai_res:
                chart_part = ai_res.split("CHART:")[1].strip()
                result_parts = [p.strip() for p in chart_part.split("|")]
                if len(result_parts) >= 1 and result_parts[0].upper() == "YES":
                    visual_hint = {
                        "show": True,
                        "type": result_parts[1] if len(result_parts) > 1 else "Bar",
                        "x": result_parts[2] if len(result_parts) > 2 else None,
                        "y": result_parts[3] if len(result_parts) > 3 else "Count",
                        "title": result_parts[4] if len(result_parts) > 4 else "Data Analysis"
                    }
        except Exception as e:
            log_debug(f"AI Summary/Visual analysis failed: {e}")
            pass

    if not summary or len(str(summary).strip()) < 2:
        summary = "I found the requested data. See the details below."

    return {"answer": summary, "data": data, "sql": sql, "visual_hint": visual_hint}
