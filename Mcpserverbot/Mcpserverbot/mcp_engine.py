# mcp_engine.py
import os
import json
import re
import asyncio
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
import nest_asyncio
import truststore

# Use OS-managed trusted certificates for outbound HTTPS requests.
truststore.inject_into_ssl()

import requests
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
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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
# Gemini LLM call
# ------------------------------------------------
def _gemini_chat(prompt: str, model: str = None, timeout: int = 300) -> str:
    """LLM call using Gemini API only with automatic exponential backoff retries for 429 errors."""
    model = model or LLM_MODEL

    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key not set. Please add GEMINI_API_KEY to your .env file.")

    # Use the supported lower-quota-pressure Flash-Lite model as fallback.
    gemini_model = model if "gemini" in model.lower() else "gemini-2.5-flash-lite"

    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    max_retries = 5
    backoff = 2
    for attempt in range(max_retries):
        try:
            log_debug(f"Calling Gemini API: {gemini_model} (attempt {attempt+1}/{max_retries})")
            resp = requests.post(gemini_url, json=payload, headers=headers, timeout=timeout)
            
            if resp.status_code == 429:
                if attempt == max_retries - 1:
                    resp.raise_for_status()
                sleep_time = backoff ** attempt
                log_debug(f"Received 429 (Too Many Requests). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
                
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            is_429 = False
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:
                    is_429 = True
            
            if is_429 and attempt < max_retries - 1:
                sleep_time = backoff ** attempt
                log_debug(f"Received 429 exception. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            
            log_debug(f"Gemini API call failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                if is_429:
                    raise RuntimeError(
                        "Gemini API rate limit reached (HTTP 429). "
                        "Please retry shortly or check the API quota."
                    )
                raise RuntimeError(f"Gemini API Error: {e}")


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

def _include_order_metric_in_select(sql: str) -> str:
    """Expose ranked metric values so the answer and chart can use them."""
    if not sql or re.search(r"\bSELECT\s+\*", sql, re.IGNORECASE):
        return sql

    select_match = re.search(r"\bSELECT\s+(.*?)\s+FROM\s", sql, re.IGNORECASE | re.DOTALL)
    order_match = re.search(
        r"\bORDER\s+BY\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?)\s*(?:ASC|DESC)?",
        sql,
        re.IGNORECASE,
    )
    if not select_match or not order_match:
        return sql

    selected = select_match.group(1)
    metric_expr = order_match.group(1)
    metric_name = metric_expr.split(".")[-1]
    if re.search(rf"\b{re.escape(metric_name)}\b", selected, re.IGNORECASE):
        return sql

    metric_select = f"{metric_expr} AS {metric_name}" if "." in metric_expr else metric_expr
    return sql[:select_match.start(1)] + f"{selected}, {metric_select}" + sql[select_match.end(1):]


def _numeric_value(value):
    """Parse common database numeric values without treating identifiers as metrics."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[$,\s]", "", str(value))
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _human_label(column: str) -> str:
    return str(column).replace("_", " ").strip()


def _normalize_business_question(question: str) -> str:
    """Normalize common executive BI terms, including selected Indian-language phrases."""
    normalized = str(question or "").strip()
    replacements = {
        "predict next quarter sales": (
            "अगली तिमाही की बिक्री का अनुमान",
            "अगली तिमाही की बिक्री की भविष्यवाणी",
            "அடுத்த காலாண்டு விற்பனையை கணிக்க",
            "తదుపరి త్రైమాసిక అమ్మకాలను అంచనా",
        ),
        "which region is underperforming": (
            "कौन सा क्षेत्र कम प्रदर्शन कर रहा है",
            "எந்த பகுதி குறைவாக செயல்படுகிறது",
            "ఏ ప్రాంతం తక్కువ పనితీరు కనబరుస్తోంది",
        ),
        "why did profit drop this month": (
            "इस महीने लाभ क्यों गिरा",
            "இந்த மாதம் லாபம் ஏன் குறைந்தது",
            "ఈ నెల లాభం ఎందుకు తగ్గింది",
        ),
    }
    lowered = normalized.lower()
    for canonical, variants in replacements.items():
        if any(variant.lower() in lowered for variant in variants):
            return canonical
    return normalized


def _format_result_value(column: str, value: Any) -> str:
    numeric = _numeric_value(value)
    if numeric is None:
        return str(value)
    if any(word in str(column).lower() for word in ("year", "date")) and numeric.is_integer():
        return str(int(numeric))
    financial_words = ("amount", "salary", "budget", "sales", "revenue", "cost", "price", "pay")
    if any(word in str(column).lower() for word in financial_words):
        return f"${numeric:,.2f}"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.2f}"


def _result_columns(data: List[Dict[str, Any]]):
    """Find usable dimensions, metrics, and temporal fields in returned rows."""
    columns = list(data[0].keys()) if data and isinstance(data[0], dict) else []
    identifier_words = ("_id", " id", "code", "phone", "zip", "postal")
    time_words = ("date", "year", "month", "quarter", "week", "day")
    temporal = [c for c in columns if any(word in str(c).lower() for word in time_words)]
    metrics = []
    for column in columns:
        name = str(column).lower().replace("_", " ")
        values = [row.get(column) for row in data if row.get(column) is not None]
        if not values or column in temporal or any(word in name for word in identifier_words):
            continue
        if sum(_numeric_value(value) is not None for value in values) / len(values) >= 0.8:
            metrics.append(column)
    dimensions = [
        c for c in columns
        if c not in metrics and not any(word in str(c).lower().replace("_", " ") for word in identifier_words)
    ]
    return dimensions, metrics, temporal


def _grounded_result_summary(question: str, data: Any) -> str:
    """Build an answer from actual rows so model outages cannot hide successful results."""
    if not isinstance(data, list) or not data:
        return "Query completed, but no matching information was found in the database."
    if not isinstance(data[0], dict):
        return f"Found {len(data)} matching results."

    q_lower = question.lower()
    dimensions, metrics, temporal = _result_columns(data)
    first = data[0]
    metric = metrics[0] if metrics else None
    dimension = next((col for col in dimensions if col not in temporal), None)
    identifier = next(
        (
            col for col in first
            if any(word in str(col).lower().replace("_", " ") for word in (" id", "code", "number", "reference"))
        ),
        None,
    )
    rank_words = ("highest", "largest", "maximum", "max ", "top ", "most", "lowest", "smallest", "minimum", "least")
    is_ranked = any(word in q_lower for word in rank_words) or "underperform" in q_lower
    rank_direction = "lowest" if any(
        word in q_lower for word in ("lowest", "smallest", "minimum", "least", "underperform")
    ) else "highest"

    if temporal and metric:
        values = [_numeric_value(row.get(metric)) for row in data]
        clean_values = [value for value in values if value is not None]
        is_prediction = any(word in q_lower for word in ("predict", "prediction", "forecast", "estimate", "projection"))
        if is_prediction and len(clean_values) >= 2:
            intervals = len(clean_values) - 1
            monthly_change = (clean_values[-1] - clean_values[0]) / intervals
            horizon = 3 if any(word in q_lower for word in ("quarter", "quarterly")) else 1
            projections = [max(0.0, clean_values[-1] + monthly_change * index) for index in range(1, horizon + 1)]
            projected_value = sum(projections) if horizon > 1 else projections[0]
            period_label = "next quarter" if horizon == 3 else "next period"
            return (
                f"Projected sales for the {period_label} are "
                f"{_format_result_value(metric, projected_value)}, using the average change "
                f"across {intervals} recorded interval(s). This is a directional forecast, "
                f"not a committed target."
            )
        if len(data) == 1:
            return (
                f"Only one time period is available: "
                f"{_human_label(metric)} is {_format_result_value(metric, first.get(metric))} "
                f"in {_format_result_value(temporal[0], first.get(temporal[0]))}."
            )
        peak_value = max(_numeric_value(row.get(metric)) or float("-inf") for row in data)
        peak_rows = [row for row in data if _numeric_value(row.get(metric)) == peak_value]
        periods = ", ".join(_format_result_value(temporal[0], row.get(temporal[0])) for row in peak_rows)
        period_phrase = f"in {periods}" if len(peak_rows) == 1 else f"in each of {periods}"
        result = (
            f"Results for {_human_label(metric)} span {len(data)} time periods. "
            f"The highest value is {_format_result_value(metric, peak_value)} "
            f"{period_phrase}."
        )
        if "profit" in q_lower and "sales" in str(metric).lower():
            result = (
                "Profit is not available in the connected data, so a profit-drop cause "
                "cannot be verified. Using sales as an indicator, " + result
            )
        if any(word in q_lower for word in ("why", "drop", "decline", "decrease")) and len(clean_values) >= 2:
            delta = clean_values[-1] - clean_values[-2]
            direction = "decreased" if delta < 0 else "increased" if delta > 0 else "did not change"
            result += (
                f" From the previous to the latest recorded period, it {direction} "
                f"by {_format_result_value(metric, abs(delta))}."
            )
        return result

    if metric and dimension:
        subject = str(first.get(dimension))
        value = _format_result_value(metric, first.get(metric))
        if is_ranked:
            ranked_values = [
                _numeric_value(row.get(metric))
                for row in data
                if _numeric_value(row.get(metric)) is not None
            ]
            if len(ranked_values) > 1 and len(set(ranked_values)) == 1:
                return (
                    f"All {len(data)} returned {_human_label(dimension)} groups are tied at "
                    f"{value} for {_human_label(metric)}; there is no single "
                    f"{rank_direction} performer."
                )
            suffix = f" Returned {len(data)} ranked results." if len(data) > 1 else ""
            if "underperform" in q_lower and str(dimension).lower() != "region":
                return (
                    f"Using {_human_label(dimension)} as the available geographic breakdown, "
                    f"{subject} has the {rank_direction} {_human_label(metric)} at {value}.{suffix}"
                )
            return f"{subject} has the {rank_direction} {_human_label(metric)} at {value}.{suffix}"
        if len(data) > 1 and any(
            word in q_lower for word in
            (" each ", " by ", " per ", "across ", "breakdown", "compare", "comparison", "budget")
        ):
            peak = max(data, key=lambda row: _numeric_value(row.get(metric)) or float("-inf"))
            return (
                f"Returned {_human_label(metric)} for {len(data)} groups. "
                f"{peak.get(dimension)} has the highest value at "
                f"{_format_result_value(metric, peak.get(metric))}."
            )
        if len(data) == 1:
            return f"{subject} has {_human_label(metric)} of {value}."

    if metric and identifier and is_ranked:
        return (
            f"{_human_label(identifier).title()} {first.get(identifier)} has the "
            f"{rank_direction} {_human_label(metric)} at "
            f"{_format_result_value(metric, first.get(metric))}."
        )

    if metric and len(data) == 1:
        return f"The {_human_label(metric)} is {_format_result_value(metric, first.get(metric))}."

    if len(data) == 1:
        facts = ", ".join(
            f"{_human_label(column)}: {value}" for column, value in list(first.items())[:4]
        )
        return f"Found one matching record: {facts}."
    return f"Found {len(data)} results matching your query."


def _grounded_visual_hint(question: str, data: Any) -> Dict[str, Any]:
    """Only request a chart when the question and returned columns support it."""
    hidden = {"show": False}
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return hidden

    q_lower = question.lower()
    dimensions, metrics, temporal = _result_columns(data)
    if not metrics:
        return hidden

    explicit = any(
        word in q_lower for word in
        ("chart", "graph", "plot", "visual", "dashboard", "power bi", "powerbi")
    )
    trend = any(
        word in q_lower for word in
        ("over time", "over the year", "over years", "by year", "yearly",
         "by month", "monthly", "trend", "by quarter", "quarterly", "predict",
         "prediction", "forecast", "drop", "decline")
    )
    compare = any(
        word in q_lower for word in
        ("top ", "bottom ", "ranking", "ranked", "compare", "comparison",
         " by ", " each ", " per ", "across ", "highest", "largest", "lowest",
         "smallest", "most", "least", "breakdown", "share", "underperform")
    )
    metric = metrics[0]
    title = question.strip().rstrip("?.!")
    numeric_values = [
        _numeric_value(row.get(metric))
        for row in data
        if _numeric_value(row.get(metric)) is not None
    ]
    has_metric_variation = len(set(numeric_values)) > 1

    if temporal and (trend or explicit) and len({str(row.get(temporal[0])) for row in data}) >= 2:
        hint = {"show": True, "type": "Line", "x": temporal[0], "y": metric, "title": title}
        if any(word in q_lower for word in ("predict", "prediction", "forecast", "estimate", "projection")):
            hint["forecast_periods"] = 3 if any(word in q_lower for word in ("quarter", "quarterly")) else 1
            hint["forecast_label"] = "next quarter" if hint["forecast_periods"] == 3 else "next period"
        if "profit" in q_lower and "sales" in str(metric).lower():
            hint["measure_note"] = "Sales indicator used because profit is not available."
        return hint

    category = next((column for column in dimensions if column not in temporal), None)
    category_count = len({str(row.get(category)) for row in data}) if category else 0
    if category and category_count >= 2 and has_metric_variation and ((compare and len(data) > 1) or explicit):
        chart_type = "Pie" if any(word in q_lower for word in ("pie", "share", "proportion")) else "Bar"
        hint = {"show": True, "type": chart_type, "x": category, "y": metric, "title": title}
        if any(word in q_lower for word in ("underperform", "lowest", "least", "bottom", "worst")):
            hint["performance_direction"] = "lowest"
            if str(category).lower() != "region":
                hint["dimension_note"] = f"Using {_human_label(category)} as the available geographic breakdown."
        return hint

    return hidden

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

    original_question = question
    question = _normalize_business_question(question)
    intent, tool_keyword = _detect_intent(question, data_source)
    log_debug(f"Intent: {intent}, Tool: {tool_keyword}, Source: {data_source}, Q: {original_question}")
    if original_question != question:
        log_debug(f"Normalized BI question: {question}")

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

                # For text content (PDF, Word, doc) — answer the user's SPECIFIC question using Gemini
                if "read_pdf" in tool_obj["name"] or "read_word" in tool_obj["name"] or "read_document" in tool_obj["name"]:
                    # Truncate document to 12000 chars to stay within Gemini context limits
                    doc_content = result_text[:12000]
                    if len(result_text) > 12000:
                        doc_content += "\n\n[Document truncated for length]"

                    # Check if user is asking a specific question or just wants the content
                    generic_kw = ["summarize", "summarise", "summary", "what is this", "what does this",
                                  "describe", "tell me about", "read", "open", "show content",
                                  "what is in", "content of"]
                    is_generic = any(k in question.lower() for k in generic_kw)

                    if is_generic:
                        # Generic: just summarize the document
                        qa_prompt = f"""You are a document analyst. Summarize the key points of the following document clearly and concisely.

Document Content:
{doc_content}

Provide a professional, well-structured summary."""
                    else:
                        # Specific Q&A: answer the exact question
                        qa_prompt = f"""You are an expert document analyst. Answer the user's question ONLY based on the document content below.

Document: {target_file}
Document Content:
{doc_content}

User's Question: {question}

Rules:
- Answer ONLY from the document. If the answer is not in the document, say: "This information is not found in the document."
- Be concise and specific.
- Quote relevant parts of the document where helpful.
- Do NOT make up information.

Answer:"""

                    try:
                        ai_answer = _gemini_chat(qa_prompt, llm_model, timeout=120)
                        log_debug(f"Document Q&A answer generated for: {question[:80]}")
                        return {
                            "answer": f"📄 **From `{target_file}`:**\n\n{ai_answer}",
                            "data": None,
                            "sql": f"Tool: {tool_obj['name']}({target_file})",
                            "visual_hint": {"show": False}
                        }
                    except Exception as e:
                        log_debug(f"Document Q&A Gemini call failed: {e}")
                        # Fallback: show raw content
                        summary = f"📄 **Content of `{target_file}`:**\n\n{result_text[:3000]}"
                        if len(result_text) > 3000:
                            summary += f"\n\n*... ({len(result_text) - 3000} more characters)*"
                        return {"answer": summary, "data": None, "sql": f"Tool: {tool_obj['name']}({target_file})", "visual_hint": {"show": False}}

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
                        "visual_hint": _grounded_visual_hint(question, data)
                    }

                return {"answer": f"Result:\n\n{result_text[:2000]}", "data": None, "visual_hint": {"show": False}}

            # ─── SHAREPOINT routing ───────────────────────────────────────
            elif tool_keyword == "sharepoint":
                tool_obj = next((t for t in tools if "sharepoint" in t["name"] and "list" in t["name"]), None)
                if not tool_obj:
                    tool_obj = next((t for t in tools if "sharepoint" in t["name"]), None)
                if not tool_obj:
                    return {"answer": "❌ SharePoint connector not active. Check your .env credentials.", "data": None, "visual_hint": {"show": False}}

                args = {}
                props = (tool_obj.get("input_schema") or {}).get("properties", {})
                if "query" in props:
                    args = {"query": question}

                log_debug(f"SharePoint call: {tool_obj['name']} args={args}")
                result_text = call_mcp_tool_sync(tool_obj["server"], tool_obj["name"], args)
                log_debug(f"SharePoint result (300): {result_text[:300]}")

                if result_text.startswith("Error") or "SharePoint Error" in result_text:
                    return {"answer": f"❌ SharePoint Error: {result_text}", "data": None, "visual_hint": {"show": False}}

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

                return {
                    "answer": summary,
                    "data": data,
                    "sql": f"Tool: {tool_obj['name']}",
                    "visual_hint": _grounded_visual_hint(question, data)
                }

            # ─── POSTGRES schema routing ──────────────────────────────────
            elif tool_keyword == "postgres":
                tool_obj = next((t for t in tools if t["name"].startswith("postgres_")), None)
                if not tool_obj:
                    return {"answer": "❌ PostgreSQL connector not active.", "data": None, "visual_hint": {"show": False}}
                result_text = call_mcp_tool_sync(tool_obj["server"], tool_obj["name"], {})
                return {"answer": f"PostgreSQL result:\n\n{result_text[:2000]}", "data": None, "visual_hint": {"show": False}}

        except Exception as e:
            log_debug(f"Tool call failed: {e}")
            return {"answer": f"❌ Tool Error: {e}", "data": None, "visual_hint": {"show": False}}

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

            def available_columns(table):
                table_columns = schema_obj.get(table, [])
                return {
                    str(col.get("Field") or col.get("column") or col.get("column_name", "")).lower()
                    for col in table_columns if isinstance(col, dict)
                }

            def first_available(table, candidates, fallback):
                names = available_columns(table)
                return next((column for column in candidates if column in names), fallback)

            def named_table(*candidates):
                return next((table_map.get(candidate) for candidate in candidates if table_map.get(candidate)), None)

            def sql_literal(value):
                return str(value).replace("'", "''")

            def employee_join_trend_sql(table):
                date_column = first_available(table, ("join_date", "hire_date", "joining_date"), "hire_date")
                year_expr = (
                    f"EXTRACT(YEAR FROM {date_column})"
                    if "postgres" in query_tool["name"].lower()
                    else f"YEAR({date_column})"
                )
                return (
                    f"SELECT {year_expr} AS hire_year, COUNT(*) AS employees_joined "
                    f"FROM {table} WHERE {date_column} IS NOT NULL "
                    f"GROUP BY {year_expr} ORDER BY hire_year"
                )

            def highest_project_budget_sql(table):
                name = first_available(table, ("project_name", "name", "title"), "project_name")
                budget = first_available(table, ("budget", "project_budget", "amount"), "budget")
                wants_visual_context = any(
                    word in q_lower
                    for word in ("chart", "graph", "visual", "compare", "comparison", "dashboard")
                )
                limit = "" if wants_visual_context else " LIMIT 1"
                return f"SELECT {name}, {budget} FROM {table} ORDER BY {budget} DESC{limit}"

            def project_end_date_sql(table):
                name = first_available(table, ("project_name", "name", "title"), "project_name")
                end_date = first_available(table, ("end_date", "project_end_date", "completion_date"), "end_date")
                date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
                if not date_match:
                    return None
                return (
                    f"SELECT {name}, {end_date} FROM {table} "
                    f"WHERE {end_date} = '{sql_literal(date_match.group(0))}'"
                )

            def highest_employee_salary_sql(table):
                name = first_available(table, ("emp_name", "full_name", "employee_name", "name"), "emp_name")
                salary = first_available(table, ("salary", "basic_salary", "net_pay"), "salary")
                return f"SELECT {name}, {salary} FROM {table} ORDER BY {salary} DESC LIMIT 10"

            def employees_by_department_sql(table):
                dept_table = table_map.get("departments") or table_map.get("department")
                employee_dept = first_available(table, ("dept_id", "department_id"), "dept_id")
                if dept_table:
                    dept_id = first_available(dept_table, ("dept_id", "department_id"), "dept_id")
                    dept_name = first_available(dept_table, ("dept_name", "department_name", "name"), "dept_name")
                    return (
                        f"SELECT d.{dept_name} AS department, COUNT(*) AS employee_count "
                        f"FROM {table} e JOIN {dept_table} d ON e.{employee_dept} = d.{dept_id} "
                        f"GROUP BY d.{dept_name} ORDER BY employee_count DESC"
                    )
                return (
                    f"SELECT {employee_dept} AS department, COUNT(*) AS employee_count "
                    f"FROM {table} GROUP BY {employee_dept} ORDER BY employee_count DESC"
                )

            def salary_by_department_sql(table):
                dept_table = table_map.get("departments") or table_map.get("department")
                employee_dept = first_available(table, ("dept_id", "department_id"), "dept_id")
                salary = first_available(table, ("salary", "basic_salary", "net_pay"), "salary")
                aggregate = "AVG" if any(word in q_lower for word in ("average", "avg", "mean")) else "SUM"
                alias = "average_salary" if aggregate == "AVG" else "total_salary"
                if dept_table:
                    dept_id = first_available(dept_table, ("dept_id", "department_id"), "dept_id")
                    dept_name = first_available(dept_table, ("dept_name", "department_name", "name"), "dept_name")
                    return (
                        f"SELECT d.{dept_name} AS department, {aggregate}(e.{salary}) AS {alias} "
                        f"FROM {table} e JOIN {dept_table} d ON e.{employee_dept} = d.{dept_id} "
                        f"GROUP BY d.{dept_name} ORDER BY {alias} DESC"
                    )
                return (
                    f"SELECT {employee_dept} AS department, {aggregate}({salary}) AS {alias} "
                    f"FROM {table} GROUP BY {employee_dept} ORDER BY {alias} DESC"
                )

            def total_orders_amount_sql(table):
                amount = first_available(table, ("order_amount", "total_amount", "amount"), "order_amount")
                return f"SELECT SUM({amount}) AS total_sales FROM {table}"

            def highest_order_sql(table):
                identifier = first_available(table, ("order_id", "sales_order_id", "id", "order_code"), "order_id")
                amount = first_available(table, ("order_amount", "total_amount", "amount"), "order_amount")
                return f"SELECT {identifier}, {amount} FROM {table} ORDER BY {amount} DESC LIMIT 1"

            def highest_order_amount_sql(table):
                amount = first_available(table, ("order_amount", "total_amount", "amount"), "order_amount")
                return f"SELECT MAX({amount}) AS highest_order_amount FROM {table}"

            def orders_over_time_sql(table):
                amount = first_available(table, ("order_amount", "total_amount", "amount"), "order_amount")
                date_column = first_available(table, ("order_date", "date", "created_at"), "order_date")
                prediction_request = any(word in q_lower for word in ("predict", "prediction", "forecast", "estimate", "projection"))
                if prediction_request or any(word in q_lower for word in ("month", "monthly", "this month")):
                    period = (
                        f"TO_CHAR({date_column}, 'YYYY-MM')"
                        if "postgres" in query_tool["name"].lower()
                        else f"DATE_FORMAT({date_column}, '%%Y-%%m')"
                    )
                    alias = "order_month"
                else:
                    period = (
                        f"EXTRACT(YEAR FROM {date_column})"
                        if "postgres" in query_tool["name"].lower()
                        else f"YEAR({date_column})"
                    )
                    alias = "order_year"
                return (
                    f"SELECT {period} AS {alias}, SUM({amount}) AS total_sales "
                    f"FROM {table} WHERE {date_column} IS NOT NULL "
                    f"GROUP BY {period} ORDER BY {alias}"
                )

            def sales_by_geography_sql(table):
                order_table = named_table("orders", "order", "sales_orders", "sales_order")
                customer_id = first_available(table, ("customer_id",), "customer_id")
                geography = first_available(table, ("region", "city", "location", "country"), "city")
                geography_alias = "region" if geography == "region" else geography
                order_customer_id = first_available(order_table or "orders", ("customer_id",), "customer_id")
                amount = first_available(order_table or "orders", ("order_amount", "total_amount", "amount"), "order_amount")
                return (
                    f"SELECT c.{geography} AS {geography_alias}, SUM(o.{amount}) AS total_sales "
                    f"FROM {table} c JOIN {order_table} o ON c.{customer_id} = o.{order_customer_id} "
                    f"GROUP BY c.{geography} ORDER BY total_sales ASC"
                )

            def customers_by_city_sql(table):
                city = first_available(table, ("city", "location", "region"), "city")
                return f"SELECT {city}, COUNT(*) AS customer_count FROM {table} GROUP BY {city} ORDER BY customer_count DESC"

            def count_rows_sql(table):
                label = table.split(".")[-1].rstrip("s")
                return f"SELECT COUNT(*) AS {label}_count FROM {table}"

            def employee_salary_aggregate_sql(table):
                salary = first_available(table, ("salary", "basic_salary", "net_pay"), "salary")
                if any(word in q_lower for word in ("average", "avg", "mean")):
                    return f"SELECT AVG({salary}) AS average_salary FROM {table}"
                if any(word in q_lower for word in ("lowest", "minimum", "min ")):
                    return f"SELECT MIN({salary}) AS lowest_salary FROM {table}"
                return f"SELECT SUM({salary}) AS total_salary FROM {table}"

            def order_amount_aggregate_sql(table):
                amount = first_available(table, ("order_amount", "total_amount", "amount"), "order_amount")
                if any(word in q_lower for word in ("average", "avg", "mean")):
                    return f"SELECT AVG({amount}) AS average_order_amount FROM {table}"
                return f"SELECT SUM({amount}) AS total_order_amount FROM {table}"

            def orders_by_customer_sql(table):
                order_table = named_table("orders", "order", "sales_orders", "sales_order")
                customer_id = first_available(table, ("customer_id",), "customer_id")
                name = first_available(table, ("customer_name", "name"), "customer_name")
                order_customer_id = first_available(order_table or "orders", ("customer_id",), "customer_id")
                if any(word in q_lower for word in ("how many", "count", "number of")):
                    return (
                        f"SELECT c.{name} AS customer, COUNT(*) AS order_count "
                        f"FROM {table} c JOIN {order_table} o ON c.{customer_id} = o.{order_customer_id} "
                        f"GROUP BY c.{name} ORDER BY order_count DESC"
                    )
                amount = first_available(order_table or "orders", ("order_amount", "total_amount", "amount"), "order_amount")
                return (
                    f"SELECT c.{name} AS customer, SUM(o.{amount}) AS total_sales "
                    f"FROM {table} c JOIN {order_table} o ON c.{customer_id} = o.{order_customer_id} "
                    f"GROUP BY c.{name} ORDER BY total_sales DESC"
                )

            def largest_customer_orders_sql(table):
                order_table = named_table("orders", "order", "sales_orders", "sales_order")
                customer_id = first_available(table, ("customer_id",), "customer_id")
                name = first_available(table, ("customer_name", "name"), "customer_name")
                order_customer_id = first_available(order_table or "orders", ("customer_id",), "customer_id")
                amount = first_available(order_table or "orders", ("order_amount", "total_amount", "amount"), "order_amount")
                return (
                    f"SELECT c.{name} AS customer, o.{amount} AS order_amount "
                    f"FROM {table} c JOIN {order_table} o ON c.{customer_id} = o.{order_customer_id} "
                    f"ORDER BY o.{amount} DESC LIMIT 10"
                )

            def top_customers_sql(table):
                order_table = named_table("orders", "order", "sales_orders", "sales_order")
                customer_id = first_available(table, ("customer_id",), "customer_id")
                name = first_available(table, ("customer_name", "name"), "customer_name")
                order_customer_id = first_available(order_table or "orders", ("customer_id",), "customer_id")
                amount = first_available(order_table or "orders", ("order_amount", "total_amount", "amount"), "order_amount")
                return (
                    f"SELECT c.{name} AS customer, SUM(o.{amount}) AS total_sales "
                    f"FROM {table} c JOIN {order_table} o ON c.{customer_id} = o.{order_customer_id} "
                    f"GROUP BY c.{name} ORDER BY total_sales DESC LIMIT 10"
                )

            def project_budget_list_sql(table):
                name = first_available(table, ("project_name", "name", "title"), "project_name")
                budget = first_available(table, ("budget", "project_budget", "amount"), "budget")
                return f"SELECT {name}, {budget} FROM {table} ORDER BY {budget} DESC"

            def department_location_sql(table):
                name = first_available(table, ("dept_name", "department_name", "name"), "name")
                location = first_available(table, ("location", "city", "office_location"), "location")
                location_match = re.search(r"\blocated\s+in\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
                if not location_match:
                    return None
                target_location = sql_literal(location_match.group(1).strip())
                return f"SELECT {name}, {location} FROM {table} WHERE {location} = '{target_location}'"

            # Common fast patterns
            FAST_PATTERNS = [
                # 1. SPECIFIC BI PATTERNS (TOP / TRENDS) - Priority over simple aggregates
                (["employees joined over the years", "employees joined over years",
                  "employees joined by year", "hires by year", "hiring trend",
                  "employee joining trend"],
                 ["employee"], employee_join_trend_sql),
                (["which project has the highest budget", "project with the highest budget",
                  "highest project budget", "largest project budget", "highest budget project"],
                 ["project"], highest_project_budget_sql),
                (["which project ends on", "project ending on", "projects ending on",
                  "project ends on"],
                 ["project"], project_end_date_sql),
                (["which employees have the highest salaries", "employees with the highest salaries",
                  "highest salaries", "highest salary", "top salary", "highest paid", "top earners"],
                 ["employee"], highest_employee_salary_sql),
                (["employees work in each department", "employees in each department",
                  "employee count by department", "employees by department",
                  "headcount by department", "headcount per department",
                  "department has the most employees", "department has most employees",
                  "most employees by department"],
                 ["employee"], employees_by_department_sql),
                (["average salary by department", "avg salary by department",
                  "mean salary by department", "total salary by department",
                  "salary by department", "department payroll"],
                 ["employee"], salary_by_department_sql),
                (["which order has the highest amount", "which order has highest amount",
                  "order with the highest amount", "which is the largest order", "highest value order"],
                 ["orders", "order", "sales_orders", "sales_order"], highest_order_sql),
                (["highest order amount", "maximum order amount", "max order amount", "largest order amount"],
                 ["orders", "order", "sales_orders", "sales_order"], highest_order_amount_sql),
                (["which customer placed the largest orders", "customer placed the largest orders",
                  "largest orders by customer", "largest order by customer"],
                 ["customers", "customer"], largest_customer_orders_sql),
                (["top customers", "best customers", "highest sales by customer"],
                 ["customers", "customer"], top_customers_sql),
                (["orders by customer", "orders per customer", "orders in each customer",
                  "orders did each customer", "sales by customer", "revenue by customer",
                  "customer sales breakdown"],
                 ["customers", "customer"], orders_by_customer_sql),
                (["sales over the years", "sales over years", "sales by year",
                  "revenue by year", "order amount by year", "yearly sales",
                  "monthly sales", "sales trend", "sales by month", "predict next quarter sales",
                  "forecast sales", "sales forecast", "predict sales", "why did profit drop",
                  "why did sales drop", "why did revenue drop", "profit drop this month"],
                 ["orders", "order"], orders_over_time_sql),
                (["which region is underperforming", "underperforming region",
                  "lowest performing region", "worst performing region",
                  "which city is underperforming", "lowest performing city"],
                 ["customers", "customer"], sales_by_geography_sql),
                (["customers by city", "customer count by city", "customers in each city",
                  "how many customers in each city", "city has the most customers",
                  "city has most customers"],
                 ["customers", "customer"], customers_by_city_sql),
                (["project budgets", "budget by project", "budgets by project",
                  "compare project budgets", "project budget comparison"],
                 ["project"], project_budget_list_sql),
                (["department is located in", "department located in", "departments located in"],
                 ["department"], department_location_sql),
                (["stock quantity", "total stock", "inventory count", "qty on hand"],
                 ["inventory", "stock"], lambda t: f"SELECT SUM(qty_on_hand) AS total_stock FROM {t}"),
                (["top selling", "best selling", "popular products"],
                 ["v_sales_by_month", "sales", "product"], lambda t: f"SELECT * FROM {t} LIMIT 10"),
                (["total expenses", "expense by region", "expenses summary"],
                 ["expenses", "region"], lambda t: f"SELECT r.name, SUM(e.amount) as total_expenses FROM public.expenses e JOIN public.regions r ON e.region_id = r.region_id GROUP BY r.name"),
                (["total revenue", "revenue by region", "sales summary"],
                 ["v_sales_by_region", "sales", "region"], lambda t: f"SELECT * FROM {t}"),
                
                # 2. LIST PATTERNS (Generic)
                (["show all employees", "list all employees", "get all employees", "all employees", "show employees", "list employees"],
                 ["employee"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all products", "list all products", "get all products", "all products", "show products", "list products"],
                 ["product"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all customers", "list all customers", "get all customers", "all customers", "show customers", "list customers"],
                 ["customer"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all users", "list all users", "get all users", "all users", "show users", "list users"],
                 ["user"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all orders", "list all orders", "get all orders", "all orders", "show orders", "list orders"],
                 ["sales_order", "order"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all invoices", "list all invoices", "get all invoices", "all invoices", "show invoices"],
                 ["invoice"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all payroll", "list all payroll", "get all payroll", "all payroll", "payroll list"],
                 ["payroll"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all departments", "list all departments", "get all departments", "all departments", "show departments"],
                 ["department"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all projects", "list all projects", "get all projects", "all projects", "show projects"],
                 ["project"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all suppliers", "list all suppliers", "get all suppliers", "all suppliers", "show suppliers"],
                 ["supplier"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all sales", "list all sales", "get all sales", "all sales", "show sales"],
                 ["sales"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all inventory", "list all inventory", "get all inventory", "all inventory", "show inventory"],
                 ["inventory"], lambda t: f"SELECT * FROM {t} LIMIT 100"),
                (["show all warehouses", "list all warehouses", "get all warehouses", "all warehouses"],
                 ["warehouse"], lambda t: f"SELECT * FROM {t} LIMIT 100"),

                # 3. AGGREGATE PATTERNS (FALLBACKS)
                (["total sales", "total revenue", "sum of sales", "total order amount"],
                 ["orders", "order"], total_orders_amount_sql),
                (["average order amount", "avg order amount", "mean order amount"],
                 ["orders", "order"], order_amount_aggregate_sql),
                (["average employee salary", "average salary", "avg salary",
                  "mean salary", "total employee salary", "salary total"],
                 ["employee"], employee_salary_aggregate_sql),
                (["total payroll", "total salary", "payroll total", "salary total"],
                 ["payroll"], lambda t: f"SELECT SUM(net_pay) AS total_payroll FROM {t}"),
                (["how many employees", "count employees", "employee count", "number of employees"],
                 ["employee"], lambda t: f"SELECT COUNT(*) AS employee_count FROM {t}"),
                (["how many customers", "count customers", "customer count"],
                 ["customer"], lambda t: f"SELECT COUNT(*) AS customer_count FROM {t}"),
                (["how many orders", "count orders", "order count", "number of orders"],
                 ["orders", "order"], count_rows_sql),
                (["how many projects", "count projects", "project count", "number of projects"],
                 ["projects", "project"], count_rows_sql),
                (["how many departments", "count departments", "department count", "number of departments"],
                 ["departments", "department"], count_rows_sql),
            ]

            import re
            for trigger_phrases, table_hints, sql_gen in FAST_PATTERNS:
                matched = False
                for phrase in trigger_phrases:
                    # Stricter word boundary matching
                    if re.search(r'\b' + re.escape(phrase) + r'\b', q_lower):
                        matched = True
                        break
                
                if matched:
                    log_debug(f"MATCHED FAST PATTERN: {trigger_phrases}")
                    match = None
                    # Phase 1: Try exact match of any hint to a table key
                    for hint in table_hints:
                        if hint in table_map:
                            match = table_map[hint]
                            log_debug(f"Exact match found for hint '{hint}': {match}")
                            break
                        # Also check common plural/singular forms
                        for variant in [f"{hint}s", hint[:-1] if hint.endswith("s") else hint]:
                            if variant in table_map:
                                match = table_map[variant]
                                log_debug(f"Variant match found for hint '{hint}' -> '{variant}': {match}")
                                break
                        if match: break
                    
                    # Phase 2: Fallback to substring match if no exact match found
                    if not match:
                        for hint in table_hints:
                            match = next((full for short, full in table_map.items() if hint in short), None)
                            if match: 
                                log_debug(f"Substring match found for hint '{hint}': {match}")
                                break
                    
                    if match:
                        fast_sql = sql_gen(match)
                        if fast_sql:
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
                    raw_sql = _gemini_chat(retry_prompt, llm_model, timeout=180)
                else:
                    raw_sql = _gemini_chat(sql_prompt, llm_model, timeout=180)

                temp_sql = _clean_sql(raw_sql)
                temp_sql = _include_order_metric_in_select(temp_sql)
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
                if "gemini api" in error_to_retry.lower():
                    break

    if not sql:
        friendly_err = error_to_retry
        error_lower = str(error_to_retry).lower()
        if "gemini api rate limit" in error_lower or "gemini api error: 429" in error_lower:
            return {
                "answer": (
                    "AI service is temporarily rate limited. "
                    "The database query could not be prepared right now. "
                    "Please retry shortly or check the Gemini API quota."
                ),
                "data": None,
                "visual_hint": {"show": False}
            }
        if "gemini api error" in error_lower:
            return {
                "answer": (
                    "AI service is temporarily unavailable, so the database query "
                    "could not be prepared. Please retry shortly."
                ),
                "data": None,
                "visual_hint": {"show": False}
            }
        if "format string" in error_lower:
            friendly_err = "The query could not be processed due to a character format issue. This usually means the keyword you searched for was not found in any related tables."
        elif "no such table" in error_lower or "not exist" in error_lower:
            friendly_err = "The requested information or category does not exist in the current database schema."
            
        return {
            "answer": f"❌ **Query Not Found in Database**\n\n{friendly_err}",
            "data": None,
            "visual_hint": {"show": False}
        }

    # ── Parse result ──
    data = None
    try:
        data = json.loads(result_text)
    except Exception:
        pass

    # Keep completed query results useful even if an LLM call is unavailable.
    # These summaries and chart axes are constrained to the columns returned.
    summary = _grounded_result_summary(question, data)
    visual_hint = _grounded_visual_hint(question, data)
    log_debug(f"Grounded answer generated. Visual: {visual_hint}")

    if not summary or len(str(summary).strip()) < 2:
        summary = "I found the requested data. See the details below."

    return {"answer": summary, "data": data, "sql": sql, "visual_hint": visual_hint}
