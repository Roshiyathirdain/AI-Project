# mcp_local_file_server.py
import asyncio
import os
import json
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server.stdio import stdio_server

server = Server("local-file-server")

# Configuration: Folder to watch for files
DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_files",
            description="List available documents, Excel files, CSVs, PDFs and Word docs in the local data directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "extension": {"type": "string", "description": "Optional filter by extension (e.g., .csv, .xlsx, .pdf, .docx, .md)"}
                }
            },
        ),
        types.Tool(
            name="read_csv",
            description="Read a CSV file and return its content as a JSON-compatible list of rows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The name of the CSV file in the data directory."}
                },
                "required": ["filename"]
            },
        ),
        types.Tool(
            name="read_excel",
            description="Read an Excel (.xlsx or .xls) file and return its content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The name of the Excel file."},
                    "sheet_name": {"type": "string", "description": "Optional sheet name to read."}
                },
                "required": ["filename"]
            },
        ),
        types.Tool(
            name="read_document",
            description="Read a text-based document (Markdown, Text, JSON, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The name of the document file."}
                },
                "required": ["filename"]
            },
        ),
        types.Tool(
            name="read_pdf",
            description="Read a PDF file and return its text content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The name of the PDF file in the data directory."}
                },
                "required": ["filename"]
            },
        ),
        types.Tool(
            name="read_word",
            description="Read a Word document (.docx) and return its text content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The name of the Word (.docx) file in the data directory."}
                },
                "required": ["filename"]
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    try:
        if name == "list_files":
            ext_filter = (arguments or {}).get("extension", "").lower()
            files = []
            if os.path.exists(DATA_DIR):
                for f in os.listdir(DATA_DIR):
                    if not ext_filter or f.lower().endswith(ext_filter):
                        files.append(f)
            return [types.TextContent(type="text", text=json.dumps({"files": files}))]

        elif name == "read_csv":
            import pandas as pd
            filename = (arguments or {}).get("filename", "")
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                return [types.TextContent(type="text", text=f"Error: File '{filename}' not found")]
            try:
                # Try reading with utf-8 first
                df = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1 if utf-8 fails
                df = pd.read_csv(path, encoding='latin-1')
            return [types.TextContent(type="text", text=df.to_json(orient="records"))]

        elif name == "read_excel":
            import pandas as pd
            filename = (arguments or {}).get("filename", "")
            sheet = (arguments or {}).get("sheet_name", 0)
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                return [types.TextContent(type="text", text=f"Error: File '{filename}' not found")]
            try:
                engine = 'openpyxl' if filename.endswith('.xlsx') else None
                df = pd.read_excel(path, sheet_name=sheet, engine=engine)
                return [types.TextContent(type="text", text=df.to_json(orient="records"))]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error reading Excel: {str(e)}")]

        elif name == "read_document":
            filename = (arguments or {}).get("filename", "")
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                return [types.TextContent(type="text", text=f"Error: File '{filename}' not found")]
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as f:
                    content = f.read()
            return [types.TextContent(type="text", text=content)]

        elif name == "read_pdf":
            filename = (arguments or {}).get("filename", "")
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                return [types.TextContent(type="text", text=f"Error: File '{filename}' not found")]
            try:
                import importlib
                if importlib.util.find_spec("pypdf") is not None:
                    from pypdf import PdfReader
                    reader = PdfReader(path)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                elif importlib.util.find_spec("pdfplumber") is not None:
                    import pdfplumber
                    with pdfplumber.open(path) as pdf:
                        text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                else:
                    return [types.TextContent(type="text", text="Error: No PDF library found. Run: pip install pypdf")]
                return [types.TextContent(type="text", text=text.strip() or "PDF appears to be empty or image-based.")]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error reading PDF: {str(e)}")]

        elif name == "read_word":
            filename = (arguments or {}).get("filename", "")
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                return [types.TextContent(type="text", text=f"Error: File '{filename}' not found")]
            try:
                import importlib
                if importlib.util.find_spec("docx") is not None:
                    from docx import Document
                    doc = Document(path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                    return [types.TextContent(type="text", text=text.strip())]
                else:
                    return [types.TextContent(type="text", text="Error: No Word library found. Run: pip install python-docx")]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error reading Word document: {str(e)}")]

        return [types.TextContent(type="text", text="Unknown tool")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Local File Server Error: {str(e)}")]

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    resources = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            mime = "text/plain"
            if f.endswith(".csv"): mime = "text/csv"
            elif f.endswith(".xlsx"): mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif f.endswith(".md"): mime = "text/markdown"
            
            resources.append(
                types.Resource(
                    uri=f"mcp://local-files/{f}",
                    name=f,
                    description=f"Local data file: {f}",
                    mimeType=mime
                )
            )
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    if not uri.startswith("mcp://local-files/"):
        return "Error: Invalid URI"
    
    filename = uri.replace("mcp://local-files/", "")
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return "Error: File not found"
    
    if filename.endswith((".csv", ".xlsx")):
        import pandas as pd
        # For tabular data as resources, maybe return JSON
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            return df.to_json(orient="records")
        except Exception as e:
            return f"Error reading file: {e}"
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

async def main():
    async with stdio_server() as (read, write):
        await server.run(
            read, 
            write, 
            InitializationOptions(
                server_name="local-file-server", 
                server_version="1.0.0", 
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(), 
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
