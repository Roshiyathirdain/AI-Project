# mcp_sharepoint_server.py
import asyncio
import os
import json
import requests
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server.stdio import stdio_server
from dotenv import load_dotenv

load_dotenv()

# SharePoint / Microsoft Graph Configuration
CLIENT_ID = os.getenv("SHAREPOINT_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHAREPOINT_CLIENT_SECRET")
TENANT_ID = os.getenv("SHAREPOINT_TENANT_ID")
SITE_ID = os.getenv("SHAREPOINT_SITE_ID")
DRIVE_ID = os.getenv("SHAREPOINT_DRIVE_ID")

server = Server("sharepoint-connector")

def get_access_token():
    import msal
    if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID]):
        raise ValueError("Missing SharePoint credentials in .env")
    
    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    app = msal.ConfidentialClientApplication(CLIENT_ID, authority=authority, client_credential=CLIENT_SECRET)
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"Could not acquire token: {result.get('error_description')}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_sharepoint",
            description="Search for files in SharePoint using Microsoft Graph API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for."}
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="list_site_files",
            description="List files in the configured SharePoint site/drive.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional subfolder path."}
                }
            },
        ),
        types.Tool(
            name="upload_file",
            description="Upload a local file to SharePoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "local_path": {"type": "string", "description": "Path to the local file."},
                    "sharepoint_path": {"type": "string", "description": "Target path in SharePoint."}
                },
                "required": ["local_path", "sharepoint_path"]
            },
        ),
        types.Tool(
            name="get_file_content",
            description="Download and read the content of a file from SharePoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "The unique ID of the file in SharePoint."},
                    "path": {"type": "string", "description": "Optional path to the file (if ID not known)."}
                }
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    try:
        token = get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        if name == "search_sharepoint":
            query = (arguments or {}).get("query", "")
            # Graph Search API
            url = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drive/root/search(q='{query}')"
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            return [types.TextContent(type="text", text=json.dumps(resp.json().get("value", []), indent=2))]

        elif name == "list_site_files":
            path = (arguments or {}).get("path", "root")
            if path == "root":
                url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root/children"
            else:
                url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{path}:/children"
            
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            return [types.TextContent(type="text", text=json.dumps(resp.json().get("value", []), indent=2))]

        elif name == "upload_file":
            local_path = (arguments or {}).get("local_path", "")
            sp_path = (arguments or {}).get("sharepoint_path", "").strip("/")
            
            if not os.path.exists(local_path):
                return [types.TextContent(type="text", text=f"Error: Local file '{local_path}' not found")]

            filename = os.path.basename(local_path)
            # Graph Upload API - using the format for uploading to a path
            if sp_path:
                url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{sp_path}/{filename}:/content"
            else:
                url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{filename}:/content"
            
            with open(local_path, "rb") as f:
                resp = requests.put(url, headers=headers, data=f)
            resp.raise_for_status()
            return [types.TextContent(type="text", text=f"Successfully uploaded {filename} to SharePoint!")]

        elif name == "get_file_content":
            item_id = (arguments or {}).get("item_id", "")
            path = (arguments or {}).get("path", "")
            
            if item_id:
                url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/items/{item_id}/content"
            elif path:
                url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{path.strip('/')}:/content"
            else:
                return [types.TextContent(type="text", text="Error: item_id or path required")]

            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            # For simplicity, returning text. For binary, might need base64 or other handling.
            return [types.TextContent(type="text", text=resp.text)]

        return [types.TextContent(type="text", text="Unknown tool")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"SharePoint Error: {str(e)}")]

async def main():
    async with stdio_server() as (read, write):
        await server.run(
            read, write,
            InitializationOptions(
                server_name="sharepoint-connector",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
