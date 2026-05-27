# automated_backup_to_sharepoint.py
import os
import pandas as pd
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import requests
import msal

load_dotenv()

# Config
PG_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "dbname": os.getenv("POSTGRES_DB")
}

SP_CONFIG = {
    "client_id": os.getenv("SHAREPOINT_CLIENT_ID"),
    "client_secret": os.getenv("SHAREPOINT_CLIENT_SECRET"),
    "tenant_id": os.getenv("SHAREPOINT_TENANT_ID"),
    "site_id": os.getenv("SHAREPOINT_SITE_ID"),
    "drive_id": os.getenv("SHAREPOINT_DRIVE_ID")
}

def export_and_upload():
    print("🚀 Starting Automated Backup: Postgres -> SharePoint")
    
    if "your_" in (SP_CONFIG["client_id"] or ""):
        print("❌ Error: Please update your SharePoint credentials in .env first.")
        return

    # 1. Export Postgres Table to CSV
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        table_name = os.getenv("BACKUP_TABLE", "employees")
        # Ensure table name is safe
        if not table_name.isalnum() and "_" not in table_name:
             raise ValueError(f"Invalid table name for backup: {table_name}")
             
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        filename = f"Backup_Postgres_{table_name}_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Exported {table_name} to {filename}")
        conn.close()
    except Exception as e:
        print(f"❌ Database Export Error: {e}")
        return

    # 2. Upload to SharePoint
    try:
        authority = f"https://login.microsoftonline.com/{SP_CONFIG['tenant_id']}"
        app = msal.ConfidentialClientApplication(SP_CONFIG['client_id'], authority=authority, client_credential=SP_CONFIG['client_secret'])
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        
        if "access_token" not in result:
            print(f"❌ Auth Error: {result.get('error_description')}")
            return

        token = result["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        url = f"https://graph.microsoft.com/v1.0/drives/{SP_CONFIG['drive_id']}/root:/Backups/{filename}:/content"
        
        with open(filename, "rb") as f:
            resp = requests.put(url, headers=headers, data=f)
        
        resp.raise_for_status()
        print(f"✅ Successfully uploaded {filename} to SharePoint /Backups/ folder!")
        
        # Cleanup
        os.remove(filename)

    except Exception as e:
        print(f"❌ SharePoint Upload Error: {e}")

if __name__ == "__main__":
    export_and_upload()
