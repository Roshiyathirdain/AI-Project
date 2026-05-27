# discover_sharepoint.py
import os
import requests
import msal
import json
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SHAREPOINT_CLIENT_ID")
CLIENT_SECRET = os.getenv("SHAREPOINT_CLIENT_SECRET")
TENANT_ID = os.getenv("SHAREPOINT_TENANT_ID")

def discover():
    print("🔍 Discovering SharePoint Site and Drive Details...")
    
    if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID]) or "your_" in (CLIENT_ID or ""):
        print("❌ Error: Please fill in CLIENT_ID, CLIENT_SECRET, and TENANT_ID in your .env file first.")
        return

    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    app = msal.ConfidentialClientApplication(CLIENT_ID, authority=authority, client_credential=CLIENT_SECRET)
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    
    if "access_token" not in result:
        print(f"❌ Could not acquire token: {result.get('error_description')}")
        return

    token = result["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Search for Site
    site_query = input("Enter your SharePoint Site Name (e.g., CodeBackup): ")
    url = f"https://graph.microsoft.com/v1.0/sites?search={site_query}"
    resp = requests.get(url, headers=headers)
    sites = resp.json().get("value", [])
    
    if not sites:
        print(f"❌ No sites found matching '{site_query}'")
        return

    print("\nFound Sites:")
    for i, s in enumerate(sites):
        print(f"{i+1}. {s['displayName']} (ID: {s['id']}) - {s['webUrl']}")
    
    choice = int(input("\nSelect site number: ")) - 1
    selected_site = sites[choice]
    site_id = selected_site["id"]

    # 2. Get Drive for that site
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    resp = requests.get(url, headers=headers)
    drives = resp.json().get("value", [])

    if not drives:
        print("❌ No drives (Document Libraries) found for this site.")
        return

    print("\nFound Document Libraries (Drives):")
    for i, d in enumerate(drives):
        print(f"{i+1}. {d['name']} (ID: {d['id']})")

    choice = int(input("\nSelect drive number: ")) - 1
    selected_drive = drives[choice]
    drive_id = selected_drive["id"]

    print("\n✅ DISCOVERY COMPLETE!")
    print(f"SHAREPOINT_SITE_ID={site_id}")
    print(f"SHAREPOINT_DRIVE_ID={drive_id}")
    
    update = input("\nUpdate .env automatically? (y/n): ")
    if update.lower() == 'y':
        with open(".env", "r") as f:
            lines = f.readlines()
        
        with open(".env", "w") as f:
            for line in lines:
                if line.startswith("SHAREPOINT_SITE_ID="):
                    f.write(f"SHAREPOINT_SITE_ID={site_id}\n")
                elif line.startswith("SHAREPOINT_DRIVE_ID="):
                    f.write(f"SHAREPOINT_DRIVE_ID={drive_id}\n")
                else:
                    f.write(line)
        print("📝 .env file updated!")

if __name__ == "__main__":
    discover()
