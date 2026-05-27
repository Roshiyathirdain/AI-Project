import requests
import json
try:
    resp = requests.get("http://localhost:11434/api/tags")
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
