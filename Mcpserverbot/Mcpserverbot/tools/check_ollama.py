import requests
import json

url = "http://localhost:11434/api/tags"
print("Checking Ollama status...")
try:
    r = requests.get(url, timeout=5)
    print(f"Ollama Status: {r.status_code}")
    print(f"Models: {r.json()}")
except Exception as e:
    print(f"Ollama not reachable: {e}")
