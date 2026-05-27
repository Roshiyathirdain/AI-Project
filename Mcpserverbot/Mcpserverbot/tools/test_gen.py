import requests
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("ZAI_API_KEY")
url = "https://api.z.ai/api/paas/v4/chat/completions"

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "glm-4.7",
    "messages": [{"role": "user", "content": "Say hello!"}],
    "stream": False
}

print(f"Testing Z.ai API with glm-4.7...")
try:
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print(f"Response: {r.json()['choices'][0]['message']['content']}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Connection Failed: {e}")
