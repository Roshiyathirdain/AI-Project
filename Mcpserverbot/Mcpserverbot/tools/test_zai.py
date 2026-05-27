
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZAI_API_KEY")

endpoints = [
    "https://api.z.ai/api/paas/v4/chat/completions",
    "https://open.bigmodel.cn/api/paas/v4/chat/completions"
]

models = [
    "glm-4-flash",
    "glm-4",
    "glm-4-air",
    "glm-4-plus",
    "glm-4v",
    "glm-3-turbo"
]

def test_api():
    if not api_key:
        print("No API Key found")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False
    }

    for url in endpoints:
        print(f"\n--- Testing Endpoint: {url} ---")
        for model in models:
            payload["model"] = model
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    print(f"[SUCCESS] Model: {model}")
                    return # Stop at first success
                else:
                    print(f"[FAILED] Model: {model} | Status: {resp.status_code} | Body: {resp.text}")
            except Exception as e:
                print(f"[ERROR] Model: {model} | {e}")

if __name__ == "__main__":
    test_api()
