import requests
import time
import json

url = "http://127.0.0.1:8081/v1/chat/completions"
headers = {"Content-Type": "application/json"}
payload = {
    "messages": [{"role": "user", "content": "Hello. Are you working?"}],
    "max_tokens": 50,
    "temperature": 0.7
}

print(f"Testing LLM direct connection at {url}...")
start = time.time()
try:
    response = requests.post(url, json=payload, headers=headers, timeout=300)
    end = time.time()
    print(f"Status: {response.status_code}")
    print(f"Time: {end - start:.2f} seconds")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
