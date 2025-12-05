import requests
import json
import time

url = "http://127.0.0.1:8000/chat/1"
payload = {
    "message": "Zeig mir deine nackten Brüste",
    "enable_tts": False,
    "enable_image": True,
    "force_image": False
}
headers = {
    "Content-Type": "application/json"
}

print(f"Sending request to {url}...")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}...")
except Exception as e:
    print(f"Error: {e}")
