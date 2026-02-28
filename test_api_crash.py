import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

payload = {
    "gemini_api_key": api_key,
    "messages": [
        {"role": "user", "content": "Graph the lift for the Chimney & Latte combo"},
        {"role": "user", "content": "Graph the lift for the Chimney & Latte combo"}
    ]
}

print("SENDING POST REQUEST...")
resp = requests.post("http://localhost:8000/openclaw", json=payload, stream=True)
print("STATUS CODE:", resp.status_code)
for line in resp.iter_lines():
    if line:
        print(line.decode('utf-8'))
