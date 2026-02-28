import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

payload = {
    "gemini_api_key": api_key,
    "messages": [
        {"role": "user", "content": "Graph the lift for the Chimney & Latte combo"}
    ]
}

print("SENDING COMBO QUERY...")
resp = requests.post("http://localhost:8000/openclaw", json=payload, stream=True)
print(f"STATUS CODE: {resp.status_code}")

all_output = []
for line in resp.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        all_output.append(decoded)
        print(decoded[:120])

has_error = any('"type": "error"' in o or '"type":"error"' in o for o in all_output)
has_tokens = any('"type": "token"' in o or '"type":"token"' in o for o in all_output)
has_trace = any('"type": "trace"' in o or '"type":"trace"' in o for o in all_output)

print(f"\n=== RESULTS ===")
print(f"Total events: {len(all_output)}")
print(f"Has errors: {has_error}")
print(f"Has tokens: {has_tokens}") 
print(f"Has traces: {has_trace}")
print("PASS" if not has_error else "FAIL")
