import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

payload = {
    "gemini_api_key": api_key,
    "messages": [
        {"role": "user", "content": "How many employees do I need in Jnah for today?"}
    ]
}

print("SENDING POST REQUEST...")
resp = requests.post("http://localhost:8000/openclaw", json=payload, stream=True)
print(f"STATUS CODE: {resp.status_code}")

all_output = []
for line in resp.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        all_output.append(decoded)
        print(decoded)

print(f"\n\n=== TOTAL SSE EVENTS: {len(all_output)} ===")
has_error = any('"type": "error"' in o or '"type":"error"' in o for o in all_output)
has_tokens = any('"type": "token"' in o or '"type":"token"' in o for o in all_output)
has_trace = any('"type": "trace"' in o or '"type":"trace"' in o for o in all_output)

print(f"Has error events: {has_error}")
print(f"Has token events: {has_tokens}")
print(f"Has trace events: {has_trace}")
if has_error:
    print("\n*** FAILURE: Error events detected ***")
else:
    print("\n*** SUCCESS: No errors! ***")
