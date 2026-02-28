import requests
import json

resp = requests.post("http://localhost:8000/tools/predict_demand", json={"branch_name": "Conut Jnah", "month": 1, "year": 2026})
data = resp.json()
print(f"Status: {resp.status_code}")
print(f"Model type: {data.get('model_type')}")
print(f"Warning: {data.get('warning', 'NONE')}")
print(f"Volume: {data.get('predicted_volume')}")
print(f"CI: {data.get('confidence_interval')}")
print(f"Full response: {json.dumps(data, indent=2)}")
