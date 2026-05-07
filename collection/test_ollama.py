import requests
import time

t0 = time.time()
r = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Rewrite casually: The cat sat on the mat.",
        "stream": False,
        "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 512},
    },
    timeout=120,
)
print(f"Status: {r.status_code}")
print(f"Time:   {time.time() - t0:.2f}s")
print(f"Response: {r.json().get('response', '')[:200]}")