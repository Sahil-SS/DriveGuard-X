import requests
import numpy as np

# Generate a valid 1-second dummy audio window (16,000 samples at 16kHz)
dummy_samples = np.random.normal(0, 0.1, 16000).tolist()

response = requests.post(
    "http://127.0.0.1:8000/api/v1/road/predict",
    json={"samples": dummy_samples}
)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())