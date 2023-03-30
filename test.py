import requests

with open("resto.wav", "rb") as f:
    files = {"file": ("sample_1.wav", f)}
    response = requests.post("http://localhost:5001/api/v1/audio?num_speakers=2", files=files)
    print(response.json())
