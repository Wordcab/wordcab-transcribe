import requests

file = "sample_1.mp3"
with open(file, "rb") as f:
    files = {"file": (file, f)}
    response = requests.post("http://localhost:5001/api/v1/audio?num_speakers=2", files=files)
    print(response.json())

# url = "https://youtu.be/M3ujv8xdK2w"
# response = requests.post(f"http://localhost:5001/api/v1/youtube?url={url}&num_speakers=1")
# print(response.json())
