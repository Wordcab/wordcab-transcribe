import json

import requests

headers = {"accept": "application/json", "Content-Type": "application/json"}
# params = {"url": "https://youtu.be/JZ696sbfPHs"}
# params = {"url": "https://youtu.be/CNzSJ5SGhqU"}
# params = {"url": "https://youtu.be/vAvcxeXtBz0"}
# params = {"url": "https://youtu.be/pmjrj_TrOEI"}
# params = {"url": "https://youtu.be/SVwLEocqK0E"}  # 2h - 3 speakers
params = {"url": "https://youtu.be/ry9SYnV3svc"}  # eng sample - 2 speakers
# params = {"url": "https://youtu.be/oAhVu3HvWnw"}
# params = {"url": "https://youtu.be/sfQMxf9Dm8I"}
# params = {"url": "https://youtu.be/uLBZf9eS4Y0"}
# params = {"url": "https://youtu.be/JJbtS8CMr80"}  # 4h - multiple speakers

data = {
    "offset_start": None,
    "offset_end": None,
    "num_speakers": -1,  # Leave at -1 to guess the number of speakers
    "diarization": True,  # Longer processing time but speaker segment attribution
    "source_lang": "en",  # optional, default is "en"
    "timestamps": "s",  # optional, default is "s". Can be "s", "ms" or "hms".
    "internal_vad": False,  # optional, default is False
    "word_timestamps": False,  # optional, default is False
}

response = requests.post(
    "http://localhost:5001/api/v1/youtube",
    headers=headers,
    params=params,
    data=json.dumps(data),
)

r_json = response.json()

with open("data/youtube_output.json", "w", encoding="utf-8") as f:
    json.dump(r_json, f, indent=4, ensure_ascii=False)
