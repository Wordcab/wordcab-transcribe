import json
import requests

headers = {"accept": "application/json", "Content-Type": "application/json"}
params = {"url": "https://github.com/Wordcab/wordcab-python/raw/main/tests/sample_1.mp3"}


data = {
    "offset_start": None,
    "offset_end": None,
    "num_speakers": -1,  # Leave at -1 to guess the number of speakers
    "diarization": True,  # Longer processing time but speaker segment attribution
    "source_lang": "en",  # optional, default is "en"
    "timestamps": "s",  # optional, default is "s". Can be "s", "ms" or "hms".
    "internal_vad": False,  # optional, default is False
    "vocab": ["Martha's Flowers", "Thomas", "Randal"],  # optional, default is None
    "word_timestamps": False,  # optional, default is False
}

response = requests.post(
    "http://localhost:5001/api/v1/audio-url",
    headers=headers,
    params=params,
    data=json.dumps(data),
)

r_json = response.json()

with open("data/audio_url_output.json", "w", encoding="utf-8") as f:
    json.dump(r_json, f, indent=4, ensure_ascii=False)
