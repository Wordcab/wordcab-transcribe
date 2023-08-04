import json
import requests


filepath = "data/short_one_speaker.mp3"

data = {
    "alignment": False,  # Longer processing time but better timestamps
    "diarization": True,  # Longer processing time but speaker segment attribution
    "dual_channel": False,  # Only for stereo audio files with one speaker per channel
    "source_lang": "ru",  # optional, default is "en"
    "timestamps": "s",  # optional, default is "s". Can be "s", "ms" or "hms".
    "word_timestamps": False,  # optional, default is False
}

with open(filepath, "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:5001/api/v1/audio",
        files=files,
        data=data,
    )

r_json = response.json()

filename = filepath.split(".")[0]
with open(f"{filename}.json", "w", encoding="utf-8") as f:
    json.dump(r_json, f, indent=4, ensure_ascii=False)
