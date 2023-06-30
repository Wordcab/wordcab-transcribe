import glob
import json
import requests
import time
from tqdm import tqdm

audio_files = glob.glob("data/audio_files/*")

results = {}
for audio in tqdm(audio_files):
    filename = audio.split("/")[-1]
    results[filename] = {}

    # Batch process
    try:
        with open(audio, "rb") as f:
            files = {"file": f}
            response = requests.post(
                "http://localhost:5001/api/v1/audio",
                files=files,
                data={
                "dual_channel": False,
                "source_lang": "en",
                "timestamps": "s",
                "diarization": False,
                "alignment": True,
                "use_batch": True,
                "word_timestamps": True
                },
            )
            if response.status_code == 200:
                data = response.json()
                utterances = data["utterances"]
                start_timestamp = utterances[0]["start"]
                results[filename]["batch"] = start_timestamp
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"Error: {e}")
        results[filename]["batch"] = None

    # Original process
    try:
        with open(audio, "rb") as f:
            files = {"file": f}
            response = requests.post(
                "http://localhost:5001/api/v1/audio",
                files=files,
                data={
                "dual_channel": False,
                "source_lang": "en",
                "timestamps": "s",
                "diarization": False,
                "alignment": True,
                "use_batch": False,
                "word_timestamps": True
                },
            )
            if response.status_code == 200:
                data = response.json()
                utterances = data["utterances"]
                start_timestamp = utterances[0]["start"]
                results[filename]["original"] = start_timestamp
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"Error: {e}")
        results[filename]["original"] = None
        
with open("data/audio_files/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
