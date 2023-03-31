# Wordcab Transcribe

FastAPI based API for transcribing audio files using [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) and [`pyannote-audio`](https://github.com/pyannote/pyannote-audio)

## Requirements

Python 3.10


## Docker commands

```bash
docker build -t wordcab-transcribe:latest .
docker run -d --name wordcab-transcribe \
    --gpus all \
    --ipc=host \
    --shm-size 64g \
    --ulimit memlock=1 \
    --ulimit stack=67108864 \
    -p 5001:5001 \
    --restart unless-stopped \
    wordcab-transcribe:latest
```

## Test the API

Once the container is running, you can test the API.

The API documentation is available at [http://localhost:5001/docs](http://localhost:5001/docs).

### Using CURL

```bash
curl -X 'POST' \
  'http://localhost:5001/api/v1/youtube' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/audio/file.wav'
```

### Using Python

```python
import requests

filepath = "/path/to/audio/file.wav"  # or mp3
files = {"file": open(filepath, "rb")}
response = requests.post("http://localhost:5001/api/v1/audio", files=files)
print(response.json())
```
