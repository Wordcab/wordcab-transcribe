# Wordcab Transcribe

FastAPI based API for transcribing audio files using [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) and [`pyannote-audio`](https://github.com/pyannote/pyannote-audio)

## Requirements

Python 3.10


## Docker commands

```bash
docker build -t asr-api:latest .
docker run --gpus all --ipc=host --shm-size 64g --ulimit memlock=1 -d --name asr-api -p 5001:5001 --restart unless-stopped asr-api:latest
```
