# ASR API

## Local commands
```bash
poetry install
poetry shell

# Run the API
poetry run python -m asr_api.main
```

## Docker commands

```bash
docker build -t asr-api:latest .
docker run --gpus all --ipc=host --shm-size 64g --ulimit memlock=1 -d --name asr-api -p 5001:5001 --restart unless-stopped asr-api:latest
```
