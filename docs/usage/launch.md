You can run the API either locally or using Docker. The API is built using FastAPI and Uvicorn. 

We are using environment variables to configure and customize the API runtime, find the list of available environment
variables in the [ENV](usage/env) section.

## Run locally

```bash
hatch run runtime:launch
```

!!! tip
    This is the recommended way to run the API in development as it's easier to debug and hot reload when code changes.

## Run using Docker

Build the image.

```bash
docker build -t wordcab-transcribe:latest .
```

Run the container.

```bash
docker run -d --name wordcab-transcribe \
    --gpus all \
    --shm-size 1g \
    --restart unless-stopped \
    -p 5001:5001 \
    -v ~/.cache:/root/.cache \
    wordcab-transcribe:latest
```

You can mount a volume to the container to load local whisper models.

If you mount a volume, you need to update the `WHISPER_MODEL` environment variable in the `.env` file.

```bash
docker run -d --name wordcab-transcribe \
    --gpus all \
    --shm-size 1g \
    --restart unless-stopped \
    -p 5001:5001 \
    -v ~/.cache:/root/.cache \
    -v /path/to/whisper/models:/app/whisper/models \
    wordcab-transcribe:latest
```

You can simply enter the container using the following command:

```bash
docker exec -it wordcab-transcribe /bin/bash
```

Check the logs to know when the API is ready.

```bash
docker logs -f wordcab-transcribe
```

This is useful to check everything is working as expected.

## Run behind a reverse proxy

We have included a `nginx.conf` file to help you get started.

```bash
# Create a docker network and connect the api container to it
docker network create transcribe
docker network connect transcribe wordcab-transcribe

# Replace /absolute/path/to/nginx.conf with the absolute path to the nginx.conf
# file on your machine (e.g. /home/user/wordcab-transcribe/nginx.conf).
docker run -d \
    --name nginx \
    --network transcribe \
    -p 80:80 \
    -v /absolute/path/to/nginx.conf:/etc/nginx/nginx.conf:ro \
    nginx

# Check everything is working as expected
docker logs nginx
```

Your API should now be exposed on port 80.

## Run `only_transcription`

You can run the API in transcription only mode by setting the `asr_type` in the `.env` file to `only_transcription`.

## Run `only_diarization`

You can run the API in diarization only mode by setting the `asr_type` in the `.env` file to `only_diarization`.

## Use remote servers

You can use remote servers for transcription and diarization by setting the `asr_type` in the `.env` file to `async` and
adding URLs to the `transcribe_server_urls` and `diarize_server_urls` environment variables.

If an async server is already running, you can simply add remote servers to the list of URLs by using the endpoints,
check [Management endpoints](usage/asr/#management-endpoints) for more details.
