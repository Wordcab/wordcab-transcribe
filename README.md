<h1 align="center">Wordcab Transcribe</h1>
<p align="center"><em>💬 Speech recognition is now a commodity</em></p>

<div align="center">
	<a  href="https://github.com/Wordcab/wordcab-transcribe/releases" target="_blank">
		<img src="https://img.shields.io/badge/release-v0.5.2-pink" />
  </a>
	<a  href="https://github.com/Wordcab/wordcab-transcribe/actions?workflow=Quality Checks" target="_blank">
		<img src="https://github.com/Wordcab/wordcab-transcribe/workflows/Quality Checks/badge.svg" />
	</a>
	<a  href="https://github.com/pypa/hatch" target="_blank">
		<img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" />
	</a>
</div>


---

FastAPI based API for transcribing audio files using [`faster-whisper`](https://github.com/guillaumekln/faster-whisper)
and [Auto-Tuning-Spectral-Clustering](https://arxiv.org/pdf/2003.02405.pdf) for diarization
(based on this [GitHub implementation](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)).

> [!IMPORTANT]\
> If you want to see the great performance of Wordcab-Transcribe compared to all the available ASR tools on the market, please check out our benchmark project: [Rate that ASR](https://github.com/Wordcab/rtasr#readme).

## Key features

- ⚡ Fast: The faster-whisper library and CTranslate2 make audio processing incredibly fast compared to other implementations.
- 🐳 Easy to deploy: You can deploy the project on your workstation or in the cloud using Docker.
- 🔥 Batch requests: You can transcribe multiple audio files at once because batch requests are implemented in the API.
- 💸 Cost-effective: As an open-source solution, you won't have to pay for costly ASR platforms.
- 🫶 Easy-to-use API: With just a few lines of code, you can use the API to transcribe audio files or even YouTube videos.
- 🤗 MIT License: You can use the project for commercial purposes without any restrictions.
 
## Requirements

### Local development

- Linux _(tested on Ubuntu Server 20.04/22.04)_
- Python >=3.8, <3.12
- [Hatch](https://hatch.pypa.io/latest/)
- [FFmpeg](https://ffmpeg.org/download.html)

#### Run the API locally 🚀

```bash
hatch run runtime:launch
```

### Deployment

- [Docker](https://docs.docker.com/engine/install/ubuntu/) _(optional for deployment)_
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) _(optional for deployment)_

#### Run the API using Docker

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

This is useful to check everything is working as expected.

### Run the API behind a reverse proxy

You can run the API behind a reverse proxy like Nginx. We have included a `nginx.conf` file to help you get started.

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

---

<details open>
<summary>⏱️ Profile the API</summary>

You can profile the process executions using `py-spy` as a profiler.

```bash
# Launch the container with the cap-add=SYS_PTRACE option
docker run -d --name wordcab-transcribe \
    --gpus all \
    --shm-size 1g \
    --restart unless-stopped \
    --cap-add=SYS_PTRACE \
    -p 5001:5001 \
    -v ~/.cache:/root/.cache \
    wordcab-transcribe:latest

# Enter the container
docker exec -it wordcab-transcribe /bin/bash

# Install py-spy
pip install py-spy

# Find the PID of the process to profile
top  # 28 for example

# Run the profiler
py-spy record --pid 28 --format speedscope -o profile.speedscope.json

# Launch any task on the API to generate some profiling data

# Exit the container and copy the generated file to your local machine
exit
docker cp wordcab-transcribe:/app/profile.speedscope.json profile.speedscope.json

# Go to https://www.speedscope.app/ and upload the file to visualize the profile
```

</details>

---

## Test the API

Once the container is running, you can test the API.

The API documentation is available at [http://localhost:5001/docs](http://localhost:5001/docs).

- Audio file:

```python
import json
import requests

filepath = "/path/to/audio/file.wav"  # or any other convertible format by ffmpeg
data = {
  "num_speakers": -1,  # # Leave at -1 to guess the number of speakers
  "diarization": True,  # Longer processing time but speaker segment attribution
  "multi_channel": False,  # Only for stereo audio files with one speaker per channel
  "source_lang": "en",  # optional, default is "en"
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
```

- YouTube video:

```python
import json
import requests

headers = {"accept": "application/json", "Content-Type": "application/json"}
params = {"url": "https://youtu.be/JZ696sbfPHs"}
data = {
  "diarization": True,  # Longer processing time but speaker segment attribution
  "source_lang": "en",  # optional, default is "en"
  "timestamps": "s",  # optional, default is "s". Can be "s", "ms" or "hms".
  "word_timestamps": False,  # optional, default is False
}

response = requests.post(
  "http://localhost:5001/api/v1/youtube",
  headers=headers,
  params=params,
  data=json.dumps(data),
)

r_json = response.json()

with open("youtube_video_output.json", "w", encoding="utf-8") as f:
  json.dump(r_json, f, indent=4, ensure_ascii=False)
```

## Running Local Models

You can link a local folder path to use a custom model. If you do so, you should mount the folder in the
docker run command as a volume, or include the model directory in your Dockerfile to bake it into the image.

**Note** that for the default `tensorrt-llm` whisper engine, the simplest way to get a converted model is to use
`hatch` to start the server locally once. Specify the `WHISPER_MODEL` and `ALIGN_MODEL` in `.env`, then run
`hatch run runtime:launch` in your terminal. This will download and convert these models.

You'll then find the converted models in `cloned_wordcab_transcribe_repo/src/wordcab_transcribe/whisper_models`.
Then in your Dockerfile, copy the converted models to the `/app/src/wordcab_transcribe/whisper_models` directory.

Example Dockerfile line for `WHISPER_MODEL`: `COPY cloned_wordcab_transcribe_repo/src/wordcab_transcribe/whisper_models/large-v3 /app/src/wordcab_transcribe/whisper_models/large-v3`
Example Dockerfile line for `ALIGN_MODEL`: `COPY cloned_wordcab_transcribe_repo/src/wordcab_transcribe/whisper_models/tiny /app/src/wordcab_transcribe/whisper_models/tiny`

## 🚀 Contributing

### Getting started

1. Ensure you have the `Hatch` installed (with pipx for example):

- [hatch](https://hatch.pypa.io/latest/install/)

2. Clone the repo

```bash
git clone
cd wordcab-transcribe
```

3. Install dependencies and start coding

```bash
hatch env create
```

4. Run tests

```bash
# Quality checks without modifying the code
hatch run quality:check

# Quality checks and auto-formatting
hatch run quality:format

# Run tests with coverage
hatch run tests:run
```

### Working workflow

1. Create an issue for the feature or bug you want to work on.
2. Create a branch using the left panel on GitHub.
3. `git fetch`and `git checkout` the branch.
4. Make changes and commit.
5. Push the branch to GitHub.
6. Create a pull request and ask for review.
7. Merge the pull request when it's approved and CI passes.
8. Delete the branch.
9. Update your local repo with `git fetch` and `git pull`.
