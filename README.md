# Wordcab Transcribe 💬

FastAPI based API for transcribing audio files using [`faster-whisper`](https://github.com/guillaumekln/faster-whisper)
and [Auto-Tuning-Spectral-Clustering](https://arxiv.org/pdf/2003.02405.pdf) for diarization 
(based on this [GitHub implementation](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)).

More details on this project on this [blog post](https://wordcab.github.io/wordcab-posts/blog/2023/03/31/wordcab-transcribe/).

## Key features

- ⚡ Fast: The faster-whisper library and CTranslate2 make audio processing incredibly fast compared to other implementations.
- 🐳 Easy to deploy: You can deploy the project on your workstation or in the cloud using Docker.
- 🔥 Batch requests: You can transcribe multiple audio files at once because batch requests are implemented in the API.
- 💸 Cost-effective: As an open-source solution, you won't have to pay for costly ASR platforms.
- 🫶 Easy-to-use API: With just a few lines of code, you can use the API to transcribe audio files or even YouTube videos.
- 🤗 Open-source (commercial-use under [WTLv0.1 license](https://github.com/Wordcab/wordcab-transcribe/blob/main/LICENSE), please reach out to `info@wordcab.com`): Our project is open-source and based on open-source libraries, allowing you to customize and extend it as needed until you don't sell this as a hosted service.

## Requirements

- Linux _(tested on Ubuntu Server 22.04)_
- Python 3.9
- Docker
- NVIDIA GPU + NVIDIA Container Toolkit

To learn more about the prerequisites to run the API, check out the [Prerequisites](https://wordcab.github.io/wordcab-posts/blog/2023/03/31/wordcab-transcribe/#prerequisites) section of the blog post.

## Docker commands

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
  "alignment": True,  # Longer processing time but better timestamps
  "diarization": True,  # Longer processing time but speaker segment attribution
  "dual_channel": False,  # Only for stereo audio files with one speaker per channel
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
  "alignment": True,  # Longer processing time but better timestamps
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

## Local testing

Before launching the API, be sure to install torch and torchaudio on your machine.

```bash
pip install --upgrade torch==1.13.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Then, you can launch the API using the following command.

```bash
poetry run uvicorn wordcab_transcribe.main:app --reload
```

## 🚀 Contributing

### Getting started

1. Ensure you have the following tools :

- [poetry](https://python-poetry.org/)
- [nox](https://nox.thea.codes/) and [nox-poetry](https://nox-poetry.readthedocs.io/)

2. Clone the repo

```bash
git clone
cd wordcab-transcribe
```

3. Install dependencies and start coding

```bash
poetry shell
poetry install --no-cache

# install pre-commit hooks
nox --session=pre-commit -- install

# open your IDE
code .
```

4. Run tests

```bash
# run all tests
nox

# run a specific session
nox --session=tests  # run tests
nox --session=pre-commit  # run pre-commit hooks

# run a specific test
nox --session=tests -- -k test_something
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

![GitHub Workflow](https://user-images.githubusercontent.com/6351798/48032310-63842400-e114-11e8-8db0-06dc0504dcb5.png)
