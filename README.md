# Wordcab Transcribe 💬

FastAPI based API for transcribing audio files using [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) and [`NVIDIA NeMo`](https://github.com/NVIDIA/NeMo)

More details on this project on this [blog post](https://wordcab.github.io/wordcab-posts/blog/2023/03/31/wordcab-transcribe/).

## Key features

- 🤗 Open-source: Our project is open-source and based on open-source libraries, allowing you to customize and extend it as needed.
- ⚡ Fast: The faster-whisper library and CTranslate2 make audio processing incredibly fast compared to other implementations.
- 🐳 Easy to deploy: You can deploy the project on your workstation or in the cloud using Docker.
- 🔥 Batch requests: You can transcribe multiple audio files at once because batch requests are implemented in the API.
- 💸 Cost-effective: As an open-source solution, you won't have to pay for costly ASR platforms.
- 🫶 Easy-to-use API: With just a few lines of code, you can use the API to transcribe audio files or even YouTube videos.

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
    wordcab-transcribe:latest
```

## Test the API

Once the container is running, you can test the API.

The API documentation is available at [http://localhost:5001/docs](http://localhost:5001/docs).

- Audio file:

```python
import requests

headers = {"accept": "application/json"}
data = {
  "source_lang": "en",  # optional, default is "en"
  "timestamps": "s",  # optional, default is "s". Can be "s", "ms" or "hms".
}

filepath = "tests/sample_1.mp3"  # or any other audio file. Prefer wav files.
with open(filepath, "rb") as f:
  files = {"file": f}
  response = requests.post(
    "http://localhost:5001/api/v1/audio",
    headers=headers,
    files=files,
    data=data,
  )
print(response.json())
```

- YouTube video:

```python
import json
import requests

headers = {"accept": "application/json", "Content-Type": "application/json"}
params = {"url": "https://youtu.be/JZ696sbfPHs"}
data = {
  "source_lang": "en",  # optional, default is "en"
  "timestamps": "s",  # optional, default is "s". Can be "s", "ms" or "hms".
}

response = requests.post(
  "http://localhost:5001/api/v1/youtube",
  headers=headers,
  params=params,
  data=json.dumps(data),
)
print(response.json())
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

1. Clone the repo

```bash
git clone
cd wordcab-ask
```

2. Install dependencies and start coding

```bash
poetry install
poetry shell

# install pre-commit hooks
nox --session=pre-commit -- install

# open your IDE
code .
```

3. Run tests

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
