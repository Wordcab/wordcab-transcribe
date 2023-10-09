<h1 align="center">Wordcab Transcribe</h1>
<p align="center"><em>💬 Speech recognition is now a commodity</em></p>

<div align="center">
	<a  href="https://github.com/Wordcab/wordcab-transcribe/releases" target="_blank">
		<img src="https://img.shields.io/badge/release-v0.5.1-pink" />
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

!!! important
    If you want to see the great performance of Wordcab-Transcribe compared to all the available ASR tools on the market, please check out our benchmark project: [Rate that ASR](https://github.com/Wordcab/rtasr#readme).

## Key features

- ⚡ Fast: The faster-whisper library and CTranslate2 make audio processing incredibly fast compared to other implementations.
- 🐳 Easy to deploy: You can deploy the project on your workstation or in the cloud using Docker.
- 🔥 Batch requests: You can transcribe multiple audio files at once because batch requests are implemented in the API.
- 💸 Cost-effective: As an open-source solution, you won't have to pay for costly ASR platforms.
- 🫶 Easy-to-use API: With just a few lines of code, you can use the API to transcribe audio files or even YouTube videos.
- 🤗 Open-source (commercial-use under [WTLv0.1 license](https://github.com/Wordcab/wordcab-transcribe/blob/main/LICENSE), please reach out to `info@wordcab.com`): Our project is open-source and based on open-source libraries, allowing you to customize and extend it as needed until you don't sell this as a hosted service.

## Requirements

### Local development

- Linux _(tested on Ubuntu Server 20.04/22.04)_
- Python >=3.8, <3.12
- [Hatch](https://hatch.pypa.io/latest/)
- [FFmpeg](https://ffmpeg.org/download.html)

### Deployment

- [Docker](https://docs.docker.com/engine/install/ubuntu/) _(optional for deployment)_
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) _(optional for deployment)_

## How to start?

You need to clone the repository and install the dependencies:

```bash
git clone https://github.com/Wordcab/wordcab-transcribe.git

cd wordcab-transcribe

hatch env create
```

Then, you can start using the API. Head to the [Usage](usage/launch) section to learn more.
