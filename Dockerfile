FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    software-properties-common \
    python3-pip

COPY requirements.txt /requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --upgrade torch==2.0.0+cu118 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118

COPY . /app
WORKDIR /app

CMD ["uvicorn", "--reload", "--host=0.0.0.0", "--port=5001", "wordcab_transcribe.main:app"]
