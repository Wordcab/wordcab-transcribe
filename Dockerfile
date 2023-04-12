FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
    
COPY requirements.txt /requirements.txt
RUN python3.10 -m pip install -r requirements.txt
RUN python3.10 -m pip install --upgrade torch==1.13.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

COPY . /app
WORKDIR /app

CMD ["uvicorn", "--reload", "--host=0.0.0.0", "--port=5001", "wordcab_transcribe.main:app"]
