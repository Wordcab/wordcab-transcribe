# Copyright (c) 2023, The Wordcab team. All rights reserved.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

COPY requirements.txt /requirements.txt
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    cmake \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install -r requirements.txt \
    && python3.10 -m pip install numpy --pre torch torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

COPY . /app
WORKDIR /app

ENTRYPOINT /docker-entrypoint.sh $0 $@
CMD ["uvicorn", "--reload", "--host=0.0.0.0", "--port=5001", "asr_api.main:app"]
