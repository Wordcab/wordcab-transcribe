FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    software-properties-common \
    python3-pip

ENV PYTHONUNBUFFERED=1
RUN python3 -m pip install pip --upgrade \
    && python3 -m pip install hatch

WORKDIR /app
COPY . .

CMD ["hatch", "run", "runtime:launch"]
