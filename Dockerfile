FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    gdb \
    curl \
    wget \
    cmake \
    ccache \
    ffmpeg \
    gnupg2 \
    openmpi-bin \
    libsndfile1 \
    libopenmpi-dev \
    build-essential \
    ca-certificates \
    software-properties-common \
    python3.10 \
    python3-pip

RUN export CUDNN_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'${CUDNN_PATH} >> ~/.bashrc

ENV MPI4PY_VERSION="3.1.5"
ENV RELEASE_URL="https://github.com/mpi4py/mpi4py/archive/refs/tags/${MPI4PY_VERSION}.tar.gz"

RUN curl -L ${RELEASE_URL} | tar -zx -C /tmp \
    && sed -i 's/>= 40\\.9\\.0/>= 40.9.0, < 69/g' /tmp/mpi4py-${MPI4PY_VERSION}/pyproject.toml \
    && pip3 install /tmp/mpi4py-${MPI4PY_VERSION} \
    && rm -rf /tmp/mpi4py*

ENV PYTHONUNBUFFERED=1
RUN python3 -m pip install pip --upgrade \
    && python3 -m pip install hatch

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["hatch", "run", "runtime:launch"]
