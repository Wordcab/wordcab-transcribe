FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

ENV PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    libsndfile1 \
    software-properties-common \
    ffmpeg \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    gnupg2 \
    wget \
    git \
    curl \
    gdb \
    openmpi-bin \
    libopenmpi-dev \
    libffi-dev \
    libssl-dev \
    python3-pip \
    libbz2-dev \
    python3-dev \
    liblzma-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    libtiff-tools=4.3.0-6ubuntu0.8 \
    libtiff5=4.3.0-6ubuntu0.8 \
    libgnutls30=3.7.3-4ubuntu1.4 \
    openssl=3.0.2-0ubuntu1.15 \
    libpam-modules=1.4.0-11ubuntu2.4 \
    libpam-modules-bin=1.4.0-11ubuntu2.4 \
    libpam-runtime=1.4.0-11ubuntu2.4 \
    libpam0g=1.4.0-11ubuntu2.4 \
    login=1:4.8.1-2ubuntu2.2 \
    passwd=1:4.8.1-2ubuntu2.2 \
    uidmap=1:4.8.1-2ubuntu2.2 \
    binutils=2.38-4ubuntu2.6

RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar -xvf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations --with-ssl && \
    make && make install && \
    cd .. && rm -r Python-3.10.12 && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip

RUN export CUDNN_PATH=$(python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'${CUDNN_PATH} >> ~/.bashrc

ENV MPI4PY_VERSION="3.1.5"
ENV RELEASE_URL="https://github.com/mpi4py/mpi4py/archive/refs/tags/${MPI4PY_VERSION}.tar.gz"

RUN curl -L ${RELEASE_URL} | tar -zx -C /tmp \
    && sed -i 's/>= 40\\.9\\.0/>= 40.9.0, < 69/g' /tmp/mpi4py-${MPI4PY_VERSION}/pyproject.toml \
    && pip install /tmp/mpi4py-${MPI4PY_VERSION} \
    && rm -rf /tmp/mpi4py*

RUN python -m pip install pip --upgrade \
    && python -m pip install hatch

WORKDIR /app
COPY . .

CMD ["hatch", "run", "runtime:launch"]
