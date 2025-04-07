ARG CUDA_IMAGE="12.4.0-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

# Install depencencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

RUN git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
RUN cd llama-cpp-python && CMAKE_ARGS="-DGGML_CUDA=on" pip3 install .
# RUN cd llama-cpp-python && git submodule update --remote --merge

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# FROM python:3.13.2
# RUN python3 -m ensurepip --default-pip
WORKDIR /app

# COPY --chown=user ./pyproject.toml pyproject.toml
COPY --chown=user . /app
RUN cd /app && python3 -m pip install --no-cache-dir --user .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
