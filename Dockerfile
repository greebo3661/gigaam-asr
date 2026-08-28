# Stack per salute-developers/GigaAM pyproject.toml: torch>=2.6, ffmpeg, pip package gigaam.
# torch 2.8 is not published for cu124; 2.6.0+cu124 matches Ubuntu 22.04 + GTX 1080 setups.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install -r requirements.txt

COPY main.py transcriber.py ./

ENV GIGAAM_MODEL=v3_e2e_rnnt \
    GIGAAM_MODEL_PATH=/models \
    DEVICE=cuda \
    FP16_ENCODER=true \
    HOST=0.0.0.0 \
    PORT=5002 \
    WARMUP_ON_START=true \
    MAX_AUDIO_DURATION_SEC=25

EXPOSE 5002

CMD ["python3", "main.py"]
