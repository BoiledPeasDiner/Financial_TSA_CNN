# ==== NVIDIA公式: CUDA 11.6 + cuDNN8 + Ubuntu 20.04 ====
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# OS依存 & ビルド系（pyenv用） + OpenCVランタイム依存
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    wget \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    liblzma-dev \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---- pyenv で Python 3.8.3 を厳密固定 ----
ENV PYENV_ROOT=/opt/pyenv
ENV PATH=$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH
RUN git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT" && \
    echo 'export PYENV_ROOT=/opt/pyenv' >> /etc/profile.d/pyenv.sh && \
    echo 'export PATH=$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH' >> /etc/profile.d/pyenv.sh && \
    pyenv install 3.8.3 && \
    pyenv global 3.8.3 && \
    python -m pip install --upgrade pip

# ---- まず Torch 系だけを個別レイヤーで（CUDA 11.6 ホイール）----
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu116 \
    torch==1.12.1+cu116 \
    torchvision==0.13.1+cu116 \
    torchaudio==0.12.1+cu116

# ---- その次にその他ライブラリ ----
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    Pillow \
    tqdm \
    scikit-learn \
    opencv-python \
    mplfinance \
    torchinfo

# （必要ならJupyterを追加。使うときだけアンコメント）
# RUN pip install --no-cache-dir \
#     jupyterlab

# 実行環境
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /workspace
CMD ["/bin/bash"]



######################################
# ビルド
# docker build -t finimg:py383-torch1121-cuda116 .

# 実行（GPU有効・カレントを /workspace にマウント）
# docker run --rm -it --gpus all `
#   -v "${PWD}:/workspace" -w /workspace `
#   finimg:py383-torch1121-cuda116

# # CUDA 動作チェック（コンテナ内）
# python - << 'PY'
# import torch
# print("Torch:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
# PY
