##########################################################################
# Ubuntu 18だったのでvscode185に下げないといけない

# # PyTorch 1.13.0 + CUDA 11.6 + cuDNN8 (Ubuntu 20.04 系)
# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# # タイムゾーン & Matplotlib をヘッドレスで
# ENV TZ=Asia/Tokyo \
#     MPLBACKEND=Agg

# # ランタイム依存（OpenCV/ffmpeg などで利用）
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
#  && rm -rf /var/lib/apt/lists/*

# # pip を最新化して必要パッケージを導入
# # torch はベースに入っているので入れ直し不要
# # torchvision は torch==1.13.0 に対応する 0.14.0 を指定（cu116 対応）
# RUN python -m pip install --upgrade pip && \
#     pip install \
#       numpy \
#       pandas \
#       matplotlib \
#       pillow \
#       scikit-learn \
#       tqdm \
#       torchinfo \
#       opencv-python-headless \
#       mplfinance \
#       torchvision==0.14.0

# # 作業ディレクトリ
# WORKDIR /workspace

# # デフォルトは bash
# CMD ["/bin/bash"]


###########################################################################
# condaなしの代わりに、Python3.8.3指定のマイクロバージョンを妥協

# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# ENV DEBIAN_FRONTEND=noninteractive \
#     TZ=Asia/Tokyo \
#     MPLBACKEND=Agg

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3 \
#     python3-pip \
#     python3-dev \
#     curl \
#     ca-certificates \
#     git \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     ffmpeg \
#  && rm -rf /var/lib/apt/lists/*

# # python コマンド名を python3 に合わせる（好み）
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
#     python -m pip install --upgrade pip

# # PyTorch 1.12.1 + cu113（pip）と要求ライブラリ
# RUN pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
#     torch==1.12.1+cu113 \
#     torchvision==0.13.1+cu113

# RUN pip install \
#     numpy \
#     pandas \
#     matplotlib \
#     pillow \
#     scikit-learn \
#     tqdm \
#     torchinfo \
#     opencv-python-headless \
#     mplfinance

# WORKDIR /workspace
# CMD ["/bin/bash"]


###########################################################################
# docker build -t py38-pt112-cu113:latest .

# docker run --rm -it --gpus all   -v "${PWD}:/workspace"   py38-pt112-cu113:latest

##########################################################################
# condaあり
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    MPLBACKEND=Agg

# 基本ツール・ランタイム依存（1行ずつ \ で継続）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    bzip2 \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Miniconda で Python 3.8.3 を厳密指定
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p $CONDA_DIR \
 && rm -f /tmp/mc.sh

RUN conda update -y -n base -c defaults conda \
 && conda install -y \
    python=3.8.3 \
 && conda clean -afy

# pip を最新化
RUN python -m pip install --upgrade pip

# PyTorch 1.12.1 (CUDA 11.3) + torchvision（対応ペア）
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113

# 要求ライブラリ（1行ずつ \ で継続）
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    pillow \
    scikit-learn \
    tqdm \
    torchinfo \
    opencv-python-headless \
    mplfinance

WORKDIR /workspace
CMD ["/bin/bash"]
###########################################################################
# docker build -t py38-pt112-cu113_withconda:latest .

# docker run --rm -it --gpus all   -v "${PWD}:/workspace"   py38-pt112-cu113_withconda:latest

##########################################################################

