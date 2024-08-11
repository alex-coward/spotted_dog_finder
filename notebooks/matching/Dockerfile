# Use the official Debian-hosted Python image
FROM python:3.9-slim-buster


# Ensure we have an up to date baseline, install dependencies
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir --upgrade pip

# Install Conda
RUN apt-get install wget && \
    mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ~/miniconda3/bin/conda init bash

# Change shell to bash from sh to use conda
SHELL ["/bin/bash", "--login", "-c"]

# Install faiss
RUN conda create --name matching python=3.9 &&\
    conda activate matching &&\
    conda install -c conda-forge faiss-gpu

# Install huggingface transformers
RUN conda activate matching && \
    conda install -c huggingface transformers && \
    conda update tokenizers

# Install pytorch
RUN conda config --add channels pytorch && \
    conda config --add channels nvidia && \
    conda activate matching && \
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8

ADD . /

# Entry point
# ENTRYPOINT ["/bin/bash"]

# Init conda environment
CMD ["-c", "conda activate matching"]



