FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt -y upgrade
RUN apt-get -y update
RUN apt -y install software-properties-common git vim htop tmux wget

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt -y upgrade
RUN apt-get -y update

RUN apt-get -y install python3.9 
RUN apt-get -y install python3-pip 
RUN apt-get -y install python3.9-distutils 
RUN apt-get -y install python3.9-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

WORKDIR /root

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

# For the line below, update the version for torch, torchvision, and torchaudio based on your CUDA version
# Install PyTorch with CUDA for systems with a GPU, and fallback to CPU-only PyTorch otherwise
RUN if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q 'True'; then \
        pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116; \
    else \
        pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1; \
    fi

RUN pip3 install numpy==1.26.4
RUN pip3 install transformers==4.21.1 scikit-learn==1.1.2 tree-sitter==0.20.0
RUN pip3 install z3
RUN pip3 install pd
RUN pip3 install pandas==2.2.3
RUN pip3 install pymoo==0.6.1.3
RUN pip3 install numba==0.60.0
RUN pip3 install pyDOE==0.3.8

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility