## Host requirements
# sudo apt-get install nvidia-container-toolkit
## Build Docker Image
# docker build -t lanegraphnet .
## Run it
# docker run -it --gpus all -v /data:/data --ipc=host lanegraphnet:latest

# Base image
FROM nvidia/cuda:11.3-devel-ubuntu20.04

# Use nvidia keyring keys as per https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub


ENV DEBIAN_FRONTEND="noninteractive"
ENV GIT_SSL_NO_VERIFY=1

# Sys requirements
RUN apt-get update && apt-get install -y ninja-build wget git python3.8 python3-pip python3-opencv zip vim

# Setup python3.8
#RUN python3.8 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN python3.8 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.8 -m  pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

RUN python3.8 -m pip install open3d numpy==1.22.1 opencv-python scipy tqdm Pillow matplotlib triangle momepy yacs wandb easydict utm pyyaml==5.4.1

