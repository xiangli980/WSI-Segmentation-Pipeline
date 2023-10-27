# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image


# start from nvidia/cuda 10.0
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda:10.2-base-ubuntu18.04
FROM nvidia/cuda:11.8.0-base-ubuntu18.04
LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL maintainer="Sayat Mimar - Sarder Lab. <sayat.mimar@ufl.edu>"

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RUN mkdir /usr/local/nvidia && ln -s /usr/local/cuda-10.0/compat /usr/local/nvidia/lib

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Remove bad repos
RUN rm \
    /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    # As of 2018-04-16 this repo has the latest release of Python 2.7 (2.7.14) \
    # add-apt-repository ppa:jonathonf/python-2.7 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    #keyboard-configuration \
    git \
    wget \
    python-qt4 \
    python3-pyqt4 \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    libpython-dev \
    libpython3-dev \
    python2.7-dev \
    python-tk \
    # We can't go higher than 3.7 and use tensorflow 1.x \
    python3.8-dev \
    python3.8-distutils \
    python3-tk \
    software-properties-common \
    libssl-dev \
    # Standard build tools \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    # needed for supporting CUDA \
    # libcupti-dev \
    # useful later \
    libmemcached-dev && \
    #apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECKPOINT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN apt-get install libxml2-dev libxslt1-dev -y
# RUN apt-get install software-properties-common -y
# RUN add-apt-repository ppa:graphics-drivers/ppa -y
# RUN apt-get update -y
# RUN apt-get upgrade -y
# RUN apt-get install nvidia-driver-455 -y

WORKDIR /
# Make Python3 the default and install pip.  Whichever is done last determines
# the default python version for pip.

#Make a specific version of python the default and install pip
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln `which python3.8` /usr/bin/python && \
    ln `which python3.8` /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    ln `which pip3` /usr/bin/pip


RUN which  python && \
    python --version
# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
# RUN curl -O https://bootstrap.pypa.io/pip/3.7/get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py

ENV build_path=$PWD/build

# HistomicsTK sepcific

# copy HistomicsTK files
ENV glom_path=$PWD/GLM
RUN mkdir -p $glom_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    #apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# RUN pip install torch
# RUN python -c 'import torch,sys;print(torch.cuda.is_available());sys.exit(not torch.cuda.is_available())'
COPY . $glom_path/
WORKDIR $glom_path

# Install HistomicsTK and its dependencies
#   Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.


RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    pip install --no-cache-dir .  && \
    pip install --no-cache-dir tensorboard cmake onnx && \
    pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 -f https://download.pytorch.org/whl/torch_stable.html && \
    rm -rf /root/.cache/pip/*

# ENV FORCE_CUDA="1"
# ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
# ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
# RUN pip install --no-cache-dir -e detectron2_repo
# Show what was installed
RUN python --version && pip --version && pip freeze

# define entrypoint through which all CLIs can be run 
WORKDIR $glom_path/segmentationCode/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint GlomSegment --help


ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]