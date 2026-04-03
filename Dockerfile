FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
curl

# Install prerequisites and add deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        && add-apt-repository ppa:deadsnakes/ppa \
        && apt-get update


# Install Python 3.10 (or 3.9, 3.11, does not really matter.)
RUN apt-get install -y python3.10
RUN apt-get install -y python3.10-dev
RUN apt-get install -y python3.10-distutils

# Set python3.10 as the default python3 (optional)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN echo "----------- PYTHON VERSION -----------"
RUN python3 --version

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN apt update
RUN apt install nano

RUN ln -s /usr/bin/python3 /usr/bin/python

# Copying source files for fnofound & localNO
COPY fnofound ./fnofound

COPY install_dependencies.sh .
COPY run_inference.sh .
COPY multiphys_pretrain.py .

COPY mamba ./mamba

RUN mkdir experiments/
RUN cd experiments/

RUN mkdir logs/
RUN mkdir data/
RUN mkdir pretrained_models/
RUN mkdir scripts/
RUN cd ..

COPY experiments/scripts/poseidon.py ./experiments/scripts/
COPY experiments/scripts/poseidon_single_problem.py ./experiments/scripts/

RUN chmod +x install_dependencies.sh
RUN ./install_dependencies.sh

CMD ["/bin/bash"]
