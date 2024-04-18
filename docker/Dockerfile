# --------- Def base image
FROM nvidia/cuda:11.3.1-base-ubuntu18.04

# --------- My consts
ARG ENVIRONMENT_NAME="face_detection_age"

ARG MINICONDA_VERSION="py38_23.11.0-1"
ARG MINICONDA_FILENAME="Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh"
ARG YAML_original_PATH="./env/${ENVIRONMENT_NAME}_from-history.yaml"

# --------- Copy env file to image
ENV env_ENVIRONMENT_NAME=${ENVIRONMENT_NAME}

ENV MINICONDA_COPIED_NAME=${MINICONDA_FILENAME}
ENV MINICONDA_DOCKER_FILEPATH=/tmp/${MINICONDA_COPIED_NAME}

# Install prerequisites
RUN apt-get update ; apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# for cv2 pre's docker exclusive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0

# --------- Copy from system to container
ADD ${YAML_original_PATH} /src/environment.yaml
ADD ./dependencies/deepface-50105b21e6bd2c5d490483558b78a92701539752.zip /src/dependencies/

# Update environment.yaml
RUN sed -i 's|../dependencies/deepface-50105b21e6bd2c5d490483558b78a92701539752.zip|/src/dependencies/deepface-50105b21e6bd2c5d490483558b78a92701539752.zip|g' /src/environment.yaml

# --------- Load and install Miniconda
ENV MINICONDA_PATH=/opt/miniconda/

# Download Miniconda installer
RUN cd ./tmp && curl -L -O https://repo.continuum.io/miniconda/${MINICONDA_FILENAME}

RUN chmod +x ${MINICONDA_DOCKER_FILEPATH}
RUN mkdir /root/.conda
RUN bash -c "${MINICONDA_DOCKER_FILEPATH} -b -f -p ${MINICONDA_PATH}"

# Initializes Conda for bash shell interaction.
RUN ${MINICONDA_PATH}/bin/conda init bash

# --------- Create and activate the environment
RUN ${MINICONDA_PATH}/bin/conda env create -q -f /src/environment.yaml -n ${ENVIRONMENT_NAME} --yes
RUN export PATH="${MINICONDA_PATH}/envs/${ENVIRONMENT_NAME}/bin:$PATH"
RUN echo conda activate ${ENVIRONMENT_NAME} >> /root/.bashrc

RUN rm -rf /tmp/*
