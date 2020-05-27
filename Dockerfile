FROM python:3.6-buster

ADD requirements.txt /root/requirements.txt
RUN mkdir -p /root/.mujoco && \
    wget https://www.roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && \
    mv mujoco200_linux /root/.mujoco/mujoco200 && \
    rm mujoco200_linux.zip
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ADD mjkey.txt /root/.mujoco/mjkey.txt
RUN pip3 install -r /root/requirements.txt
RUN pip3 install fiber
ADD . /root/
