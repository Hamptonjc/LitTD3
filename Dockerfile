FROM pytorch/pytorch:latest

RUN conda create --name TD3

RUN apt update && apt install build-essential swig -y

RUN conda run -n TD3 pip install gym[box2d] pyglet pytorch_lightning rich

RUN apt-get update && \
  apt-get install git ffmpeg libsm6 libxext6 xvfb mesa-utils libgl1-mesa-glx python-opengl -y

RUN echo 'Xvfb&' >> /root/.bashrc

RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;4m\]\]Lit-TD3-Docker\[⚡\]\[$(tput sgr0)\]\[$(tput sgr0)\]:\w\\$\[$(tput sgr0)\] "' >> /root/.bashrc

ENV CONDA_DEFAULT_ENV TD3

# Install mujoco
RUN conda run -n TD3 pip install mujoco

RUN mkdir /root/.mujoco/ && \
    apt-get install wget gcc libosmesa6-dev libglfw3 -y && \
    wget -P /root/.mujoco/ https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xf /root/.mujoco/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco && \
    rm /root/.mujoco/mujoco210-linux-x86_64.tar.gz && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> /root/.bashrc && \
    ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so


RUN conda run -n TD3 pip install imageio


