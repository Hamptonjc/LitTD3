FROM pytorch/pytorch:latest

RUN conda create --name TD3

RUN conda run -n TD3 pip install gym[box2d] pyglet pytorch_lightning rich

RUN apt-get update && \
  apt-get install ffmpeg libsm6 libxext6 xvfb mesa-utils libgl1-mesa-glx python-opengl -y

RUN echo 'Xvfb&' >> /root/.bashrc

RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;30m\]Lit-TD3-Docker\[🤖\]\[$(tput sgr0)\]\[$(tput sgr0)\]:\w\\$\[$(tput sgr0)\] "' >> /root/.bashrc

ENV CONDA_DEFAULT_ENV TD3
