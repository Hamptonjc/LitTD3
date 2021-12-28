FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.10

RUN conda run -n lightning pip install gym pyglet

ENV TZ=America/New_York

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#RUN apt-get update && \
#  apt-get install ffmpeg libsm6 libxext6 xvfb mesa-utils libgl1-mesa-glx pyopengl -y

RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;30m\]Lit-TD3-Docker\[🤖\]\[$(tput sgr0)\]\[$(tput sgr0)\]:\w\\$\[$(tput sgr0)\] "' >> /root/.bashrc

RUN echo 'conda activate lightning' >> /root/.bashrc
