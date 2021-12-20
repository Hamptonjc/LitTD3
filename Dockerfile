FROM pytorchlightning/pytorch_lightning:base-conda-py3.7-torch1.10

RUN conda run -n lightning pip install gym pyglet

RUN apt-get update && apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev

RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;30m\]Lit-TD3-Docker\[🤖\]\[$(tput sgr0)\]\[$(tput sgr0)\]:\w\\$\[$(tput sgr0)\] "' >> /root/.bashrc

RUN echo 'conda activate lightning' >> /root/.bashrc
