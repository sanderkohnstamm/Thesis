
FROM ultralytics/yolov5:v6.1

RUN apt update
RUN apt install -y bc
RUN apt install -y tmux
RUN python3 -m pip -q install --upgrade pip

RUN pip3 install torch torchvision torchaudio
RUN pip install wandb
RUN pip install wandb --upgrade


WORKDIR /project