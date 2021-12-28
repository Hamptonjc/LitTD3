#! /bin/bash

docker run \
	--mount type=bind,source="$(pwd)",target=/LitTD3 \
    -it \
    -e DISPLAY=${HOSTNAME}$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:$HOME/.Xauthority \
    --name "$USER"-LitTD3-container \
    -w /LitTD3 \
    --shm-size=16g \
    --network=host \
    --rm \
    lit_td3:latest \
    /bin/bash \
