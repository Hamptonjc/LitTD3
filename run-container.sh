#! /bin/bash

docker run \
	--mount type=bind,source="$(pwd)",target=/LitTD3 \
    --gpus all \
    -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
	--name "$USER"-LitTD3-container \
	-w /LitTD3 \
	--shm-size=16g \
	--network=host \
	--rm \
	lit_td3:latest \
	/bin/bash \