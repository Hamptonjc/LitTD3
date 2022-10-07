#! /bin/bash

docker run \
	--mount type=bind,source="$(pwd)",target=/LitTD3 \
	--gpus all \
	-it \
	--name "$USER"-LitTD3-container \
	-w /LitTD3 \
	--shm-size=16g \
	--rm \
	hamptonjc/lit_td3:latest \
	/bin/bash 
