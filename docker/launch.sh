#!/bin/bash

TAG='funabiki/intelligent_signal:1.11.0-cuda11.3-cudnn8-devel'
PROJECT_DIR="$(cd "$(dirname "${0}")/.." || exit; pwd)"

# build
cd "$(dirname "${0}")/.." || exit
DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} docker

# run
docker run -it --rm \
  --shm-size=8g \
  --gpus all \
  -v "${PROJECT_DIR}:/workspace" \
  -w "/workspace" \
  "${TAG}"
