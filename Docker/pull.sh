#!/usr/bin/env bash

REPOSITORY="hrcnthu/sam2_2204"
TAG="gpu"

IMG="${REPOSITORY}:${TAG}"

docker pull "${IMG}"
