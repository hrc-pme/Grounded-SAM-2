#!/usr/bin/env bash

REPOSITORY="hrcnthu/sam2_2204"
TAG="gpu"

IMG="${REPOSITORY}:${TAG}"

docker image push "${IMG}"
