#!/usr/bin/env bash

DOCKER_NAME="pointnav_semantic"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(pwd)/data:/data\
    -v $(pwd)/Neural-SLAM:/Neural-SLAM\
    -v $(pwd)/configs/challenge_objectnav2020.local.rgbd.yaml:/challenge_objectnav2020.local.rgbd.yaml \
    -v $(pwd)/semantic:/semantic\
    --runtime=nvidia \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_pointnav2020.local.rgbd.yaml" \
    ${DOCKER_NAME}\

