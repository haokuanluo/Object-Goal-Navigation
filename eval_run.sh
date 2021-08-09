#!/usr/bin/env bash

DOCKER_BUILDKIT=1 docker build . --build-arg INCUBATOR_VER=$(date +%s) --file neural_slam_eval.Dockerfile -t pointnav_eval
sh eval_locally_pointnav_rgbd.sh --docker-name pointnav_eval
#python images/image.py
