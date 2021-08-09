#!/usr/bin/env bash

DOCKER_BUILDKIT=1 docker build . --build-arg INCUBATOR_VER=$(date +%s) --file neural_slam_semantic.Dockerfile -t pointnav_semantic
sh semantic_locally_pointnav_rgbd.sh --docker-name pointnav_semantic
#python images/image.py
