#!/usr/bin/env bash

DOCKER_BUILDKIT=1 docker build . --build-arg INCUBATOR_VER=$(date +%s) --file objnav.Dockerfile -t objnav
sh objnav_locally_pointnav_rgbd.sh --docker-name objnav
#python images/image.py
