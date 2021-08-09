#!/usr/bin/env bash

docker build . --build-arg INCUBATOR_VER=$(date +%s) --file neural_slam_train.Dockerfile -t pointnav_train
sh train_locally_pointnav_rgbd.sh --docker-name pointnav_train
#python images/image.py
