#!/usr/bin/env bash

# 32 50
# 31 200
# 33 300
# 34 300 10
python Neural-SLAM/main.py -n1 --print_images 0 --auto_gpu_config 0 --split val --eval 1 --train_global 0 --train_local 0 --train_slam 0 --load_local pretrained_models/model_best.local --load_slam pretrained_models/model_best.slam --load_global pretrained_models/model_best.global --num_episodes 1000 --max_episode_length 500 --dump_location ./habitat-challenge-data/eval36/

