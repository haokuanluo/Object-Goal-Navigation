#!/usr/bin/env bash


python Neural-SLAM/main.py -n3 --print_images 0 --auto_gpu_config 0 --split train --eval 0 --train_global 1 --train_local 1 --train_slam 1 --load_local habitat-challenge-data/tmp22/models/exp1/model_best.local --load_slam habitat-challenge-data/tmp22/models/exp1/model_best.slam --dump_location ./habitat-challenge-data/tmp23/

