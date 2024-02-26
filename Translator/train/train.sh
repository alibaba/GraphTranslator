#!/bin/bash

current_time=$(date +"%Y-%m-%d-%H-%M-%S")

nohup python -u train.py > train_$current_time.log 2>&1 &
