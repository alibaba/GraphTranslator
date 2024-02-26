#!/bin/bash

current_time=$(date +"%Y-%m-%d-%H-%M-%S")

nohup python -u producer.py > producer_$current_time.log 2>&1 &

