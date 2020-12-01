#!/usr/bin/env bash

set -e

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
for GPU_ID in $(seq $((NUM_GPUS - 1)) -1 0); do
    Proc=$(nvidia-smi -q -i 1 -d PIDS | grep Processes | cut -d : -f 2)
    if [ $Proc = None ]; then
        echo $GPU_ID
        exit 0
    fi
done
