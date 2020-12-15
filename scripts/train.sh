#!/usr/bin/env bash

set -xe

# FNAME=$(basename "$0")
# DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# GPU_ID=$(/home/longtu/minNMT/scripts/empty_gpu.sh)
# if [ ! -z $GPU_ID ]; then
#     export CUDA_VISIBLE_DEVICES=$GPU_ID
# else
#     sleep 10
# fi
CUDA_VISIBLE_DEVICES=5 \
python /clwork/vincentzlt/minNMT/train.py \
    --default_root_dir=/clwork/vincentzlt/minNMT/exps \
    --gradient_clip_val=0 \
    --gpus=1 \
    --progress_bar_refresh_rate=1 \
    --accumulate_grad_batches=1 \
    --max_epochs=10 \
    --max_steps=105000 \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --limit_test_batches=1.0 \
    --val_check_interval=0.1 \
    --log_every_n_steps=1 \
    --weights_summary=top \
    --deterministic=true \
    --reload_dataloaders_every_epoch=true \
    --train_pkl=/clwork/vincentzlt/minNMT/data/stanford.wmt14/train.pkl \
    --val_pkl=/clwork/vincentzlt/minNMT/data/stanford.wmt14/val.pkl \
    --src_test=/clwork/vincentzlt/minNMT/data/stanford.wmt14/test.en.id \
    --trg_test=/clwork/vincentzlt/minNMT/data/stanford.wmt14/test.de.id \
    --batch_size=25000 \
    --num_workers=40 \
    --seed=$RANDOM \
    --src_lang=en \
    --trg_lang=de \
    --vocab_size=37000 \
    --d_model=512 \
    --nhead=8 \
    --num_encoder_layers=6 \
    --num_decoder_layers=6 \
    --dim_feedforward=2048 \
    --dropout=0.1 \
    --activation=relu \
    --warmup=4000 \
    --bpe_file=/clwork/vincentzlt/minNMT/data/stanford.wmt14/share.bpe.37000 \
    --lenpen=0.6 \
    --beam_size=4 \
    --ckpt_steps=1500
