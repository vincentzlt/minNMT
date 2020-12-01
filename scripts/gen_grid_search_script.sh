#!/usr/bin/env bash

set -ex
FNAME=$(basename "$0")
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# change --accumulate_grad_batches=13
mkdir $DIR/search_accumulate_grad_batches -p
for i in $(seq 2 2 20); do
    sed -E "s/accumulate_grad_batches=[0-9]*/accumulate_grad_batches=${i}/g" $DIR/train.sh \
        >$DIR/search_accumulate_grad_batches/train_${i}.sh
done
