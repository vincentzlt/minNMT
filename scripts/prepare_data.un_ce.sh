#!/usr/bin/env bash

set -e

PROG_DIR=$(dirname "$(readlink -f "$0")")/..

# download stanford wmt14 data and de-moses them to plain text
DATA_DIR=${1:-data/un_ce}
[[ ! -d $DATA_DIR ]] && mkdir -p $DATA_DIR
cd $DATA_DIR

SRC_DIR=/clwork/vincentzlt/data/UN_data

cp $SRC_DIR/en-zh/UNv1.0.en-zh.zh train.zh
cp $SRC_DIR/en-zh/UNv1.0.en-zh.en train.en
cp $SRC_DIR/testsets/devset/UNv1.0.devset.zh val.zh
cp $SRC_DIR/testsets/devset/UNv1.0.devset.en val.en
cp $SRC_DIR/testsets/testset/UNv1.0.testset.zh test.zh
cp $SRC_DIR/testsets/testset/UNv1.0.testset.en test.en

wc -l {train,val,test}.{zh,en}

cd -
