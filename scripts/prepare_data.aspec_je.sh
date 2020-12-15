#!/usr/bin/env bash

set -e

PROG_DIR=$(dirname "$(readlink -f "$0")")/..

# download stanford wmt14 data and de-moses them to plain text
DATA_DIR=${1:-data/aspec_je}
[[ ! -d $DATA_DIR ]] && mkdir -p $DATA_DIR
cd $DATA_DIR

SRC_DIR=/cldata/ASPEC/ASPEC-JE/

cat $SRC_DIR/train/train-1.txt | awk -F' \\|\\|\\| ' '{print $4}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py" >train.ja
cat $SRC_DIR/train/train-1.txt | awk -F' \\|\\|\\| ' '{print $5}' >train.en
cat $SRC_DIR/dev/dev.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py" >val.ja
cat $SRC_DIR/dev/dev.txt | awk -F' \\|\\|\\| ' '{print $4}' >val.en
cat $SRC_DIR/test/test.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py" >test.ja
cat $SRC_DIR/test/test.txt | awk -F' \\|\\|\\| ' '{print $4}' >test.en

wc -l {train,val,test}.{ja,en}

cd -
