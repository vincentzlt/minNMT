#!/usr/bin/env bash

set -e

PROG_DIR=$(dirname "$(readlink -f "$0")")/..

DATA_DIR=${1:-data/aspec_jc}
[[ ! -d $DATA_DIR ]] && mkdir -p $DATA_DIR
cd $DATA_DIR

SRC_DIR=/cldata/ASPEC/ASPEC-JC

cat $SRC_DIR/train/train.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >train.ja
cat $SRC_DIR/train/train.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >train.zh
cat $SRC_DIR/dev/dev.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >val.ja
cat $SRC_DIR/dev/dev.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >val.zh
cat $SRC_DIR/test/test.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >test.ja
cat $SRC_DIR/test/test.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >test.zh

wc -l {train,val,test}.{ja,zh}

cd -
