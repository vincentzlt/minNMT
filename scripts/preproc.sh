#!/usr/bin/env bash

set -xe
PROG_DIR=$(dirname "$(readlink -f "$0")")/..
DATA_DIR=${1:-data/wmt14.en-de}
VOCAB_SIZE=${2:-37000}
SRC_LANG=${3:-en}
TRG_LANG=${3:-de}
BPE_FILE=share.bpe.$VOCAB_SIZE

cd $DATA_DIR
# check files
[ -f train.$SRC_LANG ] || {
    echo "train.$SRC_LANG not found!"
    exit 1
}
[ -f train.$TRG_LANG ] || {
    echo "train.$TRG_LANG not found!"
    exit 1
}
[ -f val.$SRC_LANG ] || {
    echo "val.$SRC_LANG not found!"
    exit 1
}
[ -f val.$TRG_LANG ] || {
    echo "val.$TRG_LANG not found!"
    exit 1
}
[ -f test.$SRC_LANG ] || {
    echo "test.$SRC_LANG not found!"
    exit 1
}
[ -f test.$TRG_LANG ] || {
    echo "test.$TRG_LANG not found!"
    exit 1
}
wc -l {train,val,test}.{$SRC_LANG,$TRG_LANG}

# train bpe
if [ ! -f $BPE_FILE ]; then
    cat train.* >tmp
    yttm bpe --model $BPE_FILE --vocab_size $VOCAB_SIZE --data tmp
    rm tmp
fi
yttm vocab --model $BPE_FILE >$BPE_FILE.vocab

# apply bpe
for f in {train,val,test}.{$SRC_LANG,$TRG_LANG}; do
    fout=$f.id
    if [[ ! -f $fout ]]; then
        echo "encode $f with bpe ... "
        yttm encode --model $BPE_FILE --output_type id --bos --eos <$f >$fout
    fi
done
wc -l {train,val,test}.{$SRC_LANG,$TRG_LANG}.id

# read to pandas.DataFrame
for f in val test train; do
    fout=$f.pkl
    if [ ! -f $fout ]; then
        python $PROG_DIR/utils/id2df.py $SRC_LANG $f.$SRC_LANG.id $TRG_LANG $f.$TRG_LANG.id $fout
        if [ $f == train ]; then
            python $PROG_DIR/utils/rm_long.py train.pkl val.pkl test.pkl train.pkl
        fi
    fi
done

cd -
