#!/usr/bin/env bash

set -e

PROG_DIR=$(dirname "$(readlink -f "$0")")/..

# download stanford wmt14 data and de-moses them to plain text
DATA_DIR=${1:-data/aspec_jc}
[[ ! -d $DATA_DIR ]] && mkdir -p $DATA_DIR
echo work on $DATA_DIR
cd $DATA_DIR

for d in char word bpe.32000; do
    [ -d $d ] || mkdir $d
    echo "process $d ..."
    if [ $d == char ]; then
        for f in {train,val,test}.??; do
            if [ ${f##*.} == ja -o ${f##*.} == zh ]; then
                cat $f | parallel --pipe -k "sed 's/\(.\)/\1 /g'" >$d/$f
            else
                cp $f $d/$f
            fi
        done
    elif [ $d == word ]; then
        for f in {train,val,test}.??; do
            if [ ${f##*.} == ja ]; then
                cat $f | parallel --pipe -k "mecab -Owakati" >$d/$f
            elif [ ${f##*.} == zh ]; then
                cat $f | parallel --pipe -k "python -m jieba -d ' ' -q" >$d/$f
            else
                cp $f $d/$f
            fi
        done

    elif [ $d == bpe.32000 ]; then
        cat train.?? >tmp
        yttm bpe --data tmp --model $d/bpe.32000.share --vocab_size 32000
        for f in {train,val,test}.??; do
            yttm encode --model $d/bpe.32000.share --output_type subword <$f >$d/$f
        done
        rm tmp
    fi

    wc -l $d/{train,val,test}.??
done

cd -
