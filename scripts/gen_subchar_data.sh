#!/usr/bin/env bash

set -e

PROG_DIR=$(dirname "$(readlink -f "$0")")/..

# download stanford wmt14 data and de-moses them to plain text
DATA_DIR=${1:-data/aspec_jc}
[[ ! -d $DATA_DIR ]] && mkdir -p $DATA_DIR
echo work on $DATA_DIR
cd $DATA_DIR

for d in ideo ideo_finest stroke; do
    [ -d $d ] || mkdir $d
    echo "process $d ..."
    for f in {train,val,test}.??; do
        if [ ${f##*.} == ja -o ${f##*.} == zh ]; then
            python /clwork/vincentzlt/minNMT/utils/subchar.py $d <$f >$d/$f &
        else
            cp $f $d
        fi
    done
    wait
    for sub_d in bpe.32000 char; do
        [ -d $d/$sub_d ] || mkdir $d/$sub_d
        if [ $sub_d == char ]; then
            for f in $d/{train,val,test}.??; do
                if [ ${f##*.} == ja -o ${f##*.} == zh ]; then
                    cat $f | parallel --pipe -k "sed 's/\(.\)/\1 /g'" >$d/$sub_d/${f##*/}
                else
                    cp $f $d/$sub_d/${f##*/}
                fi
            done

        elif
            [ $sub_d == bpe.32000 ]
        then
            cat $d/train.?? >$d/tmp
            yttm bpe --data $d/tmp --model $d/$sub_d/bpe.32000.share --vocab_size 32000
            for f in $d/{train,val,test}.??; do
                yttm encode --model $d/$sub_d/bpe.32000.share --output_type subword <$f >$d/$sub_d/${f##*/}
            done
            rm $d/tmp
        fi
    done
done

cd -
