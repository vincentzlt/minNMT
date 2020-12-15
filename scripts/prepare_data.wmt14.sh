#!/usr/bin/env bash

set -x
set -e

PROG_DIR=$(dirname "$(readlink -f "$0")")/..

# download stanford wmt14 data and de-moses them to plain text
DATA_DIR=${1:-data/wmt14.en-de}
[[ ! -d $DATA_DIR ]] && mkdir -p $DATA_DIR
cd $DATA_DIR

# download train data

wget -c http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz &
wget -c http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz &
wget -c http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz &
wait

# create train

if [[ ! -f train.de ]] || [[ ! -f train.en ]]; then
    tar vxfz training-parallel-europarl-v7.tgz --wildcards --no-anchored 'training/*de-en*'
    tar vxfz training-parallel-commoncrawl.tgz --wildcards --no-anchored '*de-en*'
    tar vxfz training-parallel-nc-v9.tgz --wildcards --no-anchored 'training/*de-en*'

    cat training/europarl-v7.de-en.de training/news-commentary-v9.de-en.de commoncrawl.de-en.de >train.de
    cat training/europarl-v7.de-en.en training/news-commentary-v9.de-en.en commoncrawl.de-en.en >train.en

    rm -rf training commoncrawl*
fi

# filter out cjk characters
paste train.{en,de} |
    python $PROG_DIR/utils/not_cjk.py >tmp
cat tmp | cut -f 1 >train.en
cat tmp | cut -f 2 >train.de
rm tmp

# download val and test data
wget -c http://www.statmt.org/wmt14/dev.tgz
tar vxfz dev.tgz
cat dev/newstest2013-src.en.sgm | sed -e 's/<[^>]*>//g' | sed '/^$/d' >val.en
cat dev/newstest2013-ref.de.sgm | sed -e 's/<[^>]*>//g' | sed '/^$/d' >val.de
rm -rf dev

wget -c http://www.statmt.org/wmt14/test-full.tgz
tar vxfz test-full.tgz
cat test-full/newstest2014-deen-src.en.sgm | sed -e 's/<[^>]*>//g' | sed '/^$/d' >test.en
cat test-full/newstest2014-deen-ref.de.sgm | sed -e 's/<[^>]*>//g' | sed '/^$/d' >test.de
rm -rf test-full
cd -
