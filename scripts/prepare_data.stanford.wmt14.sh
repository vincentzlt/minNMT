#!/usr/bin/env bash

set -e

PROG_DIR=$(dirname "$(readlink -f "$0")")/..

# download stanford wmt14 data and de-moses them to plain text
DATA_DIR=${1:-data/stanford.wmt14}
[[ ! -d $DATA_DIR ]] && mkdir -p $DATA_DIR
cd $DATA_DIR

wget -c https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
wget -c https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
wget -c https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en && mv newstest2013.en val.en
wget -c https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de && mv newstest2013.de val.de
wget -c https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en && mv newstest2014.en test.en
wget -c https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de && mv newstest2014.de test.de

wc -l {train,val,test}.{en,de}

cd -
