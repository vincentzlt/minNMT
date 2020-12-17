#!/usr/bin/env bash

set -e
trap 'rm -f "$TMPFILE"' EXIT
TMPFILE=$(mktemp -p $(pwd))

PROG_DIR=$(dirname "$(readlink -f "$0")")/..
CHAR_SPLIT="sed 's/\(.\)/\1 /g'"
MECAB_SPLIT="mecab -Owakati"
JIEBA_SPLIT="python -m jieba -d ' ' -q"
REPLACE_SPACE="sed 's/ /█/g'"
TO_SUBCHAR="python $PROG_DIR/utils/subchar.py"
VOCAB="python $PROG_DIR/utils/vocab.py"
TO_ID="python $PROG_DIR/utils/to_id.py"

DATA_DIR=${1:-data/aspec_jc}
[[ ! -d $DATA_DIR ]] && mkdir -p $DATA_DIR
cd $DATA_DIR

echo "prepare raw data + NFKC norm ($(pwd)) ... "
SRC_DIR=/cldata/ASPEC/ASPEC-JC
cat $SRC_DIR/train/train.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >train.ja
cat $SRC_DIR/train/train.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >train.zh
cat $SRC_DIR/dev/dev.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >val.ja
cat $SRC_DIR/dev/dev.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >val.zh
cat $SRC_DIR/test/test.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >test.ja
cat $SRC_DIR/test/test.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k "python $PROG_DIR/utils/norm_text.py " >test.zh
echo "done"
wc -l {train,val,test}.{ja,zh}

SPLIT_DIR=char
echo "prepare char level data ($(pwd)/$SPLIT_DIR) ... "
[ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
for f in {train,val,test}.{ja,zh}; do
    cat $f | parallel --pipe -k $CHAR_SPLIT >$SPLIT_DIR/$f
done
cat $SPLIT_DIR/train.{ja,zh} | $VOCAB >$SPLIT_DIR/train.share.vocab
echo "vocab size $(wc -l $SPLIT_DIR/train.share.vocab) ..."
for f in $SPLIT_DIR/{train,val,test}.{ja,zh}; do
    cat $f | parallel --pipe -k "$TO_ID $SPLIT_DIR/train.share.vocab" >$f.id
done
echo "done"

SPLIT_DIR=word
echo "prepare word data ($(pwd)/$SPLIT_DIR) ... "
[ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
for f in {train,val,test}.ja; do
    cat $f | parallel --pipe -k $MECAB_SPLIT >$SPLIT_DIR/$f
done
for f in {train,val,test}.zh; do
    cat $f | parallel --pipe -k $JIEBA_SPLIT >$SPLIT_DIR/$f
done
cat $SPLIT_DIR/train.{ja,zh} | $VOCAB >$SPLIT_DIR/train.share.vocab
head -n 32000 $SPLIT_DIR/train.share.vocab >$TMPFILE
echo "vocab size reduced from $(wc -l $SPLIT_DIR/train.share.vocab) to $(wc -l $TMPFILE) ..."
mv $TMPFILE $SPLIT_DIR/train.share.vocab
for f in $SPLIT_DIR/{train,val,test}.{ja,zh}; do
    cat $f | parallel --pipe -k "$TO_ID $SPLIT_DIR/train.share.vocab" >$f.id
done
echo "done"

SPLIT_DIR=bpe.32000
echo "prepare bpe.32000 data ($(pwd)/$SPLIT_DIR) ... "
[ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
cat train.{ja,zh} >$TMPFILE
yttm bpe --data $TMPFILE --model $SPLIT_DIR/bpe.32000.share --vocab_size 32000
yttm vocab --model $SPLIT_DIR/bpe.32000.share >$SPLIT_DIR/bpe.32000.share.vocab
echo "vocab size $(wc -l $SPLIT_DIR/bpe.32000.share.vocab) ..."
for f in {train,val,test}.{ja,zh}; do
    yttm encode --model $SPLIT_DIR/bpe.32000.share --output_type id --bos --eos <$f >$SPLIT_DIR/$f.id
done
echo "done"

for SPLIT_DIR in ideo ideo_finest stroke; do
    echo "prepare $SPLIT_DIR data ($(pwd)/$SPLIT_DIR) ... "
    [ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
    for f in {train,val,test}.{ja,zh}; do
        cat $f | parallel --pipe -k "$TO_SUBCHAR $SPLIT_DIR" | parallel --pipe -k $REPLACE_SPACE >$SPLIT_DIR/$f
    done
    cd $SPLIT_DIR
    SPLIT_DIR=char
    echo "prepare char level data ($(pwd)/$SPLIT_DIR) ... "
    [ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
    for f in {train,val,test}.{ja,zh}; do
        cat $f | parallel --pipe -k $CHAR_SPLIT >$SPLIT_DIR/$f
    done
    cat $SPLIT_DIR/train.{ja,zh} | $VOCAB >$SPLIT_DIR/train.share.vocab
    echo "vocab size $(wc -l $SPLIT_DIR/train.share.vocab) ..."
    for f in $SPLIT_DIR/{train,val,test}.{ja,zh}; do
        cat $f | parallel --pipe -k "$TO_ID $SPLIT_DIR/train.share.vocab" >$f.id
    done
    echo "done"
    SPLIT_DIR=bpe.32000
    echo "prepare bpe.32000 data ($(pwd)/$SPLIT_DIR) ... "
    [ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
    cat train.{ja,zh} >$TMPFILE
    yttm bpe --data $TMPFILE --model $SPLIT_DIR/bpe.32000.share --vocab_size 32000
    yttm vocab --model $SPLIT_DIR/bpe.32000.share >$SPLIT_DIR/bpe.32000.share.vocab
    echo "vocab size $(wc -l $SPLIT_DIR/bpe.32000.share.vocab) ..."
    for f in {train,val,test}.{ja,zh}; do
        yttm encode --model $SPLIT_DIR/bpe.32000.share --output_type id --bos --eos <$f >$SPLIT_DIR/$f.id
    done
    echo "done"
    cd ..
done
cd ..
