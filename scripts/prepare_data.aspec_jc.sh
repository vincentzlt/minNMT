#!/usr/bin/env bash

set -e
trap 'rm -f "$TMPFILE"' EXIT
TMPFILE=$(mktemp -p $(pwd))

PROG_DIR=$(dirname "$(readlink -f "$0")")/..
PARA="parallel --pipe -k"
NORM_TEXT="python $PROG_DIR/utils/norm_text.py"
CHAR_SPLIT="sed 's/\(.\)/\1 /g'"
MECAB_SPLIT="mecab -Owakati"
JIEBA_SPLIT="python -m jieba -d ' ' -q"
REPLACE_SPACE="sed 's/ /█/g'"
TO_SUBCHAR="python $PROG_DIR/utils/subchar.py"
VOCAB="python $PROG_DIR/utils/vocab.py"
TO_ID="python $PROG_DIR/utils/to_id.py"
TO_PYTHON_LIST="python $PROG_DIR/utils/to_python_list.py"

SRC_DIR=${1:-/pvc/data/ASPEC/ASPEC-JC}
DATA_DIR=${2:-data/aspec_jc}

[ -d $DATA_DIR ] || mkdir -p $DATA_DIR && cd $DATA_DIR

echo "prepare raw data + NFKC norm ($(pwd)) ... "
cat $SRC_DIR/train/train.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k $NORM_TEXT >train.ja
cat $SRC_DIR/train/train.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k $NORM_TEXT >train.zh
cat $SRC_DIR/dev/dev.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k $NORM_TEXT >val.ja
cat $SRC_DIR/dev/dev.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k $NORM_TEXT >val.zh
cat $SRC_DIR/test/test.txt | awk -F' \\|\\|\\| ' '{print $2}' | parallel --pipe -k $NORM_TEXT >test.ja
cat $SRC_DIR/test/test.txt | awk -F' \\|\\|\\| ' '{print $3}' | parallel --pipe -k $NORM_TEXT >test.zh
echo "done"
wc -l {train,val,test}.{ja,zh}

SUB_DIR=char
echo "prepare ${SUB_DIR} level data ($(pwd)/$SUB_DIR) ... "
[ -d $SUB_DIR ] || mkdir $SUB_DIR && cd $SUB_DIR
for f in {train,val,test}.{ja,zh}; do
    cat ../$f | $PARA $CHAR_SPLIT >$f
done
cat train.{ja,zh} | $VOCAB >train.share.vocab
echo "vocab size $(wc -l train.share.vocab | cut -d ' ' -f1) ..."
for f in {train,val,test}.{ja,zh}; do
    cat $f | $PARA "$TO_ID train.share.vocab" | $PARA $TO_PYTHON_LIST >$f.id
done
cd ..
echo "done"

SUB_DIR=word
echo "prepare ${SUB_DIR} level data ($(pwd)/$SUB_DIR) ... "
[ -d $SUB_DIR ] || mkdir $SUB_DIR && cd $SUB_DIR
for f in {train,val,test}.ja; do
    cat ../$f | $PARA $MECAB_SPLIT >$f
done
for f in {train,val,test}.zh; do
    cat ../$f | $PARA $JIEBA_SPLIT >$f
done
cat train.{ja,zh} | $VOCAB 32000 >train.share.vocab
echo "vocab size $(wc -l train.share.vocab | cut -d ' ' -f1) ..."
for f in {train,val,test}.{ja,zh}; do
    cat $f | $PARA "$TO_ID train.share.vocab" | $PARA $TO_PYTHON_LIST >$f.id
done
cd ..
echo "done"

SUB_DIR=bpe.32000
echo "prepare ${SUB_DIR} level data ($(pwd)/$SUB_DIR) ... "
[ -d $SUB_DIR ] || mkdir $SUB_DIR && cd $SUB_DIR
cat ../train.{ja,zh} >$TMPFILE
yttm bpe --data $TMPFILE --model bpe.32000.share --vocab_size 32000
yttm vocab --model bpe.32000.share >bpe.32000.share.vocab
echo "vocab size $(wc -l bpe.32000.share.vocab | cut -d ' ' -f1) ..."
for f in {train,val,test}.{ja,zh}; do
    yttm encode --model bpe.32000.share --output_type id <../$f | $PARA $TO_PYTHON_LIST >$f.id
done
cd ..
echo "done"

# for SPLIT_DIR in ideo ideo_finest stroke; do
#     echo "prepare $SPLIT_DIR data ($(pwd)/$SPLIT_DIR) ... "
#     [ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
#     for f in {train,val,test}.{ja,zh}; do
#         cat $f | parallel --pipe -k "$TO_SUBCHAR $SPLIT_DIR" | parallel --pipe -k $REPLACE_SPACE >$SPLIT_DIR/$f
#     done
#     cd $SPLIT_DIR
#     SPLIT_DIR=char
#     echo "prepare char level data ($(pwd)/$SPLIT_DIR) ... "
#     [ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
#     for f in {train,val,test}.{ja,zh}; do
#         cat $f | parallel --pipe -k $CHAR_SPLIT >$SPLIT_DIR/$f
#     done
#     cat $SPLIT_DIR/train.{ja,zh} | $VOCAB >$SPLIT_DIR/train.share.vocab
#     echo "vocab size $(wc -l $SPLIT_DIR/train.share.vocab) ..."
#     for f in $SPLIT_DIR/{train,val,test}.{ja,zh}; do
#         cat $f | parallel --pipe -k "$TO_ID $SPLIT_DIR/train.share.vocab" | parallel --pipe -k $TO_PYTHON_LIST >$f.id
#     done
#     echo "done"
#     SPLIT_DIR=bpe.32000
#     echo "prepare bpe.32000 data ($(pwd)/$SPLIT_DIR) ... "
#     [ -d $SPLIT_DIR ] || mkdir $SPLIT_DIR
#     cat train.{ja,zh} >$TMPFILE
#     yttm bpe --data $TMPFILE --model $SPLIT_DIR/bpe.32000.share --vocab_size 32000
#     yttm vocab --model $SPLIT_DIR/bpe.32000.share >$SPLIT_DIR/bpe.32000.share.vocab
#     echo "vocab size $(wc -l $SPLIT_DIR/bpe.32000.share.vocab) ..."
#     for f in {train,val,test}.{ja,zh}; do
#         yttm encode --model $SPLIT_DIR/bpe.32000.share --output_type id <$f | parallel --pipe -k $TO_PYTHON_LIST >$SPLIT_DIR/$f.id
#     done
#     echo "done"
#     cd ..
# done
cd ..
