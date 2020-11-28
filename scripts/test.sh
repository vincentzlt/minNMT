set -e
set -x

src_lang=${1:-en}
trg_lang=${2:-de}
src=${3}
trg=${4}
hyp=${5}
bpe=${6}
gpu=${7}
ckpt=${8}
batch_size=${9:-5000}
beam_size=${10:-4}
lenpen=${11:-0.6}

src_tok=$src.tok
src_id=$src_tok.id
hyp_tok=$hyp.tok
hyp_id=$hyp_tok.id

sacremoses -l $src_lang tokenize <$src >$src_tok
yttm encode --model $bpe --output_type id --bos --eos <$src_tok >$src_id
python /storage07/user_data/zhanglongtu01/minNMT/search.py \
    --input $src_id \
    --ckpt $ckpt \
    --gpu $gpu \
    --batch_size $batch_size \
    --beam_size $beam_size \
    --lenpen $lenpen \
    >$hyp_id

yttm decode --model $bpe --ignore_ids 2,3 <$hyp_id >$hyp_tok
sacremoses -l $trg_lang detokenize <$hyp_tok >$hyp
<$hyp | sacrebleu $trg
