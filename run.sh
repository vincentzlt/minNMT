set -x
set -e

python mt.py \
    "--gpus=3," \
    "--accumulate_grad_batches=10" \
    "--gradient_clip_val=1.0" \
    "--seed=12345" \
    "--default_root_dir=exp/1" \
    "--model=my_transformer" \
    "--vocab_size=37000" \
    "--dim=512" \
    "--dropout=0.1" \
    "--lr_scale=1" \
    "--warmup=4000" \
    "--loss=label_smooth" \
    "--train_path=/storage07/user_data/zhanglongtu01/minNMT/data/train.pkl.gz" \
    "--val_path=/storage07/user_data/zhanglongtu01/minNMT/data/val.pkl.gz" \
    "--test_path=/storage07/user_data/zhanglongtu01/minNMT/data/test.pkl.gz" \
    "--src_vocab_path=/storage07/user_data/zhanglongtu01/minNMT/data/share.vocab.en" \
    "--trg_vocab_path=/storage07/user_data/zhanglongtu01/minNMT/data/share.vocab.de" \
    "--slang=en" \
    "--tlang=de" \
    "--is_moses" \
    "--bpe=/storage07/user_data/zhanglongtu01/minNMT/data/bpe.37k.share" \
    "--batch_size=2500" \
    "--ckpt_save_interval=1500"
