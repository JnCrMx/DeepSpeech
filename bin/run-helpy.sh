#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

train_file=../helpy/HelpySpeech/all.csv
test_file=../helpy/HelpySpeech/all.csv
checkpoint_dir=../helpy/HelpySpeech/checkpoints

lm=../helpy/HelpySpeech/lm.binary
trie=../helpy/HelpySpeech/trie

export_dir=../helpy/HelpySpeech/out

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py \
  --train_files "$train_file" \
  --test_files "$test_file" \
  --train_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epochs 200 \
  --checkpoint_dir "$checkpoint_dir" \
  --lm_binary_path "$lm" \
  --lm_trie_path "$trie" \
  --export_dir "$export_dir" \
  "$@"

