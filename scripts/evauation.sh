#!/usr/bin/env bash

: ${nlu_url:=ccoreilly/wav2vec2-large-100k-voxpopuli-catala}

# Full
# : ${slurp_testset_no_examples:=None}
# random 10 samples
: ${slurp_testset_no_examples:=10}



echo 'Using samples= '$slurp_testset_no_examples

python evaluation.py \
    --slurp_testset_filepath=data/slurp_testset.jsonl \
    --slurp_testset_audiodir=data/audio/slurp_real \
    --slurp_testset_no_examples=$slurp_testset_no_examples \
    --nlu_url=$nlu_url
