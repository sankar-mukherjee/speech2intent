#!/usr/bin/env bash

: ${push_to_hub:=0}
: ${use_gpu:=0}

python text_intent_classifer/train.py \
    --slurp_train_filepath=data/train.jsonl \
    --slurp_val_filepath=data/devel.jsonl \
    --output_dir=models/slurp-intent_baseline-distilbert-base-uncased \
    --nlu_url=distilbert-base-uncased \
    --push_to_hub=$push_to_hub \
    --use_gpu=$use_gpu
