#!/usr/bin/env bash

python text_intent_classifer/infer.py \
    --text=This was a masterpiece. \
    --model_dir=models/slurp-intent_baseline-distilbert-base-uncased
