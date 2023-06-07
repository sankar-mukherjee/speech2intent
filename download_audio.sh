#!/bin/bash

mkdir -p data/audio

wget --help | grep -q '\--show-progress' && \
  PROGRESS_OPT="--show-progress --progress=bar:force" || PROGRESS_OPT=""

echo "Downloading slurp audio data..."
wget -c -q $PROGRESS_OPT \
     https://zenodo.org/record/4274930/files/slurp_real.tar.gz \
     -O data/audio/slurp_real.tar.gz 2>&1 | tee data/audio/slurp_real_download.log 

wget -c -q $PROGRESS_OPT \
     https://zenodo.org/record/4274930/files/slurp_synth.tar.gz \
     -O data/audio/slurp_synth.tar.gz 2>&1 | tee data/audio/slurp_synth_download.log

echo "Extracting packages to audio/slurp*"
tar -zxvf data/audio/slurp_real.tar.gz -C data/audio
tar -zxvf data/audio/slurp_synth.tar.gz -C data/audio
