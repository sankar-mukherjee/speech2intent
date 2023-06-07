# Speech to intent classification

Its built with FastAPI, pydantic, huggingface.

# Setup

Build the docker
> sh docker_build.sh

go inside docker
> docker run -it -v $PWD:/workspace/speech2intent/ speech2intent bash


# Details
To start the REST API server (main.py) run 
>  uvicorn main:app --reload

Go to http://localhost:8000/docs#/

There are 3 routes and input and expected output examples are give here. 
### Main API

```
/speech2intent

input
{
  "nlu_url": "cartesinus/slurp-intent_baseline-xlm_r-en",
  "speech_file_path": "calendar_entry.wav"
}

output
{
  "text": "MAKE A CALINBE IN TROFORTCOMODEL",
  "intent": "recommendation_locations"
}
```

### NLU API

```
/nlu/nlu_text_to_intent

input
{
  "nlu_url": "cartesinus/slurp-intent_baseline-xlm_r-en",
  "text_input": "play my favorite song from last year"
}

output
{
  "intent": "play_music"
}
```

### ASR API

```
/asr/run_asr

input
{
  "speech_file_path": "calendar_entry.wav"
}

output
{
  "transcript": "MAKE A CALINBE IN TROFORTCOMODEL"
}
```
# Models and Functionalities

huggingface models
* ASR: facebook/wav2vec2-base-960h
* NLU: cartesinus/slurp-intent_baseline-xlm_r-en
---
* ASR model was converted to onnx model and saved in models/wav2vec2-base-960h.onnx 
> python convert_torch_to_onnx.py --model=facebook/wav2vec2-base-960h


* When the server starts 
    * nlu model loaded cartesinus/slurp-intent_baseline-xlm_r-en
    * onnxruntime session was created with models/wav2vec2-base-960h.onnx

# Evaluation

Classifer performance is measured via the ‘headset’ subset of the SLURP test set.

* full
> python evaluation.py \
--slurp_testset_filepath=data/slurp_testset.jsonl \
--slurp_testset_audiodir=data/audio/slurp_real \
--nlu_url=ccoreilly/wav2vec2-large-100k-voxpopuli-catala

* random 10 samples
> python evaluation.py \
--slurp_testset_filepath=data/slurp_testset.jsonl \
--slurp_testset_audiodir=data/audio/slurp_real \
--nlu_url=ccoreilly/wav2vec2-large-100k-voxpopuli-catala \
--slurp_testset_no_examples=10

random 10 samples results
```
WER: 0.453
╒═════════════════════╤═════════════╤══════════╤═════════════╕
│ Intent (scen_act)   │   Precision │   Recall │   F-Measure │
╞═════════════════════╪═════════════╪══════════╪═════════════╡
│ OVERALL             │      0.4000 │   0.4000 │      0.4000 │
╘═════════════════════╧═════════════╧══════════╧═════════════╛ 
```

* Download SLURP data via **download_audio.sh**
* https://github.com/pswietojanski/slurp/blob/master/dataset/slurp/test.jsonl  --> data/slurp_testset.jsonl 
