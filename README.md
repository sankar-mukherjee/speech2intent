# Speech to intent classification

Its built with FastAPI, pydantic, huggingface.

# Setup

Build the docker
> sh scripts/docker_build.sh

Run
> docker run -p 8008:8000 -d speech2intent:latest

wait for couple of minutes and then go to http://127.0.0.1:8008/docs#/


Alternatively you can run this command to go inside container 
> docker run -p 8008:8000 -it -v $PWD:/workspace/speech2intent/ speech2intent:latest bash

and then run 
> uvicorn main:app --reload


# Details

There are 3 routes and input and expected output examples are given here. 
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

When starting the server for the first time:

* ASR model ```facebook/wav2vec2-base-960h``` was downloaded and converted to onnx model and saved in ```models/wav2vec2-base-960h.onnx ```
> python convert_torch_to_onnx.py --model=facebook/wav2vec2-base-960h

* onnxruntime session was created with ```models/wav2vec2-base-960h.onnx```
* nlu model loaded ```cartesinus/slurp-intent_baseline-xlm_r-en```

#### NOTE: ```"speech_file_path"```: in the all two APIs can only take wav files which are mounted with the container.

# Evaluation


Classifer performance is measured via the ‘headset’ subset of the SLURP test set.

* Download SLURP data via 
> sh scripts/download_audio.sh 

This will create a ```audio``` folder inside ```data``` folder.

* copy https://github.com/pswietojanski/slurp/blob/master/dataset/slurp/test.jsonl  to  ```data/slurp_testset.jsonl``` 

* Evaluate random 10 samples from ```data/slurp_testset.jsonl```
> sh scripts/evauation.sh

* For full dataset evaluation change  ```${slurp_testset_no_examples:=None}``` in ```scripts/evauation.sh```


* random 10 samples evaluation results
```
WER: 0.453
╒═════════════════════╤═════════════╤══════════╤═════════════╕
│ Intent (scen_act)   │   Precision │   Recall │   F-Measure │
╞═════════════════════╪═════════════╪══════════╪═════════════╡
│ OVERALL             │      0.4000 │   0.4000 │      0.4000 │
╘═════════════════════╧═════════════╧══════════╧═════════════╛ 
```

# Finetune New Model on SLURP dataset

A fintuned model  
> sankar1535/slurp-intent_baseline-distilbert-base-uncased

was created via finuning pretrained huggingface model

> distilbert-base-uncased

train and infer Command. 
> sh scripts/train_text_intent_classifer.sh

> sh scripts/infer_text_intent_classifer.sh

Traning data is taken from https://github.com/pswietojanski/slurp/blob/master/dataset/slurp/
