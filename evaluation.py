import argparse
import random

import jiwer

from main import speech_to_intent
from modules.eval_metrics.metrics import ErrorMetric
from modules.util import format_results, load_gold_data
from settings import speech2intent_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slurp_testset_filepath",
        type=str,
        default="data/slurp_testset.jsonl",
    )
    parser.add_argument(
        "--slurp_testset_no_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--slurp_testset_audiodir",
        type=str,
        default="data/audio/slurp_real",
    )
    parser.add_argument(
        "--nlu_url",
        type=str,
        default="ccoreilly/wav2vec2-large-100k-voxpopuli-catala",
        help="Model HuggingFace ID",
    )
    args = parser.parse_args()

    slurp_testset_audiodir = args.slurp_testset_audiodir
    slurp_testset_filepath = args.slurp_testset_filepath
    slurp_testset_no_examples = args.slurp_testset_no_examples
    nlu_url_id = args.nlu_url

    examples = load_gold_data(slurp_testset_filepath)
    # take only 'headset' files
    headset_examples = {key: value for key, value in examples.items() if 'headset' in key}

    if slurp_testset_no_examples:
        headset_examples = dict(random.sample(headset_examples.items(), slurp_testset_no_examples))

    WER = []
    intent_f1 = ErrorMetric.get_instance(metric="f1")

    total_examples = len(headset_examples)
    for i, headset_id in enumerate(list(headset_examples)):
        headset_example = headset_examples.pop(headset_id)

        # dont covert .flac to .wav
        # audio_data, sample_rate = sf.read(gold_id)
        # sf.write(wav_path, audio_data, sample_rate)

        # speech2intent
        result = speech_to_intent(speech2intent_input(
            speech_file_path=slurp_testset_audiodir+'/'+headset_id,
            nlu_url=nlu_url_id))
        # WER
        WER.append(jiwer.wer(headset_example["text"].lower(), result['text'].lower()))
        # intent add
        intent_f1("{}_{}".format(headset_example["scenario"], headset_example["action"]) , result['intent'])

        print(str(i)+'/'+str(total_examples), end='\r')
        print('', end='', flush=True)

    print('WER: '+ str(round(sum(WER)/len(WER), 3)))

    results = intent_f1.get_metric()
    print(format_results(results=results, 
                         label="intent (scen_act)",
                         full=False), "\n")

    print('done')
    