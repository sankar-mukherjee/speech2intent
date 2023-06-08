import numpy as np
import onnxruntime as rt
import soundfile as sf
from fastapi import FastAPI
from transformers import (AutoTokenizer, Wav2Vec2Processor,
                          AutoModelForSequenceClassification)

from settings import (asr_input, nlu_input, nlu_model, speech2intent_input,
                      speech2intent_output)


def load_asr_session():
    """
    load the onnx model
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    ONNX_PATH = 'models/wav2vec2-base-960h.onnx'

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = rt.InferenceSession(ONNX_PATH, sess_options)

    print("onnxruntime session created with "+ ONNX_PATH)
    return processor, session

def load_nlu_session(nlu_model_url):
    """
    load the nlu model
    """
    tokenizer = AutoTokenizer.from_pretrained(nlu_model_url)
    model = AutoModelForSequenceClassification.from_pretrained(nlu_model_url)

    print("nlu model loaded " + nlu_model_url)
    return tokenizer, model

# preload the models
current_nlu_model_url = nlu_model().url
session_nlu_tokenizer, session_nlu_model = load_nlu_session(current_nlu_model_url)
asr_processor, asr_session = load_asr_session()

app = FastAPI()

# ASR API
@app.post("/asr/run_asr")
def run_asr(user_input: asr_input):
    speech_array, _ = sf.read(user_input.speech_file_path)

    features = asr_processor(speech_array, sampling_rate=16000, return_tensors="pt")
    input_values = features.input_values

    onnx_outputs = asr_session.run(None, {asr_session.get_inputs()[0].name: input_values.numpy()})[0]
    prediction = np.argmax(onnx_outputs, axis=-1)
    transcript = asr_processor.decode(prediction.squeeze().tolist())

    # print('run_asr finished')
    return {"transcript": transcript}

# NLU API
@app.post("/nlu/nlu_text_to_intent")
def nlu_text_to_intent(user_input: nlu_input):
    global session_nlu_tokenizer, session_nlu_model, current_nlu_model_url

    text_input = user_input.text_input
    nlu_model_url = user_input.nlu_url

    # Check if the nlu_model_url has changed
    if current_nlu_model_url != nlu_model_url:
        current_nlu_model_url = nlu_model_url
        session_nlu_tokenizer, session_nlu_model = load_nlu_session(nlu_model_url)

    inputs = session_nlu_tokenizer(text_input, return_tensors="pt")
    logits = session_nlu_model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    output = session_nlu_model.config.id2label[predicted_class_id]

    # print('nlu_text_to_intent finished')
    return {"intent": output}


# Main API
@app.post("/speech2intent", response_model=speech2intent_output)
def speech_to_intent(user_input: speech2intent_input):
    nlu_url = user_input.nlu_url
    speech_file_path = user_input.speech_file_path

    # Call ASR API
    asr_response = run_asr(asr_input(speech_file_path=speech_file_path))

    # Call NLU API
    nlu_response = nlu_text_to_intent(
            nlu_input(
            nlu_url=nlu_url,
            text_input=asr_response["transcript"]
            )
        )

    return {"text": asr_response["transcript"], "intent": nlu_response['intent']}

@app.get("/")
async def get_home():
    return {"message": "speech_to_intent classifier"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
