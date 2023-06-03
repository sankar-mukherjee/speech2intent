from fastapi import FastAPI, Form
from transformers import Wav2Vec2Processor
import onnxruntime as rt
import soundfile as sf
import numpy as np
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
from pydantic import BaseModel
import uvicorn

# custom types
class speech2intent_input(BaseModel):
    nlu_url: str = 'cartesinus/slurp-intent_baseline-xlm_r-en'
    speech_file_path: str

class asr_input(BaseModel):
    speech_file_path: str

class nlu_input(BaseModel):
    nlu_model_path: str = 'cartesinus/slurp-intent_baseline-xlm_r-en'
    text_input: str

    def get_model_path():
        return 'cartesinus/slurp-intent_baseline-xlm_r-en'

class speech2intent_output(BaseModel):
    text: str
    intent: str


def load_asr_session():
    """
    load the onnxruntime
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    ONNX_PATH = 'models/wav2vec2-base-960h.onnx'

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = rt.InferenceSession(ONNX_PATH, sess_options)

    print("onnxruntime session created")
    return processor, session

def load_nlu_session(nlu_model_path):
    """
    load the nlu model
    """
    tokenizer = AutoTokenizer.from_pretrained(nlu_model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(nlu_model_path)

    print("nlu model loaded")
    return tokenizer, model

global nlu_tokenizer, nlu_model

nlu_tokenizer, nlu_model = load_nlu_session(nlu_input.get_model_path())
asr_processor, asr_session = load_asr_session()

app = FastAPI()

# ASR API
@app.post("/asr/run_asr")
def run_asr(user_input: asr_input):
    print(user_input)
    speech_array, _ = sf.read(user_input.speech_file_path)

    features = asr_processor(speech_array, sampling_rate=16000, return_tensors="pt")
    input_values = features.input_values

    onnx_outputs = asr_session.run(None, {asr_session.get_inputs()[0].name: input_values.numpy()})[0]
    prediction = np.argmax(onnx_outputs, axis=-1)
    transcript = asr_processor.decode(prediction.squeeze().tolist())

    print('run_asr finished')
    return {"transcript": transcript}

def check_if_exists(nlu_model_path, nlu_tokenizer, nlu_model):
    if nlu_model_path != nlu_input.nlu_model_path:
        nlu_tokenizer, nlu_model = load_nlu_session(nlu_model_path)
    return nlu_tokenizer, nlu_model


# NLU API
@app.post("/nlu/nlu_text_to_intent")
def nlu_text_to_intent(user_input: nlu_input):
    text_input = user_input.text_input
    nlu_model_path = user_input.nlu_model_path

    # new nlu model
    nlu_tokenizer, nlu_model = check_if_exists(nlu_model_path, nlu_tokenizer, nlu_model)

    inputs = nlu_tokenizer(text_input, return_tensors="pt")
    logits = nlu_model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    output = nlu_model.config.id2label[predicted_class_id]

    print('nlu_text_to_intent finished')
    return {"intent": output}


# Main API
@app.post("/speech2intent", response_model=speech2intent_output)
def speech_to_intent(user_input: speech2intent_input):
    nlu_url = user_input.nlu_url
    speech_file_path = user_input.speech_file_path

    # Call ASR API
    a = asr_input
    setattr(a, 'speech_file_path', speech_file_path)
    asr_response = run_asr(a)

    # Call NLU API
    setattr(nlu_input, 'nlu_model_path', nlu_url)
    setattr(nlu_input, 'text_input', asr_response["transcript"])
    nlu_response = nlu_text_to_intent(nlu_input)

    return {"text": asr_response["transcript"], "intent": nlu_response['intent']}

@app.get("/")
async def get_home():
    return {"message": "speech_to_intent"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
