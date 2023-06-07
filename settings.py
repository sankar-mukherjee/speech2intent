from pydantic import BaseModel, FilePath, validator


class asr_input(BaseModel):
    speech_file_path: FilePath

    @validator('speech_file_path')
    def check_file_extension(cls, file_path):
        allowed_extensions = ['.wav', '.mp3', '.flac']
        if not str(file_path).lower().endswith(tuple(allowed_extensions)):
            raise ValueError(f"Unsupported file extension. Only {', '.join(allowed_extensions)} files are allowed.")
        return file_path

class nlu_model(BaseModel):
    url: str = 'cartesinus/slurp-intent_baseline-xlm_r-en'

class nlu_input(BaseModel):
    nlu_url: str = nlu_model().url
    text_input: str

class speech2intent_input(BaseModel):
    nlu_url: str = nlu_model().url
    speech_file_path: str

class speech2intent_output(BaseModel):
    text: str
    intent: str
