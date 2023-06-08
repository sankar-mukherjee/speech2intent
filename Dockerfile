FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV PYTHONPATH /workspace/speech2intent
WORKDIR /workspace/speech2intent

ADD requirements.txt .
RUN pip install -r requirements.txt

# Expose port 8000 for the FastAPI application
EXPOSE 8000

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
