FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV PYTHONPATH /workspace/vallex
WORKDIR /workspace/vallex

ADD requirements.txt .
RUN pip install -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
