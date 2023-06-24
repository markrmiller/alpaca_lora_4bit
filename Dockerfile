# syntax = docker/dockerfile:experimental

# Dockerfile is split into parts because we want to cache building the requirements and downloading the model, both of which can take a long time.

FROM nvidia/cuda:11.7.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip3 install --upgrade pip 

# Some of the requirements expect some python packages in their setup.py, just install them first.
RUN --mount=type=cache,target=/root/.cache/pip pip install --user torch==2.0.0
RUN --mount=type=cache,target=/root/.cache/pip pip install --user semantic-version==2.10.0 requests tqdm



#\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# The docker build environment has trouble detecting CUDA version, build for all reasonable archs
ENV TORCH_CUDA_ARCH_LIST="8.6" #you NEED to add your own GPU ARCH Number. https://stackoverflow.com/questions/68496906/pytorch-installation-for-different-cuda-architectures

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\



COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install --user -r requirements.txt

# -------------------------------

#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y git python3 python3-pip

RUN ln -s `which python3` /usr/bin/python


# Copy the installed packages from the first stage
COPY --from=builder /root/.local /root/.local

RUN mkdir alpaca_lora_4bit
WORKDIR alpaca_lora_4bit

COPY ./requirements.txt .

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/HazyResearch/flash-attention.git --no-build-isolation



COPY . .


# Run the server
WORKDIR /alpaca_lora_4bit


ENTRYPOINT python finetune.py ./models/filtered.json \
    --ds_type=txt \
    --lora_out_dir=./test/ \
    --llama_q4_config_dir=./models/airoboros-7b-4bit-128g/ \
    --llama_q4_model=./models/airoboros-7b-4bit-128g/airoboros-7b-gpt4-1.4-GPTQ-4bit-128g.no-act.order.safetensors \
    --mbatch_size=1 \
    --batch_size=1 \
    --epochs=1 \
    --lr=3e-4 \
    --cutoff_len=4192 \
    --lora_r=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --warmup_steps=5 \
    --save_steps=100 \
    --save_total_limit=3 \
    --logging_steps=5 \
    --groupsize=128 \
    --xformers \
    --backend=cuda \
    --grad_chckpt \
    --val_set_size=19238 \
    --ds_type=alpaca
