# syntax = docker/dockerfile:experimental

# Dockerfile is split into parts because we want to cache building the requirements and downloading the model, both of which can take a long time.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip3 install --upgrade pip 

# Some of the requirements expect some python packages in their setup.py, just install them first.
RUN --mount=type=cache,target=/root/.cache/pip pip install --user torch==2.0.0
RUN --mount=type=cache,target=/root/.cache/pip pip install --user semantic-version==2.10.0 requests tqdm

# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# The docker build environment has trouble detecting CUDA version, build for all reasonable archs
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6" 
# you NEED to add your own GPU ARCH Number. https://developer.nvidia.com/cuda-gpus
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\



COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install --user -r requirements.txt

# -------------------------------

#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y git python3 python3-pip

RUN ln -s `which python3` /usr/bin/python

RUN apt install -y wget

# Copy the installed packages from the first stage
COPY --from=builder /root/.local /root/.local

RUN mkdir alpaca_lora_4bit
WORKDIR alpaca_lora_4bit

COPY ./requirements.txt .

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/HazyResearch/flash-attention.git --no-build-isolation

#for the backups
RUN pip install dropbox

COPY . .
#RUN mkdir models/output

# Run the server
WORKDIR /alpaca_lora_4bit

ENTRYPOINT ["python", "finetune.py", "./models/Data.json"]
           
CMD [	    "--lora_out_dir=./output/", \
            "--llama_q4_config_dir=./models/llama-7b-4bit-128g/", \
            "--llama_q4_model=llama7b-gptq-4bit-128g.safetensors", \
            "--mbatch_size=1", \
            "--batch_size=1", \
            "--epochs=3", \
            "--lr=6e-4", \
            "--cutoff_len=4096", \
            "--lora_r=2", \
            "--lora_alpha=8", \
            "--lora_dropout=0", \
            "--warmup_steps=5", \
            "--save_steps=500", \
            "--save_total_limit=1", \
            "--logging_steps=5", \
            "--groupsize=128", \
            "--xformers", \
            "--backend=triton", \
            "--grad_chckpt", \
            "--val_set_size=7853", \
            "--ds_type=alpaca", \
            "--weight_decay=0.1", \
            "--adam_beta1=0.9", \
            "--adam_beta2=0.99", \
            "--adam_epsilon=1e-5", \
            "--verbose", \
            "--local_rank=1"]''', \
            "--resume_checkpoint=/models/llama-7b-4bit-128g/"] #cant get resume to work, maybe i just load lora every time? but does it resume?
