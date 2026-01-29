#!/bin/bash

# Safety: exit if anything fails
set -e

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate taubench-py312

# Choose which GPU this vLLM server uses
export CUDA_VISIBLE_DEVICES=1

# Optional but recommended on clusters
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_DISABLE_TELEMETRY=1

# Start vLLM server
vllm serve Qwen/Qwen3-14B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --port 8001
