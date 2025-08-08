#!/bin/bash
#SBATCH --job-name=output_temp
#SBATCH --output=output_temp
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB

# Store the following variables so downloads do not go to the home directory.
# Make sure to change <your_username> to your actual username on the server.
export TRANSFORMERS_CACHE="/local/scratch/<your_username>/huggingface"
export HF_HOME="/local/scratch/<your_username>/hf_cache"
export HF_DATASETS_CACHE="/local/scratch/<your_username>/hf_datasets"
export PIP_CACHE_DIR="/local/scratch/<your_username>/pip_cache"
export TORCH_HOME="/local/scratch/<your_username>/torch_cache"
export PYTORCH_HOME="/local/scratch/<your_username>/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set up environment
export PYTHONPATH=$(pwd):$PYTHONPATH

source myenv/bin/activate
pwd

# Source the .env file to load environment variables
source .env

echo "Running Python script..."

# A minimal example of running a Python script:
python hello_h100.py

# Uncomment the following 5 lines to run a low-memory version of the Meta-Llama-3-70B-Instruct script:
# python llm_example_generation_scripts/Meta-Llama-3-70B-Instruct.py \
#  --mode nf4 \
#  --temperature 0.3 \
#  --input_file my_questions.csv \
#  --output_file answers_nf4.csv

echo "Done!"