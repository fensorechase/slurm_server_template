from dotenv import load_dotenv
import os
import argparse
import logging
import csv
import pandas as pd
import torch
from torch import cuda, bfloat16
import transformers
from transformers import BitsAndBytesConfig

load_dotenv()
hf_auth = os.getenv("HF_AUTH_TOKEN")

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting script...")

# -----------------------------
# CLI Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Run Llama 3.1 70B Instruct QA inference")

parser.add_argument("--mode", type=str, required=True,
                    choices=["max_performance", "int8", "nf4", "fp4", "fp8"],
                    help="Quantization/precision mode.")

parser.add_argument("--temperature", type=float, default=0.1,
                    help="Generation temperature.")

parser.add_argument("--input_file", type=str, required=True,
                    help="Path to input CSV with questions.")

parser.add_argument("--output_file", type=str, required=True,
                    help="Path to output CSV for generated answers.")

parser.add_argument("--local_weights", type=str, default=None,
                    help="Path to local weights dir (if loading from cache). If None, will download from HuggingFace.")

parser.add_argument("--max_new_tokens", type=int, default=512,
                    help="Maximum number of new tokens to generate.")

args = parser.parse_args()

# -----------------------------
# Model Setup
# -----------------------------
model_id = 'meta-llama/Llama-3.1-70B-Instruct'
model_name = 'Meta_Llama_3_1_70B'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# -----------------------------
# Quantization Modes
# -----------------------------
bnb_config_map = {
    "max_performance": None,  # BF16 full precision
    "int8": BitsAndBytesConfig(load_in_8bit=True),
    "nf4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    ),
    "fp4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='fp4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    ),
    "fp8": None  # Will set torch_dtype for FP8 below
}

quant_config = bnb_config_map[args.mode]

# -----------------------------
# Model Load Path Logic
# -----------------------------
if args.local_weights:
    model_load_path = args.local_weights
    logger.info(f"Loading model from local path: {model_load_path}")
else:
    model_load_path = model_id
    logger.info(f"Downloading model from HuggingFace: {model_id}")

# -----------------------------
# Load Model Config
# -----------------------------
model_config = transformers.AutoConfig.from_pretrained(
    model_load_path,
    token=hf_auth
)

# -----------------------------
# Model Loading Based on Mode
# -----------------------------
if args.mode == "max_performance":
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_load_path,
        torch_dtype=bfloat16,
        config=model_config,
        device_map="auto",
        token=hf_auth
    )
elif args.mode == "fp8":
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_load_path,
        torch_dtype=torch.float8_e4m3fn,
        config=model_config,
        device_map="auto",
        token=hf_auth
    )
else:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_load_path,
        config=model_config,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_auth
    )

model.eval()
logger.info(f"Model loaded in {args.mode} mode.")

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_load_path,
    token=hf_auth
)
logger.info("Tokenizer loaded.")

# -----------------------------
# Load Input CSV
# -----------------------------
questions_df = pd.read_csv(args.input_file)
logger.info(f"Loaded input CSV: {args.input_file} with {len(questions_df)} rows.")

# -----------------------------
# Text Generation Pipeline
# -----------------------------
"""
Learn more about huggingface pipelines:
- You can use a prebuilt `pipeline` specifically for text generation, text classification, etc.

https://huggingface.co/docs/transformers/en/main_classes/pipelines
"""
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    temperature=args.temperature,
    max_new_tokens=args.max_new_tokens,
    do_sample=True
)
logger.info(f"Text generation pipeline ready (temp={args.temperature}).")

# -----------------------------
# Prepare Output CSV
# -----------------------------
output_headers = [
    'qid', 'qtype', 'unique_question_id', 'question_text',
    f'{model_name}_answer_vanilla', 'answer_text',
    'document_id', 'document_source', 'document_url', 'document_focus'
]
with open(args.output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(output_headers)

# -----------------------------
# Prompts
# -----------------------------
basic_system_prompt = (
    "You are a helpful assistant answering user questions. Your responses should be informative, concise, and clear."
)

# -----------------------------
# Process Questions
# -----------------------------
for _, row in questions_df.iterrows():
    original_question = row['question_text']
    logger.info(f"Processing question: {original_question}")

    messages = [
        {"role": "system", "content": basic_system_prompt},
        {"role": "user", "content": original_question}
    ]

    response = generate_text(messages)
    model_answer = response[0]["generated_text"].strip()

    output_row = [
        row['qid'],
        row['qtype'],
        row['unique_question_id'],
        row['question_text'],
        model_answer,
        row['answer_text'],
        row['document_id'],
        row['document_source'],
        row['document_url'],
        row['document_focus']
    ]

    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_row)

logger.info(f"Completed! Answers saved to {args.output_file}.")
