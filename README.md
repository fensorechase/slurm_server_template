# GPU SLURM Server: Getting Started Running (Large) Language Models

Contains template code for setting up and running a Python project on a SLURM GPU server.

(Pre-requisite: Set up your GitHub account on the server.)

## Getting started

1. Clone this GitHub repo (or your GitHub repo) in your scratch space on the server. Your ```/local/scratch/<yourusername>``` space is where you should store all of your files and where you should download all cached files.

2. Set up a new virtual environment for your Python project:

* Ensure virtualenv is installed: ```pip install --user virtualenv```
* Create virtual environment: ```virtualenv -p python3 myenv```
* Then activate the virtualenv: ```source /myenv/bin/activate```
* Make sure you are installing items to your local scratch directory and not to the home directory -- You need to determine the absolute path where your venv directory is located, lets call it YOURVENVDIR. Edit the venv/bin/activate script with vim and add these lines at the end: export PATH=/usr/local/cuda/bin:$PATH export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:(specify your YOURVENVDIR)/venv/lib/python3.10/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
* Before starting any project, remember to install any packages needed to run your scripts:  ```pip install -r requirements.txt```
For example, you can also install these libraries:
* Install CUDA PyTorch wheels: ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```
If you use TensorFlow, install ```pip install tensorflow``` Please allow several minutes for the installation to complete.
  * Install tensorrt ```pip install tensorrt```
  * Install tensorrt-libs ```pip install --extra-index-url https://pypi.nvidia.com tensorrt-libs```
* To deactivate the virtual env, run ```deactivate```


3. Store your secrets/API keys/passwords in a local .env file -- remember to include this .env file in your .gitignore.

## Submitting your first batch job

1. Create a shell script (.sh script) -- a is a minimal starter template for a shell script can be found in: ```template.sh```

* Purpose of an .sh file: a replicable set of instruction to (i) set up your enviornment/ provision your computational resources, (ii) run multiple scripts in sequence.
* Parameters you may change in the .sh file, depending on your needs:

```python
#SBATCH --gres=gpu:2 # This dictates how many GPUs you are requesting -- either 1 or 2 (maximum 2)
#SBATCH --mem=80GB # This dictates how many GB memory you are requesting (between 40GB - 160) -- if less than 40GB do not use H100.

source myenv/bin/activate # Activates the virtual environment you should have created before running the .sh script. 
```

2. Choose a Python script you would like to run -- for example, ```hello_h100.py```
3. To run your first job on the H100 GPU, from the directory ```/local/scratch/<your_username>/<project_folder>``` which contains the ```template.sh``` script, we can to submit a batch job:

```bash
sbatch template.sh
```

* This will submit the job to be run. **You can check the progress of your job in two ways**:
* a. Run ```squeue``` from the terminal, which will show you the order of jobs from you and other users in the server queue -- this may look like:

```bash
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             50085      h100  somejob     usr76 PD       0:00      1 (QOSMaxGRESPerUser)
             50084      h100  somejob     usr76 PD       0:00      1 (QOSMaxGRESPerUser)
             50083      h100  somejob     usr76 PD       0:00      1 (QOSMaxGRESPerUser)
             50097      h100  somejob     usr76 PD       0:00      1 (QOSMaxGRESPerUser)
             50082      h100  somejob     usr76  R    3:54:44      1 h100server
             50096      h100  dif_job1    usr39  R    2:57:55      1 h100server
             50067      h100  dif_job2    usr39  R 1-01:36:00      1 h100server
```

* b. Run ```cat output_temp``` to view contents of the log file -- this will show any print or log statements from your Python script and batch script.


## Example LLM generation script

Now, let's try loading an LLM and running generation via a batch job! We'll use the pre-built script ```llm_example_generation_scripts/Meta-Llama-3-70B-Instruct.py```. Make sure to update your template.sh script to contain the command to run ```llm_example_generation_scripts/Meta-Llama-3-70B-Instruct.py``` -- for example

Example run: (max quality generation (BF16), loading from local cache):

```bash
python Meta-Llama-3-70B-Instruct.py \
  --mode max_performance \
  --temperature 0.1 \
  --input_file /local/scratch/cfensor/rag-llm/medquad_inputs/medquad_NIDDK_qa_dataset.csv \
  --output_file /local/scratch/cfensor/rag-llm/medquad_outputs/results.csv \
  --local_weights /local/scratch/cfensor/huggingface_cache/Meta_Llama_3_1_70B
```

Example run (memory-efficient NF4 mode, first time downloading weights from HF):

```bash
python Meta-Llama-3-70B-Instruct.py \
  --mode nf4 \
  --temperature 0.3 \
  --input_file my_questions.csv \
  --output_file answers_nf4.csv
```

### Argument options for this example LLM script:

**--mode** flag → chooses from 5 quantization/precision modes (performance → efficiency).
**--temperature** flag → controls generation randomness.
**--input_file** & --output_file flags → CSV paths for input & output.
**--local_weights** flag → loads weights from ```/local/scratch/cfensor/huggingface_cache/<dir>``` or downloads from HuggingFace if not set.
**--max_new_tokens** is set to 512 by default. You can adjust this to allow for longer answers.


### Model Loading Modes & Quantization Options

When running the script an NVIDIA H100, you can choose between **maximum quality** and **maximum efficiency** depending on your VRAM budget and performance needs.

---

#### 1. **Max Quality (Full Precision BF16)**  

**Best for:** Maximum accuracy, fully utilizing H100 Tensor Cores.

```python
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Best for H100 tensor cores
    device_map="auto"
)
```

Notes:

* Memory: ~140 GB (FP16) → ~80 GB (BF16)
* Pros: No quantization artifacts, best for research/benchmarking
* Cons: Requires multiple H100s or a single 80 GB H100

#### 2. High Performance + Lower Memory (8-bit)

Best for: High accuracy with ~40–50 GB VRAM usage.

```python
bnb_config = transformers.BitsAndBytesConfig(
    load_in_8bit=True
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Notes:

* Memory: ~40–50 GB
* Pros: Keeps accuracy close to BF16, reduced VRAM
* Cons: Slight latency increase

#### 3. Balanced Performance/Memory (4-bit NF4)

Best for: Excellent trade-off between speed, memory, and accuracy.

```python 
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Notes:

* Memory: ~20–24 GB
* Pros: Good accuracy for most use cases
* Cons: Slight drop in long-context performance

#### 4. Max Efficiency (4-bit FP4 or INT4)

Best for: Minimum VRAM usage (~18–20 GB).

```python
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='fp4',  # or 'int4'
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Notes:

* Memory: ~18–20 GB
* Pros: Runs on smaller VRAM GPUs
* Cons: More noticeable quality drop vs NF4

Recommendation:

* If you have 80 GB H100 → Use BF16 for best quality.
* If you have 48–64 GB VRAM → Use 8-bit mode.
* If you have <24 GB VRAM → Use 4-bit NF4 (or FP4/INT4 for even smaller VRAM).
