# LLM-finetuning

Various strategies are presented to fine-tune LLMs, using CPT, SFT and DPO/ORPO.

## Step 1: Install Hugging Face Alignment Handbook code

```
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

### Step 2: Train models

Each of the folders include YAML configs for the individual models. From each directory, the whole training series is executed like so:

```
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_cpt.py config_full_CPT-Base.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_sft.py config_full_SFT-Base.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_orpo.py config_full_ORPO-Base.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_dpo.py config_full_DPO-Base.yaml

ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_cpt.py config_full_CPT-Instruct.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_sft.py config_full_SFT-Instruct.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_orpo.py config_full_ORPO-Instruct.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_dpo.py config_full_DPO-Instruct.yaml
```
Check to make sure the input and output files in each of the YAML files are correct. E.g., the CPT run should produce the input to the SFT run, and the SFT run should produce the input to ORPO/DPO. 


### Step 3 (optional): Model Merging 

```
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .  # install the package and make scripts available
```

```python
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random

from tqdm.notebook import tqdm
!wandb login <WANB_TOKEN>
token = 'hf_XXXXXXXXXXXXXXXXXXXXX'
from huggingface_hub import login
login(token=token)
import torch
torch.cuda.get_device_name(), torch.cuda.device_count()
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)
    
from accelerate import infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import load_dataset
from datasets import IterableDataset

from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from transformers import AutoConfig
from transformers import BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import multiprocessing
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

device = torch.device("cuda")
torch.version.cuda, torch.__version__, torch.backends.cudnn.version(), torch.backends.cudnn.enabled, torch.cuda.is_available()

def load_model (model_name='lamm-mit/llama-3-1-base-bio-V20_SFT_ORPO', chat_template=None,
                #compile_mode="max-autotune", 
                compile_mode=None,
                attn_implementation="flash_attention_2",
                quant=False):
     
    bnb_config4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        #bnb_4bit_use_double_quant=True,
    )

    bnb_config4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if quant==False:
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # revision='1a02ed84a2b3525452f7cffd0c1002cc657bd3e9', #epoch 1
            trust_remote_code=True,
            use_cache=False,
            device_map="auto",
            torch_dtype =torch.bfloat16,
            attn_implementation=attn_implementation,
        ) 
        #model.config.use_cache = False
    else:
        print ("Use quantized model.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # revision='xxxxxxxxxxxxxxxxxxxxx', 
            trust_remote_code=True,
            use_cache=False,
            quantization_config =bnb_config4bit,
            device_map="auto",
            torch_dtype =torch.bfloat16,
            attn_implementation=attn_implementation,
        )

    if compile_mode != None: 
        model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(model.forward, mode=compile_mode, fullgraph=True )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_bos_token=False
                                             )
    if chat_template != None:
        print("Set custom chat template.")
        tokenizer.chat_template=chat_template

    eos_token=tokenizer.eos_token_id
    eos_token 
    return model, tokenizer

try:
    del model
    del tokenizer
except:
    print ()
    
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_SLERP_and_save (source='lamm-mit/mistral-7B-v0.3-Base-CPT', 
                    base_model='lamm-mit/mistralai-Mistral-7B-Instruct-v0.3',
                    upload_to_hub=False, private=True,
                    attn_filter = [0, 0.5, 0.3, 0.7, 1],
                    mlp_filter  = [1, 0.5, 0.7, 0.3, 0],
                    hub_id='lamm-mit',modifier='',
                       ):


    source_base = source.split('/')[-1]
    MODEL_NAME = source_base + f"_SLERP{modifier}"
     
    yaml_config = f"""
    slices:
      - sources:
          - model: {source}
            layer_range: [0, 32]
          - model: {base_model}
            layer_range: [0, 32]
    merge_method: slerp
    base_model: {base_model}
    parameters:
      t:
        - filter: self_attn
          value: {attn_filter}
        - filter: mlp
          value: {mlp_filter}
        - value: 0.5
    dtype: bfloat16
    
    """
    
    # Save config as yaml file
    with open('config.yaml', 'w', encoding="utf-8") as f:
        f.write(yaml_config)
         
    !mergekit-yaml --cuda --copy-tokenizer --allow-crimes --out-shard-size 5B --lazy-unpickle config.yaml {MODEL_NAME}

    if upload_to_hub:
        print ("Pushing to hub...")
        model, tokenizer,=load_model (model_name=MODEL_NAME)
        
        tokenizer.push_to_hub (hub_id+'/'+MODEL_NAME, private=private)
        model.push_to_hub (hub_id+'/'+MODEL_NAME, private=private)
    
        try:
            del model
            del tokenizer
        except:
            print ()

        print ("HF hub model name: ", hub_id+'/'+MODEL_NAME)
            
    return MODEL_NAME
```

Then, you can merge in this way:

```python
model_name=merge_SLERP_and_save (source='lamm-mit/mistral-7B-v0.3-Instruct-CPT', 
                    base_model='lamm-mit/mistralai-Mistral-7B-Instruct-v0.3',
                    upload_to_hub=True, #modifier='_var_H',
                    attn_filter =[0, 0.5, 0.3, 0.7, 1],
                    mlp_filter= [1, 0.5, 0.7, 0.3, 0],
                    #attn_filter =[0, 0.3, 0.5, 0.7, 1],
                    #mlp_filter= [1, 0.7, 0.5, 0.3, 0],
                       )
#If you need to manually copy/push tokenizer, e.g. for Mistral models 
tokenizer = AutoTokenizer.from_pretrained('lamm-mit/mistralai-Mistral-7B-Instruct-v0.3', trust_remote_code=True, add_bos_token=False)
tokenizer.push_to_hub ('lamm-mit/'+model_name, private=True)
```
